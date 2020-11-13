"""Train CLRS"""
#coding=utf-8
import cv2
import os
import sys
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from gluoncv import utils as gutils
from gluoncv.data.batchify import Tuple, Stack, Pad 
from gluoncv.utils.metrics.voc_detection import VOC07MApMetric

sys.path.append(os.path.expanduser('~/gluon-ocr'))
from gluonocr.model_zoo import get_clrs
from gluonocr.data import CLRSDataset, CLRSTrainTransform
from gluonocr.data import PointAugmenter
from gluonocr.loss import CLRSLoss
from config import args

gutils.random.seed(args.seed)

class Trainer(object):
    def __init__(self):
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
        self.ctx = ctx if ctx != [] else [mx.cpu()]
        if args.syncbn and len(self.ctx) > 1:
            self.net = get_clrs(args.network, args.num_layers, pretrained_base=True,
                                norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                norm_kwargs={'num_devices': len(self.ctx)})
            self.async_net = get_clrs(args.network, args.num_layers, pretrained_base=False)  # used by cpu worker
        else:
            self.net = get_clrs(args.network, args.num_layers, pretrained_base=True) 
            self.async_net = self.net
        model_name = '%s-%s%d-clrs'%(args.dataset_name, args.network, args.num_layers)
        
        if not os.path.exists(args.save_prefix):
            os.mkdir(args.save_prefix)
        args.save_prefix += model_name
        if args.export_model:
            self.export_model()
        self.init_model()
        self.net.collect_params().reset_ctx(self.ctx)

        self.train_dataloader, self.val_dataloader, self.val_metric = self.get_dataloader()
        self.loss = CLRSLoss()
        self.sum_loss = mx.metric.Loss('SumLoss')
        self.cls_loss = mx.metric.Loss('SoftmaxLoss')
        self.box_loss = mx.metric.Loss('SmothL1Loss')
        self.seg_loss = mx.metric.Loss('DiceLoss')

    def init_model(self):
        if args.resume.strip():
            self.net.load_parameters(args.resume.strip())
            self.async_net.load_parameters(args.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize(init=mx.init.Xavier())
                self.async_net.initialize(init=mx.init.Xavier())

    def get_dataloader(self):
        augment = PointAugmenter()
        with mx.autograd.train_mode():
            inp = mx.nd.zeros((1, 3, args.data_shape, args.data_shape), self.ctx[0])
            _, _, anchors, _ = self.net(inp)
        anchors = anchors.as_in_context(mx.cpu())
        tg_fn   = CLRSTrainTransform(anchors, negative_mining_ratio=-1)
        train_dataset = CLRSDataset(args.train_img_dir, 
                                    args.train_lab_dir,
                                    augment, mode='train', 
                                    img_size=(args.data_shape, args.data_shape))
        val_dataset  = CLRSDataset(args.train_img_dir, 
                                    args.train_lab_dir, 
                                    mode='val',
                                    img_size=(args.data_shape, args.data_shape))
        val_metric = VOC07MApMetric(iou_thresh=0.5, class_names=val_dataset.classes)
        if args.num_samples < 0:
            args.num_samples = len(train_dataset)
        train_dataloader = DataLoader(train_dataset.transform(tg_fn), batch_size=args.batch_size, 
                                      last_batch='discard', shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=True)
        val_batchify_fn = Tuple(Stack(), Pad(pad_val=-1), Stack(), Stack())
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size//len(self.ctx),
                                    batchify_fn=val_batchify_fn, pin_memory=True,
                                    num_workers=args.num_workers, last_batch='keep')
        return train_dataloader, val_dataloader, val_metric

    def export_model(self):
        self.net.export_block(args.save_prefix, args.resume.strip(), self.ctx)
        sys.exit()

    def train(self):
        if args.lr_decay_period > 0:
            lr_decay_epoch = list(range(args.lr_decay_period, args.epochs, args.lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
        lr_decay_epoch = [e - args.warmup_epochs for e in lr_decay_epoch]
        num_batches = args.num_samples // args.batch_size
        lr_scheduler = gutils.LRSequential([
            gutils.LRScheduler('linear', base_lr=0, target_lr=args.lr,
                        nepochs=args.warmup_epochs, iters_per_epoch=num_batches),
            gutils.LRScheduler(args.lr_mode, base_lr=args.lr,
                        nepochs=args.epochs - args.warmup_epochs,
                        iters_per_epoch=num_batches,
                        step_epoch=lr_decay_epoch,
                        step_factor=args.lr_decay, power=2),
            ])

        trainer = gluon.Trainer(self.net.collect_params(), 'sgd',
            {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler}) #

        # set up logger
        logging.basicConfig()
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        log_file_path = args.save_prefix + '_train.log'
        log_dir = os.path.dirname(log_file_path)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        fh = logging.FileHandler(log_file_path)
        logger.addHandler(fh)
        logger.info(args)

        logger.info('Start training from [Epoch {}]'.format(args.start_epoch))
        best_loss, best_map = 1000, 0
        for epoch in range(args.start_epoch, args.epochs):
            tic = time.time()
            btic = time.time()
            self.net.hybridize()
            for i, batch in enumerate(self.train_dataloader):
                data     = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx)
                cls_targ = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx)
                box_targ = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx)
                box_mask = gluon.utils.split_and_load(batch[3], ctx_list=self.ctx)
                seg_gt   = gluon.utils.split_and_load(batch[4], ctx_list=self.ctx)
                mask     = gluon.utils.split_and_load(batch[5], ctx_list=self.ctx)
                sum_losses, cls_losses, box_losses, seg_losses = [], [], [], []
                with mx.autograd.record():
                    for d, ct, bt, bm, sg, m in zip(data, cls_targ, box_targ, box_mask, seg_gt, mask):
                        cls_pred, box_pred, _, seg_pred = self.net(d)
                        pred = {'cls_pred':cls_pred, 'box_pred':box_pred, 'seg_pred':seg_pred}
                        lab  = {'cls_targ':ct, 'box_targ':bt, 'box_mask':bm, 'seg_gt':sg, 'mask':m}
                        loss, metrics = self.loss(pred, lab)
                        sum_losses.append(loss)
                        cls_losses.append(metrics['cls_loss'])
                        box_losses.append(metrics['box_loss'])
                        seg_losses.append(metrics['seg_loss'])
                    mx.autograd.backward(sum_losses)
                trainer.step(1)
                self.sum_loss.update(0, sum_losses)
                self.cls_loss.update(0, cls_losses)
                self.box_loss.update(0, box_losses)
                self.seg_loss.update(0, seg_losses)
                
                if args.log_interval and not (i + 1) % args.log_interval:
                    name0, loss0 = self.sum_loss.get()
                    name1, loss1 = self.cls_loss.get()
                    name2, loss2 = self.box_loss.get()
                    name3, loss3 = self.seg_loss.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i+1, trainer.learning_rate, args.batch_size/(time.time()-btic), name0, loss0, name1, loss1, name2, loss2,  name3, loss3))
                btic = time.time()
            name0, loss0 = self.sum_loss.get()
            name1, loss1 = self.cls_loss.get()
            name2, loss2 = self.box_loss.get()
            name3, loss3 = self.seg_loss.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, time.time()-tic, name0, loss0, name1, loss1, name2, loss2, name3, loss3))

            
            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                mean_ap, seg_loss = self.validate(epoch, logger)
                if mean_ap[-1] > best_map and seg_loss < best_loss:
                    best_map, best_loss = mean_ap[-1], seg_loss
                    self.net.save_parameters('{:s}_best.params'.format(args.save_prefix))
                if args.save_interval and (epoch+1) % args.save_interval == 0:
                    self.net.save_parameters('{:s}_{:04d}_{:.3f}.params'.format(args.save_prefix, epoch+1, best_map))

    def validate(self, epoch, logger):
        if self.val_dataloader is None:
            return 0
        logger.info('Start validate.')
        self.val_metric.reset()
        self.seg_loss.reset()
        # set nms threshold and topk constraint
        self.net.set_nms(nms_thresh=0.45, nms_topk=1000, post_nms=400)
        self.net.hybridize()
        tic = time.time()
        for batch in self.val_dataloader:
            data   = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
            label  = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx, batch_axis=0)
            seg_gt = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx, batch_axis=0)
            mask   = gluon.utils.split_and_load(batch[3], ctx_list=self.ctx, batch_axis=0)
            det_bboxes = []
            det_ids = []
            det_scores = []
            gt_bboxes = []
            gt_ids = []
            gt_difficults = []
            for x, y, seg, m in zip(data, label, seg_gt, mask):
                ids, scores, bboxes, seg_maps = self.net(x)
                det_ids.append(ids)
                det_scores.append(scores)
                # clip to image size
                det_bboxes.append(bboxes.clip(0, batch[0].shape[2]))
                # split ground truths
                gt_ids.append(y.slice_axis(axis=-1, begin=4, end=5))
                gt_bboxes.append(y.slice_axis(axis=-1, begin=0, end=4))
                gt_difficults.append(y.slice_axis(axis=-1, begin=5, end=6) if y.shape[-1] > 5 else None)
                loss = self.loss.seg_loss(seg_maps, seg, m)
                self.seg_loss.update(0, loss)
                #self.visualize(ids, scores, bboxes, seg_maps)
            # update metric
            self.val_metric.update(det_bboxes, det_ids, det_scores, gt_bboxes, gt_ids, gt_difficults)
            
        name3, loss3 = self.seg_loss.get()
        map_name, mean_ap = self.val_metric.get()
        val_msg = '\n'.join(['{}={}'.format(k, v) for k, v in zip(map_name, mean_ap)])
        logger.info('[Epoch {}] Validation: \n{}'.format(epoch, val_msg))
        logger.info('{}={:.3f}'.format(name3, loss3))
        self.seg_loss.reset()
        return mean_ap, loss3
    
    def visualize(self, ids, scores, boxes, seg_maps):
        boxes = boxes.asnumpy()[0]
        seg_maps = seg_maps.asnumpy()[0]
        ids      = ids.asnumpy()[0, :, 0]
        scores   = scores.asnumpy()[0, :, 0]
        _, h, w  = seg_maps.shape
        bina_map = np.zeros((h, w, 3), dtype=np.uint8)
        bina_map[seg_maps[0]>0.6, :] = [255, 255, 255]
        bina_map[seg_maps[1]>0.6, :] = [255, 0, 0]
        bina_map[seg_maps[2]>0.6, :] = [0, 255, 0]
        bina_map[seg_maps[3]>0.6, :] = [0, 0, 255]
        
        for i, s, box in zip(ids, scores, boxes):
            if i < 0 or s< 0.3:
                continue
            box = np.array(box).astype(np.int32)
            if i == 0:
                cv2.rectangle(bina_map, (box[0], box[1]), (box[2], box[3]), (255, 255, 255), 2)
            elif i == 1:
                cv2.rectangle(bina_map, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
            elif i == 2:
                cv2.rectangle(bina_map, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
            else:
                cv2.rectangle(bina_map, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imwrite('pred_res.jpg', bina_map)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
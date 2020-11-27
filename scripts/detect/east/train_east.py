#coding=utf-8
import cv2
import os
import sys
import logging
import warnings
import time
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from gluoncv import utils as gutils 

sys.path.append(os.path.expanduser('~/gluon-ocr'))
from gluonocr.model_zoo import get_east
from gluonocr.data import EASTDataset 
from gluonocr.data import PointAugmenter
from gluonocr.loss import EASTLoss
from config import args

gutils.random.seed(args.seed)

class Trainer(object):
    def __init__(self):
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
        self.ctx = ctx if ctx != 0 else [mx.cpu()]
        if args.syncbn and len(self.ctx) > 1:
            self.net = get_east(args.network, args.num_layers, pretrained_base=True,
                                norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                norm_kwargs={'num_devices': len(self.ctx)})
            self.async_net = get_east(args.network, args.num_layers, pretrained_base=False)  # used by cpu worker
        else:
            self.net = get_east(args.network, args.num_layers, pretrained_base=True) 
            self.async_net = self.net

        model_name = '%s-%s%d-east'%(args.dataset_name, args.network, args.num_layers)
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        if not os.path.exists(args.save_prefix):
            os.mkdir(args.save_prefix)
        args.save_prefix += model_name
        if args.export_model:
            self.export_model()
        self.init_model()
        self.net.collect_params().reset_ctx(self.ctx)
        self.loss = EASTLoss(lambd=2.0)
        self.sum_loss  = mx.metric.Loss('SumLoss')
        self.l1_loss   = mx.metric.Loss('SmoothL1Loss')
        self.bce_loss  = mx.metric.Loss('BalanceCELoss')

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
        train_dataset = EASTDataset(args.train_img_dir, 
                                    args.train_lab_dir,
                                    augment, mode='train',
                                    img_size=(args.data_shape, args.data_shape))
        val_dataset  = EASTDataset(args.train_img_dir, 
                                    args.train_lab_dir, mode='val',
                                    img_size=(args.data_shape, args.data_shape))
        args.num_samples = len(train_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                      last_batch='discard', shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=False)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, 
                                    num_workers=args.num_workers, last_batch='keep')
        return train_dataloader, val_dataloader

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
        best_loss = 1000

        for epoch in range(args.start_epoch, args.epochs):
            tic = time.time()
            btic = time.time()
            self.net.hybridize()
            for i, batch in enumerate(self.train_dataloader):
                data  = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx)
                score = gluon.utils.split_and_load(batch[1], ctx_list=self.ctx)
                mask  = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx)
                geo_map = gluon.utils.split_and_load(batch[3], ctx_list=self.ctx)
                sum_losses, bce_losses, l1_losses = [], [], []
                with mx.autograd.record():
                    for d, s, m, gm in zip(data, score, mask, geo_map):
                        pred_score, pred_geo = self.net(d)
                        pred = {'score':pred_score, 'geo_map':pred_geo}
                        lab  = {'gt':s, 'mask':m, 'geo_map':gm}
                        loss, metric = self.loss(pred, lab)
                        sum_losses.append(loss)
                        bce_losses.append(metric['bce_loss'])
                        l1_losses.append(metric['l1_loss'])
                    mx.autograd.backward(sum_losses)
                trainer.step(1)
                #mx.nd.waitall()
                self.sum_loss.update(0, sum_losses)
                self.l1_loss.update(0, l1_losses)
                self.bce_loss.update(0, bce_losses)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name0, loss0 = self.sum_loss.get()
                    name1, loss1 = self.l1_loss.get()
                    name2, loss2 = self.bce_loss.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i+1, trainer.learning_rate, args.batch_size/(time.time()-btic), name0, loss0, name1, loss1, name2, loss2))
                btic = time.time()
            name0, loss0 = self.sum_loss.get()
            name1, loss1 = self.l1_loss.get()
            name2, loss2 = self.bce_loss.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, time.time()-tic, name0, loss0, name1, loss1, name2, loss2))

            if not (epoch + 1) % args.val_interval:
                # consider reduce the frequency of validation to save time
                mean_loss = self.validate(logger)
                if mean_loss < best_loss:
                    best_loss = mean_loss
                    self.net.save_parameters('{:s}_best.params'.format(args.save_prefix))
                if args.save_interval and (epoch+1) % args.save_interval == 0:
                    self.net.save_parameters('{:s}_{:04d}_{:.3f}.params'.format(args.save_prefix, epoch+1, mean_loss))

    def validate(self, logger):
        if self.val_dataloader is None:
            return 0
        logger.info('Start validate.')
        self.sum_loss.reset()
        self.l1_loss.reset()
        self.bce_loss.reset()
        tic = time.time()
        for batch in self.val_dataloader:
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
            labs = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 4)]
            for it, x in enumerate(data):
                score, geo_map = self.net(x)
                pred = {'score':score, 'geo_map':geo_map}
                lab  = {'gt':labs[0][it], 'mask':labs[1][it], 'geo_map':labs[2][it]}
                loss, metric = self.loss(pred, lab)
                self.sum_loss.update(0, loss)
                self.l1_loss.update(0, metric['l1_loss'])
                self.bce_loss.update(0, metric['bce_loss'])
        
        name0, loss0 = self.sum_loss.get()
        name1, loss1 = self.l1_loss.get()
        name2, loss2 = self.bce_loss.get()
        
        logger.info('Evaling cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    time.time()-tic, name0, loss0, name1, loss1, name2, loss2))
        self.sum_loss.reset()
        self.l1_loss.reset()
        self.bce_loss.reset()
        return loss0

    def export_model(self):
        self.net.export_block(args.save_prefix, args.resume.strip(), self.ctx)
        sys.exit()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
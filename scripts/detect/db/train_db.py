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

sys.path.append(os.path.expanduser('~/demo/gluon-ocr'))
from gluonocr.model_zoo import get_db
from gluonocr.data import DBDataset 
from gluonocr.data import PointAugmenter
from gluonocr.loss import DBLoss
from gluonocr.post_process import DBPostProcess
from gluonocr.utils.detect_metric import DetectionIoUEvaluator
from config import args

gutils.random.seed(args.seed)

class Trainer(object):
    def __init__(self):
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',')]
        self.ctx = ctx if ctx != 0 else [mx.cpu()]
        if args.syncbn and len(self.ctx) > 1:
            self.net = get_db(args.network, args.num_layers, pretrained_base=True,
                                norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                norm_kwargs={'num_devices': len(self.ctx)})
            self.async_net = get_db(args.network, args.num_layers, pretrained_base=False)  # used by cpu worker
        else:
            self.net = get_db(args.network, args.num_layers, pretrained_base=True) 
            self.async_net = self.net

        model_name = '%s-resnet%d-db'%(args.dataset_name, args.num_layers)
        
        if not os.path.exists(args.save_prefix):
            os.mkdir(args.save_prefix)
        args.save_prefix += model_name
        self.init_model()
        self.net.hybridize()
        self.net.collect_params().reset_ctx(self.ctx)
        if args.export_model:
            self.export_model()
        
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        self.loss = DBLoss()
        self.post_proc = DBPostProcess()
        self.metric    = DetectionIoUEvaluator() 
        self.sum_loss  = mx.metric.Loss('SumLoss')
        self.bce_loss  = mx.metric.Loss('BalanceCELoss')
        self.l1_loss   = mx.metric.Loss('L1Loss')
        self.dice_loss = mx.metric.Loss('DiceLoss')

    def init_model(self):
        if args.resume.strip():
            self.net.load_parameters(args.resume.strip())
            self.async_net.load_parameters(args.resume.strip())
        else:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                self.net.initialize(init=mx.init.Xavier())
                self.async_net.initialize(init=mx.init.Xavier())

    def get_lr_scheduler(self):
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
        return lr_scheduler
    
    def get_dataloader(self):
        augment = PointAugmenter()
        train_dataset = DBDataset(args.train_img_dir, 
                                  args.train_lab_dir,
                                  augment, mode='train',
                                  img_size=(args.data_shape, args.data_shape))
        val_dataset  = DBDataset(args.train_img_dir, 
                                 args.train_lab_dir, mode='val',
                                 img_size=(args.data_shape, args.data_shape))

        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                                      last_batch='discard', shuffle=True, 
                                      num_workers=args.num_workers, pin_memory=True)
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
        best_loss = 10000

        for epoch in range(args.start_epoch, args.epochs):
            tic = time.time()
            btic = time.time()
            self.net.hybridize()
            
            for i, batch in enumerate(self.train_dataloader):
                data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
                labs = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 5)]
                sum_losses, bce_losses, l1_losses, dice_losses = [], [], [], []
                with mx.autograd.record():
                    for it, x in enumerate(data):
                        bina, thresh, thresh_bina = self.net(x)
                        pred = {'binary':bina, 'thresh':thresh, 'thresh_binary':thresh_bina}
                        lab  = {'gt':labs[0][it], 'mask':labs[1][it], 'thresh_map':labs[2][it], 'thresh_mask':labs[3][it]}
                        loss, metric = self.loss(pred, lab)
                        sum_losses.append(loss)
                        bce_losses.append(metric['bce_loss'])
                        l1_losses.append(metric['l1_loss'])
                        dice_losses.append(metric['thresh_loss'])
                    mx.autograd.backward(sum_losses)
                trainer.step(1)
                self.sum_loss.update(0, sum_losses)
                self.bce_loss.update(0, bce_losses)
                self.l1_loss.update(0, l1_losses)
                self.dice_loss.update(0, dice_losses)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name0, loss0 = self.sum_loss.get()
                    name1, loss1 = self.bce_loss.get()
                    name2, loss2 = self.l1_loss.get()
                    name3, loss3 = self.dice_loss.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, i+1, trainer.learning_rate, args.batch_size/(time.time()-btic), name0, loss0, name1, loss1, name2, loss2, name3, loss3))
                btic = time.time()
            
            name0, loss0 = self.sum_loss.get()
            name1, loss1 =self.bce_loss.get()
            name2, loss2 = self.l1_loss.get()
            name3, loss3 = self.dice_loss.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                        epoch, time.time()-tic, name0, loss0, name1, loss1, name2, loss2, name3, loss3))
            
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
        self.bce_loss.reset()
        self.l1_loss.reset()
        self.dice_loss.reset()
        tic = time.time()
        for batch in self.val_dataloader:
            data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx, batch_axis=0)
            labs = [gluon.utils.split_and_load(batch[it], ctx_list=self.ctx, batch_axis=0) for it in range(1, 5)]
            for it, x in enumerate(data):
                bina, thresh, thresh_bina = self.net(x)
                pred = {'binary':bina, 'thresh':thresh, 'thresh_binary':thresh_bina}
                lab  = {'gt':labs[0][it], 'mask':labs[1][it], 'thresh_map':labs[2][it], 'thresh_mask':labs[3][it]}
                loss, metric = self.loss(pred, lab)
                self.sum_loss.update(0, loss)
                self.bce_loss.update(0, metric['bce_loss'])
                self.l1_loss.update(0, metric['l1_loss'])
                self.dice_loss.update(0, metric['thresh_loss'])
        
        name0, loss0 = self.sum_loss.get()
        name1, loss1 = self.bce_loss.get()
        name2, loss2 = self.l1_loss.get()
        name3, loss3 = self.dice_loss.get()
        
        logger.info('Evaling cost: {:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}, {}={:.3f}'.format(
                    time.time()-tic, name0, loss0, name1, loss1, name2, loss2, name3, loss3))
        self.sum_loss.reset()
        self.bce_loss.reset()
        self.l1_loss.reset()
        self.dice_loss.reset()
        return loss0

    def export_model(self):
        self.net.export_block(args.save_prefix, args.resume.strip(), self.ctx)
        sys.exit()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
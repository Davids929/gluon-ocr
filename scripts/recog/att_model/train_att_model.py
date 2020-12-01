#coding=utf-8
import os
import sys
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from mxnet.gluon.data import DataLoader
from gluoncv import utils as gutils
sys.path.append(os.path.expanduser('~/gluon-ocr'))
from gluonocr.model_zoo import get_att_model
from gluonocr.data import FixSizeDataset, BucketDataset 
from gluonocr.data import BucketSampler, Augmenter
from gluonocr.utils.recog_metric import RecogAccuracy
from config import args

gutils.random.seed(args.seed)
class Trainer(object):
    def __init__(self):
        
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        self.ctx = ctx if ctx else [mx.cpu()]
        self.train_dataloader, self.val_dataloader = self.get_dataloader()
        # load voc size
        voc_size = self.train_dataloader._dataset.voc_size
        start = self.train_dataloader._dataset.start_sym
        end   = self.train_dataloader._dataset.end_sym
        decoder_kwargs = {'voc_size':voc_size}
        if args.syncbn and len(self.ctx) > 1:
            self.net = get_att_model(args.network, args.num_layers, 
                                    pretrained_base=True, 
                                    norm_layer=gluon.contrib.nn.SyncBatchNorm,
                                    norm_kwargs={'num_devices': len(self.ctx)},
                                    start_symbol=start, end_symbol=end,
                                    decoder_kwargs=decoder_kwargs)
        
            self.async_net = get_att_model(args.network, args.num_layers, 
                                    start_symbol=start, end_symbol=end, 
                                    decoder_kwargs=decoder_kwargs)  # used by cpu worker
        else:
            self.net = get_att_model(args.network, args.num_layers, 
                                    pretrained_base=True, 
                                    start_symbol=start, end_symbol=end,
                                    decoder_kwargs=decoder_kwargs)
            self.async_net = self.net

        model_name = '%s-%s%d-att-model'%(args.dataset_name, args.network, args.num_layers)
        if not os.path.exists(args.save_prefix):
            os.mkdir(args.save_prefix)
        args.save_prefix += model_name
        
        self.net.hybridize()
        if args.export_model:
            self.export_model()
        self.init_model()
        self.net.collect_params().reset_ctx(self.ctx)
        self.loss = gluon.loss.SoftmaxCELoss()
        self.loss_metric = mx.metric.Loss('SoftmaxCELoss')
        self.acc_metric  = RecogAccuracy()

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
        augment = Augmenter()
        if args.bucket_mode:
            dataset_fn = BucketDataset
        else:
            dataset_fn = FixSizeDataset

        train_dataset = dataset_fn(args.train_data_path, args.voc_path, 
                                   short_side=args.short_side,
                                   fix_width=args.fix_width,
                                   augment_fn=augment,
                                   max_len=args.max_len,
                                   start_sym=0, end_sym=1)
        val_dataset  = dataset_fn(args.val_data_path, args.voc_path, 
                                   short_side=args.short_side,
                                   fix_width=args.fix_width,
                                   max_len=args.max_len,
                                   start_sym=0, end_sym=1)
        
        if args.num_samples < 0:
            args.num_samples = len(train_dataset)

        if args.bucket_mode:
            train_sampler = BucketSampler(args.batch_size, train_dataset.bucket_dict,
                                        shuffle=True, last_batch='discard')
            val_sampler = BucketSampler(1, val_dataset.bucket_dict,
                                        shuffle=False, last_batch='keep')
            batch_size  = None
        else:
            train_sampler, val_sampler = None, None
            batch_size  = args.batch_size

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                                      batch_sampler=train_sampler, pin_memory=True,
                                      num_workers=args.num_workers)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, 
                                    batch_sampler=val_sampler, pin_memory=True,
                                    num_workers=args.num_workers)
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
            {'wd': args.wd, 'momentum': args.momentum, 'lr_scheduler': lr_scheduler})

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
        best_acc = [0]

        for epoch in range(args.start_epoch, args.epochs):
            tic = time.time()
            btic = time.time()
            self.net.hybridize()

            for i, batch in enumerate(self.train_dataloader):
                src_data = gluon.utils.split_and_load(batch[0], ctx_list=self.ctx)
                mask     = batch[1][:,:, ::32, ::8]
                src_mask = gluon.utils.split_and_load(mask, ctx_list=self.ctx)
                src_targ = gluon.utils.split_and_load(batch[2], ctx_list=self.ctx)
                tag_lab  = gluon.utils.split_and_load(batch[3], ctx_list=self.ctx)
                tag_mask = gluon.utils.split_and_load(batch[4], ctx_list=self.ctx)
                l_list = []
                with mx.autograd.record():
                    for sd, sm, st, tl, tm in zip(src_data, src_mask, src_targ, tag_lab, tag_mask):
                        states = self.net.begin_state(bs, sd.context)
                        outputs = self.net(sd, sm, st)
                        loss = self.loss(outputs, tl, tm.expand_dims(axis=2))
                        l_list.append(loss)
                    mx.autograd.backward(l_list)
                trainer.step(args.batch_size)
                mx.nd.waitall()
                self.acc_metric.update(outputs, tl, tm)
                self.loss_metric.update(0, l_list)
                if args.log_interval and not (i + 1) % args.log_interval:
                    name1, acc1  = self.acc_metric.get()
                    name2, loss2 = self.loss_metric.get()
                    logger.info('[Epoch {}][Batch {}], LR: {:.2E}, Speed: {:.3f} samples/sec, {}={:.3f}, {}={:.3f}'.format(
                            epoch, i, trainer.learning_rate, args.batch_size/(time.time()-btic), name1, acc1, name2, loss2))
                btic = time.time()

            name1, acc1  = self.acc_metric.get()
            name2, loss2 = self.loss_metric.get()
            logger.info('[Epoch {}] Training cost: {:.3f}, {}={:.3f}, {}={:.3f}'.format(
                epoch, (time.time()-tic), name1, acc1, name2, loss2))
            if not epoch % args.val_interval:
                name, current_acc = self.evaluate()
                logger.info('[Epoch {}] Validation: {}={:.3f}'.format(epoch, name, current_acc))
            
            if current_acc > best_acc[0]:
                best_acc[0] = current_acc
                self.net.save_parameters('{:s}_best.params'.format(args.save_prefix, epoch, current_acc))
                with open(args.save_prefix + '_best_map.log', 'a') as f:
                    f.write('{:04d}:\t{:.4f}\n'.format(epoch, current_acc))
            if args.save_interval and epoch % args.save_interval == 0:
                self.net.save_parameters('{:s}_{:04d}_{:.4f}.params'.format(args.save_prefix, epoch, current_acc))
            self.acc_metric.reset()
            self.loss_metric.reset()

    def evaluate(self):
        self.acc_metric.reset()
        self.net.hybridize()
        for i, data in enumerate(self.val_dataloader):
            s_data = data[0].as_in_context(self.ctx[0])
            s_mask = data[1].as_in_context(self.ctx[0])
            s_mask = s_mask[:, :, ::32, ::8]
            t_label = data[3]
            t_mask = data[4]
            bs, seq_len = t_label.shape
            targ_inp = self.net.begin_inp(bs, seq_len, self.ctx[0])
            out    = self.net(s_data, s_mask, targ_inp)
            self.acc_metric.update(out, t_label, t_mask)
        name, acc = self.acc_metric.get()
        self.acc_metric.reset()
        return name, acc

    def export_model(self):
        data = mx.nd.ones((1, 3, 32, 128), ctx=self.ctx[0])
        mask = mx.nd.ones((1, 1, 1, 16), ctx=self.ctx[0])
        states = self.net.begin_state(1, self.ctx[0])
        self.net.load_parameters(args.resume.strip())
        self.net.collect_params().reset_ctx(self.ctx)
        outs = self.net(data, mask, *states)
        self.net.export(args.save_prefix)
        print('Export model successfully.')
        sys.exit()

if __name__ == '__main__':
    trainer = Trainer()
    trainer.train()
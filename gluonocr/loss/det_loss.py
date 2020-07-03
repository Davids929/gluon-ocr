#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.loss import Loss
from .base_loss import *

class DBLoss(Loss):
    def __init__(self, eps=1e-6, l1_scale=10, bce_scale=5, weight=1., batch_axis=0, **kwargs):
        super(DBLoss, self).__init__(weight, batch_axis, **kwargs)

        self.l1_scale  = l1_scale
        self.bce_scale = bce_scale
        self.dice_loss = DiceLoss(eps=eps)
        self.l1_loss   = MaskL1Loss()
        self.bce_loss  = BalanceCELoss(eps=eps) 

    def forward(self, pred, batch):
        bce_loss = self.bce_loss(pred['binary'], batch['gt'], batch['mask'])
        metrics  = dict(bce_loss=bce_loss)
        if 'thresh' in pred:
            l1_loss = self.l1_loss(pred['thresh'], batch['thresh_map'], batch['thresh_mask'])
            metrics['l1_loss'] = l1_loss
            dice_loss = self.dice_loss(pred['thresh_binary'], batch['gt'], batch['mask'])
            metrics['thresh_loss'] = dice_loss
            loss = dice_loss + self.l1_scale * l1_loss + bce_loss * self.bce_scale
        else:
            loss = bce_loss
        return loss, metrics

class EASTLoss(Loss):
    def __init__(self, lambd=1.0, rho=1.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(EASTLoss, self).__init__(weight, batch_axis, **kwargs)
        self._lambd = lambd
        self._rho   = rho
        self._eps   = eps
        self.seg_loss = BalanceCELoss(eps=eps)
        
    def forward(self, pred, batch):
        pred_score = pred['score']
        pred_geo   = pred['geo_map']
        lab_score  = batch['gt']
        lab_geo    = batch['geo_map']
        lab_mask   = batch['mask']
        seg_loss   = self.seg_loss(pred_score, lab_score, lab_mask)
        norm_weight = lab_geo.slice_axis(axis=1, begin=8, end=None)
        lab_geo   = lab_geo.slice_axis(axis=1, begin=0, end=8)
        lab_score = mx.nd.repeat(lab_score, repeats=8, axis=1)
        
        mask      = lab_mask * lab_score
        l1_loss   = mx.nd.abs(lab_geo - pred_geo)
        l1_loss   = mx.nd.where(l1_loss > self._rho, l1_loss - 0.5*self._rho, 
                                (0.5 / self._rho) * mx.nd.square(l1_loss))
        l1_loss  = norm_weight * mx.nd.mean(l1_loss, axis=1, keepdims=True) * mask
        l1_loss  = mx.nd.sum(l1_loss, axis=self._batch_axis, exclude=True) / (mx.nd.sum(mask)+self._eps)

        loss = self._lambd * seg_loss + l1_loss
        metrics = dict(bce_loss=seg_loss, l1_loss=l1_loss)
        return loss, metrics
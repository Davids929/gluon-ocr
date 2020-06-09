#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.loss import Loss, _apply_weighting
import numpy as np

class DiceLoss(Loss):
    def __init__(self, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self.eps = eps

    def hybrid_forward(self, F, pred, label, mask, sample_weight=None):
        
        mask  = _apply_weighting(mask, self._weight, sample_weight)
        inter = F.sum(pred*label*mask, axis=self._batch_axis, exclude=True)
        union = F.sum(pred*mask)  + F.sum(label*mask) + self.eps
        loss  = 1 - 2.0*inter/union
        return loss

class MaskL1Loss(Loss):
    def __init__(self, weight=1., batch_axis=0, **kwargs):
        super(MaskL1Loss, self).__init__(weight, batch_axis, **kwargs)
    
    def hybrid_forward(self, F, pred, label, mask):
        mask_sum = F.sum(mask)
        loss = F.abs(label - pred)*mask
        loss = _apply_weighting(F, loss, self._weight)
        loss = F.sum(loss, axis=self._batch_axis, exclude=True)
        return loss/(mask_sum+1)

class BalanceL1Loss(Loss):
    def __init__(self, negative_ratio=3.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(BalanceL1Loss, self).__init__(weight, batch_axis, **kwargs)
        self.negative_ratio = negative_ratio
        self.eps = eps

    def hybrid_forward(self, F, pred, label, mask):
        positive = (label * mask)
        negative = ((1 - label) * mask)
        with mx.autograd.pause():    
            positive_count = int(F.sum(positive).asscalar())
            negative_count = min(int(F.sum(negative).asscalar()),
                            int(positive_count * self.negative_ratio))
        loss = F.abs(label - pred)
        negative_loss = (loss * negative).reshape((0, 0, -1))
        rank = negative_loss.argsort(axis=1, is_ascend=0).argsort(axis=1)
        hard_negative = rank < negative_count
        negative_loss = F.where(hard_negative>0, negative_loss, F.zeros_like(negative_loss))
        positive_loss = loss * positive
        balance_loss  = (F.sum(positive_loss, axis=self._batch_axis, exclude=True) + \
                        F.sum(negative_loss, axis=self._batch_axis, exclude=True)) / \
                        (positive_count + negative_count + self.eps)
        return balance_loss


class BalanceCELoss(Loss):
    def __init__(self, negative_ratio=3.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(BalanceCELoss, self).__init__(weight, batch_axis, **kwargs)
        self.negative_ratio = negative_ratio
        self.eps = eps

    def hybrid_forward(self, F, pred, label, mask):
        positive = (label * mask)
        negative = ((1 - label) * mask)
        with mx.autograd.pause():    
            positive_count = int(F.sum(positive).asscalar())
            negative_count = min(int(F.sum(negative).asscalar()),
                                int(positive_count * self.negative_ratio))
        
        loss = -(F.log(pred + self.eps) * label + F.log(1. - pred + self.eps) * (1. - label))
        negative_loss = (loss * negative).reshape((0, 0, -1))
        rank = negative_loss.argsort(axis=1, is_ascend=0).argsort(axis=1)
        hard_negative = rank < negative_count
        negative_loss = F.where(hard_negative>0, negative_loss, F.zeros_like(negative_loss))

        positive_loss = loss * positive
        balance_loss  = (F.sum(positive_loss, axis=self._batch_axis, exclude=True) + \
                         F.sum(negative_loss, axis=self._batch_axis, exclude=True)) / \
                          (positive_count + negative_count + self.eps)
        return balance_loss

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
    def __init__(self, alpha=1.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(EASTLoss, self).__init__(weight, batch_axis, **kwargs)
        self.alpha = alpha
        self.dice_loss = DiceLoss(eps=eps)

    def forward(self, pred, batch):
        f_score = pred['score']
        f_geo   = pred['geo_map']
        l_score = batch['gt']
        l_geo   = batch['geo_map']
        l_mask  = batch['mask']
        dice_loss = self.dice_loss(f_score, l_score, l_mask)
        f_geo_split = f_geo.slice_axis(axis=1, begin=0, end=None)
        l_geo_split = l_geo.slice_axis(axis=1, begin=0, end=None)
        smooth_l1 = 0
        for i in range(8):
            in_loss = mx.nd.abs(l_geo_split[i] - f_geo_split[i])
            in_loss = mx.nd.where(in_loss > l_score, in_loss - 0.5, mx.nd.square(in_loss))
            out_loss = l_geo_split[-1] / 8 * in_loss * l_score
            smooth_l1 += out_loss
        
        smooth_l1_loss = mx.nd.mean(smooth_l1 * l_score, axis=self._batch_axis, exclude=True)
        loss = self.alpha * dice_loss + smooth_l1_loss
        metrics = dict(dice_loss=dice_loss, smooth_l1_loss=smooth_l1_loss)
        return loss, metrics
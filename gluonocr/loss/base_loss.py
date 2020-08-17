#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.loss import Loss, _apply_weighting
import numpy as np

class DiceLoss(Loss):
    def __init__(self, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(DiceLoss, self).__init__(weight, batch_axis, **kwargs)
        self._eps = eps

    def hybrid_forward(self, F, pred, label, mask):
        inter = F.sum(pred*label*mask, axis=self._batch_axis, exclude=True)
        union = F.sum(pred*mask, axis=self._batch_axis, exclude=True) + \
                F.sum(label*mask, axis=self._batch_axis, exclude=True) + self._eps
        loss  = 1 - 2.0*inter/union
        return loss

class MaskL1Loss(Loss):
    def __init__(self, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(MaskL1Loss, self).__init__(weight, batch_axis, **kwargs)
        self._eps = eps

    def hybrid_forward(self, F, pred, label, mask):
        mask_sum = F.sum(mask)
        loss = F.abs(label - pred)*mask
        loss = _apply_weighting(F, loss, self._weight)
        loss = F.sum(loss, axis=self._batch_axis, exclude=True)
        return loss/(mask_sum+self._eps)

class MaskSmoothL1Loss(Loss):
    def __init__(self, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(MaskSmoothL1Loss, self).__init__(weight, batch_axis, **kwargs)
        self._eps = eps
        
    def hybrid_forward(self, F, pred, label, mask):
        loss = F.smooth_l1((pred - label) * mask, scalar=1.0)
        loss = F.sum(loss, axis=self._batch_axis, exclude=True)/(F.sum(mask)+self._eps)
        return loss

class BalanceL1Loss(Loss):
    def __init__(self, negative_ratio=3.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(BalanceL1Loss, self).__init__(weight, batch_axis, **kwargs)
        self.negative_ratio = negative_ratio
        self._eps = eps

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
                        (positive_count + negative_count + self._eps)
        return balance_loss


class BalanceCELoss(Loss):
    def __init__(self, negative_ratio=3.0, eps=1e-6, weight=1., batch_axis=0, **kwargs):
        super(BalanceCELoss, self).__init__(weight, batch_axis, **kwargs)
        self.negative_ratio = negative_ratio
        self._eps = eps

    def hybrid_forward(self, F, pred, label, mask):
        positive = (label * mask)
        negative = ((1 - label) * mask)
        with mx.autograd.pause():    
            positive_count = int(F.sum(positive).asscalar())
            negative_count = min(int(F.sum(negative).asscalar()),
                                int(positive_count * self.negative_ratio))
        
        loss = -(F.log(pred + self._eps) * label + F.log(1. - pred + self._eps) * (1. - label))
        negative_loss = (loss * negative).reshape((0, 0, -1))
        rank = negative_loss.argsort(axis=1, is_ascend=0).argsort(axis=1)
        hard_negative = rank < negative_count
        negative_loss = F.where(hard_negative>0, negative_loss, F.zeros_like(negative_loss))

        positive_loss = loss * positive
        balance_loss  = (F.sum(positive_loss, axis=self._batch_axis, exclude=True) + \
                         F.sum(negative_loss, axis=self._batch_axis, exclude=True)) / \
                          (positive_count + negative_count + self._eps)
        return balance_loss

class SoftmaxCELoss(gluon.loss.Loss):
    def __init__(self, axis=-1, alpha=0.25, gamma=2, batch_axis=0, **kwargs):
        super(SoftmaxCELoss, self).__init__(None, batch_axis, **kwargs)
        self._axis = axis
        self._alpha = alpha
        self._gamma = gamma

    def hybrid_forward(self, F, output, label):
        Nc = (label==1).sum()
        mask = F.expand_dims(label!=-1, axis=-1)
        output = F.softmax(output)
        # picl first element if label<=0, second element if label==1, third if label>=2
        pj = output.pick(label, axis=self._axis, keepdims=True)
        # loss = - self._alpha * ((1 - pj) ** self._gamma) * (pj + 1e-5).log()
        loss = -(pj + 1e-5).log()
        loss = F.broadcast_mul(loss, mask)
        return loss.sum(axis=self._batch_axis, exclude=True)/(Nc + 1e-5)




class BoxIOULoss(Loss):
    """
    iou_type: one of ['iou', 'giou', 'diou', 'ciou'].
    """
    def __init__(self, iou_loss_type='iou', 
                 weight=1.0, batch_axis=0, eps=1e-12, **kwargs):
        super(BoxIOULoss, self).__init__(weight, batch_axis, **kwargs)
        assert iou_loss_type in ['iou', 'giou', 'diou', 'ciou']
        self.iou_loss_type = iou_loss_type
        self._eps = eps

    def hybrid_forward(self, F, box_pred, box_lab, box_mask):
        """
        box_pred shape: [batch, num_boxes, 4]
        box_lab  shape: [batch, num_boxes, 4]
        box_mask shape: [batch, num_boxes]
        """
        
        p = box_pred.split(axis=-1, num_outputs=4, squeeze_axis=True)
        t = box_lab.split(axis=-1, num_outputs=4, squeeze_axis=True)
        p_width  = F.maximum(p[2] - p[0], 0)
        p_height = F.maximum(p[3] - p[1], 0)
        t_width  = F.maximum(t[2] - t[0], 0)
        t_height = F.maximum(t[3] - t[1], 0)
        p_area = p_width * p_height
        t_area = t_width * t_height
        
        intersect_xmin = F.maximum(p[0], t[0])
        intersect_ymin = F.maximum(p[1], t[1])
        intersect_xmax = F.minimum(p[2], t[2])
        intersect_ymax = F.minimum(p[3], t[3])
        intersect_width  = F.maximum(intersect_xmax - intersect_xmin, 0)
        intersect_height = F.maximum(intersect_ymax - intersect_ymin, 0)
        
        intersect_area = intersect_width * intersect_height
        union_area = F.maximum(p_area + t_area - intersect_area, self._eps)
        iou_v = F.divide(intersect_area, union_area)
        if self.iou_loss_type=="iou":
            loss = (1 - iou_v)*box_mask
            return F.sum(loss, axis=0, exclude=True)

        enclose_xmin = F.minimum(p[0], t[0])
        enclose_ymin = F.minimum(p[1], t[1])
        enclose_xmax = F.maximum(p[2], t[2])
        enclose_ymax = F.maximum(p[3], t[3])
        if self.iou_loss_type == "giou":
            enclose_width = F.maximum(0, enclose_xmax - enclose_xmin)
            enclose_height = F.maximum(0, enclose_ymax - enclose_ymin)
            enclose_area = F.maximum(enclose_width * enclose_height, self._eps)
            giou_v = iou_v - F.divide((enclose_area - union_area), enclose_area)
            loss = (1 - giou_v) * box_mask
            return F.sum(loss, axis=0, exclude=True)

        p_center = F.stack((p[0] + p[2]) / 2, (p[1] + p[3]) / 2, axis=-1)
        t_center = F.stack((t[0] + t[2]) / 2, (t[1] + t[3]) / 2, axis=-1)
        enclose_hw  = F.stack(enclose_ymax - enclose_ymin, enclose_xmax - enclose_xmin, axis=-1)
        euclidean   = F.norm(t_center - p_center, ord=2, axis=-1)
        diag_length = F.norm(enclose_hw, ord=2, axis=-1)
        diag_length = F.maximum(diag_length, self._eps)
        diou_v = iou_v - F.divide(F.square(euclidean), F.square(diag_length))
        if self.iou_loss_type == "diou":
            loss = (1 - diou_v)*box_mask
            return F.sum(loss, axis=0, exclude=True)
        
        p_ratio = F.devide(p_width, F.maximum(p_height, self._eps))
        t_ratio = F.devide(t_width, F.maximum(t_height, self._eps))
        arctan = mx.nd.arctan(p_ratio - t_ratio)
        v = 4*F.square(arctan/np.pi)
        alpha  = F.divide(v, F.maximum(1-iou_v+v, self._eps))
        ciou_v = diou_v - alpha*v
        loss   = (1 - ciou_v)*box_mask 
        return F.sum(loss, axis=0, exclude=True)
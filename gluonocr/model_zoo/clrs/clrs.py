#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from .anchor import CLRSAnchorGenerator
from gluoncv.nn.coder import NormalizedBoxCenterDecoder, MultiPerClassDecoder

__all__ = ['CLRS', 'get_clrs']

class DM(nn.HybridBlock):
    def __init__(self, channels=128, ksize=2, strides=2, pad=0, **kwargs):
        super(DM, self).__init__(**kwargs)
        with self.name_scope():
            self.deconv = nn.HybridSequential()
            self.deconv.add(nn.Conv2DTranspose(channels, ksize, strides))
            self.deconv.add(nn.Conv2D(channels, 3, 1, 1))
            self.deconv.add(nn.BatchNorm())
    
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(channels, 3, 1, 1))
            self.conv.add(nn.BatchNorm())
            self.conv.add(nn.Activation('relu'))
            self.conv.add(nn.Conv2D(channels, 3, 1, 1))
            self.conv.add(nn.BatchNorm())

    def hybrid_forward(self, F, x1, x2):
        x1 = self.deconv(x1)
        x2 = self.conv(x2)
        return F.relu(x1*x2)

class PM(nn.HybridBlock):
    def __init__(self, channels=256, k=4, num_classes=4, **kwargs):
        super(PM, self).__init__(**kwargs)
        with self.name_scope():
            self.skip = nn.HybridSequential()
            self.skip.add(nn.Conv2D(channels, 1, 1))
            self.bone = nn.HybridSequential()
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.conf = nn.Conv2D(k*(num_classes+1), 3, 1, 1)
            self.loc  = nn.Conv2D(k*4, 3, 1, 1)

    def hybrid_forward(self, F, x):
        x1 = self.skip(x)
        x2 = self.bone(x)
        x  = F.relu(x1 + x2)
        score  = self.conf(x)
        offset = self.loc(x) 
        return score, offset

class SM(nn.HybridBlock):
    def __init__(self, channels, n_scale=2, **kwargs):
        super(SM, self).__init__(**kwargs)
        self.n_scale = n_scale
        with self.name_scope():
            self.skip = nn.HybridSequential()
            self.skip.add(nn.Conv2D(channels, 1, 1))
            self.skip.add(nn.BatchNorm())
            self.bone = nn.HybridSequential()
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.BatchNorm())
            self.bone.add(nn.Activation('relu'))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.BatchNorm())
            self.bone.add(nn.Activation('relu'))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.BatchNorm())

    def hybrid_forward(self, F, x):
        x1 = self.skip(x)
        x2 = self.bone(x)
        x  = F.relu(x1 + x2)
        if self.n_scale > 1:
            x = F.UpSampling(x, scale=self.n_scale, sample_type='nearest')
        return x

class SegPred(nn.HybridBlock):
    def __init__(self, channels, **kwargs):
        super(SegPred, self).__init__(**kwargs)
        with self.name_scope():
            self.sms = nn.HybridSequential() 
            self.sms.add(SM(channels, 16))
            self.sms.add(SM(channels, 8))
            self.sms.add(SM(channels, 4))
            self.sms.add(SM(channels, 2))
            self.sms.add(SM(channels, 1))

            self.tail = nn.HybridSequential()
            self.tail.add(nn.Conv2D(channels, 1, 1))
            self.tail.add(nn.BatchNorm())
            self.tail.add(nn.Activation('relu'))
            self.tail.add(nn.Conv2DTranspose(channels, 2, 2))
            self.tail.add(nn.Conv2D(channels, 3, 1, 1))
            self.tail.add(nn.BatchNorm())
            self.tail.add(nn.Activation('relu'))
            self.tail.add(nn.Conv2DTranspose(4, 2, 2))
            self.tail.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, *xs):
        feats = []
        for x, block in zip(xs, self.sms):
            feat = block(x)
            feats.append(feat)
        fuse_feat = F.relu(F.add_n(*feats))
        return self.tail(fuse_feat)

class CLRS(nn.HybridBlock):
    def __init__(self, stages, sizes, ratios, steps, dm_channels=256, 
                 pm_channels=256, sm_channels=32, 
                 stds=(0.1, 0.1, 0.2, 0.2), nms_thresh=0.45, 
                 nms_topk=1000, post_nms=400, 
                 anchor_alloc_size=256, ctx=mx.cpu(),
                 norm_layer=nn.BatchNorm, norm_kwargs=None,**kwargs):
        super(CLRS, self).__init__(**kwargs)
        
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

        with self.name_scope():
            self.stages = nn.HybridSequential()
            for i in range(len(stages)):
                self.stages.add(stages[i])
            # extra layers
            self.extras = nn.HybridSequential()
            self.extras.add(self._extra_layer(256, 512))
            self.extras.add(self._extra_layer(128, 256))
            self.extras.add(self._extra_layer(128, 256, strides=1))
            self.extras.add(self._extra_layer(128, 256, strides=1))
            self.dms = nn.HybridSequential()
            for i in range(6):
                strides = 2 if i > 1 else 1
                ksize   = 2 if strides==2 else 3 
                self.dms.add(DM(dm_channels, ksize, strides=strides, pad=ksize-2))
            self.pms = nn.HybridSequential()
            self.anchor_generators = nn.HybridSequential()
            asz = anchor_alloc_size
            for i, (s, r, st) in enumerate(zip(sizes, ratios, steps)): 
                self.pms.add(PM(pm_channels, len(s)))
                anchor_generator = CLRSAnchorGenerator(i, (512, 512), s, r, st, alloc_size=(asz, asz))
                self.anchor_generators.add(anchor_generator)
                asz = max(asz // 2, 16)
            self.seg_pred = SegPred(sm_channels)
            self.bbox_decoder = NormalizedBoxCenterDecoder(stds)
            self.cls_decoder = MultiPerClassDecoder(4+1, thresh=0.01)

    def _extra_layer(self, in_channels, out_channels, strides=2, 
                    norm_layer=nn.BatchNorm, norm_kwargs=None):
        layer = nn.HybridSequential()
        layer.add(nn.Conv2D(in_channels, 1, 1))
        layer.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        layer.add(nn.Activation('relu'))
        layer.add(nn.Conv2D(out_channels, 3, strides, strides-1))
        layer.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
        layer.add(nn.Activation('relu'))
        return layer

    def set_nms(self, nms_thresh=0.45, nms_topk=400, post_nms=100):
        """Set non-maximum suppression parameters.

        Parameters
        ----------
        nms_thresh : float, default is 0.45.
            Non-maximum suppression threshold. You can specify < 0 or > 1 to disable NMS.
        nms_topk : int, default is 400
            Apply NMS to top k detection results, use -1 to disable so that every Detection
             result is used in NMS.
        post_nms : int, default is 100
            Only return top `post_nms` detection results, the rest is discarded. The number is
            based on COCO dataset which has maximum 100 objects per image. You can adjust this
            number if expecting more objects. You can use -1 to return all detections.

        Returns
        -------
        None

        """
        self._clear_cached_op()
        self.nms_thresh = nms_thresh
        self.nms_topk = nms_topk
        self.post_nms = post_nms

    def hybrid_forward(self, F, x):
        feats = []
        for block in self.stages:
            x = block(x)
            feats.append(x)

        for block in self.extras:
            x = block(x)
            feats.append(x) 

        F11 = feats[-1]
        F10 = self.dms[0](F11, feats[-2])
        F9  = self.dms[1](F10, feats[-3])
        F8  = self.dms[2](F9, feats[-4])
        F7  = self.dms[3](F8, feats[-5])
        F4  = self.dms[4](F7, feats[-6])
        F3  = self.dms[5](F4, feats[-7])

        feats = [F3, F4, F7, F8, F9, F10, F11]
        box_preds, cls_preds = [], []
        for feat, block in zip(feats, self.pms):
            pred = block(feat)
            cls_preds.append(F.flatten(F.transpose(pred[0], (0, 2, 3, 1))))
            box_preds.append(F.flatten(F.transpose(pred[1], (0, 2, 3, 1))))
        cls_preds = F.concat(*cls_preds, dim=1).reshape((0, -1, 5))
        box_preds = F.concat(*box_preds, dim=1).reshape((0, -1, 4))
        anchors   = [F.reshape(ag(feat), shape=(1, -1))
                   for feat, ag in zip(feats, self.anchor_generators)]
        anchors = F.concat(*anchors, dim=1).reshape((1, -1, 4))
        
        seg_maps = self.seg_pred(F9, F8, F7, F4, F3)
        if mx.autograd.is_training():
            return [cls_preds, box_preds, anchors, seg_maps]
        
        bboxes = self.bbox_decoder(box_preds, anchors)
        cls_ids, scores = self.cls_decoder(F.softmax(cls_preds, axis=-1))
        results = []
        for i in range(4):
            cls_id = cls_ids.slice_axis(axis=-1, begin=i, end=i+1)
            score = scores.slice_axis(axis=-1, begin=i, end=i+1)
            # per class results
            per_result = F.concat(*[cls_id, score, bboxes], dim=-1)
            results.append(per_result)
        result = F.concat(*results, dim=1)
        if self.nms_thresh > 0 and self.nms_thresh < 1:
            result = F.contrib.box_nms(
                result, overlap_thresh=self.nms_thresh, topk=self.nms_topk, valid_thresh=0.01,
                id_index=0, score_index=1, coord_start=2, force_suppress=False)
            if self.post_nms > 0:
                result = result.slice_axis(axis=1, begin=0, end=self.post_nms)
        ids = F.slice_axis(result, axis=2, begin=0, end=1)
        scores = F.slice_axis(result, axis=2, begin=1, end=2)
        bboxes = F.slice_axis(result, axis=2, begin=2, end=6)
        return ids, scores, bboxes, seg_maps

    def export_block(self, prefix, param_path, ctx=mx.cpu()):
        if not isinstance(ctx, list):
            ctx = [ctx]
        data = mx.nd.ones((1, 3, 512, 512), dtype='float32', ctx=ctx[0])
        self.load_parameters(param_path)
        self.set_nms(0.45, 1000, 400)
        self.hybridize()
        self.collect_params().reset_ctx(ctx)
        pred1 = self(data)
        self.export(prefix, epoch=0)
        print('Successfully export model!')

def get_clrs(backbone_name, num_layers, 
             norm_layer=nn.BatchNorm, norm_kwargs=None,
             pretrained_base=False, ctx=mx.cpu()):
    # import sys, os
    # sys.path.append(os.path.expanduser('~/demo/gluon-ocr/gluonocr/model_zoo'))
    # from resnet import get_resnet
    # from mobilenetv3 import get_mobilenet_v3
    # from resnext import get_resnext
    from ..resnet import get_resnet
    from ..mobilenetv3 import get_mobilenet_v3
    from ..resnext import get_resnext

    if backbone_name.lower() == 'resnet':
        base_net = get_resnet(1, num_layers, strides=[(2, 2), (1,1), (2,2), (2,2), (1,1)],
                            pretrained=pretrained_base, norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        ids = [5, 6, 8]
    elif backbone_name.lower() == 'resnext':
        base_net = get_resnext(num_layers, strides=[(2, 2), (1,1), (2,2), (2,2), (1,1)],
                              pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        ids = [5, 6, 8]
    elif backbone_name.lower() == 'mobilenetv3':
        if num_layers == 24:
            ids = [4, 6, 17]
            base_net = get_mobilenet_v3('small', strides=[(2,2), (2,2), (2,2), (1,1)],
                              pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        elif num_layers == 32:
            ids = [6, 9, 21]
            base_net = get_mobilenet_v3('large',  pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        else:
            raise ValueError('The num_layers of moblienetv3 must be 24 or 32.')
        
    else:
        raise ValueError('The %s is not support.'%backbone_name)
    
    stages = [base_net.features[:ids[0]], 
              base_net.features[ids[0]:ids[1]],
              base_net.features[ids[1]:ids[2]],
            ]
    sizes  = [[4, 6, 8, 10, 12, 16], [20, 24, 28, 32], [36, 40, 44, 48],
              [56, 64, 72, 80], [88, 96, 104, 112], [124, 136, 148, 160],
              [184, 208, 232, 256]]
    ratios = [[1.0] for _ in range(7)]
    steps  = [4, 8, 16, 32, 64, 86, 128]
    net = CLRS(stages, sizes, ratios, steps)
    return net

if __name__ == '__main__':
    x = mx.nd.ones((1, 3, 512, 512))
    net = get_clrs('resnet', 34)
    net.initialize()
    with mx.autograd.record():
        output = net(x) 

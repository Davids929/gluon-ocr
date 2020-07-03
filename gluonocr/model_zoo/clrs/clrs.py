#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class DM(nn.HybridBlock):
    def __init__(self, channels=128, **kwargs):
        super(DM, self).__init__(**kwargs)
        with self.name_scope():
            self.deconv = nn.HybridSequential()
            self.deconv.add(nn.Conv2DTranspose(channels, kernel_size=2, strides=2, padding=0))
            self.deconv.add(nn.Conv2D(channels, 3, 1))
            self.deconv.add(nn.BatchNorm())
    
            self.conv = nn.HybridSequential()
            self.conv.add(nn.Conv2D(channels, 3, 1))
            self.conv.add(nn.BatchNorm())
            self.conv.add(nn.Activation('relu'))
            self.conv.add(nn.Conv2D(channels, 3, 1))
            self.conv.add(nn.BatchNorm())

    def hybrid_forward(self, F, x1, x2):
        x1 = self.deconv(x1)
        x2 = self.conv(x2)
        return F.relu(x1*x2)

class PM(nn.HybridBlock):
    def __init__(self, channels=256, k=4, q=4, **kwargs):
        super(PM, self).__init__(**kwargs)
        with self.name_scope():
            self.skip = nn.HybridSequential()
            self.skip.add(nn.Conv2D(channels, 1, 1))
            self.bone = nn.HybridSequential()
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.bone.add(nn.Conv2D(channels, 1, 1))
            self.conf = nn.Conv2D(k*q*2, 3, 1)
            self.loc  = nn.Conv2D(k*q*4, 3, 1)

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
            x = F.UpSampling(x, scale=self.n_scale, sample_type='bilinear')
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

    def hybrid_forward(self, F, xs):
        feats = []
        for x, block in zip(xs, self.sms):
            feat = block(x)
            feats.append(feat)
        fuse_feat = F.relu(F.add_n(*feats))
        return self.tail(fuse_feat)


class CLRS(nn.HybridBlock):
    def __init__(self, stages, dm_channels=256, pm_channels=256, sm_channels=32, **kwargs):
        super(CLRS, self).__init__(**kwargs)
        with self.name_scope():
            self.stages = stages
            self.extras = nn.HybridSequential()
            self.extras.add(self._extra_layer())
            self.seg_pred = SegPred(sm_channels)


    def _extra_layer(self, in_channels, out_channels):
        layer = nn.HybridSequential()
        layer.add(nn.Conv2D(in_channels, 1, 1))
        layer.add(nn.Activation('relu'))
        layer.add(nn.Conv2D(out_channels, 3, 2, 1))
        layer.add(nn.Activation('relu'))
        return layer


    def get_feats(self, F, x):
        pass

    def hybrid_forward(self, x):
        feats = []
        for block in self.stages:
            x = block(x)
            feats.append(x)
        

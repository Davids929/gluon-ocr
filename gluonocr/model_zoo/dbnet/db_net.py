#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class DBNet(gluon.HybridBlock):
    def __init__(self, stages, inner_channels=256, k=10, use_bias=False, 
                 adaptive=False, norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        super(DBNet, self).__init__(**kwargs)
    
        self.k = k
        self.adaptive = adaptive
        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.ins_proj = nn.HybridSequential()
            self.outs     = nn.HybridSequential()
            for i in range(len(stages)):
                self.stages.add(stages[i])
                self.ins_proj.add(nn.Conv2D(inner_channels, 1, use_bias=use_bias))
                self.outs.add(nn.Conv2D(inner_channels//4, 3, padding=1, use_bias=use_bias))

            self.binarize = nn.HybridSequential()

            self.binarize.add(nn.Conv2D(inner_channels//4, 3, padding=1, use_bias=use_bias))
            self.binarize.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.binarize.add(nn.Activation('relu'))
            self.binarize.add(nn.Conv2DTranspose(inner_channels//4, 2, 2))
            self.binarize.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.binarize.add(nn.Activation('relu'))
            self.binarize.add(nn.Conv2DTranspose(1, 2, 2))
            self.binarize.add(nn.Activation('sigmoid'))
            if adaptive:
                self.thresh = nn.HybridSequential()
                self.thresh.add(nn.Conv2D(inner_channels//4, 3, padding=1, use_bias=use_bias))
                self.thresh.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.thresh.add(nn.Activation('relu'))
                self.thresh.add(nn.Conv2DTranspose(inner_channels//4, 2, 2))
                self.thresh.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
                self.thresh.add(nn.Activation('relu'))
                self.thresh.add(nn.Conv2DTranspose(1, 2, 2))
                self.thresh.add(nn.Activation('sigmoid'))

    def hybrid_forward(self, F, x):
        features = []
        for stage in self.stages:
            x = stage(x)
            features.append(x)
        
        for i, block in enumerate(self.ins_proj):
            features[i] = block(features[i])
        in2, in3, in4, in5 = features

        out4 = F.contrib.BilinearResize2D(in5, like=in4, mode='like') + in4
        out3 = F.contrib.BilinearResize2D(out4, like=in3, mode='like') + in3
        out2 = F.contrib.BilinearResize2D(out3, like=in2, mode='like') + in2

        features = [out2, out3, out4, in5]
        output = []
        for feat, block in zip(features, self.outs):
            out = block(feat)
            out = F.contrib.BilinearResize2D(out, like=in2, mode='like')
            output.append(out)
        
        fuse = F.concat(*output, dim=1)
        binary = self.binarize(fuse)
        # if not mx.autograd.is_training():
        #     return binary
        if self.adaptive:
            temp = F.contrib.BilinearResize2D(binary, like=fuse, mode='like')
            fuse = F.concat(fuse, temp, dim=1)
            thresh = self.thresh(fuse)
            thresh_binary = 1.0/(1.0 + F.exp(-self.k*(binary-thresh)))
            return binary, thresh, thresh_binary
        else:
            return binary

from gluoncv.model_zoo import get_model
def resnet_db(num_layers, pretrained_base=False, ctx=mx.cpu(), **kwargs):
    base_net = get_model('resnet%d_v1'%num_layers, pretrained=pretrained_base)
    stages = [base_net.features[:5], 
              base_net.features[5:6],
              base_net.features[6:7],
              base_net.features[7:8]]
    net = DBNet(stages, **kwargs)
    return net

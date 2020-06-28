#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn

class AttEncoder(nn.HybridBlock):
    def __init__(self, stages, hidden_dim=256, num_layers=2,
                 match_dim=512, dropout=0.1, rnn_type='lstm',
                 **kwargs):

        super(AttEncoder, self).__init__(**kwargs)
        with self.name_scope():
            self.stages = stages
            if rnn_type.lower() == 'lstm':
                self.rnn = gluon.rnn.LSTM(hidden_size=hidden_dim, num_layers=num_layers,
                                             dropout=dropout, bidirectional=True, layout='NTC')
            elif rnn_type.lower() == 'gru':
                self.rnn = gluon.rnn.GRU(hidden_size=hidden_dim, num_layers=num_layers,
                                             dropout=dropout, bidirectional=True, layout='NTC')
            else:
                raise ValueError('The rnn type is fault!')
            self.pre_compute = nn.Dense(match_dim, activation='tanh', flatten=False)

    def hybrid_forward(self, F, x, mask):
        x = self.stages(x)
        x = F.broadcast_mul(x, mask)
        x = F.transpose(x, axes=(0, 3, 2, 1))
        x = F.reshape(x, shape=(0, -3, 0))
        mask = F.transpose(mask, axes=(0, 1, 3, 2))
        mask = F.reshape(mask, shape=(0, 0, -1))
        
        output = self.rnn(x)
        out_proj = self.pre_compute(output)
        return output, out_proj, mask

def get_encoder(backbone_name, num_layers, pretrained_base=False, ctx=mx.cpu(),
                norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
    from ..resnet import get_resnet
    from ..resnext import get_resnext
    from ..vgg import get_vgg
    from ..mobilenetv3 import get_mobilenet_v3
    
    if backbone_name.lower() == 'resnet':
        base_net = get_resnet(1, num_layers, strides=[(1, 1), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'resnext':
        base_net = get_resnext(num_layers, strides=[(1, 1), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'vgg':
        base_net = get_vgg(num_layers, strides=[(2, 2), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        backbone = base_net.features[:-4]
    elif backbone_name.lower() == 'mobilenetv3':
        if num_layers == 24:
            base_net = get_mobilenet_v3('small', strides=[(2, 2), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        elif num_layers == 32:
            base_net = get_mobilenet_v3('large', strides=[(2, 2), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        else:
            raise ValueError('The num_layers of moblienetv3 must be 24 or 32.')
        backbone = base_net.features[:-4]
    else:
        raise ValueError('Please input right backbone name.')
    encoder = AttEncoder(backbone, **kwargs)
    return encoder

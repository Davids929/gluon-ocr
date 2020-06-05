#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn


class Attencoder(nn.HybridBlock):
    def __init__(self, 
                 stages, 
                 hidden_dim=256, 
                 num_layers=2,
                 match_dim=512,
                 dropout=0.1, 
                 rnn_type='lstm',
                 **kwargs)

        super(Attencoder, self).__init__(**kwargs)
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
        x = F.reshape(x, shape=(0, 0, -1))
        output = F.transpose(x, axes=(0, 2, 1))
        output = self.rnn(output)
        output = F.broadcast_mul(output, mask)
        out_proj = self.pre_compute(output)
        return output, out_proj

def resnet_encoder(num_layers, base_kwargs, encode_kwargs):
    from ..resnet import get_resnet
    base_net = get_resnet(1, num_layers, used='recog', **base_kwargs)
    backbone = base_net[:-4]
    encoder = Attencoder(backbone, **encode_kwargs)
    return encoder

def vgg_encoder(num_layers, base_kwargs, encode_kwargs):
    from ..vgg import get_vgg
    base_net = get_vgg(num_layers, used='recog', **base_kwargs)
    backbone = base_net[:-4]
    encoder = Attencoder(backbone, **encode_kwargs)
    return encoder
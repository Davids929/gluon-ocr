# coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from ..nn import STN

class CRNN(nn.HybridBlock):
    def __init__(self,
                 stages, 
                 hidden_size=256, 
                 num_layers=2, 
                 dropout=0.1, 
                 voc_size=37, 
                 use_bilstm=True, 
                 use_stn=False, 
                 **kwargs):
        super(CRNN, self).__init__(**kwargs)
        
        self.use_stn = use_stn
        with self.name_scope():
            self.stages = stages
            if use_stn:
                self.stn = STN()
        self.lstm = gluon.rnn.LSTM(hidden_size, num_layers, layout='NTC',
                                   dropout=dropout, bidirectional=use_bilstm)
        self.fc   = nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, x):
        if self.use_stn:
            x = self.stn(x)
        x = self.stages(x)
        x = F.reshape(x, (0, 0, -1))
        x = F.transpose(x, axes=(0, 2, 1))
        x = self.lstm(x)
        x = self.fc(x)
        return x

def resnet_crnn(num_layers, base_kwargs, decode_kwargs):
    from ..resnet import get_resnet
    base_net = get_resnet(1, num_layers, used='recog', **base_kwargs)
    backbone = base_net[:-4]
    crnn = CRNN(backbone, **decode_kwargs)
    return crnn

def vgg_crnn(num_layers, base_kwargs, decode_kwargs):
    from ..vgg import get_vgg
    base_net = get_vgg(num_layers, used='recog', **base_kwargs)
    backbone = base_net[:-4]
    crnn = CRNN(backbone, **decode_kwargs)
    return crnn

def mobilenet_v3_crnn(model_name, base_kwargs, decode_kwargs):
    from ..mobilenetv3 import get_mobilenet_v3
    base_net = get_mobilenet_v3(model_name, used='recog', **base_kwargs):
    backbone = base_net[:-6]
    crnn = CRNN(backbone, **decode_kwargs)
    return crnn
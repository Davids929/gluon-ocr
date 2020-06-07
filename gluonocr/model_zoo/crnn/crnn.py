# coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from ...nn import STN

class CRNN(nn.HybridBlock):
    def __init__(self, stages, hidden_size=256, num_layers=2, dropout=0.1, 
                 voc_size=37, use_bilstm=True, use_stn=False, **kwargs):
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

def get_crnn(backbone_name, num_layers, base_kwargs={}, decode_kwargs={}):
    from ..resnet import get_resnet
    from ..resnext import get_resnext
    from ..vgg import get_vgg
    from ..mobilenetv3 import get_mobilenet_v3
    
    if backbone_name.lower() == 'resnet':
        base_net = get_resnet(1, num_layers, used_recog=True, **base_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'resnext':
        base_net = get_resnext(num_layers, used_recog=True, **base_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'vgg':
        base_net = get_vgg(num_layers, used_recog=True, **base_kwargs)
        backbone = base_net.features[:-4]
    elif backbone_name.lower() == 'mobilenetv3':
        if num_layers == 24:
            base_net = get_mobilenet_v3('small', use_recog=True, **base_kwargs)
        elif num_layers == 32:
            base_net = get_mobilenet_v3('large', use_recog=True, **base_kwargs)
        else:
            raise ValueError('The num_layers of moblienetv3 must be 24 or 32.')
        backbone = base_net.features[:-4]
    else:
        raise ValueError('Please input right backbone name.')
    crnn = CRNN(backbone, **decode_kwargs)
    return crnn

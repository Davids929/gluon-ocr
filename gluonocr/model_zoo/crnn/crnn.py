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
            self.dropout = nn.Dropout(dropout)
            self.fc   = nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, x):

        if self.use_stn:
            x = self.stn(x)
        x = self.stages(x)
        x = F.transpose(x, axes=(0, 3, 2, 1))
        x = F.reshape(x, (0, -3, 0))
        x = self.lstm(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
    def export_block(self, prefix, param_path, ctx=mx.cpu()):
        if not isinstance(ctx, list):
            ctx = [ctx]
        data = mx.nd.ones((1, 3, 32, 320), dtype='float32', ctx=ctx[0])
        self.load_parameters(param_path)
        self.hybridize()
        self.collect_params().reset_ctx(ctx)
        pred1 = self(data)
        self.export(prefix, epoch=0)
        print('Successfully export model!')
    

def get_crnn(backbone_name, num_layers, pretrained_base=False, ctx=mx.cpu(),
             norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
    from ..resnet import get_resnet
    from ..resnext import get_resnext
    from ..vgg import get_vgg
    from ..mobilenetv3 import get_mobilenet_v3
    
    if backbone_name.lower() == 'resnet':
        base_net = get_resnet(1, num_layers, strides=[(2, 1), (1, 1), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'resnext':
        base_net = get_resnext(num_layers, strides=[(2, 1), (1, 1), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        backbone = base_net.features[:-1]
    elif backbone_name.lower() == 'vgg':
        base_net = get_vgg(num_layers, strides=[(2, 1), (2, 2), (2, 2), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
    elif backbone_name.lower() == 'mobilenetv3':
        if num_layers == 24:
            base_net = get_mobilenet_v3('small', strides=[(2, 2), (2, 1), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        elif num_layers == 32:
            base_net = get_mobilenet_v3('large', strides=[(2, 2), (2, 1), (2, 1), (2, 1)], 
                              pretrained=pretrained_base, norm_layer=norm_layer, 
                              norm_kwargs=norm_kwargs)
        else:
            raise ValueError('The num_layers of moblienetv3 must be 24 or 32.')
        backbone = base_net.features[:-4]
    else:
        raise ValueError('Please input right backbone name.')
    crnn = CRNN(backbone, **kwargs)
    return crnn

from mxnet import gluon
from mxnet.gluon import nn
import mxnet as mx

class EAST(nn.HybridBlock):
    def __init__(self, stages, channels=[128, 128, 128, 32], 
                 norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
        
        super(EAST, self).__init__(**kwargs)
        self.norm_layer  = norm_layer
        self.norm_kwargs = norm_kwargs

        with self.name_scope():
            self.stages = nn.HybridSequential()
            self.convs  = nn.HybridSequential()
            for i in range(len(stages)):
                self.stages.add(stages[i])
            for i in range(3):
                self.convs.add(self._make_layers(channels[i]))
                
            self.pred_score = nn.HybridSequential()
            self.pred_score.add(nn.Conv2D(channels[-1], 3, 1, 1))
            self.pred_score.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.pred_score.add(nn.Activation('relu'))
            self.pred_score.add(nn.Conv2D(1, 1, 1))
            self.pred_score.add(nn.Activation('sigmoid'))

            self.pred_geo   = nn.HybridSequential()
            self.pred_geo.add(nn.Conv2D(channels[-1], 3, 1, 1))
            self.pred_geo.add(norm_layer(**({} if norm_kwargs is None else norm_kwargs)))
            self.pred_geo.add(nn.Activation('relu'))
            self.pred_geo.add(nn.Conv2D(8, 1, 1))
            self.pred_geo.add(nn.Activation('sigmoid'))

    def _make_layers(self, channel, ksize=3, stride=1, padding=1, act_type='relu'):
        layer = nn.HybridSequential()
        layer.add(nn.Conv2D(channel, 1, 1))
        layer.add(self.norm_layer(**({} if self.norm_kwargs is None else self.norm_kwargs)))       
        layer.add(nn.Activation(act_type))
        layer.add(nn.Conv2D(channel, ksize, stride, padding))
        layer.add(self.norm_layer(**({} if self.norm_kwargs is None else self.norm_kwargs)))       
        layer.add(nn.Activation(act_type))
        return layer

    def hybrid_forward(self, F, x):
        feats = []
        for block in self.stages:
            x = block(x)
            feats.append(x)
        feats = feats[::-1]
        h = feats[0]
        for i in range(3):
            h = F.UpSampling(h, scale=2, sample_type='nearest')
            h = F.concat(h, feats[i+1], dim=1)
            h = self.convs[i](h)
        scores = self.pred_score(h)
        geometrys = (self.pred_geo(h) - 0.5) * 2 * 800
        return scores, geometrys

    def export_block(self, prefix, param_path, ctx=mx.cpu()):
        if not isinstance(ctx, list):
            ctx = [ctx]
        data = mx.nd.ones((1, 3, 512, 512), dtype='float32', ctx=ctx[0])
        self.load_parameters(param_path)
        self.hybridize()
        self.collect_params().reset_ctx(ctx)
        pred1 = self(data)
        self.export(prefix, epoch=0)
        print('Export model successfully!')

def get_east(backbone_name, num_layers, pretrained_base=False, ctx=mx.cpu(),
             norm_layer=nn.BatchNorm, norm_kwargs=None, **kwargs):
    from ..resnet import get_resnet
    from ..mobilenetv3 import get_mobilenet_v3
    from ..resnext import get_resnext
    if backbone_name.lower() == 'resnet':
        base_net = get_resnet(1, num_layers, pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        ids = [5, 6, 7, 8]
    elif backbone_name.lower() == 'resnext':
        base_net = get_resnext(num_layers, pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        ids = [5, 6, 7, 8]
    elif backbone_name.lower() == 'mobilenetv3':
        if num_layers == 24:
            ids = [4, 6, 12, 17]
            base_net = get_mobilenet_v3('small', pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        elif num_layers == 32:
            ids = [6, 9, 15, 21]
            base_net = get_mobilenet_v3('large', pretrained=pretrained_base,
                              norm_layer=norm_layer, norm_kwargs=norm_kwargs)
        else:
            raise ValueError('The num_layers of moblienetv3 must be 24 or 32.')
        
    else:
        raise ValueError('Input right backbone name.')
    stages = [base_net.features[:ids[0]], 
              base_net.features[ids[0]:ids[1]],
              base_net.features[ids[1]:ids[2]],
              base_net.features[ids[2]:ids[3]],
             ]
    net = EAST(stages, **kwargs)
    return net
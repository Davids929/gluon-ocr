#coding=utf-8
import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from .att_encoder import get_encoder
from .att_decoder import AttDecoder

class AttModel(nn.HybridBlock):
    def __init__(self, encoder, decoder, start_symbol=0, end_symbol=1, **kwargs):
        super(AttModel, self).__init__(**kwargs)
        self.start_symbol = start_symbol
        self.end_symbol   = end_symbol
        with self.name_scope():
            self.encoder = encoder
            self.decoder = decoder

    def hybrid_forward(self, F, x, mask, targ_input):

        if isinstance(x, mx.ndarray.NDArray):
            batch_size = x.shape[0]
            state = self.begin_state(func=mx.ndarray.zeros, ctx=x.context, 
                                     batch_size=batch_size, dtype=x.dtype)
        else:
            state = self.begin_state(func=mx.symbol.zeros)

        en_out, en_proj, mask = self.encoder(x, mask)
        states = [en_out, en_proj, mask, state]
        tag_input = F.transpose(targ_input, axes=(1, 0)).expand_dims(axis=-1)

        def train_func(out, states):
            outs = self.decoder(out, states)
            return outs[0], outs[1:]

        if mx.autograd.is_training():
            outputs, states = F.contrib.foreach(train_func, tag_input, states)
            outputs = F.reshape(outputs, shape=(0, -3, 0))
            outputs = F.transpose(outputs, axes=(1, 0, 2))
            return outputs

        def test_func(inp, states):
            outs = self.decoder(*states)
            pred = F.softmax(outs[0], axis=-1)
            pred = F.argmax(pred, axis=-1)
            states = [pred*inp] + list(outs[1:])
            return pred, states
        
        first_input = F.slice_axis(tag_input, axis=0, begin=0, end=1)
        first_input = F.squeeze(first_input, axis=0)*self.start_symbol
        states = [first_input] + states
        outputs, states = F.contrib.foreach(test_func, tag_input, states)
        outputs = F.reshape(outputs, (0, -1))
        outputs = F.transpose(outputs, axes=(1, 0))
        return outputs

    def __call__(self, data, mask, targ_input, **kwargs):
    
        return super(AttModel, self).__call__(data, mask, targ_input, **kwargs)

    def export_block(self, prefix, param_path, ctx=mx.cpu()):
        if not isinstance(ctx, list):
            ctx = [ctx]
        data = mx.nd.ones((1, 3, 32, 128), ctx=ctx[0])
        mask = mx.nd.ones((1, 1, 1, 16), ctx=ctx[0])
        targ_input = mx.nd.ones((1, 10), ctx=ctx[0])
        self.hybridize()
        #self.load_parameters(param_path)
        self.initialize()
        self.collect_params().reset_ctx(ctx)
        outs = self.__call__(data, mask, targ_input)
        self.export(prefix)
        print('Export model successfully.')

    def begin_state(self, **kwargs):
        return self.decoder.lstm.begin_state(**kwargs)

def get_att_model(backbone_name, num_layers, pretrained_base=False, ctx=mx.cpu(),
                  norm_layer=nn.BatchNorm, norm_kwargs=None, start_symbol=0, end_symbol=1, 
                  encoder_kwargs={}, decoder_kwargs={}):
    encoder = get_encoder(backbone_name, num_layers, pretrained_base=pretrained_base, ctx=ctx,
                          norm_layer=norm_layer, norm_kwargs=norm_kwargs, **encoder_kwargs)
    decoder = AttDecoder(**decoder_kwargs)
    net = AttModel(encoder, decoder, start_symbol=start_symbol, end_symbol=end_symbol)
    return net


def resnet18_att_model(**kwargs):
    return get_att_model('resnet', 18, **kwargs)

def resnet34_att_model(**kwargs):
    return get_att_model('resnet', 34, **kwargs)

def resnet50_att_model(**kwargs):
    return get_att_model('resnet', 50, **kwargs)

def mobilenet_small_att_model(**kwargs):
    return get_att_model('mobilenetv3', 24, **kwargs)

def mobilenet_large_att_model(**kwargs):
    return get_att_model('mobilenetv3', 32, **kwargs)
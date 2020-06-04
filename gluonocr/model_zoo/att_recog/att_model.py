#coding=utf-8
import mxnet as mx
from mxnet import gluon
import numpy as np
from mxnet.gluon import nn
from .att_encoder import *
from .att_decoder import *

class AttModel(object):
    def __init__(self, encoder_fn, encoder_kwargs, decoder_fn, decoder_kwargs):
        self.encoder = encoder_fn(**encoder_kwargs)
        self.decoder = decoder_fn(**decoder_kwargs)

    def train(self, x, mask, tag_input):
        en_out, en_proj = self.encoder(x, mask)
        with mx.autograd.pause():
            seq_len = tag_input.shape[-1]
            bs, time_steps = en_out.shape[:2]
        states = self.decoder.begin_state(bs, time_steps, en_out.context)
        output_list = []
        for i in range(seq_len):
            curr_inp = tag_input[:, i].expand_dims(axis=1)
            outputs = self.decoder(en_out, en_proj, mask, *states, curr_inp)
            output_list.append(outputs[0])
            states = outputs[1:]
        pred = mx.nd.concat(*output_list, dim=1)
        return pred

    def test(self, x, mask, start_symbol=0, end_symbol=1):
        en_out, en_proj = self.encoder(x, mask)
        bs, time_steps = en_out.shape[:2]
        curr_inp = mx.nd.ones(shape=(bs, 1), dtype='float32', ctx=en_out.context)*start_symbol
        ids_list = []
        states = self.decoder.begin_state(bs, time_steps, en_out.context)
        for j in range(time_steps):
            outputs = self.decoder(en_out, en_proj, mask, *states, curr_inp)
            states  = outputs[1:]
            pred    =  mx.nd.softmax(outputs[0], axis=-1)
            pred_id = pred.argmax(axis=-1)
            curr_inp = pred_id
            ids = pred_id.asnumpy().astype('int32')[:,0]
            if np.sum(ids==end_symbol) == bs:
                break
            ids_list.append(ids)
        ids_list = np.stack(ids_list, axis=-1)
        return ids_list

    def __call__(self, mode, *args):
        if mode == 'train':
            return self.train(*args)
        else:
            return self.test(*args)
    
    def export(self, path, **kwargs):
        self.encoder.export(path+'-encoder', **kwargs)
        self.decoder.export(path+'-decoder', **kwargs)

    def initalize(self, **kwargs):
        self.encoder.initalize(**kwargs)
        self.decoder.initalize(**kwargs)
    
    def hybridlize(self, **kwargs):
        self.encoder.hybridlize(**kwargs)
        self.decoder.hybridlize(**kwargs)
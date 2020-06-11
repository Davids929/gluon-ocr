#coding=utf-8
import mxnet as mx
from mxnet import gluon
import numpy as np
from mxnet.gluon import nn
from .att_encoder import *
from .att_decoder import *

class AttModel(nn.HybridBlock):
    def __init__(self, encoder, decoder, **kwargs):
        super(AttModel, self).__init__(**kwargs)
        self.start_symbol = 0
        self.end_symbol   = 1
        with self.name_scope():
            self.encoder = encoder
            self.decoder = decoder

    # def hybrid_forward(self, F, x, mask, tag_input, state):
    #     en_out, en_proj = self.encoder(x, mask)
    #     states   = [en_out, en_proj, mask, state[0], state[1]]
    #     tag_input = F.transpose(tag_input, axes=(1, 0)).expand_dims(axis=-1)
    #     outputs, states = F.contrib.foreach(self.decoder, tag_input, states)
    #     outputs = F.reshape(outputs, shape=(0, -3, 0))
    #     outputs = F.transpose(outputs, axes=(1, 0, 2))
    #     return outputs

    def hybrid_forward(self, F, x, mask, state):
        en_out, en_proj = self.encoder(x, mask)
        sum_mask = F.sum_axis(mask, axis=-1)
        curr_inp = F.ones_like(sum_mask).as_in_context(en_out.context)*self.start_symbol
        ids_list = []
        states   = [en_out, en_proj, mask, state[0], state[1]]
        
        def cond(out, state):
            pred = F.softmax(out, axis=-1)
            pred = F.argmax(pred, axis=-1)
            ones = F.ones_like(pred)
            ends = ones*self.end_symbol
            return F.sum(ones) != F.sum(ends==pred)

        def func(out, state):
            out, state = self.decoder(out, state)
            pred = F.softmax(out, axis=-1)
            pred = F.argmax(pred, axis=-1)
            return pred, [pred, state]

        loop_vars = [curr_inp, states]
        outputs, states = F.contrib.while_loop(cond, func, loop_vars, max_iterations=10)
        outputs = F.reshape(outputs, (0, -1))
        outputs = F.reshape(outputs, (1, 0))
        return outputs

def get_att_model():
    pass
#coding=utf-8
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from ...nn.attention_cell import _get_attention_cell


class AttDecoder(nn.HybridBlock):
    def __init__(self, embed_dim=256, match_dim=256, hidden_dim=256, 
                 voc_size=37, num_layers=2, dropout=0.1, bilstm=False,
                 attention_cell='scaled_luong', **kwargs):

        super(AttDecoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        with self.name_scope():
            self.attention = _get_attention_cell(attention_cell, units=match_dim, 
                                                dropout=dropout)
            self.embedding = nn.Embedding(voc_size, embed_dim)
            self.lstm      = gluon.rnn.LSTM(hidden_dim, num_layers, dropout=dropout,
                                            bidirectional=bilstm, layout='NTC')
            
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, cur_input, en_out, en_proj, en_mask, h, c):

        state = [h, c]
        pre_output_list = F.SliceChannel(h, num_outputs=self.num_layers,
                                         axis=0, squeeze_axis=False)
        pre_output = pre_output_list[self.num_layers-1]
        pre_output = F.transpose(pre_output, axes=(1, 0, 2))
        cur_input  = self.embedding(cur_input)
        att_input  = pre_output + cur_input
        att_out, att_weight = self.attention(att_input, en_proj, en_out, en_mask)
        lstm_in = F.concat(cur_input, att_out, dim=2)
        lstm_out, state = self.lstm(lstm_in, state)
        #lstm_out = F.concat(lstm_in, lstm_out, dim=2)
        output = self.dropout(lstm_out)
        output = self.fc(output)
        return output, en_out, en_proj, en_mask, state[0], state[1]

    def begin_state(self, batch_size, ctx):
        h_state = nd.zeros(shape=(self.num_layers, batch_size, self.hidden_dim), dtype='float32', ctx=ctx)
        c_state = nd.zeros(shape=(self.num_layers, batch_size, self.hidden_dim), dtype='float32', ctx=ctx)
        # pre_att = nd.zeros(shape=(batch_size, seq_len, 1), dtype='float32', ctx=ctx)
        # sum_att = nd.zeros(shape=(batch_size, seq_len, 1), dtype='float32', ctx=ctx)
        state = [h_state, c_state]
        return state
#coding=utf-8
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn
from ...nn.attention_cell import _get_attention_cell
from ...nn.rnn_layer import RNNLayer

class AttDecoder(nn.HybridBlock):
    def __init__(self, embed_dim=512, match_dim=512, hidden_size=512, 
                 voc_size=37, num_layers=2, dropout=0.1, bilstm=False,
                 attention_cell='mlp_luong', **kwargs):

        super(AttDecoder, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        with self.name_scope():
            self.attention = _get_attention_cell(attention_cell, units=match_dim, 
                                                dropout=dropout)
            self.embedding = nn.Embedding(voc_size, embed_dim)
            
            self.lstm    = RNNLayer('lstm', num_layers, 
                                 hidden_size, dropout=dropout, 
                                 bidirectional=bilstm, layout='NTC')

            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, cur_input, en_out, en_proj, en_mask, state):
        cur_input  = self.embedding(cur_input)
        mask = en_mask.expand_dims(axis=1)
        att_out, att_weight = self.attention(cur_input, en_proj, en_out, mask)
        lstm_in = F.concat(cur_input, att_out, dim=2)
        lstm_out, state = self.lstm(lstm_in, state, mask=en_mask)
        #lstm_out = F.concat(lstm_in, lstm_out, dim=2)
        output = self.dropout(lstm_out)
        output = self.fc(output)
        return output, en_out, en_proj, en_mask, state


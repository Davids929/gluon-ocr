#coding=utf-8
import mxnet as mx
from mxnet import gluon, nd
from mxnet.gluon import nn

class BaseAttention(nn.HybridBlock):
    def __init__(self, 
                 match_dim=512, 
                 dropout=0.1, 
                 **kwargs):
        super(BaseAttention, self).__init__(**kwargs)
        with self.name_scope():
            self.inp_dense = nn.Dense(match_dim, activation='tanh', flatten=False)
            self.attention = nn.HybridSequential()
            self.attention.add(nn.Activation('tanh'))
            self.attention.add(nn.Dropout(dropout))
            self.attention.add(nn.Dense(1, flatten=False))

    def hybrid_forward(self, F, en_out, en_proj, en_mask, cur_input):
    
        att_input = self.inp_dense(cur_input)
        # energy shape: [batch_size, seq_len, 1]
        energy = self.attention(att_input)
        energy = energy * en_mask + (1.0 - en_mask) * (-10000.0)
        cur_att   = F.softmax(energy, axis=1)
        cur_att   = cur_att.transpose(axes=(0, 2, 1))
        output    = F.batch_dot(cur_att, en_out)
        return output

class AttDecoder(nn.HybridBlock):
    def __init__(self, 
                 hidden_dim=256, 
                 embed_dim=512, 
                 match_dim=512, 
                 voc_size=5994, 
                 num_layers=2, 
                 dropout=0.1, 
                 bilstm=False,
                 **kwargs):

        super(AttDecoder, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        with self.name_scope():
            self.attention = BaseAttention(match_dim, dropout)
            self.embedding = nn.Embedding(voc_size, embed_dim)
            self.lstm      = gluon.rnn.LSTM(hidden_dim, num_layers, dropout=dropout,
                                            bidirectional=bilstm, layout='NTC')
            
            self.dropout = nn.Dropout(dropout)
            self.fc      = nn.Dense(voc_size, flatten=False)

    def hybrid_forward(self, F, en_out, en_proj, en_mask, h, c, cur_input):
        state = [h, c]
        pre_output_list = F.SliceChannel(h, num_outputs=self.num_layers,
                                         axis=0, squeeze_axis=False)
        pre_output = pre_output_list[self.num_layers-1]
        pre_output = F.transpose(pre_output, axes=(1, 0, 2))
        cur_input  = self.embedding(cur_input)
        att_input  = F.concat(pre_output, cur_input, dim=2)
        att_out    = self.attention(en_out, en_proj, en_mask, att_input)
        lstm_in = F.concat(cur_input, att_out, dim=2)
        lstm_out, state = self.lstm(lstm_in, state)
        output = F.concat(lstm_in, lstm_out, dim=2)
        output = self.dropout(output)
        output = self.fc(output)
        return output, state[0], state[1]

    def begin_state(self, batch_size, seq_len, ctx):
        h_state = nd.zeros(shape=(self.num_layers, batch_size, self.hidden_dim), dtype='float32', ctx=ctx)
        c_state = nd.zeros(shape=(self.num_layers, batch_size, self.hidden_dim), dtype='float32', ctx=ctx)
        # pre_att = nd.zeros(shape=(batch_size, seq_len, 1), dtype='float32', ctx=ctx)
        # sum_att = nd.zeros(shape=(batch_size, seq_len, 1), dtype='float32', ctx=ctx)
        state = [h_state, c_state]
        return state
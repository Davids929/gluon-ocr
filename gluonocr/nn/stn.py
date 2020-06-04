#coding=utf-8
from mxnet.gluon import nn

class STN(nn.HybridBlock):
    def __init__(self, hidden_size=32, **kwargs):
        super(STN, self).__init__(**kwargs)
        with self.name_scope():
            self.loc = nn.HybridSequential()
            self.loc.add(nn.Conv2D(hidden_size, 3, strides=1, padding=1))
            self.loc.add(nn.Activation('relu'))
            self.loc.add(nn.Conv2D(2, 3, strides=1, padding=1))
            self.loc.add(nn.Activation('tanh'))

    def hybrid_forward(self, F, x):
        warp_matrix = self.loc(x)
        grid = F.GridGenerator(warp_matrix, transform_type='warp')
        out = F.BilinearSampler(x, grid)
        return out
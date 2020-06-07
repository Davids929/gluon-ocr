#coding=utf-8
import os
import logging
import warnings
import time
import numpy as np
import mxnet as mx
from mxnet import nd
from mxnet import gluon
from gluoncv import utils as gutils
from gluonocr.model_zoo import get_crnn
from gluonocr.data import BucketDataset, BucketSampler, Augmenter
from config import args

gutils.random.seed(args.seed)
class Trainer(object):
    def __init__(self):
        ctx = [mx.gpu(int(i)) for i in args.gpus.split(',') if i.strip()]
        self.ctx = ctx if ctx else [mx.cpu()]
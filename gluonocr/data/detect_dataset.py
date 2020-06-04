import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms

class SegDataset(Dataset):
    def __init__(self):
        pass
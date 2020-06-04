#coding=utf-8
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class BoxAugmenter(object):
    def __init__(self, configs):
        self.aug = get_aug_seq(configs)

    def get_aug_seq(self, configs):
        seqence = []
        for cfg in configs:
            seqence.append(getattr(iaa, cfg[0]))(**cfg[1])
        seq = iaa.Sequential(seqence, random_order=True)
        return seq

class PointAugmenter(DetectAugmenter):
    def __init__(self, configs):
        super(SegmentAugmenter, self).__init__(configs)



#coding=utf-8
import numpy as np
import imgaug.augmenters as iaa
import random
class Augmenter(object):
    def __init__(self, configs):
        if configs == []:
            self.seq = self.get_default_seq()
        else:
            self.seq = self.get_aug_seq(configs)

    def __call__(self, img):
        return self.seq(img)
    
    def get_aug_seq(self, configs):
        seqence = []
        for cfg in configs:
            seqence.append(getattr(iaa, cfg[0]))(**cfg[1])
        seq = iaa.Sequential(seqence, random_order=True)
        return seq
        
    def get_default_seq(self):
        seqence = [iaa.LinearContrast((0.8, 1.2)),
                   iaa.Grayscale((0,1.0)),
                   iaa.GaussianBlur((0, 0.5)),
                   iaa.Sharpen(alpha=(0, 0.2), lightness=(0.85, 1.2)),
                   iaa.Add((-10, 10), per_channel=0.5),
                   ]
        seq = iaa.Sequential(seqence, random_order=True)
        return seq

class SynthLines(object):
    def __init__(self, short_side=32, ):
        self.short_side = short_side

    def __call__(self, ):
        pass

    def synth_lines(self, nums=10000):
        pass
#coding=utf-8
import numpy as np
import imgaug.augmenters as iaa
import random
class Augmenter(object):
    def __init__(self, configs=[]):
        if configs == []:
            self.seq = self.get_default_seq()
        else:
            self.seq = self.get_aug_seq(configs)

    def __call__(self, img):
        img = self.seq.augment_image(img)
        return img
    
    def get_aug_seq(self, configs):
        seqence = []
        for cfg in configs:
            fn = getattr(iaa, cfg[0])
            if isinstance(cfg[1], list):
                seqence.append(fn(*cfg[1]))
            elif isinstance(cfg[1], dict):
                seqence.append(fn(**cfg[1]))
            else:
                raise ValueError("Augmenter parameter must be dict or list.")
        seq = iaa.Sequential(seqence, random_order=True)
        return seq
        
    def get_default_seq(self):
        seqence = [iaa.LinearContrast((0.8, 1.2)),
                   iaa.Grayscale((0, 1.0)),
                   iaa.GaussianBlur((0, 2)),
                   iaa.Multiply((0.8, 1.2)),
                   iaa.Add((-15, 15), per_channel=0.5),
                   ]
        seq = iaa.SomeOf(3, seqence, random_order=True)
        return seq

class SynthLines(object):
    def __init__(self, short_side=32):
        self.short_side = short_side

    def __call__(self, ):
        pass

    def synth_lines(self, nums=10000):
        pass
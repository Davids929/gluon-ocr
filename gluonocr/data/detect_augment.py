#coding=utf-8
import numpy as np
import imgaug
import imgaug.augmenters as iaa

class MaskAugmenter(object):
    def __init__(self, configs=[]):
        if len(configs)==0:
            self.aug = self.get_default_aug()
        else:
            self.aug = self.get_aug_seq(configs)    

    def __call__(self, img, mask):
        img, mask = self.aug(image=img, segmentation_maps=mask)
        return img, mask

    def get_aug_seq(self, configs):
        seqence = []
        for cfg in configs:
            seqence.append(getattr(iaa, cfg[0]))(**cfg[1])
        seq = iaa.Sequential(seqence, random_order=True)
        return seq

    def get_default_aug(self):
        pixel_seq = iaa.SomeOf(3, [iaa.LinearContrast((0.8, 1.2)),
                                   iaa.Multiply((0.8, 1.2)),
                                   iaa.Grayscale((0,1.0)),
                                   iaa.GaussianBlur((0, 1)),
                                   iaa.Add((-10, 10), per_channel=0.5),
                                   ])
        
        posi_seq  = iaa.SomeOf(2, [iaa.Affine(rotate=(-10, 10)),
                                   iaa.Fliplr(0.5),
                                   iaa.Crop(percent=([0.05, 0.1], [0.05, 0.1], 
                                                     [0.05, 0.1], [0.05, 0.1])),
                                   iaa.Resize((0.5, 3.0)),
                                    ])

        seq = iaa.Sequential([pixel_seq, posi_seq], random_order=True)
        return seq

class PointAugmenter(MaskAugmenter):
    def __init__(self, configs=[]):
        super(PointAugmenter, self).__init__(configs)
    
    def __call__(self, img, ploys):
        aug = self.aug.to_deterministic()
        origin_shape = img.shape
        img = aug.augment_image(img)
        poly_list = []
        for poly in ploys:
            keypoints = [imgaug.Keypoint(p[0], p[1]) for p in poly]
            keypoints = aug.augment_keypoints(
                            [imgaug.KeypointsOnImage(keypoints, shape=origin_shape)])[0].keypoints
            poly = [[p.x, p.y] for p in keypoints]
            poly_list.append(np.array(poly))
        return img, poly_list
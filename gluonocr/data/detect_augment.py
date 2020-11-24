#coding=utf-8
import cv2
import numpy as np
import imgaug
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

__all__ = ['MaskAugmenter', 'PointAugmenter', 'RandomCropData']

class MaskAugmenter(object):
    def __init__(self, configs=[]):
        if len(configs)==0:
            self.aug = self.get_default_aug()
        else:
            self.aug = self.get_aug_seq(configs)    

    def __call__(self, img, mask):
        mask = SegmentationMapsOnImage(mask, shape=img.shape)
        img, mask = self.aug(image=img, segmentation_maps=mask)
        return img, mask.get_arr()

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

    def get_default_aug(self):
        pixel_seq = iaa.SomeOf(3, [iaa.LinearContrast((0.8, 1.2)),
                                   iaa.Multiply((0.8, 1.2)),
                                   iaa.GaussianBlur((0, 1)),
                                   iaa.Add((-10, 10), per_channel=0.2)
                                   ])
        
        posi_seq  = iaa.SomeOf(1, [iaa.Affine(rotate=(-10, 10)),
                                   iaa.Fliplr(0.5),
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

class RandomCropData(object):
    
    def __init__(self, size=(512, 512), keep_ratio=True, 
                max_tries=20, min_crop_side_ratio=0.6,
                require_original_image=False):
        self.size = size
        self.keep_ratio = keep_ratio
        self.max_tries = max_tries
        self.min_crop_side_ratio = min_crop_side_ratio
        self.require_original_image = require_original_image

    def __call__(self, data):
        return self.process(data)

    def process(self, data):
        im = data['image']
        text_polys = data['polygons']
        ignore_tags = data['ignore_tags']

        all_care_polys = [
            text_polys[i] for i, tag in enumerate(ignore_tags) if not tag
        ]
        # 计算crop区域
        crop_x, crop_y, crop_w, crop_h = self.crop_area(im, all_care_polys)
        # crop 图片 保持比例填充
        scale_w = self.size[0] / crop_w
        scale_h = self.size[1] / crop_h
        scale   = min(scale_w, scale_h)
        h = int(crop_h * scale)
        w = int(crop_w * scale)
        if self.keep_ratio:
            padimg = np.zeros((self.size[1], self.size[0], im.shape[2]), im.dtype)
            padimg[:h, :w] = cv2.resize(
                im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w], (w, h))
            img = padimg
        else:
            img = cv2.resize(im[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w],
                            tuple(self.size))
        # crop 文本框
        text_polys_crop = []
        ignore_tags_crop = []
        for poly, tag in zip(text_polys, ignore_tags):
            poly = ((poly - (crop_x, crop_y)) * scale).tolist()
            if not self.is_poly_outside_rect(poly, 0, 0, w, h):
                text_polys_crop.append(poly)
                ignore_tags_crop.append(tag)

        data['image'] = img
        data['polygons'] = np.array(text_polys_crop)
        data['ignore_tags'] = ignore_tags_crop
        return data

    def is_poly_outside_rect(self, poly, x, y, w, h):
        poly = np.array(poly)
        if poly[:, 0].max() < x or poly[:, 0].min() > x + w:
            return True
        if poly[:, 1].max() < y or poly[:, 1].min() > y + h:
            return True
        return False

    def split_regions(self, axis):
        regions = []
        min_axis = 0
        for i in range(1, axis.shape[0]):
            if axis[i] != axis[i-1] + 1:
                region = axis[min_axis:i]
                min_axis = i
                regions.append(region)
        return regions

    def random_select(self, axis, max_size):
        xx = np.random.choice(axis, size=2)
        xmin = np.min(xx)
        xmax = np.max(xx)
        xmin = np.clip(xmin, 0, max_size - 1)
        xmax = np.clip(xmax, 0, max_size - 1)
        return xmin, xmax

    def region_wise_random_select(self, regions, max_size):
        selected_index = list(np.random.choice(len(regions), 2))
        selected_values = []
        for index in selected_index:
            axis = regions[index]
            xx = int(np.random.choice(axis, size=1))
            selected_values.append(xx)
        xmin = min(selected_values)
        xmax = max(selected_values)
        return xmin, xmax

    def crop_area(self, img, polys):
        h, w, _ = img.shape
        h_array = np.zeros(h, dtype=np.int32)
        w_array = np.zeros(w, dtype=np.int32)
        for points in polys:
            points = np.round(points, decimals=0).astype(np.int32)
            minx = np.min(points[:, 0])
            maxx = np.max(points[:, 0])
            w_array[minx:maxx] = 1
            miny = np.min(points[:, 1])
            maxy = np.max(points[:, 1])
            h_array[miny:maxy] = 1
        # ensure the cropped area not across a text
        h_axis = np.where(h_array == 0)[0]
        w_axis = np.where(w_array == 0)[0]

        if len(h_axis) == 0 or len(w_axis) == 0:
            return 0, 0, w, h

        h_regions = self.split_regions(h_axis)
        w_regions = self.split_regions(w_axis)

        for i in range(self.max_tries):
            if len(w_regions) > 1:
                xmin, xmax = self.region_wise_random_select(w_regions, w)
            else:
                xmin, xmax = self.random_select(w_axis, w)
            if len(h_regions) > 1:
                ymin, ymax = self.region_wise_random_select(h_regions, h)
            else:
                ymin, ymax = self.random_select(h_axis, h)

            if xmax - xmin < self.min_crop_side_ratio * w or ymax - ymin < self.min_crop_side_ratio * h:
                # area too small
                continue
            num_poly_in_rect = 0
            for poly in polys:
                if not self.is_poly_outside_rect(poly, xmin, ymin, xmax - xmin, ymax - ymin):
                    num_poly_in_rect += 1
                    break

            if num_poly_in_rect > 0:
                return xmin, ymin, xmax - xmin, ymax - ymin

        return 0, 0, w, h

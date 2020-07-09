#coding=utf-8

import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from .detect_dataset import DBDataset
from . import normalize_fn
from .detect_augment import PointAugmenter, RandomCropData
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator

class CLRSDataset(DBDataset):
    CLASSES = ('top-left', 'top-right', 'bottom-right', 'bottom-left')

    def __init__(self, img_dir, lab_dir, augment_fns=None, min_text_size=6, 
                 img_size=(640, 640), debug=False, mode='train'):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.debug   = debug
        self.mode    = mode
        self.img_size      = img_size
        self.augment_fns   = augment_fns
        self.min_text_size = min_text_size
        self.imgs_type   = ['jpg', 'jpeg', 'png', 'bmp']
        self.imgs_list, self.labs_list = self._get_items(img_dir, lab_dir)
        self.random_crop = RandomCropData(size=img_size)

    @property
    def classes(self):
        return type(self).CLASSES

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs_list[idx])
        lab_path = os.path.join(self.lab_dir, self.labs_list[idx])
        img_np   = cv2.imread(img_path)
        if img_np is None:
            return self.__getitem__(idx-1)
        
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        polygons, ignore_tags = self._load_ann(lab_path)

        if len(polygons) == 0:
            return self.__getitem__(idx-1)
        if self.augment_fns is not None:
            img_np, polygons = self.augment_fns(img_np, polygons)
        if self.mode != 'train':
            img_np, polygons = self.image_resize(img_np, polygons, self.img_size)
        data = {'image':img_np, 'polygons':polygons, 'ignore_tags':ignore_tags}
        if self.mode == 'train':
            data = self.random_crop(data)
        boxes, seg_gt, mask = self.gen_gt(data, self.img_size)
        if self.debug:
            pass
        image = self.padd_image(data['image'], self.img_size, layout='HWC')
        image = mx.nd.array(image, dtype='float32')
        image = normalize_fn(image)
        seg_gt = mx.nd.array(seg_gt, dtype='float32')
        mask   = mx.nd.array(mask, dtype='float32').expand_dims(axis=0)
        return image, boxes, seg_gt, mask

    def get_tight_rect(self, bb):
        ## bb: [x1, y1, x2, y2.......]
        bb = np.array(bb, dtype=np.float32)
        rect   = cv2.minAreaRect(bb)
        points = cv2.boxPoints(rect)
        points = list(points)
        ps = sorted(points, key = lambda x:x[0])

        if ps[1][1] > ps[0][1]:
            px1 = ps[0][0]
            py1 = ps[0][1]
            px4 = ps[1][0]
            py4 = ps[1][1]
        else:
            px1 = ps[1][0]
            py1 = ps[1][1]
            px4 = ps[0][0]
            py4 = ps[0][1]
        if ps[3][1] > ps[2][1]:
            px2 = ps[2][0]
            py2 = ps[2][1]
            px3 = ps[3][0]
            py3 = ps[3][1]
        else:
            px2 = ps[3][0]
            py2 = ps[3][1]
            px3 = ps[2][0]
            py3 = ps[2][1]
        return [px1, py1, px2, py2, px3, py3, px4, py4]

    def gen_gt(self, data, img_size=(512, 512)):
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        boxes   = []
        tl_mask = np.zeros(img_size, dtype=np.uint8)
        tr_mask = np.zeros(img_size, dtype=np.uint8)
        br_mask = np.zeros(img_size, dtype=np.uint8)
        bl_mask = np.zeros(img_size, dtype=np.uint8)
        mask    = np.ones(img_size, dtype=np.uint8)
        for i in range(polygons.shape[0]):
            polygon = polygons[i]
            polygon[:, 0] = np.clip(polygon[:, 0], 0, img_size[1])
            polygon[:, 1] = np.clip(polygon[:, 1], 0, img_size[0])
            x1, y1, x2, y2, x3, y3, x4, y4 = self.get_tight_rect(polygon)
            ## get box
            side1 = math.sqrt(math.pow(x2 - x1, 2) + math.pow(y2 - y1, 2))
            side2 = math.sqrt(math.pow(x3 - x2, 2) + math.pow(y3 - y2, 2))
            side3 = math.sqrt(math.pow(x4 - x3, 2) + math.pow(y4 - y3, 2))
            side4 = math.sqrt(math.pow(x1 - x4, 2) + math.pow(y1 - y4, 2))
            h = min(side1 + side3, side2 + side4)/2.0
            if h < self.min_text_size or ignore_tags[i]:
                cv2.fillPoly(mask, polygons[i].astype(np.int32)[np.newaxis, :, :], 0)
                continue
            theta = math.atan2(y2 - y1, x2 - x1)
            boxes.append(np.array([x1 - h/2, y1 - h/2, x1 + h/2, y1 + h/2, 0]))
            boxes.append(np.array([x2 - h/2, y2 - h/2, x2 + h/2, y2 + h/2, 1]))
            boxes.append(np.array([x3 - h/2, y3 - h/2, x3 + h/2, y3 + h/2, 2]))
            boxes.append(np.array([x4 - h/2, y4 - h/2, x4 + h/2, y4 + h/2, 3]))
            c1_x, c2_x = (x1 + x2)/2.0, (x2 + x3)/2.0
            c3_x, c4_x, c_x = (x3 + x4)/2.0, (x4 + x1)/2.0, (x1 + x2 + x3 + x4)/4.0 
            c1_y, c2_y  = (y1 + y2)/2.0, (y2 + y3)/2.0 
            c3_y, c4_y, c_y = (y3 + y4)/2.0, (y4 + y1)/2.0, (y1 + y2 + y3 + y4)/4.0
            poly = np.array([[x1,y1], [c1_x,c1_y], [c_x, c_y], [c4_x, c4_y]], dtype=np.int32)
            cv2.fillPoly(tl_mask, poly[np.newaxis, :, :], 1)
            poly = np.array([[c1_x,c1_y], [x2,y2], [c2_x, c2_y], [c_x,c_y]], dtype=np.int32)
            cv2.fillPoly(tr_mask, poly[np.newaxis, :, :], 1)
            poly = np.array([[c_x,c_y], [c2_x,c2_y], [c_x,c_y], [c3_x,c3_y]], dtype=np.int32)
            cv2.fillPoly(br_mask, poly[np.newaxis, :, :], 1)
            poly = np.array([[c4_x,c4_y], [c_x,c_y], [c3_x,c3_y], [x4,y4]], dtype=np.int32)
            cv2.fillPoly(bl_mask, poly[np.newaxis, :, :], 1)

        seg_gt = np.stack([tl_mask, tr_mask, br_mask, bl_mask], axis=0)
        if boxes == []:
            boxes = [[-1, -1, -1, -1, -1]]
        boxes  = np.array(boxes)
        return boxes, seg_gt, mask

class TargetGenerator(object):
    def __init__(self, anchors, iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2), **kwargs):
        self._anchors = anchors
        self._target_generator = SSDTargetGenerator(
                    iou_thresh=iou_thresh, stds=box_norm, **kwargs)

    def __call__(self, img, label, seg_gt, mask):
        gt_bboxes = mx.nd.array(label[np.newaxis, :, :4])
        gt_ids = mx.nd.array(label[np.newaxis, :, 4:5])
        cls_targets, box_targets, _ = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0], seg_gt, mask
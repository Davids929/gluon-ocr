import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from . import normalize_fn
from .make_seg_data import MakeSegDetectorData, MakeBorderMap
from .detect_augment import PointAugmenter

class DBDataset(Dataset):
    def __init__(self, img_dir, lab_dir, augment_fns=None, img_size=(640, 640),
                 min_text_size=8, shrink_ratio=0.4, debug=False):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.debug   = debug
        self.img_size    = img_size
        self.augment_fns = augment_fns
        self.imgs_type   = ['jpg', 'jpeg', 'png', 'bmp']
        self.imgs_list, self.labs_list = self._get_items(img_dir, lab_dir)
        self.get_label  = MakeSegDetectorData(min_text_size=min_text_size, shrink_ratio=shrink_ratio)
        self.get_border = MakeBorderMap(shrink_ratio=shrink_ratio)

    def _get_items(self, img_dir, lab_dir):
        file_list = os.listdir(img_dir)
        imgs_list, labs_list = [], [] 
        for file in file_list:
            tp = file.split('.')[-1]
            if tp.lower() not in self.imgs_type:
                continue
            lab_name = file + '.txt'
            if not os.path.exists(os.path.join(lab_dir, lab_name)):
                lab_name = file[:-len(tp)] + 'txt'
                if not os.path.exists(os.path.join(lab_dir, lab_name)):
                    continue
            imgs_list.append(file)
            labs_list.append(lab_name)
        return imgs_list, labs_list

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.imgs_list[idx])
        lab_path = os.path.join(self.lab_dir, self.labs_list[idx])
        img_np   = cv2.imread(img_path)
        if img_np is None:
            return self.__getitem__(idx-1)
        polygons, ignore_tags = self._load_ann(lab_path)
        if len(polygons) == 0:
            return self.__getitem__(idx-1)
        if self.augment_fns is not None:
            img_np, polygons = self.augment_fns(img_np, polygons)
        img_np, polygons = self.image_resize(img_np, polygons)
        data = {'image':img_np, 'polygons':polygons, 'ignore_tags':ignore_tags}
        data = self.get_label(data)
        data = self.get_border(data)
        image = mx.nd.array(data['image'], dtype='float32')
        if self.debug:
            aa =  data['gt'][0]*255
            cv2.imwrite('gt.jpg',aa.astype('uint8'))
            bb = data['thresh_map']*255 * data['thresh_mask']
            cv2.imwrite('thresh.jpg', bb.astype('uint8'))
            cv2.imwrite('image.jpg', image.astype('uint8').asnumpy())
        image = normalize_fn(image)
        gt    = mx.nd.array(data['gt'], dtype='float32')
        mask  = mx.nd.array(data['mask'], dtype='float32').expand_dims(axis=0)
        thresh_map = mx.nd.array(data['thresh_map'], dtype='float32').expand_dims(axis=0)
        thresh_mask= mx.nd.array(data['thresh_mask'], dtype='float32').expand_dims(axis=0)
        return image, gt, mask, thresh_map, thresh_mask

    def image_resize(self, img, polys):
        h, w = img.shape[:2]
        scale = self.img_size[0]/max(h,w)
        new_h = int(h*scale)
        new_w = int(w*scale)
        img   = cv2.resize(img, (new_w, new_h)) 
        new_img = np.zeros((self.img_size[1], self.img_size[0], img.shape[2]), img.dtype)
        new_img[:new_h, :new_w, :] img
        ploy_list = []
        for poly in polys:
            poly = (np.array(poly)*scale).tolist()
            ploy_list.append(poly)
        return new_img, ploy_list

    def _load_ann(self, lab_path):
        with open(lab_path, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
        ploy_list = []
        ignore_list = []
        for l in lines:
            lst  = l.strip().split(',')
            num_points = ((len(lst)-1)//2)*2
            try:
                points = [float(i) for i in lst[:num_points]]
            except:
                print(lab_path, ',', l)
                continue
            poly = np.array(points).reshape(-1, 2).tolist()
            text = ','.join(lst[8:])
            ignore = text == '###' or text == ''
            ploy_list.append(text)
            ignore_list.append(ignore)

        return ploy_list, ignore_list
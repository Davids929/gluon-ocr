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
        img_np, polygons = self.image_resize(img_np, polygons, self.img_size)
        data = {'image':img_np, 'polygons':polygons, 'ignore_tags':ignore_tags}
        data = self.get_label(data)
        data = self.get_border(data)
        
        if self.debug:
            aa =  data['gt']*255
            cv2.imwrite('gt.jpg',aa.astype('uint8'))
            bb = data['thresh_map']*255 * data['thresh_mask']
            cv2.imwrite('thresh.jpg', bb.astype('uint8'))
            image = data['image'].astype('uint8')
            for poly in polygons:
                cv2.polylines(image, [poly.astype('int32')], True, (0, 255, 0), 2)
            cv2.imwrite('image.jpg', image)
        image = self.padd_image(data['image'], self.img_size, layout='HWC')
        image = mx.nd.array(image, dtype='float32')
        image = normalize_fn(image)
        gt    = self.padd_image(data['gt'], self.img_size, layout='HW')
        gt    = mx.nd.array(gt, dtype='float32').expand_dims(axis=0)
        mask  = mx.nd.array(data['mask'], dtype='float32').expand_dims(axis=0)
        thresh_map = self.padd_image(data['thresh_map'], self.img_size, layout='HW')
        thresh_map = mx.nd.array(thresh_map, dtype='float32').expand_dims(axis=0)
        thresh_mask= self.padd_image(data['thresh_mask'], self.img_size, layout='HW')
        thresh_mask= mx.nd.array(thresh_mask, dtype='float32').expand_dims(axis=0)
        return image, gt, mask, thresh_map, thresh_mask

    def image_resize(self, img, polys, img_size):
        h, w = img.shape[:2]
        scale = img_size[0]/max(h,w)
        new_h = int(h*scale)
        new_w = int(w*scale)
        img   = cv2.resize(img, (new_w, new_h)) 
        # new_img = np.zeros((img_size[1], img_size[0], img.shape[2]), img.dtype)
        # new_img[:new_h, :new_w, :] = img
        ploy_list = []
        for poly in polys:
            ploy_list.append(poly*scale)
        return img, ploy_list

    def padd_image(self, img, size, layout="CHW"):
        if layout == "CHW":
            c, h, w = img.shape
            new_img = np.zeros((c, size[0], size[1]))
            new_img[:, :h, :w] = img
        elif layout == "HWC":
            h, w, c = img.shape
            new_img = np.zeros((size[0], size[1], c))
            new_img[:h, :w, :] = img
        elif layout == "HW":
            h, w = img.shape
            new_img = np.zeros((size[0], size[1]))
            new_img[:h, :w] = img
        else:
            raise ValueError('Layout type is not support.')
        return new_img
    def _load_ann(self, lab_path):
        with open(lab_path, 'r', encoding='utf-8') as fi:
            lines = fi.readlines()
        ploy_list = []
        ignore_list = []
        for l in lines:
            lst  = l.strip().split(',')
            num_points = 8#((len(lst)-1)//2)*2
            try:
                points = [float(i) for i in lst[:num_points]]
            except:
                print(lab_path, ',', l)
                continue
            poly = np.array(points).reshape(-1, 2)
            text = ','.join(lst[num_points:])
            ignore = text == '###' or text == ''
            ploy_list.append(poly)
            ignore_list.append(ignore)

        return ploy_list, ignore_list

class EASTDataset(DBDataset):
    def __init__(self, img_dir, lab_dir, augment_fns=None, img_size=(640, 640),
                 min_text_size=8, shrink_ratio=0.4, debug=False):
        super(EASTDataset, self).__init__(img_dir, lab_dir, augment_fns=augment_fns, 
                                          img_size=img_size,min_text_size=min_text_size, 
                                          shrink_ratio=shrink_ratio, debug=debug)
       
        self.get_label  = MakeSegDetectorData(min_text_size=min_text_size, 
                                              shrink_ratio=shrink_ratio,
                                              gen_geometry=True)
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
        img_np, polygons = self.image_resize(img_np, polygons, self.img_size)
        data = {'image':img_np, 'polygons':polygons, 'ignore_tags':ignore_tags}
        data = self.get_label(data)
       
        if self.debug:
            aa = data['gt']*255
            cv2.imwrite('gt.jpg',aa.astype('uint8'))
            bb = data['mask']*255
            cv2.imwrite('thresh.jpg', bb.astype('uint8'))
            image = data['image'].astype('uint8')
            for poly in polygons:
                cv2.polylines(image, [poly.astype('int32')], True, (0, 255, 0), 2)
            cv2.imwrite('image.jpg', image)
        image = self.padd_image(data['image'], self.img_size, layout='HWC')
        image = mx.nd.array(image, dtype='float32')
        image = normalize_fn(image)
        gt    = self.padd_image(data['gt'], self.img_size, layout='HW')
        gt    = mx.nd.array(gt[::4, ::4], dtype='float32').expand_dims(axis=0)
        mask  = self.padd_image(data['mask'], self.img_size, layout='HW')
        mask  = mx.nd.array(mask[::4, ::4], dtype='float32').expand_dims(axis=0)
        geo_map = self.padd_image(data['geo_map'], self.img_size, layout='CHW')
        geo_map = mx.nd.array(geo_map[:, ::4, ::4], dtype='float32')
        return image, gt, mask, geo_map

import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from . import normalize_fn
from .make_seg_data import MakeShrinkMap, MakeBorderMap
from .detect_augment import PointAugmenter, RandomCropData
from gluoncv.model_zoo.ssd.target import SSDTargetGenerator

__all__ = ['DBDataset', 'EASTDataset', 'CLRSDataset', 'CLRSTrainTransform']

class DBDataset(Dataset):
    def __init__(self, img_dir, lab_dir, augment_fns=None, img_size=(640, 640),
                 min_text_size=8, shrink_ratio=0.4, debug=False, mode='train'):
        self.img_dir = img_dir
        self.lab_dir = lab_dir
        self.debug   = debug
        self.mode    = mode
        self.img_size    = img_size
        self.augment_fns = augment_fns
        self.imgs_type   = ['jpg', 'jpeg', 'png', 'bmp']
        self.imgs_list, self.labs_list = self._get_items(img_dir, lab_dir)
        self.random_crop = RandomCropData(size=img_size)
        self.get_label   = MakeShrinkMap(min_text_size=min_text_size, shrink_ratio=shrink_ratio)
        self.get_border  = MakeBorderMap(shrink_ratio=shrink_ratio)

    def _get_items(self, img_dirs, lab_dirs):
        if not isinstance(img_dirs, list):
            img_dirs = [img_dirs]
        if not isinstance(lab_dirs, list):
            lab_dirs = [lab_dirs]
        imgs_list, labs_list = [], [] 
        for img_dir, lab_dir in zip(img_dirs, lab_dirs):
            file_list = os.listdir(img_dir)
            for file in file_list:
                tp = file.split('.')[-1]
                if tp.lower() not in self.imgs_type:
                    continue
                lab_name = file + '.txt'
                lab_path = os.path.join(lab_dir, lab_name)
                if not os.path.exists(lab_path):
                    lab_name = file[:-len(tp)] + 'txt'
                    lab_path = os.path.join(lab_dir, lab_name)
                    if not os.path.exists(lab_path):
                        continue
                imgs_list.append(os.path.join(img_dir, file))
                labs_list.append(lab_path)
        return imgs_list, labs_list

    def __len__(self):
        return len(self.imgs_list)
    
    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.imgs_list[idx])
        # lab_path = os.path.join(self.lab_dir, self.labs_list[idx])
        img_path, lab_path = self.imgs_list[idx], self.labs_list[idx]
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
        data = self.get_label(data)
        data = self.get_border(data)
        
        if self.debug:
            gt =  data['gt']*255
            cv2.imwrite('gt.jpg',gt.astype('uint8'))
            mask = data['mask']*255
            cv2.imwrite('mask.jpg', mask.astype('uint8'))
            thresh_map = data['thresh_map']*255 
            cv2.imwrite('thresh.jpg', thresh_map.astype('uint8'))
            thresh_mask = data['thresh_mask']*255
            cv2.imwrite('thresh_mask.jpg', thresh_mask.astype('uint8'))
            image = data['image'].astype('uint8')
            for poly in data['polygons']:
                cv2.polylines(image, [poly.astype('int32')], True, (0, 255, 0), 2)
            cv2.imwrite('image.jpg', image)
        
        image = self.padd_image(data['image'], self.img_size, layout='HWC')
        image = mx.nd.array(image, dtype='float32')
        image = normalize_fn(image)
        gt    = self.padd_image(data['gt'], self.img_size, layout='HW')
        gt    = mx.nd.array(gt, dtype='float32').expand_dims(axis=0)
        mask  = self.padd_image(data['mask'], self.img_size, layout='HW')
        mask  = mx.nd.array(mask, dtype='float32').expand_dims(axis=0)
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
        ploy_list = []
        for poly in polys:
            ploy_list.append(poly*scale)
        return img, ploy_list

    def padd_image(self, img, size, layout="CHW"):
        if layout == "CHW":
            c, h, w = img.shape
            new_img = np.zeros((c, size[0], size[1]))
            new_img[:, :h, :w] = img.copy()
        elif layout == "HWC":
            h, w, c = img.shape
            new_img = np.zeros((size[0], size[1], c))
            new_img[:h, :w, :] = img.copy()
        elif layout == "HW":
            h, w = img.shape
            new_img = np.zeros((size[0], size[1]))
            new_img[:h, :w] = img.copy()
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
                 min_text_size=8, shrink_ratio=0.3, debug=False, mode='train'):
        super(EASTDataset, self).__init__(img_dir, lab_dir, augment_fns=augment_fns, 
                                          img_size=img_size,min_text_size=min_text_size, 
                                          shrink_ratio=shrink_ratio, debug=debug)
       
        self.get_label.gen_geometry = True

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.imgs_list[idx])
        # lab_path = os.path.join(self.lab_dir, self.labs_list[idx])
        img_path, lab_path = self.imgs_list[idx], self.labs_list[idx]
        img_np   = cv2.imread(img_path)
        if img_np is None:
            return self.__getitem__(idx-1)
        if random.uniform(0, 1) > 0.5:
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
        data = self.get_label(data)
        if self.debug:
            score = data['gt']*255
            cv2.imwrite('gt.jpg',score.astype('uint8'))
            mask = data['mask']*255
            cv2.imwrite('mask.jpg', mask.astype('uint8'))
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
        self.random_crop = RandomCropData(size=img_size, max_tries=5)

    @property
    def classes(self):
        return type(self).CLASSES

    def __getitem__(self, idx):
        # img_path = os.path.join(self.img_dir, self.imgs_list[idx])
        # lab_path = os.path.join(self.lab_dir, self.labs_list[idx])
        img_path, lab_path = self.imgs_list[idx], self.labs_list[idx]
        img_np   = cv2.imread(img_path)
        if img_np is None:
            return self.__getitem__(idx-1)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        polygons, ignore_tags = self._load_ann(lab_path)
        if len(polygons) == 0:
            return self.__getitem__(idx-1)
        
        if self.augment_fns is not None:
            img_np, polygons = self.augment_fns(img_np, polygons)
        data = {'image':img_np, 'polygons':polygons, 'ignore_tags':ignore_tags}
        if self.mode == 'train':
            data = self.random_crop(data)
        else:
            data['image'], data['polygons'] = self.image_resize(data['image'], 
                                                                data['polygons'], 
                                                                self.img_size)

        boxes, seg_gt, mask = self.gen_gt(data, self.img_size)
        if len(boxes)<4:
            return self.__getitem__(idx-1)
    
        image = self.padd_image(data['image'], self.img_size, layout='HWC')
        if self.debug:
            img_np = image.copy()
            show = np.zeros(img_np.shape, dtype=np.uint8)
            show[seg_gt[0]==1, :] = [255, 255, 255]
            show[seg_gt[1]==1, :] = [255, 0, 0]
            show[seg_gt[2]==1, :] = [0, 255, 0]
            show[seg_gt[3]==1, :] = [0, 0, 255]
            for i,box in enumerate(boxes):
                if box[4] == 0:
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 255, 255), 2)
                elif box[4] == 1:
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                elif box[4] == 2:
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                else:
                    cv2.rectangle(img_np, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                if (i+1)%4==0 and box[0]>-1:
                    tl = [(boxes[i-3][0] + boxes[i-3][2])/2, (boxes[i-3][1] + boxes[i-3][3])/2]
                    tr = [(boxes[i-2][0] + boxes[i-2][2])/2, (boxes[i-2][1] + boxes[i-2][3])/2]
                    br = [(boxes[i-1][0] + boxes[i-1][2])/2, (boxes[i-1][1] + boxes[i-1][3])/2]
                    bl = [(boxes[i][0] + boxes[i][2])/2, (boxes[i][1] + boxes[i][3])/2]
                    contour = np.array([tl, tr, br, bl], dtype=np.int32)
                    cv2.drawContours(img_np, [contour], -1, (255, 255, 0), 2)
            seg_mask = np.stack([mask, mask, mask], axis=-1)*255
            show = np.concatenate([img_np, show, seg_mask], axis=1).astype(np.uint8)
            show = cv2.cvtColor(show, cv2.COLOR_RGB2BGR)
            cv2.imwrite('clrs_label.jpg', show)

        image = mx.nd.array(image, dtype='uint8')
        image = normalize_fn(image)
        boxes = mx.nd.array(boxes, dtype='float32')
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
        polygons    = data['polygons']
        ignore_tags = data['ignore_tags']
        boxes   = []
        tl_mask = np.zeros(img_size, dtype=np.uint8)
        tr_mask = np.zeros(img_size, dtype=np.uint8)
        br_mask = np.zeros(img_size, dtype=np.uint8)
        bl_mask = np.zeros(img_size, dtype=np.uint8)
        mask    = np.ones(img_size, dtype=np.uint8)
        for i in range(len(polygons)):
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
            #theta = math.atan2(y2 - y1, x2 - x1)
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
            poly = np.array([[c_x,c_y], [c2_x,c2_y], [x3,y3], [c3_x,c3_y]], dtype=np.int32)
            cv2.fillPoly(br_mask, poly[np.newaxis, :, :], 1)
            poly = np.array([[c4_x,c4_y], [c_x,c_y], [c3_x,c3_y], [x4,y4]], dtype=np.int32)
            cv2.fillPoly(bl_mask, poly[np.newaxis, :, :], 1)

        seg_gt = np.stack([tl_mask, tr_mask, br_mask, bl_mask], axis=0)
        if boxes == []:
            boxes = [[-1, -1, -1, -1, -1]]
        else:
            boxes  = np.array(boxes, dtype=np.float32)
            if np.isinf(boxes).sum() > 0:
                print('label has inf.')
        return boxes, seg_gt, mask

class CLRSTrainTransform(object):
    def __init__(self, anchors, iou_thresh=0.5, box_norm=(0.1, 0.1, 0.2, 0.2), **kwargs):
        self._anchors = anchors
        self._target_generator = SSDTargetGenerator(
                    iou_thresh=iou_thresh, stds=box_norm, **kwargs)

    def __call__(self, img, label, seg_gt, mask):
        label = label.expand_dims(0)
        gt_bboxes = label[:, :, :4]
        gt_ids = label[:, :, 4:5]
        cls_targets, box_targets, box_mask = self._target_generator(
            self._anchors, None, gt_bboxes, gt_ids)
        return img, cls_targets[0], box_targets[0], box_mask, seg_gt, mask

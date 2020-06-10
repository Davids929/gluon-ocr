#coding=utf-8
import argparse
import os
import cv2
import sys
import numpy as np
import math
import mxnet as mx
from mxnet import gluon
sys.path.append(os.path.expanduser('~/demo/gluon-ocr'))
from gluonocr.utils.east_postprocess import EASTPostPocess

parser = argparse.ArgumentParser(description='Text Recognition Training')
parser.add_argument('--json-path', type=str, help='model file path.')
parser.add_argument('--params-path', type=str, help='params file path.')
parser.add_argument('--image-path', type=str, help='image path')
parser.add_argument('--result-dir', type=str, default='./demo_results/', help='path to save results')
parser.add_argument('--image-short-side', type=int, default=736,
                    help='The threshold to replace it in the representers')
parser.add_argument('--score-thresh', type=float, default=0.8,
                    help='The threshold to replace it in the representers')
parser.add_argument('--conver-thresh', type=float, default=0.1,
                    help='The threshold to replace it in the representers')
parser.add_argument('--nms-thresh', type=float, default=0.2,
                    help='The threshold to replace it in the representers')
parser.add_argument('--visualize', action='store_true',
                    help='visualize maps in tensorboard')
parser.add_argument('--polygon', action='store_true',
                    help='output polygons if true')
parser.add_argument('--gpu-id', type=int, default='0')

img_types = ['jpg', 'png', 'jpeg', 'bmp']

class Demo(object):
    def __init__(self, args):
        self.args = args
        if args.gpu_id:
            self.ctx = mx.cpu()
        else:
            self.ctx = mx.gpu(int(args.gpu_id))
        self.net = gluon.SymbolBlock.imports(args.json_path, ['data'],
                                             args.params_path, ctx=self.ctx)
        
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)
        params    = {'score_thresh':args.score_thresh,
                     'cover_thresh':args.cover_thresh,
                     'nms_thresh':args.nms_thresh} 
        self.postpro = EASTPostPocess(params)

    def resize_image(self, img, max_scale=1024):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args.image_short_side
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
            if new_width > max_scale:
                new_width = max_scale
                new_height = int(math.ceil(new_width / width * height / 32) * 32)
        else:
            new_width = self.args.image_short_side
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
            if new_height> max_scale:
                new_height = max_scale
                new_width = int(math.ceil(new_height / height * width / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        #img = np.rot90(img, 3)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        return img, original_shape
        
        
    def inference(self, image, visualize=False):
        if os.path.isdir(image):
            file_list  = os.listdir(image)
            image_list = [os.path.join(image, i) for i in file_list if i.split('.')[-1].lower() in img_types]
        else:
            image_list = [image]
        for image_path in image_list:
            img, origin_shape = self.load_image(image_path)
            origin_h, origin_w = origin_shape
            h, w = img.shape[2:]
            score, geo_map = self.net(img)
            print(image_path)
            score = score.asnumpy()
            geo_map = geo_map.asnumpy()
            ratio_h = 4*origin_h/h
            ratio_w = 4*origin_w/w
            boxes = self.postpro(score, geo_map, [(ratio_h, ratio_w)])
            if len(boxes) == 0:
                return 0
            boxes[:, :, 0] = np.clip(boxes[:, :, 0], 0, origin_w)
            boxes[:, :, 1] = np.clip(boxes[:, :, 1], 0, origin_h)
            if not os.path.isdir(self.args.result_dir):
                os.mkdir(self.args.result_dir)
            if visualize:
                vis_image = self.visualize(image_path, score, boxes)
                cv2.imwrite(os.path.join(self.args.result_dir, os.path.basename(image_path)), vis_image)

    def visualize_heatmap(self, headmap, canvas=None):
        
        if canvas is None:
            pred_image = (headmap * 255).astype(np.uint8)
            pred_image = np.stack([pred_image]*3, axis=-1)
        else:
            pred_image = (np.expand_dims(headmap, 2) + 1) / 2 * canvas.astype('float32')
            pred_image = pred_image.astype(np.uint8)
        return pred_image

    def visualize(self, image_path, pred, boxes):
        
        original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        #original_image = np.rot90(original_image, 3)
        original_shape = original_image.shape
        pred_canvas = original_image.copy().astype(np.uint8)
        origin_h, origin_w = original_shape[:2]
        heatmap  = cv2.resize(pred, (origin_w, origin_h))
        heatmap  = self.visualize_heatmap(heatmap, original_image)
        bina_map = pred>self.args.thresh
        bina_map = cv2.resize(bina_map.astype('uint8')*255, (origin_w, origin_h), 
                            interpolation=cv2.INTER_NEAREST)
        bina_map = np.stack([bina_map]*3, axis=-1)
        
        for box in boxes:
            box = np.array(box).astype(np.int32).reshape(-1, 2)
            cv2.polylines(pred_canvas, [box], True, (0, 255, 0), 2)
        pred_canvas = np.concatenate((pred_canvas, bina_map, heatmap), axis=1)
        return pred_canvas
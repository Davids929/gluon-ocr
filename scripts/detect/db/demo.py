#!python3
import argparse
import os
import cv2
import numpy as np
import math
import mxnet as mx
from mxnet import gluon
import sys
sys.path.append(os.path.expanduser('~/gluon-ocr'))
from gluonocr.post_process import DBPostProcess

parser = argparse.ArgumentParser(description='Text Detection inference.')
parser.add_argument('--model-path', type=str, help='model file path.')
parser.add_argument('--params-path', type=str, help='params file path.')
parser.add_argument('--image-path', type=str, help='image path')
parser.add_argument('--result-dir', type=str, default='./demo_results/', help='path to save results')
parser.add_argument('--image-short-side', type=int, default=736,
                    help='The threshold to replace it in the representers')
parser.add_argument('--thresh', type=float, default=0.3,
                    help='The threshold to replace it in the representers')
parser.add_argument('--box-thresh', type=float, default=0.6,
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
        self.net = gluon.SymbolBlock.imports(args.model_path, ['data'],
                                             args.params_path, ctx=self.ctx)
        
        self.mean = (0.485, 0.456, 0.406)
        self.std  = (0.229, 0.224, 0.225)
        self.struct = DBPostProcess(thresh=args.thresh, box_thresh=args.box_thresh)

    def resize_image(self, img, max_scale=2048):
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
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=self.mean, std=self.std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        return img, original_shape
        
        
    def inference(self, image, visualize=False):
        if os.path.isdir(image):
            image = os.path.expanduser(image)
            file_list  = os.listdir(image)
            image_list = [os.path.join(image, i) for i in file_list if i.split('.')[-1].lower() in img_types]
        else:
            image_list = [os.path.expanduser(image)]
        
        if not os.path.isdir(self.args.result_dir):
            os.mkdir(self.args.result_dir)
        
        for image_path in image_list:
            img, origin_shape = self.load_image(image_path)
            origh_h, origin_w = origin_shape
            outputs = self.net(img)
            print(image_path)
            if isinstance(outputs, (tuple,list)):
                pred = outputs[0]
            else:
                pred = outputs
            pred = pred.asnumpy()[0,0]
            
            if self.args.polygon:
                boxes, scores = self.struct.polygons_from_bitmap(pred, origin_w, origh_h)
            else:
                boxes, scores = self.struct.boxes_from_bitmap(pred, origin_w, origh_h)
            
            save_name = '.'.join(os.path.basename(image_path).split('.')[:-1]) + '.txt'
            save_path = os.path.join(self.args.result_dir, save_name)
            self.save_detect_res(boxes, save_path)

            if visualize:
                vis_image = self.visualize(image_path, pred, boxes)
                cv2.imwrite(os.path.join(self.args.result_dir, os.path.basename(image_path)), vis_image)

    def save_detect_res(self, boxes, save_path):
        num = boxes.shape[0]
        boxes = np.reshape(boxes.astype(np.int32), (num, 8)).tolist()
        fi_w  = open(save_path, 'w', encoding='utf-8') 
        for i in range(num):
            box = boxes[i]
            text = ','.join([str(j) for j in box]) + '\n'
            fi_w.write(text)
        fi_w.close()

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

if __name__ == '__main__':
    args = parser.parse_args()
    demo = Demo(args)
    demo.inference(args.image_path, visualize=args.visualize)
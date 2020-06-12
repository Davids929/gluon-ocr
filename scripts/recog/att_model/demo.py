#coding=utf-8
import argparse
import os
import cv2
import sys
import numpy as np
import math
import time
import mxnet as mx
from mxnet import gluon
sys.path.append(os.path.expanduser('~/demo/gluon-ocr'))

parser = argparse.ArgumentParser(description='Text Recognition inference.')
parser.add_argument('--model-path', type=str, help='model file path.')
parser.add_argument('--params-path', type=str, help='params file path.')
parser.add_argument('--image-path', type=str, help='image path')
parser.add_argument('--short-side', type=int, default=32,
                    help='The short side of image.')
parser.add_argument('--voc-path', type=str, default='',
                        help='the path of vocabulary.')
parser.add_argument('--sos', type=int, default=0,
                    help='start symbol.')
parser.add_argument('--eos', type=int, default=1,
                    help='end symbol.')            
parser.add_argument('--gpu-id', type=str, default='0')

img_types = ['jpg', 'png', 'jpeg', 'bmp']
mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

class Demo(object):
    def __init__(self, args):
        self.args = args
        if not args.gpu_id:
            self.ctx = mx.cpu()
        else:
            self.ctx = mx.gpu(int(args.gpu_id))
        self.net = gluon.SymbolBlock.imports(args.model_path, ['data0', 'data1', 'data2', 'data3', 'data4'],
                                             args.params_path, ctx=self.ctx)
        self.word2id = self.load_voc(args.voc_path)
        self.words   = list(self.word2id.keys())
        
    def load_voc(self, voc_path):
        word2id_dict = {'<s>':args.sos, '</s>':args.eos}
        with open(voc_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        idx = len(word2id_dict)
        for line in line_list:
            word = line.strip('\n')[0]
            word2id_dict[word] = idx
            idx = idx + 1
        return word2id_dict

    def resize_image(self, img, max_scale=1440, min_divisor=8):
        height, width = img.shape[:2]
        if width/height> max_scale/args.short_side:
            img = cv2.resize(img, (max_scale, args.short_side))
        else:
            new_height = args.short_side
            new_width = int(math.ceil(new_height / height * width / min_divisor) * min_divisor)
            img = cv2.resize(img, (new_width, new_height))
        return img
        
    def load_data(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        h, w = img.shape
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        mask  = mx.nd.ones(shape=(1, w//8), dtype='float32', ctx=self.ctx)
        ## decoder lstm state
        h_state = mx.nd.zeros(shape=(2, 1, 512), dtype='float32', ctx=self.ctx)
        c_state = mx.nd.zeros(shape=(2, 1, 512), dtype='float32', ctx=self.ctx)
        return img, mask, h_state, c_state
    
    def ids2text(self, ids):
        chrs = []
        for idx in ids:
            if idx == args.eos:
                break
            chrs.append(self.words[int(idx)])
        text = ''.join(chrs)
        return text

    def inference(self, image, visualize=False):
        if os.path.isdir(image):
            file_list  = os.listdir(image)
            image_list = [os.path.join(image, i) for i in file_list if i.split('.')[-1].lower() in img_types]
        else:
            image_list = [image]
        for image_path in image_list:
            data = self.load_data(image_path)
            time1 = time.time()
            outs  = self.net(*data)
            time2 = time.time()
            outs  = outs.asnumpy()[0].tolist()
            text  = self.ids2text(outs)
            print(image_path, 'recog results:', text) 
    
if __name__ == '__main__':
    args = parser.parse_args()
    
    demo = Demo(args)
    demo.inference(args.image_path, visualize=args.visualize)
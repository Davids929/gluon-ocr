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
sys.path.append(os.path.expanduser('~/gluon-ocr'))

parser = argparse.ArgumentParser(description='Text Recognition inference.')
parser.add_argument('--model-path', type=str, help='model file path.')
parser.add_argument('--params-path', type=str, help='params file path.')
parser.add_argument('--image-path', type=str, help='image path')
parser.add_argument('--short-side', type=int, default=32,
                    help='The short side of image.')
parser.add_argument('--voc-path', type=str, default='',
                        help='the path of vocabulary.')
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
        self.net = gluon.SymbolBlock.imports(args.model_path, ['data0', 'data1'],
                                             args.params_path, ctx=self.ctx)
        self.word2id = self.load_voc(args.voc_path)
        self.words   = list(self.word2id.keys())
        self.voc_size = len(self.words)

    def load_voc(self, voc_path):
        word2id_dict = {}
        with open(voc_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        idx = len(word2id_dict)
        for line in line_list:
            word = line.strip('\n')[0]
            word2id_dict[word] = idx
            idx = idx + 1
        return word2id_dict

    def resize_image(self, img, max_scale=1440, min_divisor=4):
        height, width = img.shape[:2]
        if width/height> max_scale/args.short_side:
            img = cv2.resize(img, (max_scale, args.short_side))
        else:
            new_height = args.short_side
            new_width = int(math.ceil(new_height / height * width / min_divisor) * min_divisor)
            img = cv2.resize(img, (new_width, new_height))
        return img
        
    def load_data(self, image_path):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        h, w = img.shape[:2]
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = np.stack([img, img, img], axis=-1)
        img = mx.nd.array(img)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        return img
    
    def ctc_ids2text(self, ids, blank):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        n = len(ids)
        words = []
        for i in range(n):
            if ids[i]!=blank and (not (i>0 and ids[i-1]==ids[i])):
                words.append(self.words[int(ids[i])])
        text = ''.join(words)
        return text

    def inference(self, image):
        if os.path.isdir(image):
            file_list  = os.listdir(image)
            image_list = [os.path.join(image, i) for i in file_list if i.split('.')[-1].lower() in img_types]
        else:
            image_list = [image]
        image_list.sort()
        text_list = []
        for image_path in image_list:
            data  = self.load_data(image_path)
            time1 = time.time()
            pred  = self.net(data)
            time2 = time.time()
            pred  = mx.nd.softmax(pred)
            pred  = mx.nd.argmax(pred, axis=-1) 
            outs  = pred.asnumpy()[0].tolist()
            text  = self.ctc_ids2text(outs, self.voc_size)
            text_list.append(text)
            print(image_path, 'recog results:', text) 
        return text_list

    def evaluate(self, image):
        with open(image, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        nums  = len(line_list)
        count = 0
        for line in line_list:
            path, lab = line.strip('\n').split('\t')
            try:
                pred = self.inference(path)
            except:
                continue
            if pred[0] == lab:
                count += 1
        print('accuracy:', count/nums)

    def test(self, img_dir, save_path):
        file_list  = os.listdir(img_dir)
        image_list = [i for i in file_list if i.split('.')[-1].lower() in img_types]
        image_list.sort()
        fi_w = open(save_path, 'w', encoding='utf-8')
        for img in image_list:
            path = os.path.join(img_dir, img)
            pred = self.inference(path)
            fi_w.write(img + '\t' + pred[0] + '\n')

        fi_w.close()

if __name__ == '__main__':
    args = parser.parse_args()
    
    demo = Demo(args)
    save_path = '/home/idcard/data/scene_text_lines/test_results0728.txt'
    demo.test(args.image_path, save_path)
#!python3
import argparse
import os
import cv2
import numpy as np
from PIL import ImageFont, Image, ImageDraw
import math
import mxnet as mx
from mxnet import gluon
import sys
sys.path.append(os.path.expanduser('~/demo/gluon-ocr'))
from gluonocr.post_process import DBPostProcess
from gluonocr.data import crop_patch

parser = argparse.ArgumentParser(description='End2end ocr recognition inference.')
parser.add_argument('--db-model-path', type=str, help='DB model file path.')
parser.add_argument('--db-params-path', type=str, help='DB params file path.')
parser.add_argument('--crnn-model-path', type=str, help='CRNN model file path.')
parser.add_argument('--crnn-params-path', type=str, help='CRNN params file path.')
parser.add_argument('--image-path', type=str, help='image path')
parser.add_argument('--voc-path', type=str, help='the path of vocabulary.')
parser.add_argument('--font-path', type=str, help='the path of font.')
parser.add_argument('--result-dir', type=str, default='./demo_results/', help='path to save results')
parser.add_argument('--min-scale', type=int, default=736)
parser.add_argument('--max-scale', type=int, default=2048)
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

mean = (0.485, 0.456, 0.406)
std  = (0.229, 0.224, 0.225)

class OCRModel(object):
    def __init__(self, args):
        if args.gpu_id<0:
            self.ctx = mx.cpu()
        else:
            self.ctx = mx.gpu(int(args.gpu_id))
        self.db = gluon.SymbolBlock.imports(args.db_model_path, ['data'],
                                             args.db_params_path, ctx=self.ctx)
        self.crnn = gluon.SymbolBlock.imports(args.crnn_model_path, ['data'],
                                              args.crnn_params_path, ctx=self.ctx)
        
        self.post_process = DBPostProcess(thresh=args.thresh, box_thresh=args.box_thresh)

        with open(args.voc_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        self.words = [line.strip('\n')[0] for line in line_list]
        self.voc_size = len(self.words)
        self.args = args

    def resize_image(self, img, min_scale=736, min_divisor=32, max_scale=3072):
        height, width, _ = img.shape
        if height < width:
            new_height = min_scale
            new_width = int(math.ceil(new_height / height * width / min_divisor) * min_divisor)
            if new_width > max_scale:
                new_width = max_scale
                new_height = int(math.ceil(new_width / width * height / min_divisor) * min_divisor)
        else:
            new_width = min_scale
            new_height = int(math.ceil(new_width / width * height / min_divisor) * min_divisor)
            if new_height> max_scale:
                new_height = max_scale
                new_width = int(math.ceil(new_height / height * width / min_divisor) * min_divisor)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

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

    def detect(self, img_np):
        origh_h, origin_w = img_np.shape[:2]
        img_np = self.resize_image(img_np, min_scale=self.args.min_scale, max_scale=self.args.max_scale)
        img = mx.nd.array(img_np)
        img = mx.nd.image.to_tensor(img)
        img = mx.nd.image.normalize(img, mean=mean, std=std)
        img = img.expand_dims(0).as_in_context(self.ctx)
        outputs = self.db(img)
        pred = outputs[0].asnumpy()[0,0]
        boxes, scores = self.post_process.boxes_from_bitmap(pred, origin_w, origh_h)
        return boxes, scores

    def load_imgs(self, img_np, boxes):
        n = boxes.shape[0]
        batchs = np.zeros((n, 32, self.args.max_scale, 3), dtype=np.uint8)
        w_list = []
        for i,box in enumerate(boxes):
            img = crop_patch(img_np, box)
            img = self.resize_image(img, min_scale=32, min_divisor=4, max_scale=self.args.max_scale)
            h, w = img.shape[:2]
            batchs[i, :, :w, :] = img
            w_list.append(w)
        
        sort_ids = np.argsort(w_list)
        batchs = batchs[sort_ids]
        w_list = np.array(w_list)[sort_ids]
        max_w  = w_list[-1]
        batchs = batchs[:, :, :max_w, :]
        return batchs, w_list, sort_ids

    def batch_recog(self, img_np, boxes, bs=16):
        if len(boxes) == 0 : 
            return []
        batchs, w_list, sort_ids = self.load_imgs(img_np, boxes)
        batchs = mx.nd.array(batchs)
        batchs = mx.nd.image.to_tensor(batchs)
        batchs = mx.nd.image.normalize(batchs, mean=mean, std=std)
        batchs = batchs.as_in_context(self.ctx)
        n = len(w_list)
        num_steps = int(np.ceil(n/bs))
        start = 0
        text_list = []
        for i in range(num_steps):
            end = start + bs
            if end > n:
                end = n
                bs  = end - start
            data  = batchs[start:end, :, :, :w_list[end-1]]
            pred  = self.crnn(data)
            pred  = mx.nd.softmax(pred)
            pred  = mx.nd.argmax(pred, axis=-1) 
            outs  = pred.asnumpy()
            for j in range(bs):
                text = self.ctc_ids2text(outs[j], self.voc_size)
                text_list.append(text)
            start = end

        texts = ['']*n
        for i, idx in enumerate(sort_ids):
            texts[idx] = text_list[i]
        return texts

    def inference(self, image, visualize=True):
        if os.path.isdir(image):
            file_list  = os.listdir(image)
            image_list = [os.path.join(image, i) for i in file_list if i.split('.')[-1].lower() in img_types]
        else:
            image_list = [image]
        
        if not os.path.isdir(self.args.result_dir):
            os.mkdir(self.args.result_dir)
        
        for image_path in image_list:
            img_np = cv2.imread(image_path)
            print(image_path)
            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)            
            boxes, scores = self.detect(img_np)
            texts = self.batch_recog(img_np, boxes)

            if visualize:
                vis_image = self.visualize(img_np, boxes, texts)
                cv2.imwrite(os.path.join(self.args.result_dir, os.path.basename(image_path)), vis_image)
    
    def visualize(self, image, boxes, texts):
        h, w = image.shape[:2]
        img_pil  = Image.fromarray(255*np.ones((h, w, 3)).astype('uint8'))
        draw     = ImageDraw.Draw(img_pil)

        for box, text in zip(boxes, texts):
            box = np.array(box).astype(np.int32)
            cv2.polylines(image, [box], True, (0, 0, 255), 2)
            draw.polygon(box, outline='blue')
            box_height = math.sqrt((box[0][0] - box[3][0])**2 + (box[0][1] - box[3][1])**2)
            box_width  = math.sqrt((box[0][0] - box[1][0])**2 + (box[0][1] - box[1][1])**2)
            font_size  = max(int(0.8*min(box_height, box_width)), 10)
            font = ImageFont.truetype(self.args.font_path, size=font_size)
            draw.text((box[0][0], box[0][1]), text, font=font, fill=(0, 0, 0))
        new_np = np.concatenate([image, np.array(img_pil).astype(np.uint8)], axis=1)
        new_np = cv2.cvtColor(new_np, cv2.COLOR_RGB2BGR) 
        return new_np

if __name__ == '__main__':
    args = parser.parse_args()
    demo = OCRModel(args)
    demo.inference(args.image_path, visualize=args.visualize)
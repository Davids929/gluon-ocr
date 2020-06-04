#coding=utf-8
import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from mxnet.gluon.data.vision import transforms

transform_fn = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

class CTCFixSizeDataset(Dataset):
    def __init__(self, line_path, voc_path, augmnet_fn=None, short_side=32, fix_width=256, max_len=60):

        self.short_side  = short_side
        self.fix_width   = fix_width
        self.max_len     = max_len
        self.augmnet_fn  = augmnet_fn
        self.word2id   = self._load_voc_dict(voc_path)
        self.word_list = list(self.word2id.keys())
        self.imgs_list, self.labs_list = self._load_imgs(line_path)

    def _load_voc_dict(self, dict_path):
        word2id_dict = {}
        with open(dict_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        idx = 0
        for line in line_list:
            word = line.strip()[0]
            word2id_dict[word] = idx
            idx = idx + 1
        return word2id_dict

    @property
    def voc_size(self):
        return len(self.word_list)

    def _load_imgs(self, line_path):
        imgs_list = []
        labs_list = []
        if not isinstance(line_path, list):
            line_path = [line_path]
        for path in line_path:
            with open(path, 'r', encoding='utf-8') as fi:
                #line_list = fi.readlines()
                for i, line in enumerate(fi):
                    lst = line.strip().split('\t')
                    if len(lst) == 1:
                        continue
                    img_path = lst[0]
                    label    = lst[1]
                    if not os.path.exists(img_path):
                        continue
                    if label == '###':
                        continue
                    imgs_list.append(img_path)
                    labs_list.append(label)
        return imgs_list, labs_list

    def __len__(self):
        return len(self.imgs_list)

    def text2ids(self, text, text_len):
        ids       = mx.nd.ones(shape=(text_len), dtype='float32')*-1
        ids_mask  = mx.nd.zeros(shape=(text_len), dtype='float32')
        char_list = list(text)
        for i, ch in enumerate(char_list):
            if ch in self.word_list: 
                ids[i] = self.word2id[ch]  
            else:
                continue
            ids_mask[i] = 1.0
        return ids, ids_mask

    def ids2text(self, ids, blank):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        n = len(ids)
        words = []
        for i in range(n):
            if ids[i]!=blank and (not (i>0 and ids[i-1]==ids[i])):
                words.append(self.word_list[i])
        text = ''.join(words)
        return text

    def image_resize(self, img_np, max_width=512):
        h, w = img_np.shape[:2]
        if h > 4*w:
            img_np = np.rot90(img_np)
            h, w = w, h
        if self.fix_width is not None:
            img_np = cv2.resize(img_np, (self.fix_width, self.short_side))
            return img_np

        w = int(math.ceil(w*self.short_side/h/8))*8
        if w> max_width:
            w = max_width
        img_np = cv2.resize(img_np, (w, self.short_side))
        return img_np

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        text     = self.labs_list[idx]
        img_np   = cv2.imread(img_path)
        img_np   = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np   = self.image_resize(img_np)
        if self.augment_fn is not None:
            img_np = self.augment_fn(img_np)
        h, w = img_np.shape[:2]
        img_nd   = mx.nd.array(img_np)
        img_nd = transform_fn(img_nd)
        lab, lab_mask = self.text2ids(text, self.max_len)
        return img_nd, lab, lab_mask, idx

class AttFixedSizeDataset(CTCFixSizeDataset):
    def __init__(self, line_path, voc_path, augmnet_fn=None, short_side=32, fix_width=256, 
                 max_len=60, start_sym=0, end_sym=1):

        super(AttFixedSizeDataset, self).__init__(line_path, voc_path, augmnet_fn=None, 
                                                  short_side=32, fix_width=256, max_len=60)
        self.start_sym = start_sym
        self.end_sym   = end_sym

    def _load_voc_dict(self, dict_path):
        word2id_dict = {}
        word2id_dict = {'<s>':self.start_sym, '</s>':self.end_sym}
        with open(dict_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        idx = 2
        for line in line_list:
            word = line.strip('\n')[0]
            word2id_dict[word] = idx
            idx = idx + 1
        return word2id_dict

    def text2ids(self, text, text_len):
        ids       = mx.nd.ones(shape=(text_len), dtype='float32')*self.end_sym
        ids_mask  = mx.nd.zeros(shape=(text_len), dtype='float32')
        char_list = list(text)
        char_list = char_list + ['</s>']
        for i, ch in enumerate(char_list):
            if ch in self.word_list: 
                ids[i] = self.word2id[ch]  
            else:
                continue
            ids_mask[i] = 1.0
        return ids, ids_mask

    def ids2text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        text_list = []
        for i in ids:
            int_i = int(i)
            if int_i == self.end_sym:
                break
            text_list.append(self.word_list[int_i])
        return text_list

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        text     = self.labs_list[idx]
        img_np   = cv2.imread(img_path)
        img_np   = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np   = self.image_resize(img_np)
        if self.augment_fn is not None:
            img_np = self.augment_fn(img_np)
        h, w = img_np.shape[:2]
        img_nd   = mx.nd.array(img_np)
        img_nd = transform_fn(img_nd)
        img_mask = mx.nd.ones(shape=(1, h, w), dtype='float32')
        lab, lab_mask = self.text2ids(text, self.max_len)
        targ_data = mx.nd.ones(shape=(self.max_len), dtype='float32')*self.end_sym
        targ_data[0] = self.start_sym
        targ_data[1:] = lab[:-1]
        return img_nd, img_mask, targ_data, lab, lab_mask, idx


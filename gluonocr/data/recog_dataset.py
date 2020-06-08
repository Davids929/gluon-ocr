#coding=utf-8
import os
import cv2
import math
import random
import numpy as np
import mxnet as mx
from mxnet.gluon.data import Dataset
from . import normalize_fn
class FixSizeDataset(Dataset):
    def __init__(self, line_path, voc_path, augmnet_fn=None, short_side=32, 
                 fix_width=256, max_len=60, start_sym=None, end_sym=None):

        self.short_side  = short_side
        self.fix_width   = fix_width
        self.max_len     = max_len
        self.add_symbol  = False if start_sym==None or end_sym==None else True
        self.start_sym   = start_sym
        self.end_sym     = end_sym if self.add_symbol else -1
        self.augmnet_fn  = augmnet_fn
        self.word2id   = self._load_voc_dict(voc_path)
        self.word_list = list(self.word2id.keys())
        self.imgs_list, self.labs_list = self._get_items(line_path)

    def _load_voc_dict(self, dict_path):
        word2id_dict = {}
        if self.add_symbol:
             word2id_dict = {'<s>':self.start_sym, '</s>':self.end_sym}
        with open(dict_path, 'r', encoding='utf-8') as fi:
            line_list = fi.readlines()
        idx = len(word2id_dict)
        for line in line_list:
            word = line.strip()[0]
            word2id_dict[word] = idx
            idx = idx + 1
        return word2id_dict

    @property
    def voc_size(self):
        return len(self.word_list)

    def _get_items(self, line_path):
        imgs_list = []
        labs_list = []
        if not isinstance(line_path, list):
            line_path = [line_path]
        for path in line_path:
            with open(path, 'r', encoding='utf-8') as fi:
                for i, line in enumerate(fi):
                    lst = line.strip().split('\t')
                    if len(lst) < 2:
                        continue
                    img_path = lst[0]
                    label    = lst[1]
                    if not os.path.exists(img_path):
                        continue
                    if label == '###' or len(label)>self.max_len-1:
                        continue
                    imgs_list.append(img_path)
                    labs_list.append(label)
        return imgs_list, labs_list

    def __len__(self):
        return len(self.imgs_list)

    def text2ids(self, text, text_len):
        ids       = mx.nd.ones(shape=(text_len), dtype='float32')*self.end_sym
        ids_mask  = mx.nd.zeros(shape=(text_len), dtype='float32')
        char_list = list(text)
        if self.add_symbol:
            char_list.append('</s>')
        for i, ch in enumerate(char_list):
            if ch in self.word_list: 
                ids[i] = self.word2id[ch]  
            else:
                continue
            ids_mask[i] = 1.0
        return ids, ids_mask

    def ctc_ids2text(self, ids, blank):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        n = len(ids)
        words = []
        for i in range(n):
            if ids[i]!=blank and (not (i>0 and ids[i-1]==ids[i])):
                words.append(self.word_list[i])
        text = ''.join(words)
        return text

    def att_ids2text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        text_list = []
        for i in ids:
            int_i = int(i)
            if int_i == self.end_sym:
                break
            text_list.append(self.word_list[int_i])
        return text_list

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
        img_nd   = normalize_fn(img_nd)
        img_mask = mx.nd.ones((1, h, w), dtype='float32')
        lab, lab_mask = self.text2ids(text, self.max_len)
        if not self.add_symbol:
            return img_nd, img_mask, lab, lab_mask, idx

        targ_data = mx.nd.ones(shape=(self.max_len), dtype='float32')*self.end_sym
        targ_data[0] = self.start_sym
        targ_data[1:] = lab[:-1]
        return img_nd, img_mask, targ_data, lab, lab_mask, idx

class BucketDataset(FixSizeDataset):
    def __init_(self, line_path, voc_path, augmnet_fn=None, short_side=32, 
                fix_width=None, max_len=60, start_sym=None, end_sym=None, 
                split_width_len=128, split_text_len=10,):
    
        super(BucketDataset, self).__init__(line_path, voc_path, augmnet_fn=augmnet_fn, 
                                            short_side=short_side, fix_width=None, max_len=max_len,
                                            start_sym=start_sym, end_sym=end_sym)
        
        self.split_width_len = split_width_len
        self.split_text_len  = split_text_len
        self.gen_bucket()

    def _get_bucket_key(self, img_shape, text_len):
        h, w = img_shape[:2]
        text_ratio = math.ceil((text_len+1)/self.split_text_len)
        text_len = self.split_text_len*text_ratio
        if h > 4*w:
            w, h = img_shape[:2]
        if w/h > self.max_width/self.short_side:
            return (self.short_side, self.max_width, text_len)
        ratio = math.ceil(self.short_side * w / h / self.split_width_len)
        return (self.short_side, self.split_width_len * ratio, text_len)
    
    def gen_bucket(self):
        bucket_keys, bucket_dict = [], {}
        for idx in range(len(self.imgs_list)):
            img_np = cv2.imread(self.imgs_list[idx])
            text = self.labs_list[idx]
            if img_np is None:
                continue
            if len(text) > self.max_len:
                continue
            bucket_key = self._get_bucket_key(img_np.shape, len(text))
            bucket_key = str(bucket_key)
            if bucket_key not in bucket_keys:
                bucket_keys.append(bucket_key)
                bucket_dict[bucket_key] = []
            bucket_dict[bucket_key].append(idx)

        for key in bucket_keys:
            print('bucket key:', key, 'the number of image:', len(bucket_dict[key]))
        self.bucket_dict = bucket_dict
        self.bucket_keys = bucket_keys

    def __getitem__(self, idx):
        img_path = self.imgs_list[idx]
        text = self.labs_list[idx]
        img_np = cv2.imread(img_path)
        inp_h, inp_w, lab_len = self._get_bucket_key(img_np.shape, len(text))
        img_data = mx.nd.zeros(shape=(3, inp_h, inp_w), dtype='float32')
        img_mask = mx.nd.zeros(shape=(1, inp_h, inp_w), dtype='float32')
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
        img_np = self.image_resize(img_np, max_width=self.max_width)
        
        h, w = img_np.shape[:2]
        if self.use_augment:
            img_np = self.augmenter(img_np)
        img_nd = mx.nd.array(img_np) 
        img_nd = normalize_fn(img_nd)
        img_data[:, :h, :w] = img_nd
        img_mask[:, :h, :w] = 1.0
        lab, lab_mask = self.text2ids(text, lab_len)
        if not self.add_symbol:
            return img_data, img_mask, lab, lab_mask, idx
        targ_data = mx.nd.ones(shape=(lab_len), dtype='float32')*self.end_sym
        targ_data[0] = self.start_sym
        targ_data[1:] = lab[:-1]
        return img_data, img_mask, targ_data, lab, lab_mask, idx

class Sampler(object):
    def __init__(self, idx_list):
        self.idx_list = idx_list

    def __iter__(self):
        return iter(self.idx_list)

    def __len__(self):
        return len(self.idx_list)

    def __getitem__(self, item):
        return self.idx_list[item]

class BucketSampler(object):
    '''
    last_batch : {'keep', 'discard'}
        Specifies how the last batch is handled if batch_size does not evenly
        divide sequence length.

        If 'keep', the last batch will be returned directly, but will contain
        less element than `batch_size` requires.

        If 'discard', the last batch will be discarded.

    '''
    def __init__(self, batch_size, bucket_dict, shuffle=True, last_batch='discard'):
        bucket_keys  = list(bucket_dict.keys())
        self._batch_size = batch_size
        self._last_batch = last_batch
        self.shuffle     = shuffle
        self.sampler_list = []
        for key in bucket_keys:
            self.sampler_list.append(Sampler(bucket_dict[key]))

    def __iter__(self):
        if self.shuffle:
            for sampler in self.sampler_list:
                random.shuffle(sampler.idx_list)
            random.shuffle(self.sampler_list)

        num_sampler = len(self.sampler_list)
        sampler_idx_list = list(range(num_sampler))
        start_idx_list = [0] * num_sampler
        while True:
            if sampler_idx_list == []:
                break
            samp_idx = random.sample(sampler_idx_list, 1)[0]
            _sampler = self.sampler_list[samp_idx]
            start_idx = start_idx_list[samp_idx]
            batch = []
            while True:
                if len(batch) == self._batch_size:
                    start_idx_list[samp_idx] = start_idx
                    break

                if start_idx < len(_sampler):
                    batch.append(_sampler[start_idx])
                    start_idx = start_idx + 1
                else:
                    sampler_idx_list.remove(samp_idx)
                    if self._last_batch == 'discard':
                        batch = []
                    break
            if batch:
                yield batch


    def __len__(self):
        num = 0
        for _sampler in self.sampler_list:
            if self._last_batch == 'keep':
                #num += (len(_sampler) + self._batch_size - 1) // self._batch_size
                num += math.ceil(len(_sampler/self._batch_size))
            elif self._last_batch == 'discard':
                num += len(_sampler) // self._batch_size
            else:
                raise ValueError(
                    "last_batch must be one of 'keep', 'discard', or 'rollover', " \
                    "but got %s" % self._last_batch)
        return num
#coding=utf-8
#coding=utf-8
from mxnet.metric import EvalMetric
import numpy as np
from mxnet import gluon
from editdistance import distance

class RecogAccuracy(EvalMetric):
    def __init__(self, voc_size, ctc_mode=False):
        super(RecogAccuracy, self).__init__('RecogAccuracy')
        self.ctc_mode = ctc_mode
        if ctc_mode:
            self.blank = voc_size-1
        self.eps = 1e-6

    def get_pred(self, preds, max_len=100):
        batch_size, seq_len = preds.shape[:2]
        pred_list = []
        for i in range(batch_size):
            seq = -1*np.ones([max_len])
            last_word = None
            idx = 0
            for j in range(seq_len):
                if idx >= max_len:
                    break
                curr_word = preds[i, j]
                if  curr_word!= last_word and curr_word!= self.blank:
                    seq[idx]  = curr_word
                    last_word = curr_word
                    idx += 1
            pred_list.append(seq)
        return np.array(pred_list)

    def update(self, preds, labels, mask):
        labels = labels.asnumpy().astype('int32')
        preds = preds.asnumpy()
        if len(preds.shape) != len(labels.shape):
            preds = np.argmax(preds, axis=-1).astype('int32')
        mask  = mask.asnumpy()
        seq_len = labels.shape[-1]
        if self.ctc_mode:
            preds = self.get_pred(preds)
        acc = preds[:, :seq_len] == labels
        accuracy = np.sum(acc*mask,axis=-1)/(np.sum(mask, axis=-1)+self.eps)
        accuracy = np.mean(accuracy)
        self.sum_metric += accuracy
        self.num_inst += 1

def cal_pred_acc(preds, targs):
    correct_num  = 0
    pred_sum_num = 0
    targ_sum_num = 0 
    for pred, targ in zip(preds, targs):
        pred_num = len(pred)
        targ_num = len(targ)
        dist     = distance(pred, targ)
        correct_num  += max(pred_num, targ_num) - dist
        pred_sum_num += pred_num
        targ_sum_num += targ_num
    
    precision = correct_num/pred_sum_num
    recall    = correct_num/targ_sum_num
    return precision, recall

class RecogDistanceEvaluator(object):
    def __init__(self):
        pass

    def evaluate_image(self, gt, pred):
        correct_num  = 0
        pred_sum_num = 0
        gt_sum_num   = 0
         
        for gt_text, pred_text in zip(gt, pred):
            pred_num = len(pred_text)
            gt_num   = len(gt_text)
            dist     = distance(pred_text, gt_text)
            correct_num  += max(pred_num, gt_num) - dist
            pred_sum_num += pred_num
            gt_sum_num   += gt_num
        
        precision = correct_num/pred_sum_num
        recall    = correct_num/gt_sum_num
        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                precision * recall / (precision + recall)
        per_sample_metric = {'pred_num':pred_sum_num, 
                             'gt_num':gt_sum_num,
                             'correct_num':correct_num,
                             'precision':precision,
                             'recall':recall,
                             'hmean':hmean}
        return per_sample_metric

    def combine_results(self, metric_list):
        pred_nums = 0
        gt_nums   = 0
        correct_nums = 0
        for met in metric_list:
            pred_nums += met['pred_num']
            gt_nums   += met['gt_num']
            correct_nums += met['correct_num']
        
        precision = correct_num/pred_sum_num
        recall    = correct_num/targ_sum_num
        hmean = 0 if (precision + recall) == 0 else 2.0 * \
                precision * recall / (precision + recall)
        res = {'precision':precision,
               'recall':recall,
               'hmean':hmean}
        return res
        

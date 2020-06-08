#coding=utf-8
#coding=utf-8
from mxnet.metric import EvalMetric
import numpy as np
from mxnet import gluon

class RecogAccuracy(EvalMetric):
    def __init__(self, voc_size, ctc_mode=False):
        super(RecogAccuracy, self).__init__('RecogAccuracy')
        self.ctc_mode = ctc_mode
        if ctc_mode:
            self.blank = voc_size-1


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
        preds = np.argmax(preds, axis=-1).astype('int32')
        mask  = mask.asnumpy()
        seq_len = labels.shape[-1]
        if self.ctc_mode:
            preds = self.get_pred(preds)
        acc = preds[:, :seq_len] == labels
        accuracy = np.sum(acc*mask,axis=-1)/np.sum(mask, axis=-1)
        accuracy = np.mean(accuracy)
        self.sum_metric += accuracy
        self.num_inst += 1
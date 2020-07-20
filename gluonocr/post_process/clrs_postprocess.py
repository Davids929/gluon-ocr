#coding=utf-8
import math
import cv2
import numpy as np
from ..utils.locality_aware_nms import standard_nms

"""
reference from :
https://github.com/lvpengyuan/corner/blob/master/eval_all.py
"""

class CLRSPostProcess(object):
    def __init__(self, seg_thresh=0.3, box_thresh=0.6):
        self.seg_thresh = seg_thresh
        self.box_thresh = box_thresh

    def corner2center(self, boxes):
        cx = (boxes[:, 0] + boxes[:, 2])/2
        cy = (boxes[:, 1] + boxes[:, 3])/2
        w  = boxes[:, 2] - boxes[:, 0]
        h  = boxes[:, 3] - boxes[:, 1]
        return np.stack([cx, cy, w, h], axis=-1)

    def get_scores(self, boxes, seg_maps):
        c, h, w = seg_maps.shape
        mask = np.zeros((c, h, w), dtype=np.float32)
        boxes = np.array(boxes, dtype=np.int32)
        boxes[:, ::2]  = np.clip(boxes[:, ::2], 0, w)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, h)
        scores = []
        c1_x, c1_y = (boxes[:, 0] + boxes[:, 2])/2.0, (boxes[:, 1] + boxes[:, 3])/2.0
        c2_x, c2_y = (boxes[:, 2] + boxes[:, 4])/2.0, (boxes[:, 3] + boxes[:, 5])/2.0
        c3_x, c3_y = (boxes[:, 4] + boxes[:, 6])/2.0, (boxes[:, 5] + boxes[:, 7])/2.0
        c4_x, c4_y = (boxes[:, 6] + boxes[:, 0])/2.0, (boxes[:, 7] + boxes[:, 1])/2.0
        c_x = (boxes[:, 0] + boxes[:, 2] + boxes[:, 4] + boxes[:, 6])/4.0
        c_y = (boxes[:, 1] + boxes[:, 3] + boxes[:, 5] + boxes[:, 7])/4.0
        min_x = np.min(boxes[:, ::2], axis=-1)
        max_x = np.max(boxes[:, ::2], axis=-1)
        min_y = np.min(boxes[:, 1::2], axis=-1)
        max_y = np.max(boxes[:, 1::2], axis=-1)
        for i,box in enumerate(boxes):
            minx, miny, maxx, maxy = min_x[i], min_y[i], max_x[i], max_y[i]
    
            if maxx - minx < 4 or maxy- miny < 4:
                scores.append(0)
                continue
            offset = np.array([[minx, miny]], dtype=np.int32)
            mask = np.zeros((4, maxy-miny, maxx-minx), dtype=np.float32)
            tl = np.array([[box[0], box[1]], [c1_x[i], c1_y[i]], [c_x[i], c_y[i]], [c4_x[i], c4_y[i]]], dtype=np.int32) - offset
            cv2.fillPoly(mask[0], tl[np.newaxis, :, :], 1)
            tl = np.array([[c1_x[i], c1_y[i]], [box[2], box[3]], [c2_x[i], c2_y[i]], [c_x[i], c_y[i]]], dtype=np.int32) - offset
            cv2.fillPoly(mask[1], tl[np.newaxis, :, :], 1)
            tl = np.array([[c_x[i], c_y[i]], [c2_x[i], c2_y[i]], [box[4], box[5]], [c3_x[i], c3_y[i]]], dtype=np.int32) - offset
            cv2.fillPoly(mask[2], tl[np.newaxis, :, :], 1)
            tl = np.array([[c4_x[i], c4_y[i]], [c_x[i], c_y[i]], [c3_x[i], c3_y[i]], [box[6], box[7]]], dtype=np.int32) - offset
            cv2.fillPoly(mask[3], tl[np.newaxis, :, :], 1)
            score = 0
            for j in range(4):
                score += (mask[j]*seg_maps[j, miny:maxy, minx:maxx]).sum()/(mask[j].sum())
            score = score/4.0
            scores.append(score)
        return scores

    def gen_box(self, corner1, corner2, mode):

        random_box = []

        def edge_len(x1, y1, x2, y2):
            return math.sqrt((x2 - x1)*(x2-x1) + (y2 - y1)*(y2-y1))

        def is_right_box(box):
            edge1 = edge_len(box[0], box[1], box[2], box[3])
            edge2 = edge_len(box[2], box[3], box[4], box[5])
            edge3 = edge_len(box[4], box[5], box[6], box[7])
            edge4 = edge_len(box[6], box[7], box[0], box[1])
            if edge1 > 5 and edge2>5 and edge3>5 and edge4>5:
                return True
            else:
                return False

        def get_point(x1, y1, x2, y2, theta, side):
            x3 = x1 + math.cos(theta)*side
            y3 = y1 + math.sin(theta)*side
            x4 = x2 + math.cos(theta)*side
            y4 = y2 + math.sin(theta)*side
            return x3, y3, x4, y4

        for c1 in corner1:
            for c2 in corner2:
                rat = max(c1[2], c2[2])/min(c1[2], c2[2])
                if c1[0]<c2[0] and c1[2]>5 and c2[2]>5 and rat < 1.5:
                    side = (c1[2] + c2[2])/2.0
                    if mode == 0:
                        # top_line
                        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0]) + math.pi/2
                        x3, y3, x4, y4 = get_point(c2[0], c2[1], c1[0], c1[1], theta, side)
                        box = [c1[0], c1[1], c2[0], c2[1], x3, y3, x4, y4]
                    elif mode == 1:
                        # bottom_line
                        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0]) - math.pi/2
                        x2, y2, x1, y1 = get_point(c2[0], c2[1], c1[0], c1[1], theta, side)
                        box = [x1, y1, x2, y2, c2[0], c2[1], c1[0], c1[1]]
                    elif mode == 2:
                        # left_line
                        theta = math.atan2(c2[1] - c1[1], c1[0] - c1[0]) - math.pi/2
                        x3, y3, x2, y2 = get_point(c2[0], c2[1], c1[0], c1[1], theta, side)
                        box = [c1[0], c1[1], x2, y2, x3, y3, c2[0], c2[1]]
                    else:
                        # right_line
                        theta = math.atan2(c2[1] - c1[1], c2[0] - c1[0]) + math.pi/2
                        x4, y4, x1, y1 = get_point(c2[0], c2[1], c1[0], c1[1], theta, side)
                        box = [x1, y1, c1[0], c1[1], c2[0], c2[1], x4, y4]
                    if is_right_box(box):
                        random_box.append(box)
        return random_box

    def get_boxes(self, ids, boxes, seg_maps, ratio):
        height, width = seg_maps.shape[1:3]
        boxes = self.corner2center(boxes)
        
        tls = boxes[ids[:, 0]==0, :]
        trs = boxes[ids[:, 0]==1, :]
        brs = boxes[ids[:, 0]==2, :]
        bls = boxes[ids[:, 0]==3, :]
        if len(tls) == 0 or len(trs) == 0 or len(brs) == 0 or len(bls) == 0:
            return []
        random_box, candidate_box = [], []
        # top_line
        box_list = self.gen_box(tls, trs, 0)
        random_box = random_box + box_list

        # bottom_line
        box_list = self.gen_box(bls, brs, 1)
        random_box = random_box + box_list

        # left_line
        box_list = self.gen_box(tls, bls, 2)
        random_box = random_box + box_list

        # right_line
        box_list = self.gen_box(trs, brs, 3)
        random_box = random_box + box_list

        scores = self.get_scores(random_box, seg_maps)
        for i in range(len(random_box)):
            if scores[i] > self.seg_thresh:
                candidate_box.append(random_box[i] + [scores[i]])
        candidate_box = np.array(candidate_box, dtype=np.float32)
        boxes = standard_nms(candidate_box, self.box_thresh)
        boxes = np.reshape(boxes[:, :8], (-1, 4, 2))
        boxes[:, :, 0] = np.clip(
                np.round(boxes[:, :, 0] * ratio), 0, width*ratio)
        boxes[:, :, 1] = np.clip(
                np.round(boxes[:, :, 1] * ratio), 0, height*ratio)
        return boxes


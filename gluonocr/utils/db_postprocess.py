#coding=utf-8
import cv2
import numpy as np
from shapely.geometry import Polygon
import pyclipper

class DBPostProcess(object):

    def __init__(self, thresh=0.3, box_thresh=0.7, min_size=4, 
                 min_area=40, scale_ratio=0.4, unclip_ratio=1.6, **kwargs):
        self.thresh     = thresh
        self.box_thresh = box_thresh
        self.min_size   = min_size
        self.min_area   = min_area
        self.scale_ratio = scale_ratio
        self.unclip_ratio = unclip_ratio

    def polygons_from_bitmap(self, pred, dest_width, dest_height):
        '''
        pred: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        if pred.shape[0] == 1:
            pred = pred[0]
        bitmap = pred>self.thresh
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
        
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            box = np.array(box)
            area = cv2.contourArea(box)
            if area < self.min_area:
                continue
            if sside < self.min_size + 2:
                continue
            
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            
            boxes.append(box.astype(np.int16))
            scores.append(score)
        boxes = np.array(boxes, dtype=np.int16)
        scores = np.array(scores)
        return boxes, scores

    def boxes_from_bitmap(self, pred, dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        
        if pred.shape[0] == 1:
            pred = pred[0]
        
        if not isinstance(dest_width, int):
            dest_width = dest_width.item()
            dest_height = dest_height.item()

        bitmap = pred>self.thresh
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap*255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)
        boxes = []
        scores = []

        for index in range(num_contours):
            
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
        
            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)
            area = cv2.contourArea(box)
            if area < self.min_area:
                continue
            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            
            boxes.append(box.astype(np.int16))
            scores.append(score)
        boxes = np.array(boxes, dtype=np.int16)
        scores = np.array(scores)
        return boxes, scores

    def unclip(self, box, unclip_ratio=1.6):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded

    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]

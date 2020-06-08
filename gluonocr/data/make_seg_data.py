from collections import OrderedDict

import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper

class MakeSegDetectorData(object):
    def __init__(self, min_text_size=8, shrink_ratio=0.4, gen_geometry=False):
        self.min_text_size = min_text_size
        self.shrink_ratio  = shrink_ratio
        self.gen_geometry  = gen_geometry 

    def __call__(self, data, *args, **kwargs):
        '''
        data: a dict typically returned from `MakeICDARData`,
            where the following keys are contrains:
                image*, polygons*, ignore_tags*, shape, filename
                * means required.
        '''
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        filename = data['filename']

        h, w = image.shape[:2]
        polygons, ignore_tags = self.validate_polygons(
            polygons, ignore_tags, h, w)
        gt = np.zeros((1, h, w), dtype=np.float32)
        mask = np.ones((h, w), dtype=np.float32)
        inst_mask = np.zeros((h, w), dtype=np.float32)
        geo_map   = np.zeros((h, w), dtype=np.float32)
        for i in range(polygons.shape[0]):
            polygon = polygons[i]
            height = min(np.linalg.norm(polygon[0] - polygon[3]),
                         np.linalg.norm(polygon[1] - polygon[2]))
            width = min(np.linalg.norm(polygon[0] - polygon[1]),
                        np.linalg.norm(polygon[2] - polygon[3]))
            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)
                cv2.fillPoly(gt[0], [shrinked.astype(np.int32)], 1)
                if self.gen_geometry:
                    cv2.fillPoly(inst_mask, [shrinked.astype(np.int32)], i+1)
                    xy_in_poly = np.argwhere(inst_mask == (index + 1))
                    # geo map.
                    y_in_poly = xy_in_poly[:, 0]
                    x_in_poly = xy_in_poly[:, 1]
                    polygon[:, 0] = np.minimum(np.maximum(polygon[:, 0], 0), w)
                    polygon[:, 1] = np.minimum(np.maximum(polygon[:, 1], 0), h)
                    for pno in range(4):
                        geo_channel_beg = pno * 2
                        geo_map[y_in_poly, x_in_poly, geo_channel_beg] =\
                            x_in_poly - polygon[pno, 0]
                        geo_map[y_in_poly, x_in_poly, geo_channel_beg+1] =\
                            y_in_poly - polygon[pno, 1]
                    geo_map[y_in_poly, x_in_poly, 8] = \
                        1.0 / max(min(height, width), 1.0)

        if filename is None:
            filename = ''
        data.update(image=image,
                    polygons=polygons,
                    gt=gt, mask=mask, filename=filename)
        if self.gen_geometry:
            data.update(geo_map=geo_map)
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if polygons.shape[0] == 0:
            return polygons, ignore_tags
        assert polygons.shape[0] == len(ignore_tags)

        polygons[:, :, 0] = np.clip(polygons[:, :, 0], 0, w - 1)
        polygons[:, :, 1] = np.clip(polygons[:, :, 1], 0, h - 1)

        for i in range(polygons.shape[0]):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][(0, 3, 2, 1), :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = [
            (polygon[1][0] - polygon[0][0]) * (polygon[1][1] + polygon[0][1]),
            (polygon[2][0] - polygon[1][0]) * (polygon[2][1] + polygon[1][1]),
            (polygon[3][0] - polygon[2][0]) * (polygon[3][1] + polygon[2][1]),
            (polygon[0][0] - polygon[3][0]) * (polygon[0][1] + polygon[3][1])
        ]
        return np.sum(edge) / 2.

class MakeBorderMap(object):
    
    def __init__(self, shrink_ratio=0.4, thresh_min=0.3, thresh_max=0.7, *args, **kwargs):
        self.shrink_ratio = shrink_ratio
        self.thresh_min = thresh_min
        self.thresh_max = thresh_max

    def __call__(self, data, *args, **kwargs):
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']

        canvas = np.zeros(image.shape[:2], dtype=np.float32)
        mask = np.zeros(image.shape[:2], dtype=np.float32)

        for i in range(len(polygons)):
            if ignore_tags[i]:
                continue
            self.draw_border_map(polygons[i], canvas, mask=mask)
        canvas = canvas * (self.thresh_max - self.thresh_min) + self.thresh_min
        data['thresh_map'] = canvas
        data['thresh_mask'] = mask
        return data

    def draw_border_map(self, polygon, canvas, mask):
        polygon = np.array(polygon)
        assert polygon.ndim == 2
        assert polygon.shape[1] == 2

        polygon_shape = Polygon(polygon)
        distance = polygon_shape.area * \
            (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in polygon]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND,
                        pyclipper.ET_CLOSEDPOLYGON)
        padded_polygon = np.array(padding.Execute(distance)[0])
        cv2.fillPoly(mask, [padded_polygon.astype(np.int32)], 1.0)

        xmin = padded_polygon[:, 0].min()
        xmax = padded_polygon[:, 0].max()
        ymin = padded_polygon[:, 1].min()
        ymax = padded_polygon[:, 1].max()
        width = xmax - xmin + 1
        height = ymax - ymin + 1

        polygon[:, 0] = polygon[:, 0] - xmin
        polygon[:, 1] = polygon[:, 1] - ymin

        xs = np.broadcast_to(
            np.linspace(0, width - 1, num=width).reshape(1, width), (height, width))
        ys = np.broadcast_to(
            np.linspace(0, height - 1, num=height).reshape(height, 1), (height, width))

        distance_map = np.zeros(
            (polygon.shape[0], height, width), dtype=np.float32)
        for i in range(polygon.shape[0]):
            j = (i + 1) % polygon.shape[0]
            absolute_distance = self.distance(xs, ys, polygon[i], polygon[j])
            distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
        distance_map = distance_map.min(axis=0)

        xmin_valid = min(max(0, xmin), canvas.shape[1] - 1)
        xmax_valid = min(max(0, xmax), canvas.shape[1] - 1)
        ymin_valid = min(max(0, ymin), canvas.shape[0] - 1)
        ymax_valid = min(max(0, ymax), canvas.shape[0] - 1)
        canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
            1 - distance_map[
                ymin_valid-ymin:ymax_valid-ymax+height,
                xmin_valid-xmin:xmax_valid-xmax+width],
            canvas[ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

    def distance(self, xs, ys, point_1, point_2):
        '''
        compute the distance from point to a line
        ys: coordinates in the first axis
        xs: coordinates in the second axis
        point_1, point_2: (x, y), the end of the line
        '''
        height, width = xs.shape[:2]
        square_distance_1 = np.square(
            xs - point_1[0]) + np.square(ys - point_1[1])
        square_distance_2 = np.square(
            xs - point_2[0]) + np.square(ys - point_2[1])
        square_distance = np.square(
            point_1[0] - point_2[0]) + np.square(point_1[1] - point_2[1])

        cosin = (square_distance - square_distance_1 - square_distance_2) / \
            (2 * np.sqrt(square_distance_1 * square_distance_2)+1e-6)
        square_sin = 1 - np.square(cosin)
        square_sin = np.nan_to_num(square_sin)
        result = np.sqrt(square_distance_1 * square_distance_2 *
                         square_sin / square_distance+1e-6)

        result[cosin < 0] = np.sqrt(np.fmin(
            square_distance_1, square_distance_2))[cosin < 0]
        # self.extend_line(point_1, point_2, result)
        return result

    def extend_line(self, point_1, point_2, result):
        ex_point_1 = (int(round(point_1[0] + (point_1[0] - point_2[0]) * (1 + self.shrink_ratio))),
                      int(round(point_1[1] + (point_1[1] - point_2[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_1), tuple(point_1),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        ex_point_2 = (int(round(point_2[0] + (point_2[0] - point_1[0]) * (1 + self.shrink_ratio))),
                      int(round(point_2[1] + (point_2[1] - point_1[1]) * (1 + self.shrink_ratio))))
        cv2.line(result, tuple(ex_point_2), tuple(point_2),
                 4096.0, 1, lineType=cv2.LINE_AA, shift=0)
        return ex_point_1, ex_point_2

import cv2
import os
import math
import numpy as np
import lmdb

__all__ = ['crop_patch', 'get_mini_boxes', 'create_recog_lmdb_data']

def crop_patch(img_np, box):
    def cal_len(p1, p2):
        return int(np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2))

    if isinstance(box, np.ndarray):
        box = box.astype('int32')
    else:
        box = np.array(box).astype('int32')
    
    if len(box) == 4:
        if box[0, 1] == box[1, 1] and box[0, 0] == box[-1, 0]:
            patch = img_np[box[0, 1]:box[2, 1], box[0, 0]:box[2, 0]]
            return patch 
        w = cal_len(box[0], box[1])
        h = cal_len(box[0], box[-1])
        pst1 = np.float32([box[0], box[1], box[3], box[2]])
        pst2 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        M = cv2.getPerspectiveTransform(pst1, pst2)
        patch = cv2.warpPerspective(img_np, M, (w,h))
        return patch
    else:
        new_box, _ = get_mini_boxes(box)
        return crop_patch(img_np, new_box)

def get_mini_boxes(contour):
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


"""
reference from :
https://github.com/clovaai/deep-text-recognition-benchmark/blob/master/create_lmdb_dataset.py
"""

def create_recog_lmdb_data(gtFiles, outputPath, checkValid=True):
    """
    Create LMDB dataset for training and evaluation.
    ARGS:
        outputPath : LMDB output path
        gtFile     : list of image path and label
        checkValid : if true, check the validity of every image
    """
    os.makedirs(outputPath, exist_ok=True)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1

    def writeCache(env, cache):
        with env.begin(write=True) as txn:
            for k, v in cache.items():
                txn.put(k, v)
    datalist = []
    if not isinstance(gtFiles, list):
        gtFiles = [gtFiles]
    for gtFile in gtFiles:
        with open(gtFile, 'r', encoding='utf-8') as data:
            datalist += data.readlines()

    nSamples = len(datalist)
    for i in range(nSamples):
        imagePath, label = datalist[i].strip('\n').split('\t')

        if not os.path.exists(imagePath):
            print('%s does not exist' % imagePath)
            continue

        with open(imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            try:
                if not checkImageIsValid(imageBin):
                    print('%s is not a valid image' % imagePath)
                    continue
            except:
                print('error occured', i)
                with open(outputPath + '/error_image_log.txt', 'a') as log:
                    log.write('%s-th image data occured error\n' % str(i))
                continue

        imageKey = 'image-%09d'.encode() % cnt
        labelKey = 'label-%09d'.encode() % cnt
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode()

        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt-1
    cache['num-samples'.encode()] = str(nSamples).encode()
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)


def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
    img = cv2.imdecode(imageBuf, cv2.IMREAD_GRAYSCALE)
    imgH, imgW = img.shape[0], img.shape[1]
    if imgH * imgW == 0:
        return False
    return True
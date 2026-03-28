#!/usr/bin/env python
# _*_ coding: utf-8 _*_
###############
# yolo->coco
###############

import json
import os
from datetime import datetime
import cv2
import random

import sys
from pathlib import Path
projectDir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectDir))
try:
    from Source import config
    from Source.Utility import PublicFunction
except ImportError as e:
    print(f"Import Source Failed：{e}")
    print(f"Make sure Source directory")
    sys.exit(1)

image_id = 0
annotation_id = 0


def addCatItem(coco, category_dict):
    for k, v in category_dict.items():
        category_item = dict()
        category_item['supercategory'] = 'none'
        category_item['id'] = int(k)
        category_item['name'] = v
        coco['categories'].append(category_item)


def addImgItem(coco, file_name, size):
    global image_id
    image_id += 1
    image_item = dict()
    image_item['id'] = image_id
    image_item['file_name'] = PublicFunction.getFileExtName(file_name)[1] + ".jpg"

    image_item['width'] = size[1]
    image_item['height'] = size[0]
    image_item['license'] = None
    image_item['flickr_url'] = None
    image_item['coco_url'] = None
    image_item['date_captured'] = str(datetime.today())
    coco['images'].append(image_item)
    # image_set.add(file_name)
    return image_id


def addAnnoItem(coco, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1

    annotation_item['image_id'] = image_id
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def xywhn2xywh(bbox, size):
    # YOLO to COCO
    bbox = list(map(float, bbox))
    size = list(map(float, size))
    xmin = (bbox[0] - bbox[2] / 2.) * size[1]
    ymin = (bbox[1] - bbox[3] / 2.) * size[0]
    w = bbox[2] * size[1]
    h = bbox[3] * size[0]
    box = (xmin, ymin, w, h)
    return list(map(int, box))


def yolo2coco(image_path, anno_path, json_path, balance=False, filterDir=None):
    coco = dict()
    coco['licenses'] = []
    coco['images'] = []
    coco['type'] = 'instances'
    coco['annotations'] = []
    coco['categories'] = []

    assert os.path.exists(image_path), "ERROR {} dose not exists".format(image_path)
    assert os.path.exists(anno_path), "ERROR {} dose not exists".format(anno_path)

    category_id = {1: "Silo-cave"}  # 类型数值

    addCatItem(coco, category_id)
    imagenameList = []
    if filterDir is not None:
        filterNameList = [i.split(os.sep)[-1][:-4] for i in PublicFunction.listFiles(filterDir, ".jpg")]

    for imgName in PublicFunction.listFiles(image_path, ".jpg"):
        imagenameList.append(os.path.join(image_path, imgName))
        #imagenameList.append(imgName)
    files = [os.path.join(anno_path, i) for i in PublicFunction.listFiles(anno_path, ".txt")]
    images_index_dict = dict(
        (v.split(os.sep)[-1][:-4], k) for k, v in enumerate(imagenameList))  # 索引建立字典，对应txt的文件名称一致，通过字典的形式

    existNameList = []
    noexistNameList = []
    num = 0
    for file in files:
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if os.path.getsize(file) != 0:
            if filterDir is not None:
                if file.split(os.sep)[-1][:-4] in filterNameList:
                    existNameList.append(file)
            else:
                existNameList.append(file)
            num = num + 1

        else:
            noexistNameList.append(file)
    random.shuffle(noexistNameList)

    if balance is True:

        choosenum = len(noexistNameList[:num])
        chooseNoExistNameList = noexistNameList[:choosenum]
    else:
        choosenum = len(noexistNameList)
        chooseNoExistNameList = noexistNameList[:choosenum]

    if filterDir is not None:
        chooseNoExistNameList = [os.path.join(anno_path, filtername + ".txt") for filtername in filterNameList]

    for file in set(existNameList + chooseNoExistNameList):
        if os.path.splitext(file)[-1] != '.txt' or 'classes' in file.split(os.sep)[-1]:
            continue
        if file.split(os.sep)[-1][:-4] in [n for n in images_index_dict.keys()]:
            index = images_index_dict[file.split(os.sep)[-1][:-4]]
            img = cv2.imread(imagenameList[index])
            shape = img.shape

            filename = imagenameList[index].split(os.sep)[-1]
            current_image_id = addImgItem(coco, imagenameList[index], shape)
        else:
            continue

        with open(file, 'r') as fid:
            for i in fid.readlines():
                i = i.strip().split()
                category = int(i[0]) + 1
                category_name = category_id[category]
                bbox = xywhn2xywh((i[1], i[2], i[3], i[4]), shape)
                addAnnoItem(coco, current_image_id, category, bbox)

    json.dump(coco, open(json_path, 'w'))
    print("class nums:{}".format(len(coco['categories'])))
    print("image nums:{}".format(len(coco['images'])))
    print("bbox nums:{}".format(len(coco['annotations'])))


if __name__ == '__main__':
    size = 512
    path_root = config.Combine_YOLODir
    save_path = config.Combine_COCODir
    PublicFunction.mkDir(str(save_path))

    # #Combine train data
    # train_image_path = os.path.join(path_root, "images", "train", str(size))
    # train_anno_path = os.path.join(path_root, "labels", "train", str(size))
    # json_name = 'train{size}.json'.format(size=size)
    # json_path = os.path.join(save_path, json_name)
    # yolo2coco(train_image_path, train_anno_path, json_path, balance=False, filterDir=None)
    #
    # #Combine test data
    # val_image_path = os.path.join(path_root, "images", "val", str(size))
    # val_anno_path = os.path.join(path_root, "labels", "val", str(size))
    # json_name = 'val{size}.json'.format(size=size)
    # json_path = os.path.join(save_path, json_name)
    # yolo2coco(val_image_path, val_anno_path, json_path, balance=False, filterDir=None)
    #
    #ShanZhou test data
    path_root = config.ShanZhou_YOLODir
    save_path = config.ShanZhou_COCODir
    PublicFunction.mkDir(save_path)

    val_image_path = os.path.join(path_root, "images", "val", str(size))
    val_anno_path = os.path.join(path_root, "labels", "val", str(size))
    json_name = 'val{size}.json'.format(size=size)
    json_path = os.path.join(save_path, json_name)
    yolo2coco(val_image_path, val_anno_path, json_path, balance=False, filterDir=None)

    # PingLu test data
    path_root = config.PingLu_YOLODir
    save_path = config.PingLu_COCODir
    PublicFunction.mkDir(save_path)

    val_image_path = os.path.join(path_root, "images", "val", str(size))
    val_anno_path = os.path.join(path_root, "labels", "val", str(size))
    json_name = 'val{size}.json'.format(size=size)
    json_path = os.path.join(save_path, json_name)
    yolo2coco(val_image_path, val_anno_path, json_path, balance=False, filterDir=None)

    # SanYuan test data
    path_root = config.SanYuan_YOLODir
    save_path = config.SanYuan_COCODir
    PublicFunction.mkDir(save_path)

    val_image_path = os.path.join(path_root, "images", "val", str(size))
    val_anno_path = os.path.join(path_root, "labels", "val", str(size))
    json_name = 'val{size}.json'.format(size=size)
    json_path = os.path.join(save_path, json_name)
    yolo2coco(val_image_path, val_anno_path, json_path, balance=False, filterDir=None)

    # ChunHua test data
    path_root = config.ChunHua_YOLODir
    save_path = config.ChunHua_COCODir
    PublicFunction.mkDir(save_path)

    val_image_path = os.path.join(path_root, "images", "val", str(size))
    val_anno_path = os.path.join(path_root, "labels", "val", str(size))
    json_name = 'val{size}.json'.format(size=size)
    json_path = os.path.join(save_path, json_name)
    yolo2coco(val_image_path, val_anno_path, json_path, balance=False, filterDir=None)



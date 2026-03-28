#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd
import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN_MobileNet_V3_Large_FPN_Weights
from torchvision.models.detection.retinanet import RetinaNetClassificationHead, RetinaNet_ResNet50_FPN_V2_Weights
from torchvision.models.detection.ssd import SSD300_VGG16_Weights
from torchvision.models.detection.ssd import SSDClassificationHead
from torchvision.models.detection import _utils
import time
from functools import partial
import os

import sys
from pathlib import Path
projectDir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectDir))

try:
    from Source.torchutil import utils, coco_utils, transforms as T, engine
    from Source.Utility import PublicFunction
    from Source import config
except ImportError as e:
    print(f"Import Source Failed：{e}")
    print(f"Make sure Source directory")
    sys.exit(1)

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# torch.cuda.set_device(0)


starttime = time.time()

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
print(torch.cuda.is_available())  # True = GPU / False = CPU
print(torch.version.cuda)
print(torch.backends.cudnn.version())

#
# FasterRCNN model
def get_fasterrcnn_resnet50_fpn_model(size, num_classes, weights=True):
    if weights is True:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    else:
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn()

    model.transform.min_size = (size,)
    model.transform.max_size = size
    # num_classes =kwargs["num_classes"]

    # replace the classifier with a new one, that has
    # num_classes which is user-defined
    # num_classes = 2  # 1 class (person) + background
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # size = kwargs["size"]

    return model


# MobileNet model
def get_fasterrcnn_mobilenet_v3_large_320_fpn_model(size, num_classes, weights=True):
    # num_classes = kwargs["num_classes"]
    # size = kwargs["size"]
    if weights is True:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    else:
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn()

    model.transform.min_size = (size - 10,)
    model.transform.max_size = size + 10
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


# SSD model
def get_ssd300_vgg16(size=512, num_classes=2, weights=True):
    # TODO: Since the default image size for SSD300 is 300
    # Add parameters for torchvision.models.detection.ssd300_vgg16 and SSD parameters
    # model = SSD(backbone, anchor_generator, (300, 300), num_classes, **kwargs)
    # After modification:
    # def ssd300_vgg16(
    #         *,
    #         weights: Optional[SSD300_VGG16_Weights] = None,
    #         progress: bool = True,
    #         size: Optional[int] = None,  # TODO:Add and modify
    #         num_classes: Optional[int] = None,
    #         weights_backbone: Optional[VGG16_Weights] = VGG16_Weights.IMAGENET1K_FEATURES,
    #         trainable_backbone_layers: Optional[int] = None,
    #         **kwargs: Any,
    # ) -> SSD:
    # model = SSD(backbone, anchor_generator, (size, size), num_classes, **kwargs)#TODO:Change 300 to size

    if weights is True:
        model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT, size=size
        )
    else:
        model = torchvision.models.detection.ssd300_vgg16(
        )
    # Image size for transforms.
    model.transform.min_size = (size - 10,)
    model.transform.max_size = size + 10

    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))
    print(in_channels)
    num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = SSDClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
    )

    return model


# RetinaNet model
def get_retinanet_resnet50_fpn(size=1024, num_classes=2, weights=True):
    if weights is True:
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT
        )
    else:
        model = torchvision.models.detection.retinanet_resnet50_fpn_v2(
        )

    # Retrieve the list of input channels.
    in_channels = _utils.retrieve_out_channels(model.backbone, (size, size))[0]
    num_anchors = model.head.classification_head.num_anchors
    # List containing number of anchors based on aspect ratios.
    # num_anchors = model.anchor_generator.num_anchors_per_location()
    # The classification head.
    model.head.classification_head = RetinaNetClassificationHead(
        in_channels=in_channels,
        num_anchors=num_anchors,
        num_classes=num_classes,
        norm_layer=partial(torch.nn.GroupNorm, 32)
    )
    # Image size for transforms.
    # 去掉Transform,精度就不为0了
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model


def get_transform(train=False):
    transformsList = []
    transformsList.append(coco_utils.ConvertCocoPolysToMask())
    transformsList.append(T.PILToTensor())
    if train:
        transformsList.append(T.RandomRotation())
    transformsList.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transformsList)


def getTrainDataset(train_img_folder, train_ann_file, train_transforms):

    trainDataset = coco_utils.CocoDetection(train_img_folder, train_ann_file, train_transforms)
    # https://discuss.pytorch.org/t/cuda-error-runtimeerror-cudnn-status-execution-failed/17625
    train_data_loader = torch.utils.data.DataLoader(
        trainDataset,
        batch_size=5,
        shuffle=True,
        num_workers=5,
        collate_fn=utils.collate_fn
    )


    return train_data_loader


def getTestDataset(test_img_folder, test_ann_file):
    test_transform =get_transform()
    testDataset = coco_utils.CocoDetection(test_img_folder, test_ann_file, test_transform)
    test_data_loader = torch.utils.data.DataLoader(
        testDataset,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn
    )
    return  test_data_loader

def FScore(PA, UA):
    F = 2 * PA * UA / (PA + UA)
    return F


starttime = time.time()


def train_model(num_epochs, train_data_loader, test_data_loader, modelName="FasterRCNN", **kwargs):
    torch.cuda.empty_cache()

    modelDir = kwargs["modelDir"]
    PublicFunction.mkDir(modelDir)
    if "epochList" in kwargs.keys():
        epochList = kwargs["epochList"]

    # # let's train_old it just for 2 epochs
    modelFileName = os.path.join(modelDir, '{modelName}_last.pth'.format(modelName=modelName))
    model = None
    if PublicFunction.check_existence(modelFileName):
        print(modelFileName)
        model = torch.load(modelFileName)
    else:
        if modelName == "FasterRCNN":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_fasterrcnn_resnet50_fpn_model(size=size, num_classes=num_classes, weights=True)
        elif modelName == "SSD":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            print("Please modify the SSD source code parameters first to adapt to the input image size!")
            print("Please refer to the get_ssd300_vgg16 function")
            model = get_ssd300_vgg16(size=size, num_classes=num_classes)
        elif modelName == "Retinanet":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_retinanet_resnet50_fpn(size=size, num_classes=num_classes)
        elif modelName == "MobileNet":
            size = kwargs["size"]
            num_classes = kwargs["num_classes"]
            model = get_fasterrcnn_mobilenet_v3_large_320_fpn_model(size=size, num_classes=num_classes)
        model.to(device)

    model.train()

    timeDict = {}
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.0001,
        momentum=0.9,
        weight_decay=0.0005
    )
    trainLossFileName = os.path.join(modelDir, "TrainLoss.csv")
    valLossFileName = os.path.join(modelDir, "ValLoss.csv")

    best_acc = 0
    valList = []
    PRList = []
    epoch_test_acc = -1
    starttime = time.time()

    for epoch in range(num_epochs):
        train_metric_logger = engine.train_one_epoch(model, optimizer, train_data_loader, test_data_loader, device,
                                                     epoch, print_freq=10)
        loss_str = []
        col_name_list = []
        for name, meter in train_metric_logger.meters.items():
            # loss_str.append(f"{name}: {str(meter)}")
            col_name_list.append(name)
            loss_str.append(str(meter))

        valList.append(loss_str)
        # lr_scheduler.step()
        # evaluate on the test dataset

        if epoch == 0:
            epoch_coco_evaluator, epoch_test_metric_logger = engine.evaluate(model, test_data_loader, device=device)
            epoch_test_acc = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            AP_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[2]
            AP_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[3]
            AP_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[4]
            AP_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[5]

            AR_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[6]
            AR_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[7]
            AR_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[8]
            AR_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[9]
            AR_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[10]
            AR_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[11]
            AR_6 = epoch_coco_evaluator.coco_eval['bbox'].stats[12]
            precision = epoch_coco_evaluator.coco_eval['bbox'].stats[13]

            PRList.append([AP_0, AP_1, AP_2, AP_3, AP_4, AP_5, AR_0, AR_1, AR_2, AR_3, AR_4, AR_5, AR_6, precision])

        if epoch_test_acc > best_acc:
            best_acc = epoch_test_acc
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, os.path.join(modelDir, '{modelName}_best.pth'.format(modelName=modelName)))
        if epoch == num_epochs - 1 or epoch % 5 == 0:
            # best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model, os.path.join(modelDir, '{modelName}_last.pth'.format(modelName=modelName)))

            epoch_coco_evaluator, epoch_test_metric_logger = engine.evaluate(model, test_data_loader, device=device)
            epoch_test_acc = epoch_coco_evaluator.coco_eval['bbox'].stats[0]

            AP_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[0]
            AP_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[1]
            AP_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[2]
            AP_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[3]
            AP_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[4]
            AP_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[5]

            AR_0 = epoch_coco_evaluator.coco_eval['bbox'].stats[6]
            AR_1 = epoch_coco_evaluator.coco_eval['bbox'].stats[7]
            AR_2 = epoch_coco_evaluator.coco_eval['bbox'].stats[8]
            AR_3 = epoch_coco_evaluator.coco_eval['bbox'].stats[9]
            AR_4 = epoch_coco_evaluator.coco_eval['bbox'].stats[10]
            AR_5 = epoch_coco_evaluator.coco_eval['bbox'].stats[11]
            AR_6 = epoch_coco_evaluator.coco_eval['bbox'].stats[12]
            precision = epoch_coco_evaluator.coco_eval['bbox'].stats[13]

            PRList.append([AP_0, AP_1, AP_2, AP_3, AP_4, AP_5, AR_0, AR_1, AR_2, AR_3, AR_4, AR_5, AR_6, precision])

        endtime = time.time()
        timeDict["YOLO"] = [endtime - starttime]
        csvpath = os.path.join(modelDir, str(512) + "time.csv")
        dataformat = pd.DataFrame(timeDict)
        dataformat.to_csv(csvpath)

        lossDf = pd.DataFrame(np.asarray(valList), columns=col_name_list)
        PRDf = pd.DataFrame(np.asarray(PRList), columns=["AP_0", "mAP50", "AP_2", "AP_3", "AP_4", "AP_5",
                                                         "AR_0", "AR_1", "AR_2", "AR_3", "AR_4", "AR_5", "recall",
                                                         "Precision_10"])
        lossDf.to_csv(trainLossFileName)
        PRDf.to_csv(valLossFileName)
        if "epochList" in kwargs.keys():
            epochList.append(epoch)

if __name__ == '__main__':

    # Train
    image_path = config.Combine_YOLODir
    json_path = config.Combine_COCODir
    modelNameList = ["SSD","MobileNet", "Retinanet"] + ["FasterRCNN"]
    size = 512
    num_classes = 2
    num_epochs = 1
    for modelName in modelNameList:
        for i in range(0, 3):
            train_img_folder = os.path.join(image_path, "images", r"train/{size}".format(size=size))
            train_ann_file = os.path.join(json_path, 'train{size}.json'.format(size=size))
            test_img_folder = os.path.join(image_path, "images", r"val/{size}".format(size=size))
            test_ann_file = os.path.join(json_path, 'val{size}.json'.format(size=size))
            train = True
            train_transforms, test_transform = get_transform(train), get_transform()
            train_data_loader = getTrainDataset(train_img_folder, train_ann_file, train_transforms)
            test_data_loader = getTestDataset(test_img_folder,test_ann_file)
            modelDir = os.path.join(config.ModelDir, modelName, str(size) + "_" + str(i))
            train_model(num_epochs, train_data_loader, test_data_loader, modelName=modelName, size=512,
                        num_classes=2, modelDir=modelDir)

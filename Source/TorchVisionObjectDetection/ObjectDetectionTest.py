#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np
import pandas as pd
import torch
import os
import time

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


def get_transform(train=False):
    transformsList = []
    transformsList.append(coco_utils.ConvertCocoPolysToMask())
    transformsList.append(T.PILToTensor())
    if train:
        transformsList.append(T.RandomRotation())
    transformsList.append(T.ToDtype(torch.float, scale=True))
    return T.Compose(transformsList)

def getTestDataset(test_img_folder, test_ann_file):
    test_transform=get_transform()
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


def val_test(test_data_loader, **kwargs):
    model_weight = kwargs["model_weight"]
    bestorlast = kwargs["bestorlast"]

    modelDir = PublicFunction.getFileExtName(model_weight)[0]

    #   # torch.save(model.state_dict(), model_path)但实际上它保存的不是模型文件，而是参数文件文件。在模型文件中，存储完整的模型，而在状态文件中，仅存储参数。因此，collections.OrderedDict只是模型的值。
    model = torch.load(model_weight)
    model.to(device)
    starttime = time.time()

    coco_evaluator, test_metric_logger = engine.evaluate(model, test_data_loader, device=device)

    cocoeval_result = coco_evaluator.coco_eval['bbox'].eval

    pr_array_0_5 = np.array(cocoeval_result['precision'])[0, :, 0, 0, 2].reshape((-1, 1))
    pr_array_0_75 = np.array(cocoeval_result['precision'])[5, :, 0, 0, 2].reshape((-1, 1))
    recall_array_0_5 = np.array(cocoeval_result['recall'])[0, 0, 0, 2].reshape((-1, 1))

    recall_array = np.arange(0.0, 1.01, 0.01).reshape((-1, 1))
    modeltime = None
    for name, meter in test_metric_logger.meters.items():
        if name == "model_time_sum":
            modeltime = meter

    # 保存精度
    AP_0 = coco_evaluator.coco_eval['bbox'].stats[0]
    AP_1 = coco_evaluator.coco_eval['bbox'].stats[1]
    AP_2 = coco_evaluator.coco_eval['bbox'].stats[2]
    AP_0_Array = np.full(np.shape(pr_array_0_75), AP_0)
    AP_1_Array = np.full(np.shape(pr_array_0_75), AP_1)
    AP_2_Array = np.full(np.shape(pr_array_0_75), AP_2)

    endtime = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starttime)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(endtime)))
    print(endtime - starttime)
    timesum_Array = np.full(np.shape(pr_array_0_75), modeltime)
    recall_array_0_5 = np.full(np.shape(pr_array_0_75), recall_array_0_5)

    PRNameList = ["Recall", "PR50", "PR75", "mAP50-95", "mAP50", "mAP75", "time_sum", "Recall50", "AR_6"]
    csvAccFileName = os.path.join(modelDir, "resultPR_AP_{bestorlast}.csv".format(bestorlast=bestorlast))
    dfAcc = pd.DataFrame(np.concatenate([recall_array, pr_array_0_5, pr_array_0_75,
                                         AP_0_Array, AP_1_Array, AP_2_Array, timesum_Array, recall_array_0_5], axis=1),
                         columns=PRNameList)
    dfAcc.to_csv(csvAccFileName)


def val_test_cocoeval(datasetName,test_data_loader, **kwargs):
    starttime = time.time()

    model_weight = kwargs["model_weight"]
    bestorlast = kwargs["bestorlast"]

    modelDir = PublicFunction.getFileExtName(model_weight)[0]
    valDir = os.path.join(modelDir, "val")
    PublicFunction.mkDir(valDir)

    model = torch.load(model_weight)
    model.to(device)
    model.eval()
    coco_evaluator, test_metric_logger = engine.evaluate(model, test_data_loader, device=device)

    mAP05_95 = coco_evaluator.coco_eval['bbox'].stats[0]  # IoU=0.50:0.95
    mAP50 = coco_evaluator.coco_eval['bbox'].stats[1]  # IoU=0.50
    mAP75 = coco_evaluator.coco_eval['bbox'].stats[2]  # IoU=0.75
    # AP_3 = coco_evaluator.coco_eval['bbox'].stats[3]
    # AP_4 = coco_evaluator.coco_eval['bbox'].stats[4]
    # AP_5 = coco_evaluator.coco_eval['bbox'].stats[5]
    #
    # AR_0 = coco_evaluator.coco_eval['bbox'].stats[6]
    # AR_1 = coco_evaluator.coco_eval['bbox'].stats[7]
    # AR_2 = coco_evaluator.coco_eval['bbox'].stats[8]
    # AR_3 = coco_evaluator.coco_eval['bbox'].stats[9]
    # AR_4 = coco_evaluator.coco_eval['bbox'].stats[10]
    # AR_5 = coco_evaluator.coco_eval['bbox'].stats[11]
    recall = coco_evaluator.coco_eval['bbox'].stats[12]  # Recall IOU=0.5
    precision = coco_evaluator.coco_eval['bbox'].stats[13]  # Precision IOU=0.5

    cocoeval_result = coco_evaluator.coco_eval['bbox'].eval
    pr_array_0_5 = np.array(cocoeval_result['precision'])[0, :, 0, 0, 2].reshape((-1, 1))
    # pr_array_0_75 = np.array(cocoeval_result['precision'])[5, :, 0, 0, 2].reshape((-1, 1))#0.75IOU
    recall_array = np.arange(0.0, 1.01, 0.01).reshape((-1, 1))

    mAP05_95_Array = np.full(np.shape(pr_array_0_5), mAP05_95)
    mAP50_Array = np.full(np.shape(pr_array_0_5), mAP50)
    mAP75_Array = np.full(np.shape(pr_array_0_5), mAP75)
    recall_array_0_5 = np.full(np.shape(pr_array_0_5), recall)

    precision_array_0_5 = np.full(np.shape(pr_array_0_5), precision)

    endtime = time.time()
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(starttime)))
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(endtime)))
    print(endtime - starttime)
    modeltime = endtime - starttime
    timesum_Array = np.full(np.shape(pr_array_0_5), modeltime)

    PRNameList = ["RecallX", "PR50", "mAP50-95", "mAP50", "mAP75", "time_sum", "Recall50", "precision50"]
    csvAccFileName = os.path.join(valDir,"{datasetName}_resultPR_AP_{bestorlast}.csv".format(datasetName=datasetName,bestorlast=bestorlast))
    dfAcc = pd.DataFrame(np.concatenate([recall_array, pr_array_0_5,
                                         mAP05_95_Array, mAP50_Array, mAP75_Array, timesum_Array, recall_array_0_5,
                                         precision_array_0_5], axis=1),
                         columns=PRNameList)
    dfAcc.to_csv(csvAccFileName)

def predict_test(test_data_loader, modelName="FasterRCNN", score_threshold=0.5, **kwargs):
    modelDir = kwargs["modelDir"]
    outputDir = os.path.join(modelDir, modelName + "Predict")
    PublicFunction.mkDir(outputDir)
    model_weight = kwargs["model_weight"]

    model = torch.load(model_weight)
    model.to(device)

    predict_test_logger = engine.predictTest(model, test_data_loader, device, score_threshold=score_threshold,
                                             outputDir=outputDir)


if __name__ == '__main__':


    best_Or_last = "last"  # best,last
    modelNameList = ["FasterRCNN"] + ["SSD", "MobileNet", "Retinanet"]

    for datasetName in ["Overall", "PingLu", "SanYuan", "ShanZhou", "ChunHua"]:
        if datasetName == "Overall":
            image_path = config.Combine_YOLODir
            json_path = config.Combine_COCODir
        elif datasetName == "PingLu":
            image_path = config.PingLu_YOLODir
            json_path = config.PingLu_COCODir
        elif datasetName == "SanYuan":
            image_path = config.SanYuan_YOLODir
            json_path = config.SanYuan_COCODir
        elif datasetName == "ShanZhou":
            image_path = config.ShanZhou_YOLODir
            json_path = config.ShanZhou_COCODir
        elif datasetName == "ChunHua":
            image_path = config.ChunHua_YOLODir
            json_path = config.ChunHua_COCODir
        else:
            raise ValueError("Undefined dataset")

        for modelName in modelNameList:
            for i in range(0, 3):


                test_img_folder = os.path.join(image_path, "images", "val", "{size}".format(size=512))
                test_ann_file = os.path.join(json_path, 'val{size}.json'.format(size=512))
                train = False
                test_data_loader = getTestDataset(test_img_folder, test_ann_file)

                modelDir = os.path.join(config.ModelDir, modelName, str(512) + "_" + str(i))
                model_weight = os.path.join(modelDir, r"{modelName}_{bestorlast}.pth".format(modelName=modelName, bestorlast=best_Or_last))

                # val_test( test_data_loader,modelName=modelName,modelDir=modelDir,model_weight=model_weight,bestorlast=best_Or_last)
                val_test_cocoeval(datasetName,test_data_loader, modelName=modelName, modelDir=modelDir, model_weight=model_weight, bestorlast=best_Or_last)

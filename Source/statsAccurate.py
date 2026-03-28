import numpy as np
import pandas as pd
import os

import sys
from pathlib import Path
projectDir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectDir))
# try:
#     from Source import config
#     from Source.Utility import PublicFunction
# except ImportError as e:
#     print(f"Import Source Failed：{e}")
#     print(f"Make sure Source directory")
#     sys.exit(1)
import config
from Utility import PublicFunction
def getAccAndTime(modelDirDict,file_prex=None):

    timeDict = {}
    for k in modelDirDict.keys():
        timeDict[k]=[]

    resultList=[]
    for k in timeDict.keys():
        precisionList=[]
        recallList=[]
        MAP50List=[]
        MAP50_90List=[]

        trainTimeList=[]
        fpsList=[]
        modelDir=modelDirDict[k]
        F1List=[]
        for i in range(0,3):
            modelFullDir=os.path.join(modelDir,"512_"+str(i))
            if not PublicFunction.check_existence(modelFullDir):
                continue
            if k =="YOLO" or k=="CA":
                count =500
            else:
                count = pd.read_csv(os.path.join(modelFullDir, "TrainLoss.csv")).values.shape[0]
            if file_prex is not None:
                df = pd.read_csv(os.path.join(modelFullDir, "val","{file_prex}_resultPR_AP_last.csv".format(file_prex=file_prex)))

            else:
                df = pd.read_csv(os.path.join(modelFullDir, "val","Overall_resultPR_AP_last.csv"))
            precision=df["precision50"].values[-1]
            recall = df["Recall50"].values[-1]
            map50=df["mAP50"].values[-1]
            map50_90=df["mAP50-95"].values[-1]
            F1 = 2 * precision * recall / (precision + recall)

            F1List.append(F1)
            precisionList.append(precision)
            recallList.append(recall)
            MAP50List.append(map50)
            MAP50_90List.append(map50_90)

            fps=int(5920/df["time_sum"].values[-1])
            timeFileName=os.path.join(modelFullDir,"512time.csv")
            if PublicFunction.check_existence(timeFileName):
                 seconds=pd.read_csv(timeFileName)["YOLO"].values
                 ss=int(seconds)/count*200
                 m, s = divmod(ss, 60)
                 h, m = divmod(m, 60)
                 trainTimeList.append(h)
            else:
                 trainTimeList.append(0)
            fpsList.append(fps)

        meanPrecsion=np.mean(precisionList)
        meanRecall=np.mean(recallList)
        meanF1=np.mean(F1List)
        meanMAP50=np.mean(MAP50List)
        meanMAP50_95=np.mean(MAP50_90List)
        meanTrainTime=np.mean(trainTimeList)
        meanfps=np.mean(fpsList)
        resultList.append([meanPrecsion,meanRecall,meanF1,meanMAP50,meanMAP50_95,meanTrainTime,meanfps])
        timeDict[k]=[meanPrecsion,meanRecall,meanF1,meanMAP50,meanMAP50_95,meanTrainTime,meanfps]
    if file_prex is None or  file_prex == "Combine":

        statsCsvFileName = os.path.join(config.ModelDir, r"stats.csv")
    else:
        statsCsvFileName = os.path.join(config.ModelDir, r"{file_prex}_stats.csv".format(file_prex=file_prex))
    df = pd.DataFrame(np.asarray(resultList).round(decimals=2),
                      columns= ["Precision", "Recall", "F1","MAP50", "MAP50-90","Training Time", "FPS"],index=[k for k in timeDict.keys()])
    df.to_csv(statsCsvFileName)
    return timeDict
#

if __name__ == '__main__':

    modelDirDict = {
        "FasterRCNN": os.path.join(config.ModelDir, "FasterRCNN"),
        "MobileNet": os.path.join(config.ModelDir, "MobileNet"),
        "SSD":os.path.join(config.ModelDir, "SSD"),
        "RetinaNet":os.path.join(config.ModelDir, "Retinanet"),
        "YOLO": os.path.join(config.ModelDir, "YOLOV11")
    }
    #Overall accurate
    getAccAndTime(modelDirDict)
    #Different region accurate
    # datasetName="PingLu"
    # datasetName="SanYuan"
    datasetName = "ShanZhou"
    # datasetName = "ChunHua"
    getAccAndTime(modelDirDict,datasetName)
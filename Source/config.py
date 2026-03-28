import os.path

level = 20
# rootDir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.path.pardir))
import sys
from pathlib import Path
rootDir = Path(__file__).resolve().parent.parent
sys.path.append(str(rootDir))
# rootDir = os.path.dirname(os.path.abspath(__file__))
print(rootDir)
# DatasetData directory
datasetDir = rootDir /"Data"/ "GoogleEarthData"
print(datasetDir)
#Training Data
Combine_YOLODir = datasetDir/"Combine" / "YOLO"
Combine_COCODir = datasetDir/"Combine" / "COCO"

#Test Data
#ShanZhou
ShanZhou_YOLODir = datasetDir/"ShanZhouDatasetData"/ "YOLO"
ShanZhou_COCODir = datasetDir/"ShanZhouDatasetData" / r"COCO"
#PingLu
PingLu_YOLODir = datasetDir/"PingLuDatasetData"/ "YOLO"
PingLu_COCODir = datasetDir/"PingLuDatasetData"/ "COCO"
#ChunHua
ChunHua_YOLODir = datasetDir/"ChunHuaDatasetData"/ "YOLO"
ChunHua_COCODir = datasetDir/"ChunHuaDatasetData"/ "COCO"
#SanYuan
SanYuan_YOLODir = datasetDir/"SanYuanDatasetData"/ "YOLO"
SanYuan_COCODir = datasetDir/"SanYuanDatasetData"/ "COCO"




# Models directory
ModelDir = rootDir/ "Models"
print(ModelDir)
#Results directory
ResultsDir=rootDir/ "Data/Results"
# DEM file
ExampleDir=rootDir/ "Data/Examples"
DEM_ShanZhouYunChengFileName =str(ExampleDir/ "DEM_ShanZhouYunCheng.tif")
Aspect_ShanZhouYunChengFileName = str(ExampleDir/ "Aspect_ShanZhouYunCheng.tif")
Slope_ShanZhouYunChengFileName = str(ExampleDir/ "Slope_ShanZhouYunCheng.tif")

#JPG directory
JPGDir=rootDir/ "Data/JPG"

# For cascading tile inference

CascadingTifDir = rootDir/ "Data/Examples/DiKeng20_34Example"
PredictResultDir = rootDir/ "Data/Examples/PredictResult"
CascadingResultDir = rootDir/ "Data/Examples/CascadingResult"

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import time
import os
import pandas as pd
import torch
import sys
from pathlib import Path
projectDir = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(projectDir))

try:
    from Source import config
except ImportError as e:
    print(f"Import Source.config Failed：{e}")
    print(f"Make sure Source dir")
    sys.exit(1)

# from ultralytics.utils import SETTINGS, SETTINGS_FILE
# if SETTINGS_FILE.exists():
#     os.remove(SETTINGS_FILE)
# datasets_dir = projectDir / "Data" / "GoogleEarthData"
# settings.update({"datasets_dir": str(datasets_dir)})

# projectDir=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.path.pardir))
# projectDir=str(Path(__file__).resolve().parent.parent.parent)
# print(projectDir)
# settings.update({"datasets_dir":os.path.join(projectDir,"Data","GoogleEarthData")})
# dataconfig = os.path.join(os.path.dirname(os.path.abspath(__file__)),r"SilocaveDataConfig512.yaml")

# 覆盖默认配置
# 绝对路径高于默认配置高于相对路径
# Setting the dataset directory,overriding the default directory
# Set settings.py or C:\Users\XXXXXX\AppData\Roaming\Ultralytics， "datasets_dir": "Your dataset Dir"
data_config = projectDir / "Source" / "YOLOV11ObjectDetection" / "SilocaveDataConfig512.yaml"
yaml_content = f"""
path: {str(projectDir / "Data" / "GoogleEarthData" / "Combine" / "YOLO")}
train: images/train/512
val: images/val/512
test: 
names:
  0: Silo-cave
download: false
"""
yaml_path = Path(__file__).parent / "SilocaveDataConfig512.yaml"
with open(yaml_path, "w", encoding="utf-8") as f:
    f.write(yaml_content)

default_config = projectDir / "Source" / "YOLOV11ObjectDetection" / "default.yaml"

from ultralytics import YOLO
import gc
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(torch.cuda.is_available())  # True = GPU / False = CPU
print(torch.version.cuda)
print(torch.backends.cudnn.version())

if __name__ == '__main__':


    size = 512

    timeDict = {}
    for i in range(0, 3):
        pre_model_name = 'yolo11n.pt'
        # Load a mode1
        model = YOLO(pre_model_name, task="detect")  # load a pretrained model (recommended for training)
        starttime = time.time()
        save_dir = Path(config.ModelDir) / "YOLOV11" / f"512_{i}"
        save_dir.mkdir(parents=True, exist_ok=True)

        model.train(
            data=str(data_config),
            epochs=1,
            imgsz=512,
            batch=5,
            cfg=str(default_config),
            project=str(save_dir.parent),
            name=save_dir.name,
            device=device,
            workers=0,
            exist_ok=True,
            pretrained=True
        )

        gc.collect()
        torch.cuda.empty_cache()

        endtime = time.time()
        timeDict["YOLO"] = [endtime - endtime]
        csv_path = save_dir / f"{size}_time.csv"

        dataformat = pd.DataFrame(timeDict)
        dataformat.to_csv(csv_path)

#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import numpy as np

import matplotlib.pyplot as plt
from distutils.version import LooseVersion
import matplotlib
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

def getPixels(geoCoordsList, geoInfoFileName):
    im_data, im_width, im_height, im_bands, im_geotrans, im_proj = PublicFunction.readTiff(geoInfoFileName)
    pixelsList = []
    for geoCoords in geoCoordsList:
        lon = geoCoords[0]
        lan = geoCoords[1]
        X, Y = PublicFunction.lonlat2geo(im_proj, float(lon), float(lan))  # 经纬度转地理坐标
        x, y = PublicFunction.geo2imagexy(im_geotrans, X, Y)
        pixels = im_data[:, x:x + 1, y:y + 1].astype(float).flatten()[0]
        pixelsList.append([lon, lan, pixels])
    return pd.DataFrame(np.asarray(pixelsList), columns=["LON", "LAN", "Value"])


def exportLocationData(shapeFileName, demFileName, outputFileName):
    fieldValuesMapList = PublicFunction.readShape(shapeFileName)
    GeoCoodList = []
    for fieldValuesMap in fieldValuesMapList:
        LON = fieldValuesMap["LON"]
        LAN = fieldValuesMap["LAN"]
        GeoCoodList.append([LON, LAN])

    if len(GeoCoodList) > 0:
        pixelsDf = getPixels(GeoCoodList, demFileName)
        pixelsDf.replace(9999, np.NaN, inplace=True)
        pixelsDf.dropna(axis=0, how='any', inplace=True)
        pixelsDf.to_csv(outputFileName)


def plotKDEPDF(data2,XLabel,title=None,jpgFileFullName=None):
    scaleData2= data2[~np.isnan(data2)]

    print(matplotlib.__version__)
    if LooseVersion(matplotlib.__version__) >= '2.1':
        density_param = {'density': True}
    else:
        density_param = {'normed': True}
    # Plot the three kernel density estimates
    fig = plt.figure(figsize=(2.5,2.5))
    ax = fig.add_axes([0.05, 0.15, 0.94, 0.85])
    ax.set_title(title,  x=0.5, y=0.9,fontsize=6, fontweight='bold')
    scaleData2 = scaleData2[~np.isnan(scaleData2)]
    ax.hist(scaleData2, bins=100, fc='#AAAAFF', label=XLabel+" Histogram", **density_param)

    dataArrayPercentile2=scaleData2[~np.isnan(scaleData2)]
    min=np.min(dataArrayPercentile2).astype(int)
    max=np.max(dataArrayPercentile2).astype(int)
    mean=np.median(dataArrayPercentile2).astype(int)
    Q1=np.quantile(dataArrayPercentile2,.25).astype(int)
    Q2=np.quantile(dataArrayPercentile2,.50).astype(int)
    Q3=np.quantile(dataArrayPercentile2,.75).astype(int)


    X_plot = np.linspace(min, max, 100)
    kde2 = PublicFunction.KDE(dataArrayPercentile2, bmin=1, bmax=30, interSize=10)
    kde_data2 = np.exp(kde2.score_samples(X_plot[:, np.newaxis]))
    statTxt="Min="+str(min)+"\n"+"Max="+str(max)+"\n"+"Q1="+str(Q1)+"\n"+"Q2="+str(Q2)+"\n"+"Q3="+str(Q3)+"\n"
    ax.plot(X_plot, kde_data2, color='black', alpha=1, lw=1, label="Probability Density Function")

    y=np.max(kde_data2)*0.65
    if XLabel=="Elevation" :
        x = 850
        y = np.max(kde_data2) * 0.65
        ax.text( x,y,statTxt , fontsize=5)
    elif XLabel=="Aspect" :
        x = 150
        y = np.max(kde_data2) * 0.65
        ax.text( x,y,statTxt, fontsize=5 )
    elif XLabel == "Slope":
        x = 15
        y = np.max(kde_data2) * 0.65
        ax.text(x,y, statTxt, fontsize=5)
    else:
        ax.text( X_plot[int(len(X_plot)*0.7)],np.max(kde_data2),statTxt)

    ax.set_yticklabels([])
    ax.set_yticks([])
    ax.set_xlabel(XLabel, fontsize=4, fontweight='bold')
    ax.set_ylabel(r'Density', fontsize=4, fontweight='bold')
    plt.savefig(jpgFileFullName, dpi=400.0)



if __name__ == '__main__':
    # MobileNet result
    title = "ShanZhou"
    shapeFileName = os.path.join(config.ResultsDir, "{name}Points.shp".format(name=title))

    # DEM
    demFileName = config.DEM_ShanZhouYunChengFileName
    outputFileName = os.path.join(config.ExampleDir, title + "_DEM.csv")
    exportLocationData(shapeFileName, demFileName, outputFileName)
    #plotting DEM Density
    jpgFileFullName = os.path.join(config.JPGDir, title + "_DEM.jpg")
    df2 = pd.read_csv(outputFileName)
    data2 = df2[["Value"]]
    plotKDEPDF(data2, r'Elevation', title=title, jpgFileFullName=jpgFileFullName)

    # Aspect
    aspectFileName = config.Aspect_ShanZhouYunChengFileName
    outputFileName = os.path.join(config.ExampleDir, title + "_Aspect.csv")
    exportLocationData(shapeFileName, aspectFileName, outputFileName)
    #plotting Aspect Density
    jpgFileFullName = os.path.join(config.JPGDir, title + "_Aspect.jpg")
    df2 = pd.read_csv(outputFileName)
    data2 = df2[["Value"]]
    plotKDEPDF(data2, r'Aspect', title=title, jpgFileFullName=jpgFileFullName)

    # Slope
    slopeFileName = config.Slope_ShanZhouYunChengFileName
    outputFileName = os.path.join(config.ExampleDir, title + "_Slope.csv")
    exportLocationData(shapeFileName, slopeFileName, outputFileName)
    #plotting Slope Density
    jpgFileFullName = os.path.join(config.JPGDir, title + "_Slope.jpg")
    df2 = pd.read_csv(outputFileName)
    data2 = df2[["Value"]]
    plotKDEPDF(data2, "Slope", title=title, jpgFileFullName=jpgFileFullName)





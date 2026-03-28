#!/usr/bin/env python
# _*_ coding: utf-8 _*_
import os, shutil
from osgeo import gdal
import numpy as np
from osgeo import osr
import osgeo
from osgeo import ogr
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
#
def listFiles(directory=None, ext=None):
    file_names = []
    if directory is not None:
        dir_file_lists = os.listdir(directory)
        lens = len(dir_file_lists)
        for i in range(lens):
            if ext is not None:
                if os.path.splitext(dir_file_lists[i])[1] == ext:
                    file_names.append(dir_file_lists[i])
            else:
                file_names.append(dir_file_lists[i])
        return file_names
    else:
        return file_names


def copyFile(srcfile, dstfile):
    if not os.path.isfile(srcfile):
        print("%s not exist!" % (srcfile))
    else:
        fpath, fname = os.path.split(dstfile)  # 分离文件名和路径
        if not os.path.exists(fpath):
            os.makedirs(fpath)  # 创建路径
        shutil.copyfile(srcfile, dstfile)  # 复制文件
        print("copy %s -> %s" % (srcfile, dstfile))


def mkDir(path):
    path = str(path).strip()
    path = path.rstrip("\\")
    path = path.rstrip("/")

    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
        print(path + ' Created')
        return True
    else:
        print(path + ' Exist!')
        return False


def getFileExtName(file_name):
    (file_path, temp_file_name) = os.path.split(file_name)
    (shot_name, extension) = os.path.splitext(temp_file_name)
    return file_path, shot_name, extension


def readTiff(file_name, recovery_resolution=False, resolutionTime=None, gussian=False, nodata=None):
    # im_data, im_width, im_height, im_bands, im_geotrans, im_proj
    dataset = gdal.Open(file_name)
    if dataset == None:
        raise IOError('Error:', file_name)
    im_width = dataset.RasterXSize
    im_height = dataset.RasterYSize
    im_bands = dataset.RasterCount
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    imdata = []
    if im_bands == 1:
        imdata.append(im_data)
        im_data = np.array(imdata)
    if nodata is not None:
        for i in range(im_bands):
            im_data[i][im_data[i] == nodata] = np.NaN
    im_proj = dataset.GetProjection()
    im_geotrans = dataset.GetGeoTransform()
    return im_data, im_width, im_height, im_bands, im_geotrans, im_proj


def writeTiff(file_name, im_data, im_geotrans=None, im_proj=None, bandNamesList=None, color_table=None, nodata=None):
    if color_table is not None:
        im_data = np.asarray(im_data, dtype=np.int)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    elif 'int32' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
        im_data = im_data[np.newaxis, :]
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(file_name, im_width, im_height, im_bands, datatype)  # ,options=['COMPRESS=LZW']
    if (dataset != None):
        if im_geotrans is not None:
            dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        if im_proj is not None:
            dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        band = dataset.GetRasterBand(i + 1)
        if nodata is not None:
            band.SetNoDataValue(nodata)
        if bandNamesList is not None:
            # band.SetNoDataValue(0)
            BandName = bandNamesList[i]
            band.SetDescription(BandName)  # This sets the band name!
        if color_table is not None:
            # Set the color table for your band
            if np.shape(im_data)[0] > 1:
                raise ValueError("colortable zhi neng yi ge boduan")
            else:
                band.SetColorTable(color_table)
        # print(im_data[:,1,:])
        band.WriteArray(im_data[i])
    dataset.FlushCache()
    del dataset





def parseDegressSecond(intDegree):
    strDg = str(intDegree)
    degree = strDg[0:-2]
    second = strDg[-2:]
    return int(degree), int(second)


def decimal2DegreeSecond(decimalFloat):
    degree = int(decimalFloat)
    minute = int((decimalFloat - degree) * 60)
    second = int(((decimalFloat - degree) * 60 - minute) * 60)
    return degree, minute, second


def degreeSecond2Decimal(degree, minute, second=0):
    return float(degree) + float(minute) / 60 + float(second) / 3600


def createShape(shape_name, fieldsList, tableList, geoType="point", epsg=4326, strWidth=40):
    path, filename, ext = getFileExtName(shape_name)
    gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
    gdal.SetConfigOption('SHAPE_ENCODING', 'GBK')
    typeList = []
    for t in tableList[0]:
        if isinstance(t, int):
            typeList.append(ogr.OFTInteger64)
        elif isinstance(t, float):
            typeList.append(ogr.OFTReal)
        elif isinstance(t, str):
            typeList.append(ogr.OFTString)
        else:
            typeList.append(ogr.OFTString)
    ogr.RegisterAll()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    if os.path.exists(shape_name):
        driver.DeleteDataSource(shape_name)
    ds = driver.CreateDataSource(shape_name)
    sr = osr.SpatialReference()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        sr.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    sr.ImportFromEPSG(epsg)
    shapLayer = None
    if geoType == "point":
        shapLayer = ds.CreateLayer(filename, sr, geom_type=ogr.wkbPoint)
    if geoType == "points":
        shapLayer = ds.CreateLayer(filename, sr, geom_type=ogr.wkbMultiPoint)
    if geoType == "rect":
        shapLayer = ds.CreateLayer(filename, sr, geom_type=ogr.wkbPolygon)
    if geoType == "polygon" or geoType == "poly":
        shapLayer = ds.CreateLayer(filename, sr, geom_type=ogr.wkbPolygon)
    lonIndex = None
    lanIndex = None
    index = 0
    for field in fieldsList:
        if field == "LON":
            lonIndex = index
        if field == "LAN":
            lanIndex = index
        fieldDefn = ogr.FieldDefn(str(field), typeList[index])
        if typeList[index] == ogr.OFTString:
            fieldDefn.SetWidth(strWidth)
        shapLayer.CreateField(fieldDefn)
        index = index + 1
    for obj in tableList:
        defn = shapLayer.GetLayerDefn()
        feature = ogr.Feature(defn)
        index = 0
        for field in fieldsList:
            feature.SetField(str(field), str(obj[index]))
            index = index + 1
        geoObj = None
        if geoType == "point":
            geoObj = ogr.Geometry(ogr.wkbPoint)
            geoObj.AddPoint(obj[lonIndex], obj[lanIndex], 0)
        if geoType == "rect":
            ringXleftOrigin = obj[-4]
            ringXrightOrigin = obj[-2]
            ringYtop = obj[-3]
            ringYbottom = obj[-1]
            ring = ogr.Geometry(ogr.wkbLinearRing)
            ring.AddPoint(ringXleftOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYtop)
            ring.AddPoint(ringXrightOrigin, ringYbottom)
            ring.AddPoint(ringXleftOrigin, ringYbottom)
            ring.CloseRings()
            geoObj = ogr.Geometry(ogr.wkbPolygon)
            geoObj.AddGeometry(ring)
        if geoType == "polygon" or geoType == "poly":  # 矩形元素
            geoObj = ogr.Geometry(ogr.wkbPolygon)
            ring = ogr.Geometry(ogr.wkbLinearRing)
            for ii, jj in zip(obj[lonIndex], obj[lanIndex]):
                ring.AddPoint(ii, jj, 0)
            ring.CloseRings()
            geoObj.AddGeometry(ring)
        feature.SetGeometry(geoObj)
        shapLayer.CreateFeature(feature)
        feature.Destroy()
    print("write shapefile" + shape_name)
    ds.Destroy()


def check_existence(path_of_file):
    path_of_file = os.path.abspath(path_of_file)
    return os.path.exists(path_of_file)


def readShape(shape_name, fieldList=None, encode="GBK", Area=False):
    if encode == "":
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "YES")
        gdal.SetConfigOption("SHAPE_ENCODING", "")
    else:
        gdal.SetConfigOption("SHAPE_ENCODING", "GBK")
    ogr.RegisterAll()
    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(shape_name)
    if ds is None:
        raise IOError('File error', shape_name)
        # sys.exit(1)
    in_layer = ds.GetLayer(0)
    spatialRef = in_layer.GetSpatialRef()
    if fieldList is None:
        fieldList = []
        for field in in_layer.schema:
            fieldList.append(field.name)
        print(fieldList)
    fieldValuesMapList = []
    fcount = in_layer.GetFeatureCount()
    for i in range(0, fcount):
        in_feature = in_layer.GetFeature(i)
        fieldValues = []
        fieldListXY = []
        for field in fieldList:
            fieldValue = in_feature.GetField(field)
            fieldValues.append(fieldValue)
            fieldListXY.append(field)
        fieldListXY.append("LON")
        fieldListXY.append("LAN")
        geom = in_feature.GetGeometryRef()
        geomName = geom.GetGeometryName()
        # MULTILINESTRING
        if geomName == "LINESTRING" or geomName == "MULTILINESTRING":
            # if geom.GetGeometryCount()>1:
            for k in range(geom.GetGeometryCount()):
                ring_out = geom.GetGeometryRef(k)
                X = []
                Y = []
                fieldValuesPoly = fieldValues.copy()
                for m in range(ring_out.GetPointCount()):
                    X.append(ring_out.GetX(m))
                    Y.append(ring_out.GetY(m))
                fieldValuesPoly.append(X)
                fieldValuesPoly.append(Y)
                dictionary = dict(zip(fieldListXY, fieldValuesPoly))
                fieldValuesMapList.append(dictionary)

        if geomName == "MULTIPOLYGON":
            for k in range(geom.GetGeometryCount()):
                poly = geom.GetGeometryRef(k)
                wkbPloy = poly.ExportToWkb()
                polygon = ogr.CreateGeometryFromWkb(wkbPloy)
                ring_out = polygon.GetGeometryRef(0)
                X = []
                Y = []
                fieldValuesPoly = fieldValues.copy()
                for m in range(ring_out.GetPointCount()):
                    X.append(ring_out.GetX(m))  # 经度
                    Y.append(ring_out.GetY(m))  # 纬度
                fieldValuesPoly.append(X)
                fieldValuesPoly.append(Y)
                dictionary = dict(zip(fieldListXY, fieldValuesPoly))
                # print(dictionary)
                fieldValuesMapList.append(dictionary)
        if geomName == "POLYGON":
            if geom.GetGeometryCount() > 1:
                # print(dir(geom))
                for n in range(geom.GetGeometryCount()):
                    fieldValuesCopyOut = fieldValues.copy()
                    X = []
                    Y = []
                    ring_out = geom.GetGeometryRef(n)
                    for j in range(ring_out.GetPointCount()):
                        X.append(ring_out.GetX(j))
                        Y.append(ring_out.GetY(j))
                    fieldValuesCopyOut.append(X)
                    fieldValuesCopyOut.append(Y)
                    if Area is True:
                        fieldListXY.append("Area")
                        fieldValuesCopyOut.append(geom.GetArea())
                    dictionaryOut = dict(zip(fieldListXY, fieldValuesCopyOut))
                    if n >= 1:
                        dictionaryOut["border"] = "Inner"
                    fieldValuesMapList.append(dictionaryOut)
            else:
                ring_out = geom.GetGeometryRef(0)
                X = []
                Y = []
                for j in range(ring_out.GetPointCount()):
                    X.append(ring_out.GetX(j))  # 经度
                    Y.append(ring_out.GetY(j))  # 纬度
                fieldValues.append(X)
                fieldValues.append(Y)
                if Area is True:
                    fieldListXY.append("Area")
                    fieldValues.append(geom.GetArea())
                dictionary = dict(zip(fieldListXY, fieldValues))
                fieldValuesMapList.append(dictionary)
        # for point in points:
        if geomName == "POINT":
            X = geom.GetX()
            Y = geom.GetY()
            fieldValues.append(X)
            fieldValues.append(Y)
            dictionary = dict(zip(fieldListXY, fieldValues))
            fieldValuesMapList.append(dictionary)
    ds.Destroy()
    return fieldValuesMapList


def geo2imagexy(im_geotrans, x, y):
    """
     Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
     the pixel location of a geospatial coordinate
     """
    ulX = im_geotrans[0]
    ulY = im_geotrans[3]
    xDist = im_geotrans[1]
    yDist = im_geotrans[5]

    rtnX = im_geotrans[2]
    rtnY = im_geotrans[4]
    colum = abs(int((x - ulX) / xDist))
    row = abs(int((ulY - y) / yDist))
    return row, colum

def lonlat2geo(projection, lon, lat):
    prosrs = osr.SpatialReference()
    if int(osgeo.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        prosrs.SetAxisMappingStrategy(osgeo.osr.OAMS_TRADITIONAL_GIS_ORDER)
    prosrs.ImportFromWkt(projection)
    geosrs = prosrs.CloneGeogCS()
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]


def KDE(X,bmin=0.1,bmax=1,interSize=20):


    if len(np.shape(X)) == 1:
        X = X[:, np.newaxis]
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(bmin, bmax, interSize)},
                        cv=3, n_jobs=5)  # 20-fold cross-validation
    grid.fit(X)

    kde = grid.best_estimator_
    print(kde.bandwidth)
    return kde


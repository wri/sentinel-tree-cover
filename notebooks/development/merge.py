import subprocess
import os
from osgeo import gdal
import pandas as pd

data = pd.read_csv("../../src/somalia_sept.csv")
data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
data['X_tile'] = pd.to_numeric(data['X_tile'])
data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
data['Y_tile'] = pd.to_numeric(data['Y_tile'])

'''
tifs = []
for index, row in data.iterrows():
    if os.path.isfile(f"somalia/{row['X_tile']}X{row['Y_tile']}Y_FINAL.tif"):
        tifs.append(f"somalia/{row['X_tile']}X{row['Y_tile']}Y_FINAL.tif")
print(f'there are {len(tifs)} tifs')
gdal.BuildVRT(f'out.vrt', tifs, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
#add_pixel_fn(f'{str(args.country)}.vrt')
ds = gdal.Open(f'out.vrt')
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"))
ds = gdal.Translate(f'out.tif', ds, options=translateoptions)
os.remove(f'out.vrt')
'''

tifs = []
for index, row in data.iterrows():
    if os.path.isfile(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_XY.tif"):
        tifs.append(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_XY.tif")
    elif os.path.isfile(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_X.tif"):
        tifs.append(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_X.tif")
    elif os.path.isfile(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_Y.tif"):
        tifs.append(f"smooth/{row['X_tile']}X{row['Y_tile']}Y_SMOOTH_Y.tif")
print(f'there are {len(tifs)} tifs')
gdal.BuildVRT(f'smooth.vrt', tifs, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
#add_pixel_fn(f'{str(args.country)}.vrt')
ds = gdal.Open(f'smooth.vrt')
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"))
ds = gdal.Translate(f'smooth.tif', ds, options=translateoptions)
os.remove(f'smooth.vrt')


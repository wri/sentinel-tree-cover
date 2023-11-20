import rasterio as rs
import numpy as np
import os

import os
from rasterio.merge import merge
from osgeo import gdal

tifs_folder = 'change/costarica/'
tif_out = 'costarica-gain.tif'

files_to_mosaic = [tifs_folder + x for x in os.listdir(tifs_folder) if ".tif" in x]
print(f"There are {len(files_to_mosaic)} tiles processed in {tifs_folder}, writing {tif_out}")
gdal.BuildVRT(f'out.vrt', files_to_mosaic, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
print("Built VRT")
ds = gdal.Open(f'out.vrt')
translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255 -co BIGTIFF=YES"))
print("Translate options")
ds = gdal.Translate(tif_out, ds)#, options=translateoptions)
os.remove(f'out.vrt')

'''
for file in files_to_mosaic[:9]:
    src = rs.open(file, "r")
    print(src.shape)
    src_files_to_mosaic.append(src)
        
mosaic, out_trans = merge(src_files_to_mosaic)

out_meta = src.meta.copy()
out_meta.update({"driver": "GTiff",
                 "height": mosaic.shape[1],
                 "width": mosaic.shape[2],
                 "transform": out_trans,
                 "crs": "+proj=longlat +datum=WGS84 +no_defs",
                 'compress':'lzw', 
                 'dtype': 'uint8'
                }
               )

with rs.open(tif_out, "w", **out_meta) as dest:
    dest.write(mosaic)
'''


"""

with rs.open("tmlgte10.tif") as dataset:
    out_meta = dataset.meta.copy()
    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            53435,
            400751,
        ),
        resampling=rs.enums.Resampling.nearest
    )
    
    # scale image transform
    with rs.open('TMLGTERESAMPLED.tif', "w", **out_meta) as dest:
        dest.write(data)
"""
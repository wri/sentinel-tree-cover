import os
from osgeo import gdal
from glob import glob
import pandas as pd
from downloading.io import FileUploader
import yaml

os.environ["GDAL_VRT_ENABLE_PYTHON"] = "YES"

def add_pixel_fn(filename: str) -> None:
    """inserts pixel-function into vrt file named 'filename'
    Args:
        filename (:obj:`string`): name of file, into which the function will be inserted
        resample_name (:obj:`string`): name of resampling method
    """

    header = """  <VRTRasterBand dataType="Byte" band="1" subClass="VRTDerivedRasterBand">"""
    contents = """
    <PixelFunctionType>average</PixelFunctionType>
    <PixelFunctionLanguage>Python</PixelFunctionLanguage>
    <PixelFunctionCode><![CDATA[
import numpy as np
def average(in_ar, out_ar, xoff, yoff, xsize, ysize, raster_xsize,raster_ysize, buf_radius, gt, **kwargs):
    x = np.ma.masked_greater(in_ar, 100)
    np.mean(x, axis = 0,out = out_ar, dtype = 'uint8')
    mask = np.all(x.mask,axis = 0)
    out_ar[mask]=255
]]>
    </PixelFunctionCode>"""

    lines = open(filename, 'r').readlines()
    lines[3] = header  # FIX ME: 3 is a hand constant
    lines.insert(4, contents)
    open(filename, 'w').write("".join(lines))

if __name__ == '__main__':
    import argparse
    data = pd.read_csv("processing_area_june_28.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    args = parser.parse_args()

    if os.path.exists(args.yaml_path):
        with open(args.yaml_path, 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']


    data = data[data['country'] == args.country]
    data = data.reset_index(drop = True)
    print(f"There are {len(data)} tiles for {args.country}")

    tifs = []
    for index, row in data.iterrows():
        x = str(int(row['X_tile']))
        y = str(int(row['Y_tile']))
        dir_i = f"../project-monitoring/tof-output/{x}/{y}"
        if os.path.exists(dir_i):
            files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
            if len(files) > 1:
                files = [x for x in files if "_POST" in x]
            for file in files:
                tifs.append(os.path.join(dir_i, file))


    print(f'There are {len(tifs)} / {len(data)} tiles processed')
    
    gdal.BuildVRT(f'{str(args.country)}.vrt', tifs, options=gdal.BuildVRTOptions(srcNodata=255, VRTNodata=255))
    #add_pixel_fn(f'{str(args.country)}.vrt')
    ds = gdal.Open(f'{str(args.country)}.vrt')
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255"))
    ds = gdal.Translate(f'{str(args.country)}.tif', ds, options=translateoptions)

    ds = gdal.Open(f'{str(args.country)}.tif', 1)
    band = ds.GetRasterBand(1)

    # create color table
    colors = gdal.ColorTable()

    # set color for each value
    colors.SetColorEntry(0, (237, 248, 233))
    colors.SetColorEntry(20, (237, 248, 233))
    colors.SetColorEntry(40, (161, 220, 149))
    colors.SetColorEntry(60, (104, 184, 105))
    colors.SetColorEntry(80, (34, 142, 66))
    colors.SetColorEntry(100, (7, 107, 44))

    # set color table and color interpretation
    band.SetRasterColorTable(colors)
    band.SetRasterColorInterpretation(gdal.GCI_PaletteIndex)

    # close and save file
    del band, ds

    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
    file = f'{str(args.country)}.tif'
    key = f'2020/mosaics/' + file
    uploader.upload(bucket = args.s3_bucket, key = key, file = file)
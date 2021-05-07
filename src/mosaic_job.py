import os
from osgeo import gdal
from glob import glob
import pandas as pd
from downloading.io import FileUploader
import yaml


if __name__ == '__main__':
    import argparse
    data = pd.read_csv("../notebooks/processing_area.csv")
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
        dir_i = f"../project-monitoring/tof/{x}/{y}"
        if os.path.exists(dir_i):
            files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
            for file in files:
                tifs.append(os.path.join(dir_i, file))


    print(f'There are {len(tifs)} / {len(data)} tiles processed')
    print(tifs)

    gdal.BuildVRT(f'{str(args.country)}.vrt', tifs)
    ds = gdal.Open(f'{str(args.country)}.vrt')
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW -a_nodata 255"))
    ds = gdal.Translate(f'{str(args.country)}.tif', ds, options=translateoptions)

    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
    file = f'{str(args.country)}.tif'
    key = f'2020/mosaics/' + file
    uploader.upload(bucket = args.s3_bucket, key = key, file = file)
import os
from osgeo import gdal
from glob import glob
import pandas as pd

if __name__ == '__main__':
    import argparse
    data = pd.read_csv("../notebooks/processing_area.csv")
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    args = parser.parse_args()
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
    translateoptions = gdal.TranslateOptions(gdal.ParseCommandLine("-ot Byte -co COMPRESS=LZW"))
    ds = gdal.Translate(f'{str(args.country)}.tif', ds, options=translateoptions)
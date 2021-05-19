from preprocessing.indices import evi, bi, msavi2, grndvi
from downloading.io import FileUploader

import numpy as np 
import os
import rasterio
from rasterio.transform import from_origin
from tqdm import tnrange, tqdm_notebook
from scipy.ndimage import median_filter
from skimage.transform import resize
import hickle as hkl
from time import sleep
import pandas as pd
import copy
import yaml
from glob import glob

from downloading import utils
from models import utils

def tile_images(arr: np.ndarray) -> list:
    """ Converts a 142x142 array to a 161, 24, 24 array
        
        Parameters:
         arr (np.ndaray): (142, 142) float array
    
        Returns:
         images (list): 
    """

    # Normal
    images = []
    for x_offset, cval in enumerate([x for x in range(0, 140, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            min_x = np.max([cval - 0, 0])
            max_x = np.min([cval + 24, 150])
            min_y = np.max([rval - 0, 0])
            max_y = np.min([rval + 24, 150])
            subs = arr[:, min_x:max_x, min_y:max_y]
            images.append(subs)
            
    # Upright  
    for x_offset, cval in enumerate([x for x in range(7,  140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            base_id = 9*9
            min_x = np.max([cval - 0, 0])
            max_x = np.min([cval + 24, 150])
            min_y = np.max([rval - 0, 0])
            max_y = np.min([rval + 24, 150])
            subs = arr[:, min_x:max_x, min_y:max_y]
            images.append(subs)

    return images


def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """ Generates a 2*expansion x 2*expansion bounding box of
        300 m ESA pixels
        
        Parameters:
         initial_bbx (list):
         expansion (int):
    
        Returns:
         bbx (list):
    """
    
    multiplier = 1/360
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx



def write_tif(arr: np.ndarray, point: list, x: int, y: int) -> str:
    #! TODO: Documentation
    
    file = out_folder[:-7] + f"{str(x)}X{str(y)}Y_POST.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]
    arr = arr.T.astype(np.uint8)

    transform = rasterio.transform.from_bounds(west = west, south = south,
                                               east = east, north = north,
                                               width = arr.shape[1], 
                                               height = arr.shape[0])

    print("Writing", file)
    new_dataset = rasterio.open(file, 'w', driver = 'GTiff',
                               height = arr.shape[0], width = arr.shape[1], count = 1,
                               dtype = "uint8",
                               crs = '+proj=longlat +datum=WGS84 +no_defs',
                               transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    return file

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tof/')
    parser.add_argument("--model_path", dest = 'model_path', default = '../models/master-13250/230k/')
    parser.add_argument("--db_path", dest = "db_path", default = "../notebooks/processing_area.csv")
    parser.add_argument("--ul_flag", dest = "ul_flag", default = False)
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--delete_processed", dest = "delete_processed", default = True)
    args = parser.parse_args()

    print(f'Country: {args.country} \n'
          f'Local path: {args.local_path} \n'
          f'Model path: {args.model_path} \n'
          f'DB path: {args.db_path} \n'
          f'UL Flag: {args.ul_flag} \n'
          f'S3 Bucket: {args.s3_bucket} \n'
          f'YAML path: {args.yaml_path} \n'
          f'Current dir: {os.getcwd()} \n'
          f'Delete processed: {args.delete_processed} \n')

    if os.path.exists(args.yaml_path):
        with open(args.yaml_path, 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']
        print(f"Loaded API key from {args.yaml_path}")
    else:
        raise Exception(f"No API key found at {args.yaml_path}")

    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
    data = pd.read_csv(args.db_path)

    SIZE = 10
    SIZE_N = SIZE*SIZE
    SIZE_UR = (SIZE - 1) * (SIZE - 1)

    data = data[data['country'] == args.country]
    data = data.reset_index(drop = True)
    print(f"There are {len(data)} tiles for {args.country}")

    tifs = []
    for index, row in data.iterrows():
        x = str(int(row['X_tile']))
        y = str(int(row['Y_tile']))

        #! TODO
        # For local path:
        #     - check to see if no .tif, but processed/ is not empty
        #     - if local processed/ exists, use it
        # If not local, check S3:
        #     - first check for tiles/x/y/.tif
        #           - YES, skip
        #           - NO, continue
        #     - if not exist, check for processed/x/y/.hkl
        #           - YES: download to args.local_path
        #           - NO: skip
        # If local processed/ exists:
        #     - Run predictions
        #     - delete local/processed
        #     - upload .tif -> tiles/x/y/.tif
        #     - delete s3://processed/x/y
        # Continue

        dir_i = f"{args.local_path}{x}/{y}/"
        if os.path.exists(dir_i):
            files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']


            if len(files) >= 1 and os.path.exists(dir_i + "output/"):
                try:
                    initial_bbx = [row['X'], row['Y'], row['X'], row['Y']]
                    point = make_bbox(initial_bbx, expansion = 300/30)

                    inp_folder = f'{args.local_path}{str(x)}/{str(y)}/processed/'
                    out_folder = f'{args.local_path}{str(x)}/{str(y)}/output/'
                    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
                    max_x = np.max(x_tiles) + 140

                    # This chunk of code actually runs the predictions
                    for x_tile in x_tiles:
                        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
                        max_y = np.max(y_tiles) + 140

                    # This chunk of code loads the predictions and mosaics them into a .tif for the tile
                    predictions = np.full((max_x, max_y), 0, dtype = np.uint8)
                    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
                    for x_tile in x_tiles:
                        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
                        for y_tile in y_tiles:
                            output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
                            if os.path.exists(output_file):
                                prediction = np.load(output_file)
                                prediction = prediction * 100
                                prediction = prediction.T.astype(np.uint8)
                                predictions_tile = predictions[ (x_tile ): (x_tile+140),
                                           y_tile:y_tile + 140]

                                if np.max(prediction) <= 100:
                                    existing_predictions = predictions_tile[np.logical_and(predictions_tile != 0, predictions_tile <= 100)] 
                                    current_predictions = prediction[np.logical_and(predictions_tile != 0, predictions_tile <= 100)]
                                    if current_predictions.shape[0] > 0:
                                        # Require colocation.. Here we can have a lower threshold as
                                        # the likelihood of false positives is much higher
                                        current_predictions[(current_predictions - existing_predictions) > 50] = np.min([current_predictions, existing_predictions])
                                        existing_predictions[(existing_predictions - current_predictions) > 50] = np.min([current_predictions, existing_predictions])
                                        existing_predictions = (current_predictions + existing_predictions) / 2
                         
                                    predictions_tile[predictions_tile == 0] = prediction[predictions_tile == 0]
                                else:
                                    predictions[ (x_tile ): (x_tile+140),
                                           y_tile:y_tile + 140] = prediction
                                
                    # This chunk of code removes some noisy areas
                    for x_i in range(0, predictions.shape[0] - 3):
                        for y_i in range(0, predictions.shape[1] - 3):
                            window = predictions[x_i:x_i+3, y_i:y_i+3]
                            if np.max(window) < 35:
                                if np.sum(np.logical_and(window > 10, window < 35)) > 5:
                                    predictions[x_i:x_i+3, y_i:y_i+3] = 0.

                    predictions[predictions <= .25*100] = 0.        
                    predictions = np.around(predictions / 20, 0) * 20
                    predictions[predictions > 100] = 255.
                    file = write_tif(predictions, point, x, y)
                    #key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_POST.tif'
                    #uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                    if args.delete_processed:
                        files_to_delete = glob(inp_folder + "/*/*")
                        print(f"Deleting files in {inp_folder}")
                        for file in files_to_delete:
                            os.remove(file)
                except:
                    print(f"Skipping {dir_i}")
                   

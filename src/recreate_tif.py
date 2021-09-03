import pandas as pd
import numpy as np
import copy
from glob import glob
import rasterio
import os
import yaml
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles,  download_folder
import glob
import boto3


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g


def load_mosaic_predictions(out_folder: str) -> np.ndarray:
    """
    Loads the .npy subtile files in an output folder and mosaics the overlapping predictions
    to return a single .npy file of tree cover for the 6x6 km tile
    Additionally, applies post-processing threshold rules and implements no-data flag of 255
    
        Parameters:
         out_folder (os.Path): location of the prediction .npy files 
    
        Returns:
         predictions (np.ndarray): 6 x 6 km tree cover data as a uint8 from 0-100 w/ 255 no-data flag
    """
    test = np.load(out_folder + "0/0.npy")
    SIZE = test.shape[0]
    print(SIZE)
    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
    max_x = np.max(x_tiles) + SIZE
    for x_tile in x_tiles:
        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        max_y = np.max(y_tiles) + SIZE
    predictions = np.full((max_x, max_y, len(x_tiles) * len(y_tiles)), np.nan, dtype = np.float32)
    mults = np.full((max_x, max_y, len(x_tiles) * len(y_tiles)), 0, dtype = np.float32)
    i = 0
    for x_tile in x_tiles:
        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        for y_tile in y_tiles:
            output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
            if os.path.exists(output_file):
                prediction = np.load(output_file)
                if np.sum(prediction) < SIZE*SIZE*255:
                    prediction = (prediction * 100).T.astype(np.float32)
                    predictions[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = prediction
                    fspecial_size = 35 if SIZE >= 160 else 20
                    mults[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, fspecial_size)
                i += 1

    predictions = predictions.astype(np.float32)
    predictions_range = np.nanmax(predictions, axis=-1) - np.nanmin(predictions, axis=-1)
    mean_certain_pred = np.nanmean(predictions[predictions_range < 50])
    mean_uncertain_pred = np.nanmean(predictions[predictions_range > 50])
    
    overpredict = True if (mean_uncertain_pred - mean_certain_pred) > 0 else False
    underpredict = True if not overpredict else False
    
    if SIZE == 168:
        for i in range(predictions.shape[-1]):
            if overpredict:
                problem_tile = True if np.nanmean(predictions[..., i]) > mean_certain_pred else False
            if underpredict:
                problem_tile = True if np.nanmean(predictions[..., i]) < mean_certain_pred else False
            range_i = np.copy(predictions_range)
            range_i[np.isnan(predictions[..., i])] = np.nan
            range_i = range_i[~np.isnan(range_i)]
            if range_i.shape[0] > 0:
                range_i = np.reshape(range_i, (168 // 56, 56, 168 // 56, 56))
                range_i = np.mean(range_i, axis = (1, 3))
                n_outliers = np.sum(range_i > 50)
                if n_outliers >= 2 and problem_tile:
                    predictions[..., i] = np.nan
                    mults[..., i] = 0.
    
    mults = mults / np.sum(mults, axis = -1)[..., np.newaxis]
    predictions[predictions > 100] = np.nan
    out = np.copy(predictions)
    out = np.sum(np.isnan(out), axis = (2))
    n_preds = predictions.shape[-1]

    predictions = np.nansum(predictions * mults, axis = -1)
    predictions[out == n_preds] = np.nan
    predictions[np.isnan(predictions)] = 255.
    predictions = predictions.astype(np.uint8)
    
    original_preds = np.copy(predictions)
    for x_i in range(0, predictions.shape[0] - 3):
        for y_i in range(0, predictions.shape[1] - 3):
            window = original_preds[x_i:x_i+3, y_i:y_i+3]
            if np.max(window) < 35:
                sum_under_35 = np.sum(np.logical_and(window > 10, window < 35))
                if np.logical_and(sum_under_35 > 6, sum_under_35 < 10):
                    window = 0.

            # This removes or mitigates some of the "noisiness" of individual trees
            # Which could have odd shapes depending on where they sit within or between
            # Sentinel pixels 
            if np.max(window) >= 25 and np.argmax(window) == 4:
                window_binary = window >= 25
                if np.sum(window_binary) < 4:
                    if np.sum(window_binary[1]) < 3 and np.sum(window_binary[:, 1]) < 3:
                        window[0, :] = 0
                        window[2, :] = 0
                        window[:, 0] = 0
                        window[:, 2] = 0
                    
    predictions = original_preds 
    predictions[predictions <= .20*100] = 0.        
    predictions[predictions > 100] = 255.

    return predictions



def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Makes a (min_x, min_y, max_x, max_y) bounding box that
       is 2 * expansion 300 x 300 meter ESA LULC pixels

       Parameters:
            initial_bbx (list): [min_x, min_y, max_x, max_y]
            expansion (int): 1/2 number of 300m pixels to expand

       Returns:
            bbx (list): expanded [min_x, min_y, max_x, max_y]
    """
    multiplier = 1/360 # Sentinel-2 pixel size in decimal degrees
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx


def FINAL_in_s3(file, key, apikey, apisecret, bucket):
    """
    Checks to see if a file/key pair exists locally or on s3 or neither
    """

    exists = False
    s3 = boto3.resource('s3', aws_access_key_id=apikey,
         aws_secret_access_key= apisecret)
    bucket = s3.Bucket(bucket)
    objs = list(bucket.objects.filter(Prefix=key))
    
    if len(objs) > 0:
        print(f"The s3 resource s3://{bucket}/{key} exists")
        exists = True

    if not exists:
        print(f"The s3 resource s3://{bucket}/{key} does not exist")

    return exists


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--db_path", dest = "db_path", default = "processing_area_june_28.csv")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--reprocess", dest = "reprocess", default = False)
    args = parser.parse_args()


    data = pd.read_csv(args.db_path)
    data = data[data['country'] == args.country]
    data = data.reset_index(drop = True)


    yaml_path = args.yaml_path
    with open(yaml_path, 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY= key['key']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']
    print(f"Successfully loaded key from {yaml_path}")
    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET, overwrite = True)


import time
for index, row in data.iterrows():
    x = str(int(row['X_tile']))
    y = str(int(row['Y_tile']))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    path_to_process = f"{args.local_path}{str(x)}/{str(y)}/"
    s3_path_to_process = f'2020/processed/{str(x)}/{str(y)}/'
    path_to_tile = f'{args.local_path}/{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/tiles/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'

    exists = FINAL_in_s3(path_to_tile,
                s3_path_to_tile, 
                AWSKEY, AWSSECRET, 
                'tof-output')
    
    if not exists:
        time1 = time.time()
        print(f"Fixing {x}, {y}")
        download_folder(bucket = "tof-output",
                           apikey = AWSKEY,
                           apisecret = AWSSECRET,
                           local_dir = path_to_process,
                           s3_folder = s3_path_to_process)
        if os.path.exists(f"{args.local_path}{x}/{y}/processed/"):
            try:
                smooth = load_mosaic_predictions(f"{args.local_path}/{x}/{y}/processed/")
                smooth = smooth.astype(np.uint8)

                data_tile = data.copy()
                data_tile = data_tile[data_tile['Y_tile'] == int(y)]
                data_tile = data_tile[data_tile['X_tile'] == int(x)]
                data_tile = data_tile.reset_index(drop = True)
                x = str(int(x))
                y = str(int(y))
                x = x[:-2] if ".0" in x else x
                y = y[:-2] if ".0" in y else y

                initial_bbx = [data_tile['X'][0], data_tile['Y'][0], data_tile['X'][0], data_tile['Y'][0]]
                bbx = make_bbox(initial_bbx, expansion = 300/30)

                file = write_tif(smooth, bbx, x, y, f"{args.local_path}/{x}/{y}/")
                key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_FINAL.tif'
                uploader.upload(bucket = 'tof-output', key = key, file = file)
                for folder in glob.glob(path_to_process + "processed/*/"):
                    for file in os.listdir(folder):
                        _file = folder + file
                        os.remove(_file)
                time2 = time.time()
                print(f"Finished in {time2 - time1} seconds")
            except:
                print(f"Ran into an error for {x}/{y}")
        else:
            print(f"The {x}/{y}/processed/ does not exist, skipping")

import pandas as pd
import numpy as np
from random import shuffle
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
from sentinelhub.config import SHConfig
import logging
import datetime
import os
import yaml
from tqdm import tqdm
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from skimage.transform import resize
from sentinelhub import CustomUrlParam
import math
import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc
import hickle as hkl
import boto3
from typing import Tuple, List
import warnings
from scipy import ndimage
from scipy.ndimage import median_filter, maximum_filter, percentile_filter
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
import time
import copy
from glob import glob
import rasterio
from rasterio.transform import from_origin
import shutil
import bottleneck as bn
import tensorflow as tf
import traceback
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)


from preprocessing import slope
from preprocessing import indices
from downloading.utils import tile_window, calculate_and_save_best_images, calculate_proximal_steps
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from tof import tof_downloading
from tof.tof_downloading import to_int16, to_float32
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3, download_ard_file
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder, download_file, download_single_file
from preprocessing.indices import evi, bi, msavi2, grndvi
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import label, grey_closing

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
np.seterr(invalid='ignore')
np.seterr(divide='ignore')

## GLOBAL VARIABLES ##
SIZE = 172-14
LEN = 4
WRITE_TEMP_TIFS = False
WRITE_RAW_TIFS = False
WRITE_MONTHLY_TIFS = False

""" This is the main python script used to generate the tree cover data.
The useage is as:
python3.x download_and_predict_job.py --db_path $PATH --country $COUNTRY --ul_flag $FLAG
"""
#####################################################
######### SINGLE-PURPOSE ONE I/O FUNCTIONS ##########
#####################################################

def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ Converts Sentinel 1 unitless backscatter coefficient
        to db with a min_db lower threshold
        
        Parameters:
         x (np.ndarray): unitless backscatter (T, X, Y, B) array
         min_db (int): integer from -50 to 0
    
        Returns:
         x (np.ndarray): db backscatter (T, X, Y, B) array
    """
    
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = (x + min_db) / min_db
    return np.clip(x, 0, 1)

def ndmi(arr):
    return (arr[..., 3] - arr[..., 8]) / (arr[..., 3] + arr[..., 8])


def superresolve_large_tile(arr: np.ndarray, sess) -> np.ndarray:
    """Superresolves an input tile utilizing the open tf.sess().
       Implements a lightweight version of DSen2, a CNN-based
       image superresolution model

       Reference: https://arxiv.org/abs/1803.04271

       Parameters:
            arr (arr): (?, X, Y, 10) array, where arr[..., 4:]
                       has been bilinearly upsampled

       Returns:
            superresolved (arr): (?, X, Y, 10) array
    """
    # Pad the input images to avoid border artifacts
    def _worker_fn(arr: np.ndarray, sess) -> np.ndarray:
        # Pad the input images to avoid border artifacts
        to_resolve = np.pad(arr, ((0, 0), (4, 4), (4, 4), (0, 0)), 'reflect')

        bilinear = to_resolve[..., 4:]
        resolved = sess.run([superresolve_logits], 
                     feed_dict={superresolve_inp: to_resolve,
                                superresolve_inp_bilinear: bilinear})[0]
        resolved = resolved[:, 4:-4, 4:-4, :]
        arr[..., 4:] = resolved
        return arr

    wsize = 110
    step = 110
    x_range = [x for x in range(0, arr.shape[1] - (wsize), step)] + [arr.shape[1] - wsize]
    y_range = [x for x in range(0, arr.shape[2] - (wsize), step)] + [arr.shape[2] - wsize]
    x_end = np.copy(arr[:, x_range[-1]:, ...])
    y_end = np.copy(arr[:, :, y_range[-1]:, ...])

    time1 = time.time()
    print("Starting image superresolution from 20m to 10m")
    for x in tqdm(x_range):
        for y in y_range:
            if x != x_range[-1] and y != y_range[-1]:
                to_resolve = arr[:, x:x+wsize, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)
            # The end x and y subtiles need to be done separately
            # So that a partially resolved tile isnt served as input
            elif x == x_range[-1]:
                to_resolve = x_end[:, :, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)
            elif y != y_range[-1]:
                to_resolve = y_end[:, x:x+wsize, :, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = _worker_fn(to_resolve, sess)

    time2 = time.time()
    print(f"Superresolution: {np.around(time2 - time1, 1)} seconds")
    return arr


#####################################################
##### GEOSPATIAL REFERENCING AND FILE I/O TOOLS #####
#####################################################

def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    """Makes a (min_x, min_y, max_x, max_y) bounding box that
       is 2 * expansion 300 x 300 meter ESA LULC pixels
       e.g. expansion = 10 generates a 6 x 6 km tile
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


def float_to_int16(arr, precision = 1000):
    arr[np.isnan(arr)] = -32768
    arr = np.clip(arr, 
        (-32768 / precision), (32767 / precision))
    arr = arr * precision
    arr = np.int16(arr)
    return arr


def write_train_to_tif(arr: np.ndarray,
              point: list,
              name,
              out_folder: str,
              suffix="_FINAL") -> str:

    file = out_folder + f"{name}.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]

    transform = rasterio.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])
    arr = to_int16(arr)
    
    new_dataset = rasterio.open(file,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=arr.shape[-1],
                                dtype="uint16",
                                compress='zstd',
                                predictor=2,
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
    arr = np.rollaxis(arr, 2)
    arr = np.flip(arr, axis = 0)
    new_dataset.write(arr)
    new_dataset.close()
    return file


def write_ard_to_tif(arr: np.ndarray,
              point: list,
              name,
              out_folder: str,
              suffix="_FINAL") -> str:

    file = out_folder + f"{name}.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]

    transform = rasterio.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])
    arr[arr > 0.255] = 0.255
    arr = arr / 0.255
    arr = arr * 255
    arr = np.uint8(arr)
    
    new_dataset = rasterio.open(file,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=arr.shape[-1],
                                dtype="uint8",
                                compress='zstd',
                                predictor=2,
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
    arr = np.rollaxis(arr, 2)
    arr = np.flip(arr, axis = 0)
    new_dataset.write(arr)
    new_dataset.close()
    return file


def adjust_shape(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Assures that the shape of arr is width x height
    Used to align 10, 20
    0, 160, 640 meter resolution Sentinel data
    """
    #print(f"Input array shape: {arr.shape}")
    arr = arr[:, :, :, np.newaxis] if len(arr.shape) == 3 else arr
    arr = arr[np.newaxis, :, :, np.newaxis] if len(arr.shape) == 2 else arr
    
    if arr.shape[1] < width:
        pad_amt = (width - arr.shape[1]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (1, pad_amt), (0,0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (pad_amt, pad_amt), (0,0), (0, 0)), 'edge')

    if arr.shape[2] < height:
        pad_amt = (height - arr.shape[2]) // 2
        if pad_amt == 0:
            arr = np.pad(arr, ((0, 0), (0,0), (1, 0), (0, 0)), 'edge')
        else:
            arr = np.pad(arr, ((0, 0), (0,0), (pad_amt, pad_amt), (0, 0)), 'edge')

    if arr.shape[1] > width:
        pad_amt =  (arr.shape[1] - width) // 2
        pad_amt_even = (arr.shape[1] - width) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, 1:, ...]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, pad_left:-pad_right, ...]

    if arr.shape[2] > height:
        pad_amt = (arr.shape[2] - height) // 2
        pad_amt_even = (arr.shape[2] - height) % 2 == 0
        if pad_amt == 0:
            arr = arr[:, :, 1:, :]
        elif pad_amt_even:
            pad_amt = int(pad_amt)
            arr = arr[:, :, pad_amt:-pad_amt, ...]
        else:
            pad_left = int(np.floor(pad_amt / 2))
            pad_right = int(np.ceil(pad_amt / 2))
            arr = arr[:, :, pad_left:-pad_right, ...]

    return arr.squeeze()

#####################################################
############### MODEL PREDICTION FNS ################
#####################################################

def normalize_subtile(subtile):
    for band in range(0, subtile.shape[-1]):
        mins = min_all[band]
        maxs = max_all[band]
        subtile[..., band] = np.clip(subtile[..., band], mins, maxs)
        midrange = (maxs + mins) / 2
        rng = maxs - mins
        standardized = (subtile[..., band] - midrange) / (rng / 2)
        subtile[..., band] = standardized
    return subtile
 

def predict_subtile(subtile: np.ndarray, sess: "tf.Sess", op: "tf.Tensor", size: "int") -> np.ndarray:
    """ Runs temporal (convGRU + UNET) predictions on a (12, 216, 216, 17) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
         sess (tf.Session): tensorflow session for prediction
    
        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    """
    #np.save('subtile.npy', subtile)
    if np.sum(subtile) != 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.

        temporal_model = True
        if not temporal_model:
            # If not temporal model, move time to channel axis
            subtile = np.moveaxis(subtile, 0, 2)
            subtile = subtile.reshape((SIZE+14, SIZE+14, 17*5))

        batch_x = subtile[np.newaxis].astype(np.float32)
        lengths = np.full((batch_x.shape[0]), args.length)
        preds = sess.run(op,
                              feed_dict={predict_inp:batch_x, 
                                         predict_length:lengths})

        preds = preds.squeeze()
        clip = (preds.shape[0] - size) // 2
        if clip > 0:
            preds = preds[clip:-clip, clip:-clip]
        preds = np.float32(preds)

    else:
        preds = np.full((SIZE, SIZE), 255)
        print(f"The sum of the subtile is {np.sum(subtile)}")
    
    return preds

#####################################################
############### RAW DATA DOWNLOAD FNS ###############
#####################################################\

def download_raw_tile(tile_idx: tuple, local_path: str,
                      subfolder: str = "raw") -> None:
    # Download pre-downloaded raw data from s3
    x = tile_idx[0]
    y = tile_idx[1]

    path_to_tile = f'{local_path}{str(x)}/{str(y)}/'
    s3_path_to_tile = f'{str(args.year)}/{subfolder}/{str(x)}/{str(y)}/'
    if subfolder == "tiles":
        folder_to_check = len(glob(path_to_tile + "*.tif")) > 0
    if subfolder == "processed":
        folder_to_check = os.path.exists(path_to_tile + subfolder + "/0/")
    if subfolder == "raw":
        folder_to_check = os.path.exists(path_to_tile + subfolder + "/clouds/")
    if not folder_to_check:
        print(f"Downloading {s3_path_to_tile}")
        download_folder(bucket = "tof-output",
                       apikey = AWSKEY,
                       apisecret = AWSSECRET,
                       local_dir = path_to_tile,
                       s3_folder = s3_path_to_tile)
    return None


def download_s1_tile(data: np.ndarray, bbx: list, api_key, year: int,
                     dates_sentinel_1: list, size: tuple, 
                     s1_file: str, s1_dates_file: str) -> None:
    """Downloads the sentinel 1 data for a tile using Sentinel-Hub
       
       Parameters:
            data (pd.DataFrame): dataframe of tile to be downloaded
            bbx (list): bounding box for tile
            api_key (sentinelhub.shconfig): API keys for sentinel hub
            year (int): year of imagery
            dates_sentinel_1 (list):
            size (tuple): size of array to make
            s1_file (str): path to save s1 data
            s1_dates_file (str): path to save s1 dates data

       Returns:
            None

    """
    print(f"Downloading {s1_file}")
    # Identifies if s1 needs to be Ascending or Descending
    s1_layer = tof_downloading.identify_s1_layer((data['Y'][0], data['X'][0]))

    s1, s1_dates = np.empty((0,)), np.empty((0,))
    # Some years may be entirely missing from the ESA Archive?
    # In <0.1% of cases. This just backfills with prior images
    for year in [year, year - 1,  year - 2, year - 3, year - 4, year + 1, year + 2]:
        if s1.shape[0] == 0:
            s1, s1_dates = tof_downloading.download_sentinel_1_composite(bbx,
                                           layer = s1_layer,
                                           api_key = api_key,
                                           year = year,
                                           dates = dates_sentinel_1,
                                           size = size,
                                           )

        if s1.shape[0] == 0: # If the first attempt receives no images, swap orbit
            if s1_layer == "SENT":
                s1_layer = "SENT_DESC"
            elif s1_layer == "SENT_DESC":
                s1_layer = "SENT"
            print(f'Switching to {s1_layer}')
            s1, s1_dates = tof_downloading.download_sentinel_1_composite(bbx,
                                               layer = s1_layer,
                                               api_key = api_key,
                                               year = year,
                                               dates = dates_sentinel_1,
                                               size = size,
                                               )


        if s1.shape[0] == 0: # If the second attempt receives no images, swap orbit
            s1_layer = "SENT_ALL"
            print(f'Switching to {s1_layer}')
            s1, s1_dates = tof_downloading.download_sentinel_1_composite(bbx,
                                               layer = s1_layer,
                                               api_key = api_key,
                                               year = year,
                                               dates = dates_sentinel_1,
                                               size = size,
                                               )
    # Convert s1 to monthly mosaics, and write to disk
    s1 = tof_downloading.process_sentinel_1_tile(s1, s1_dates)
    hkl.dump(to_int16(s1), s1_file, mode='w', compression='gzip')
    hkl.dump(s1_dates, s1_dates_file, mode='w', compression='gzip')


def download_tile(x: int, y: int, data: pd.DataFrame, api_key, year, initial_bbx, expansion) -> None:
    """Downloads the data for an input x, y tile centroid
       including:
        - Clouds
        - Sentinel 1
        - Sentinel 2 (10 and 20 m)
        - DEM

       Writes the raw data to the output/x/y folder as .hkl structure

       Parameters:
            x (int): x position of tile to be downloaded
            y (int): y position of tile to be downloaded
            data (pd.DataFrame): tile grid dataframe

       Returns:
            None

    """
    #data = data[data['Y_tile'] == int(y)]
    #data = data[data['X_tile'] == int(x)]
    #data = data.reset_index(drop = True)
    #x = str(int(x))
    #y = str(int(y))
    #x = x[:-2] if ".0" in x else x
   #y = y[:-2] if ".0" in y else y
        
    #initial_bbx = [data['X'][0], data['Y'][0], data['X'][0], data['Y'][0]]

    # The cloud bounding box is much larger, to ensure that the same image dates
    # Are selected in neighboring tiles
    cloud_bbx = make_bbox(initial_bbx, expansion = (expansion * 15)/30)
    bbx = make_bbox(initial_bbx, expansion = expansion/30)
    dem_bbx = make_bbox(initial_bbx, expansion = (expansion + 1)/30)
    print(f"The tile bbx is {bbx}")
        
    folder = f"{args.local_path}/{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'
    
    make_output_and_temp_folders(folder)

    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl' # deprecated?
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'

    if not (os.path.exists(clouds_file)):
        print(f"Downloading {clouds_file}")

        # Identify images with <30% cloud cover
        cloud_probs, cloud_percent, all_dates, all_local_clouds = tof_downloading.identify_clouds_big_bbx(
            cloud_bbx = cloud_bbx, 
            shadow_bbx = bbx, # deprecated
            dates = dates,
            api_key = api_key, 
            year = year
        )

        cloud_probs = cloud_probs * 255
        cloud_probs[cloud_probs > 100] = np.nan
        cloud_percent = np.nanmean(cloud_probs, axis = (1, 2))
        cloud_percent = cloud_percent / 100

        local_clouds = np.copy(all_local_clouds)
        image_dates = np.copy(all_dates)
        # This function selects the images used for processing
        # Based on cloud cover and frequency of cloud-free images
        #!RENAME subset_contiguous_sunny_dates -> image_selection
        to_remove = cloud_removal.subset_contiguous_sunny_dates(image_dates,
                                                               cloud_percent)
        if len(to_remove) > 0:
            clean_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            cloud_percent = np.delete(cloud_percent, to_remove)
            local_clouds = np.delete(local_clouds, to_remove)
        else:
            clean_dates = image_dates
        if len(clean_dates) >= 11:
            clean_dates = np.delete(clean_dates, 5)
            cloud_probs = np.delete(cloud_probs, 5, 0)
            cloud_percent = np.delete(cloud_percent, 5)
            local_clouds = np.delete(local_clouds, to_remove)

        _ = cloud_removal.print_dates(
            clean_dates, cloud_percent
        )

        # Expand the image selection if multiple of the local cloud
        # prob is above a threshold
        lowest_three_local = np.argpartition(all_local_clouds, 3)[:3]
        lowest_four_local = np.argpartition(all_local_clouds, 4)[:4]
        #lowest_ten_local = np.argpartition(all_local_clouds, 10)[:10]

        # If less than 3 images selected have <40% local cloud cover,
        # Then expand the image search
        criteria1 = (np.sum((local_clouds <= 0.3)) < 3)
        criteria2 = (np.sum((local_clouds <= 0.4)) < 4)
        criteria3 = len(local_clouds) <= 8
        criteria2 = np.logical_or(criteria2, criteria3)
        if criteria1 or criteria2:
            if len(clean_dates) <= 9:
                lowest = lowest_four_local if criteria2 else lowest_three_local
                lowest_dates = image_dates[lowest]
                existing_imgs_in_local = [x for x in clean_dates if x in image_dates[lowest]]
                images_to_add = [x for x in lowest_dates if x not in clean_dates]
                print(f"Adding these images: {images_to_add}")
                clean_dates = np.concatenate([clean_dates, images_to_add])
                clean_dates = np.sort(clean_dates)
        if len(clean_dates) <= 9:
            imgs_to_add = 9 - len(clean_dates)
            lowest_five_local = np.argpartition(all_local_clouds, 5)[:5]
            images_to_add = [x for x in lowest_five_local if x not in clean_dates][:imgs_to_add]
            clean_dates = np.concatenate([clean_dates, images_to_add])
            clean_dates = np.sort(clean_dates)

        for i, x, y in zip(clean_dates, cloud_percent, local_clouds):
            print(i, x, y)

        print(f"Downloading {len(clean_dates)} of {len(clean_dates)+len(to_remove)} total steps")
        hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
        hkl.dump(clean_dates, clean_steps_file, mode='w', compression='gzip')
    else:
        clean_dates =  np.arange(0, 9)
            
    if not (os.path.exists(s2_10_file)) and len(clean_dates) > 2:
        print(f"Downloading {s2_10_file}")
        clean_steps = list(hkl.load(clean_steps_file))
        cloud_probs = hkl.load(clouds_file)
        s2_10, s2_20, s2_dates, clm = tof_downloading.download_sentinel_2_new(bbx,
                                                     clean_steps = clean_steps,
                                                     api_key = api_key, dates = dates,
                                                     year = year, maxclouds = 1.0)

        # Ensure that L2A, L1C derived products have exact matching dates
        # As sometimes the L1C data has more dates than L2A if processing bug from provider
        to_remove_clouds = [i for i, val in enumerate(clean_steps) if val not in s2_dates]
        to_remove_dates = [val for i, val in enumerate(clean_steps) if val not in s2_dates]

        # Save all the files to disk (temporarily)
        hkl.dump(to_int16(s2_10), s2_10_file, mode='w', compression='gzip')
        hkl.dump(to_int16(s2_20), s2_20_file, mode='w', compression='gzip')
        hkl.dump(s2_dates, s2_dates_file, mode='w', compression='gzip')
        hkl.dump(clm, cloud_mask_file, mode='w', compression='gzip')
        # We need to know the size to ensure that Sentinel-1 is the same size as
        # Sentinel-2
        size = s2_20.shape[1:3]
            
    if not (os.path.exists(s1_file)) and len(clean_dates) > 2:
        download_s1_tile(data = data, 
                         bbx = bbx,
                         api_key = api_key,
                         year = year, 
                         dates_sentinel_1 = dates_sentinel_1, 
                         size = size, 
                         s1_file = s1_file, 
                         s1_dates_file = s1_dates_file)

    if not os.path.exists(dem_file) and len(clean_dates) > 2:
        print(f'Downloading DEM: {dem_file}')
        dem = tof_downloading.download_dem(dem_bbx, api_key = api_key)
        hkl.dump(dem, dem_file, mode='w', compression='gzip')

    return bbx, len(clean_dates)

#####################################################
################# ARD CREATION FNS ##################
#####################################################

def process_tile(x: int, y: int, data: pd.DataFrame, 
                 local_path: str, bbx, make_shadow: bool = False) -> np.ndarray:
    """
    Processes raw data structure (in temp/raw/*) to processed data structure
        - align shapes of different data sources (clouds / shadows / s1 / s2 / dem)
        - superresolve 20m to 10m with bilinear upsampling for DSen2 input
        - remove (interpolate) clouds and shadows

    Parameters:
         x (int): x position of tile to be downloaded
         y (int): y position of tile to be downloaded
         data (pd.DataFrame): tile grid dataframe

        Returns:
         x (np.ndarray)
         image_dates (np.ndarray)
         interp (np.ndarray)
         s1 (np.ndarray)
    """
    
    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
            
    folder = f"{local_path}{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'
    
    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    cloud_mask_file = f'{folder}raw/clouds/cloudmask_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'
    
    clouds = hkl.load(clouds_file)
    if os.path.exists(cloud_mask_file):
        # These are the S2Cloudless / Sen2Cor masks
        clm = hkl.load(cloud_mask_file).repeat(2, axis = 1).repeat(2, axis = 2)
        for i in range(0, clm.shape[0]):
            mins = np.maximum(i - 1, 0)
            maxs = np.minimum(i + 1, clm.shape[0])
            # if 2 in a row are clouds, remove
            # since the Sen2Cor mask has high FP
            sums = np.sum(clm[mins:maxs], axis = 0) == 2
            clm[mins:maxs, sums] = 0.
        print("The Sen2Cor cloud percentages are: ", np.mean(clm, axis = (1, 2)))
    else:
        clm = None

    s1 = hkl.load(s1_file)
    s1 = np.float32(s1) / 65535
    for i in range(s1.shape[0]):
        s1_i = s1[i]
        s1_i[s1_i == 1] = np.median(s1_i[s1_i < 65535], axis = 0)
        s1[i] = s1_i

    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    s1 = s1.astype(np.float32)

    s2_10 = to_float32(hkl.load(s2_10_file))        
    s2_20 = to_float32(hkl.load(s2_20_file))
    dem = hkl.load(dem_file)
    dem = median_filter(dem, size =5)
    image_dates = hkl.load(s2_dates_file)
    
    # Ensure arrays are the same dims
    width = s2_20.shape[1] * 2
    height = s2_20.shape[2] * 2
    s1 = adjust_shape(s1, width, height)
    s2_10 = adjust_shape(s2_10, width, height)
    dem = adjust_shape(dem, width, height)

    print(f'### Array shapes ### \nClouds: {clouds.shape}, \n'
          f'S1: {s1.shape} \n'
          f'S2: {s2_10.shape}, {s2_20.shape} \n'
          f'DEM: {dem.shape}')

    # Deal with cases w/ only 1 image
    if len(s2_10.shape) == 3:
        s2_10 = s2_10[np.newaxis]
    if len(s2_20.shape) == 3:
        s2_20 = s2_20[np.newaxis]

    # bilinearly upsample 20m bands to 10m for superresolution
    sentinel2 = np.zeros((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10

    # a foor loop is faster than trying to vectorize it here! 
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(
                s2_20[step,..., band], (width, height), 1
            )
    for band in range(4, 6):
        # indices 4, 5 are 40m and may be a different shape
        # this code is ugly, but it forces the arrays to match up w/ the 10/20m ones
        for step in range(sentinel2.shape[0]):
            mid = s2_20[step,..., band]
            #if args.make_training_data == True:
            #    print('wtf')
                #mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                #mid = np.mean(mid, axis = (1, 3))
                #sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            if (mid.shape[0] % 2 == 0) and (mid.shape[1] % 2) == 0:
                # So bands 4, 5 need to be bilinearly upsampled for the input to
                # The super-resolution
                mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            elif mid.shape[0] %2 != 0 and mid.shape[1] %2 != 0:
                mid_misaligned_x = mid[0, :]
                mid_misaligned_y = mid[:, 0]
                mid = mid[1:, 1:].reshape(
                    np.int32(np.floor(mid.shape[0] / 2)), 2,
                    np.int32(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, 1:, band + 4] = resize(mid, (width - 1, height - 1), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned_x.repeat(2)
                sentinel2[step, :, 0, band + 4] = mid_misaligned_y.repeat(2)
            elif mid.shape[0] % 2 != 0:
                mid_misaligned = mid[0, :]
                mid = mid[1:].reshape(np.int32(np.floor(mid.shape[0] / 2)), 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, :, band + 4] = resize(mid, (width - 1, height), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned.repeat(2)
            elif mid.shape[1] % 2 != 0:
                mid_misaligned = mid[:, 0]
                mid = mid[:, 1:]
                mid = mid.reshape(mid.shape[0] // 2, 2, np.int32(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, :, 1:, band + 4] = resize(mid, (width, height - 1), 1)
                sentinel2[step, :, 0, band + 4] = mid_misaligned.repeat(2)
    print(f"SENTINEL2, {sentinel2.shape}")
    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)
    if len(missing_px) > 0:
        print(f"Removing {missing_px} dates due to {missing_px} missing data")
        if clouds.shape[0] == len(image_dates):
            clouds = np.delete(clouds, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)
        if clm is not None:
            clm = np.delete(clm, missing_px, axis = 0)

    # Remove images that are >10% snow...
    def ndsi(arr):
        return (arr[..., 1] - arr[..., 8]) / (arr[..., 1] + arr[..., 8])

    def snow_filter(arr):
        ndsi =  (arr[..., 1] - arr[..., 8]) / (arr[..., 1] + arr[..., 8])
        ndsi[ndsi < 0.10] = 0.
        ndsi[ndsi > 0.42] = 0.42
        snow_prob = (ndsi - 0.1) / 0.32

        # NIR band threshold
        snow_prob[arr[..., 3] < 0.10] = 0.
        snow_prob[np.logical_and(arr[..., 3] > 0.35, snow_prob > 0)] = 1.

        # blue band threshold
        snow_prob[arr[..., 0] < 0.10] = 0.
        snow_prob[np.logical_and(arr[..., 0] > 0.22, snow_prob > 0)] = 1.

        # B2/B4 thrershold
        b2b4ratio = arr[..., 0] / arr[..., 2]
        snow_prob[b2b4ratio < 0.75] = 0.
        return snow_prob > 0

    ndsis = snow_filter(sentinel2)
    mean_snow_per_img = np.mean(ndsis, axis = (1, 2))
    snow = np.mean(ndsis, axis = 0)
    snow = 1 - binary_dilation(snow < 0.7, iterations = 2)
    to_remove = np.argwhere(mean_snow_per_img > 0.25).flatten()
    # CURRENTLY DEFUNCT ## 
    if (len(to_remove) > 10):# and args.snow:
        print(f"Removing {to_remove} dates due to {to_remove} snow cover")
        if clouds.shape[0] == len(image_dates):
            clouds = np.delete(clouds, to_remove, axis = 0)
        image_dates = np.delete(image_dates, to_remove)
        sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
        if clm is not None:
            clm = np.delete(clm, to_remove, axis = 0)
    
    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)
    if make_shadow:
        time1 = time.time()
        # Bounding box passed to identify_cloud_shadows to mask 
        # out non-urban areas from the false positive cloud removal
        #! TODO: https://www.sciencedirect.com/science/article/pii/S0034425718302037
        cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
        if clm is not None:
            try:
                clm[fcps] = 0.
                cloudshad = np.maximum(cloudshad, clm)
            except:
                print("Error, continuing")

        interp = cloud_removal.id_areas_to_interp(
            sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
        )
        print(f"IMAGE DATES: {image_dates}")
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        print(f"Interpolate per-image: {np.around(interp_pct, 1)}")
        # In order to properly normalize band values to gapfill cloudy areas
        # We need 10% of the image to be non-cloudy
        # So that we can identify PIFs with at least 1000 px
        # Images deleted here will get propogated in the resegmentation
        # So it should not cause boundary artifacts in the final product.
        
        water_mask = _water_ndwi(np.median(sentinel2, axis=0)) > 0.0
        means = np.mean(interp == 1, axis = (1, 2))
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten()
        if len(to_remove) > 0:
            print(f"Deleting {to_remove}")
            if clouds.shape[0] == len(image_dates):
                clouds = np.delete(clouds, to_remove, axis = 0)
            #print(clouds.shape, image_dates.shape, sentinel2.shape, interp.shape)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten()
        if len(to_remove) > 0:
            print(f"Deleting {to_remove}")
            print(clouds.shape, image_dates.shape, sentinel2.shape, interp.shape)
            if clouds.shape[0] == len(image_dates):
                clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        to_remove = np.argwhere(np.mean(interp > 0, axis = (1, 2)) > 0.9).flatten()
        if len(to_remove) > 0:
            print(f"Deleting {to_remove}")
            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")
        interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        np.save('interp.npy', interp)

        def _ndwi(arr):
            return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])

        water_mask = np.median(_ndwi(sentinel2), axis=0)
        water_mask = water_mask > 0
        def _ndbi(arr):
            return ((arr[..., 8] - arr[..., 3]) / (arr[..., 8] + arr[..., 3]))
        ndbi = np.median(_ndbi(sentinel2), axis = 0)

        if WRITE_RAW_TIFS:
            for i in range(sentinel2.shape[0]):
                write_ard_to_tif(sentinel2[i, ..., :3], bbx,
                                f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_{str(i)}_RAW", "")
        _, interp, to_remove = cloud_removal.remove_cloud_and_shadows(
                sentinel2, cloudshad, cloudshad, image_dates,
                 pfcps = fcps, 
                 sentinel1 = #np.mean(s1, axis = 0),
                 np.concatenate([np.mean(s1, axis = 0),
                  dem[..., np.newaxis],
                  water_mask[..., np.newaxis],
                  ndbi[..., np.newaxis]], axis = -1),
                mosaic = None,
            )
        #write_ard_to_tif(mosaic[..., :3], bbx,
        #                        f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_MOSAIC", "")
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        print(f"Interpolate per-image: {np.around(interp_pct, 1)}")

        # If the brightness increases by a lot, and the std decreases by a lot
        # Then we're calling that haze.
        mean_brightness_per_img = np.mean(sentinel2[..., :3], axis = -1)
        mean_brightness = np.mean(mean_brightness_per_img, axis = (1, 2))
        std_brightness = np.std(mean_brightness_per_img, axis = (1, 2))
        print("B", mean_brightness, std_brightness)
        is_haze = np.diff(mean_brightness) > (np.mean(mean_brightness) * 0.5)
        is_haze = is_haze * np.diff(std_brightness) < (np.mean(std_brightness) * -0.5)
        is_haze = np.argwhere(is_haze > 0)
        #if len(is_haze) > 0:
       #     is_haze = is_haze + 1
        #    is_haze = is_haze.flatten()
        #    to_remove = to_remove + list(is_haze)
        print(f"HAZE FLAG: {is_haze}")
        if len(to_remove) > 0:
            print(f"Deleting {to_remove}")
            clouds = np.delete(clouds, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)
            interp = np.delete(interp, to_remove, axis = 0)
            cloudshad, fcps = cloud_removal.identify_clouds_shadows(sentinel2, dem, bbx)
            
            if clm is not None:
                clm = np.delete(clm, to_remove, axis = 0)
                try:
                    cloudshad = np.maximum(cloudshad, clm)
                except:
                    print("CLM error, continuing")

            interp = cloud_removal.id_areas_to_interp(
                sentinel2, cloudshad, cloudshad, image_dates, pfcps = fcps
            )
        interp_pct = 100*np.mean(interp >0.05, axis = (1, 2))
        print(f"Interpolate per-image: {np.around(interp_pct, 1)}")

    else:
        interp = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )
        cloudshad = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    return sentinel2, image_dates, interp, s1, dem, cloudshad, snow


def make_indices(arr):
    indices = np.zeros(
        (arr.shape[0], arr.shape[1], arr.shape[2], 4), dtype = np.float32
    )
    indices[:, ..., 0] = evi(arr)
    indices[:, ...,  1] = bi(arr)
    indices[:, ...,  2] = msavi2(arr)
    indices[:, ...,  3] = grndvi(arr)
    return indices


def make_and_smooth_indices(arr, dates):
    """Calculates remote sensing indices
    (evi, bi, msavi2, grndvi) and smooths them
    with the Whittaker smoother
    """
    sm_indices = Smoother(lmbd = 100, 
                          size = 24, 
                          nbands = 4, 
                          dimx = arr.shape[1],
                          dimy = arr.shape[2], 
                          outsize = 12)

    indices = make_indices(arr)
    try:
        indices, _ = calculate_and_save_best_images(indices, dates)
    except:
        indices = np.zeros((24, arr.shape[1], arr.shape[2], 4), dtype = np.float32)
        dates = [0,]
    indices = sm_indices.interpolate_array(indices)
    return indices


def deal_w_missing_px(arr, dates, interp):
    missing_px = interpolation.id_missing_px(arr, 10)
    if len(missing_px) > 0:
        dates = np.delete(dates, missing_px)
        arr = np.delete(arr, missing_px, 0)
        interp = np.delete(interp, missing_px, 0)
        print(f"Removing {len(missing_px)} missing images, leaving {len(dates)} / {len(dates)}")

    if np.sum(arr == 0) > 0:
        for i in range(arr.shape[0]):
            arr_i = arr[i]
            arr_i[arr_i == 0] = np.median(arr, axis = 0)[arr_i == 0]

    if np.sum(arr == 1) > 0:
        for i in range(arr.shape[0]):
            arr_i = arr[i]
            arr_i[arr_i == 1] = np.median(arr, axis = 0)[arr_i == 1]
    to_remove = np.argwhere(np.sum(np.isnan(arr), axis = (1, 2, 3)) > 0).flatten()
    if len(to_remove) > 0: 
        print(f"Removing {to_remove} NA dates")
        dates = np.delete(dates, to_remove)
        arr = np.delete(arr, to_remove, 0)
        interp = np.delete(interp, to_remove, 0)
    return arr, dates, interp


def smooth_large_tile(arr, dates, interp):

    """
    Deals with image normalization, smoothing, and regular 
    timestep interpolation for the entire array all at once
    """
    time1 = time.time()
    sm = Smoother(lmbd = 100, size = 24, nbands = arr.shape[-1], 
        dimx = arr.shape[1], dimy = arr.shape[2], outsize = 12)
    
    arr, dates, interp = deal_w_missing_px(arr, dates, interp)
    #arr, dates = normalize_first_last_date(arr, dates)
    if arr.shape[-1] == 10:
        indices = make_and_smooth_indices(arr, dates)
    

    try:
        time3 = time.time()
        arr, max_distance = calculate_and_save_best_images(arr, dates)
        time4 = time.time()
    except:
        print("Skipping because of no images")
        arr = np.zeros((24, arr.shape[1], arr.shape[2], arr.shape[-1]), dtype = np.float32)
        dates = [0,]
    time3 = time.time()
    arr = sm.interpolate_array(arr)
    time4 = time.time()
    #indices = make_and_smooth_indices(arr, dates)
    
    print(f"Interpolating images: {np.around(time4 - time3, 1)} seconds")

    if arr.shape[-1] == 10:
        out = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2], 14), dtype = np.float32)
        out[..., :10] = arr 
        out[..., 10:] = indices
    else:
        out = arr
    time2 = time.time()
    print(f"Smooth/regularized time series: {np.around(time2 - time1, 1)} seconds")
    return out, dates, interp


def identify_bright_bare_surfaces(img):
    """ Cleanup predictions by removing erroneous
    false positives where NIR SWIR Ratio < 0.9, TCI > 0.2, EVI < 0.3"""
    def _evi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
        BLUE = np.clip(x[..., 0], 0, 1)
        GREEN = np.clip(x[..., 1], 0, 1)
        RED = np.clip(x[..., 2], 0, 1)
        NIR = np.clip(x[..., 3], 0, 1)
        evis = 2.5 * ((NIR - RED) / (NIR + (6 * RED) - (7.5 * BLUE) + 1))
        evis = np.clip(evis, -1.5, 1.5)
        return evis
    nir_swir_ratio = (img[..., 3] / (img[..., 8] + 0.01))
    nir_swir_ratio = nir_swir_ratio < 0.9
    nir_swir_ratio = nir_swir_ratio * (np.mean(img[..., :3], axis = -1) > 0.2)
    nir_swir_ratio = nir_swir_ratio * (_evi(img) < 0.3)
    bright_surface = np.sum((nir_swir_ratio), axis = 0) > 1
    bright_surface = binary_dilation(1 - bright_surface, iterations = 2)
    bright_surface = binary_dilation(1 - bright_surface, iterations = 1)
    blurred = distance(1 - bright_surface)
    blurred[blurred > 3] = 3
    blurred = (blurred / 3)
    if np.mean(blurred) < 0.995:
        print(f"The bright area is: {np.mean(blurred)}")
    return blurred[7:-7, 7:-7]


def process_subtiles(x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None, bbx = None, size = SIZE, train_bbx = None) -> None:
    '''Wrapper function to interpolate clouds and temporal gaps, superresolve tiles,
       calculate relevant indices, and save predicted tree cover as a .npy
       
       Parameters:
        x (int): integer representation of the x tile ID
        y (int): integer representation of the y tile ID
        s2 (arr): (n, 160, 160, 11) array of sentinel 2 + DEM
        dates (arr): (n,) array of day of year of each image 
        interp (arr): (n, 160, 160) bool array of interpolated areas
        s1 (arr): (12, 160, 160, 2) float32 array of dB sentinel 1 data
        sess (tf.Session): tensorflow sesion to use for temporal predictions

       Returns:
        None
    '''
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    
    s2 = interpolation.interpolate_na_vals(s2)
    s2 = np.float32(s2)
    
    s2_median = np.median(s2, axis = 0).astype(np.float32)
    s2_median = np.concatenate([s2_median, 
        np.median(evi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median,
        np.median(bi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(msavi2(s2), axis = 0)[..., np.newaxis]], axis = -1)
    s2_median = np.concatenate([s2_median, 
        np.median(grndvi(s2), axis = 0)[..., np.newaxis]], axis = -1)
    ard_ndmi_file = f"{args.local_path}{str(x)}/{str(y)}/ard_ndmi.hkl"
    ard_ndmi = (ndmi(s2) * 10000).astype(np.int16) // 5 * 5
    hkl.dump(ard_ndmi, ard_ndmi_file, mode='w', compression='gzip')
    np.save(f"{args.local_path}{str(x)}/{str(y)}/ard_dates.npy", dates)

    if WRITE_MONTHLY_TIFS:
        for i in range(s2.shape[0]):
            write_ard_to_tif(s2[i, ..., :3], bbx,
                             f"{str(x)}{str(y)}/{str(x)}X{str(y)}Y_{str(i)}", "")

    s2, dates, interp = smooth_large_tile(s2, dates, interp)
    s2_median = s2_median[np.newaxis]
    #med_evi = np.percentile(s2_median[..., 10].flatten(), 0.5)
    s1_median = np.median(s1, axis = 0)[np.newaxis].astype(np.float32)
    #s2_median = np.median(s2, axis = 0)[np.newaxis].astype(np.float32)

    fname = f"{str(x)}X{str(y)}Y{str(year)}"
    write_ard_to_tif(np.mean(s2[..., :3], axis = 0), bbx, fname, "")
    key = f'{str(year)}/composite/{x}/{y}/{str(x)}X{str(y)}Y.tif'
    uploader.upload(bucket = args.s3_bucket, key = key, file = fname + ".tif")
    dem_train = dem[np.newaxis, ..., np.newaxis]
    dem_train = dem_train.repeat(4, axis = 0)

    ###### MAR 22 #####
    path_to_tile = f'{args.local_path}/{str(x)}/{str(y)}/'
    if not os.path.exists(os.path.realpath(f"{path_to_tile}ard/")):
        os.makedirs(os.path.realpath(f"{path_to_tile}ard/"))
    s2med = np.mean(s2[..., :10], axis = 0)
    s1med = np.median(s1, axis = 0)
    ard_median = np.concatenate([s2med, dem[..., np.newaxis], s1med], axis = -1)
    print(f"ARD is {ard_median.shape} shape")
    hkl.dump(ard_median,
      f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl", 
      mode='w',
      compression='gzip')
    key = f'{str(year)}/ard/{x}/{y}/{str(x)}X{str(y)}Y_ard.hkl'
    uploader.upload(bucket = args.s3_bucket, key = key, 
      file = f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
    ###### MAR 22 #####

    if args.gen_composite == True:
        composite_fname = f'{args.local_path}{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_composite.hkl'
        composite = np.concatenate([s2_median, s1_median], axis = -1)
        composite = composite * 10000
        composite = np.clip(composite, -32000, 32000)
        composite = np.int16(composite)

        hkl.dump(composite,
                 composite_fname, 
                 mode='w',
                 compression='gzip')

        print(f"Saved composite to {composite_fname} of shape {composite.shape} and type {composite.dtype}")

    if args.make_training_data:
        # At the equator, a tile is 618 height 618 width
        #print(PX_x)
        #if np.logical_and(PX_y == 0, PX_x == 0):
        #    PX_y = (s2.shape[1]) // 2
        #    PX_x = (s2.shape[2]) // 2
        if int(x) > 0 and int(y) > 0:
            print(f"Plot centroid: {PX_x, PX_y}")
            s2_train = s2[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
            s1_train = s1[:, PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38, :]
            dem_train = dem[PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38]
            print(f"Plot shape: {s2_train.shape}")
        else:
            Px_y = (s2.shape[1]) // 2
            Px_x = (s2.shape[2]) // 2
            print(f"Plot centroid: {Px_x, Px_y}")
            s2_train = s2[:, Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38, :]
            s1_train = s1[:, Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38, :]
            dem_train = dem[Px_y - 38:Px_y + 38, Px_x - 38:Px_x + 38]
            print(f"Plot shape: {s2_train.shape}")

        dem_train = dem_train[np.newaxis, ..., np.newaxis]
        dem_train = dem_train.repeat(12, axis = 0)
        indices_mins = [-0.1409399364817101, -0.4973397113668104, 
            -0.09731556326714398, -0.7193834232943873]
        indices_maxes = [0.8177635298774327, 0.35768999002433816,
           0.7545951919107605, 0.7602693339366691]
        

        train_sample = np.concatenate([s2_train[..., :10], 
            dem_train, 
            s1_train,
            s2_train[..., 10:]], axis = -1)
        for i in range(13, 17):
            train_sample[..., i] = np.clip(train_sample[..., i], min_all[i], max_all[i])
            train_sample[..., i] -= min_all[i]
        train_sample[..., -1] /= 2
        train_sample = np.clip(train_sample, 0, 1)
        # So the four indices are clipped to -.14-.82, -.49 - 0.35, -0.09 to 0.75, -0.71 to 0.76
        # So we add 0.14, 0.49, 0.09, and 0.71
        # The last one has to be divided by 2!
        write_ard_to_tif(np.median(train_sample, axis = 0)[..., :3],
              train_bbx,
              f"{str(PLOTID)}",
              "train-ard/", "")
        train_sample = to_int16(train_sample)
        if not os.path.exists(os.path.realpath("train-ard")):
            os.makedirs(os.path.realpath("train-ard"))
        fout = f"train-ard/{str(PLOTID)}.hkl"
        fouttif =  f"train-ard/{str(PLOTID)}.tif"
        hkl.dump(train_sample, fout, mode = 'w', compression = 'gzip')
        print(f"Saved to {fout}")
        
        key = f'train-samples-128/{str(PLOTID)}.hkl'
        uploader.upload(bucket = args.s3_bucket, key = key, file = fout)
        key = f'train-samples-128/{str(PLOTID)}.tif'
        uploader.upload(bucket = args.s3_bucket, key = key, file = fouttif)
   
    if args.length == 4:
        s2 = np.reshape(s2, (4, 3, s2.shape[1], s2.shape[2], s2.shape[3]))
        s2 = np.median(s2, axis = 1, overwrite_input = True)
        s1 = np.reshape(s1, (4, 3, s1.shape[1], s1.shape[2], s1.shape[3]))
        s1 = np.median(s1, axis = 1, overwrite_input = True)  
    elif args.length == 1:
        s2 = np.median(s2, axis = 0, overwrite_input = True)[np.newaxis]
        s1 = np.median(s1, axis = 0, overwrite_input = True)[np.newaxis]
        s2 = np.repeat(s2, 4, axis = 0)
        s1 = np.repeat(s1, 4, axis = 0)

    #train_sample = np.concatenate([s2[..., :10], 
    #   dem_train, s1,
    #    s2[..., 10:]], axis = -1)
    #/Volumes/John/data/train-large-ard-x/
    #hkl.dump(train_sample, f"{str(x)}X{str(y)}Y.hkl", mode='w', compression='gzip')


    # The tiles_folder references the folder names (w/o boundaries)
    # While the tiles_array references the arrays themselves (w/ boudnaries)
    # These enable the predictions to overlap to reduce artifacts
    n_rows = 6 if SIZE != 222 else 7
    if args.make_training_data == True:
        n_rows = 6
    gap_x = int(np.ceil((s1.shape[1] - size) / (n_rows - 1)))
    gap_y = int(np.ceil((s1.shape[2] - size) / (n_rows - 1)))
    tiles_folder_x = np.hstack([np.arange(0, s1.shape[1] - size, gap_x), np.array(s1.shape[1] - SIZE)])
    tiles_folder_y = np.hstack([np.arange(0, s1.shape[2] - size, gap_y), np.array(s1.shape[2] - SIZE)])

    def cartesian(*arrays):
        mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
        dim = len(mesh)  # number of dimensions
        elements = mesh[0].size  # number of elements, any index will do
        flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
        reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
        return reshape

    windows = cartesian(tiles_folder_x, tiles_folder_y)
    win_sizes = np.full_like(windows, SIZE)
    tiles_folder = np.hstack([windows, win_sizes])
    tiles_folder = np.sort(tiles_folder, axis = 0)
    tiles_folder[:, 1] = np.tile(np.unique(tiles_folder[:, 1]), 
        int(len(tiles_folder[:, 1]) / len(np.unique(tiles_folder[:, 1]))))
    tiles_array = tof_downloading.make_overlapping_windows(tiles_folder, diff = 7)
    
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/processed/',
                  tiles_folder)
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/feats/',
                  tiles_folder)
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'

    gap_between_years = False
    t = 0

    def _hollstein_cld(arr):
        # Simple cloud detection algorithm
        # Generates "okay" cloud masks that are to be refined
        # From Figure 6 in Hollstein et al. 2016
        step1 = arr[..., 7] > 0.166
        step2b = arr[..., 1] > 0.21
        step3 = arr[..., 5] / arr[..., 8] < 4.292
        cl = step1 * step2b * step3
        for i in range(cl.shape[0]):
            cl[i] = binary_dilation(1 -
                                    (binary_dilation(cl[i] == 0, iterations=2)),
                                    iterations=10)
        return cl

    # Prep and predict subtiles
    if args.process is True:
        print(f"{str(x)}X{str(y)}Y: Generating predictions for {len(tiles_folder)} subtiles")
        for t in tqdm(range(len(tiles_folder))):
            time1 = time.time()
            tile_folder = tiles_folder[t]
            tile_array = tiles_array[t]
            
            start_x, start_y = tile_array[0], tile_array[1]
            folder_x, folder_y = tile_folder[0], tile_folder[1]
            end_x = start_x + tile_array[2]
            end_y = start_y + tile_array[3]

            subtile = np.copy(s2[:, start_x:end_x, start_y:end_y, :])
            subtile_median_s2 = np.copy(s2_median[:, start_x:end_x, start_y:end_y, :])
            subtile_median_s1 = np.copy(s1_median[:, start_x:end_x, start_y:end_y, :])
            interp_tile = interp[:, start_x:end_x, start_y:end_y]
            dates_tile = np.copy(dates)
            dem_subtile = dem[np.newaxis, start_x:end_x, start_y:end_y]
            s1_subtile = np.copy(s1[:, start_x:end_x, start_y:end_y, :])
            output = f"{path}{str(folder_y)}/{str(folder_x)}.npy"
            min_clear_images_per_date = np.sum(interp_tile < 0.33, axis = (0))
            no_images = False
            if np.percentile(min_clear_images_per_date, 50) < 1: # 33
                no_images = True

            to_remove = np.argwhere(np.sum(np.isnan(subtile), axis = (1, 2, 3)) > 100).flatten()
            if len(to_remove) > 0 and len(to_remove) < len(dates_tile): 
                print(f"Removing {to_remove} NA dates")
                dates_tile = np.delete(dates_tile, to_remove)
                subtile = np.delete(subtile, to_remove, 0)
                interp_tile = np.delete(interp_tile, to_remove, 0)
            if len(to_remove) >= len(dates_tile):
                no_images = True

            # Pad the corner / edge subtiles within each tile
            if subtile.shape[2] == SIZE + 7: 
                pad_u = 7 if start_y == 0 else 0
                pad_d = 7 if start_y != 0 else 0
                subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
                s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
                dem_subtile = np.pad(dem_subtile, ((0, 0,), (0, 0), (pad_u, pad_d)), 'reflect')
                subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
                subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
                min_clear_images_per_date = np.pad(min_clear_images_per_date, ((0, 0), (pad_u, pad_d)), 'reflect')
            if subtile.shape[1] == SIZE + 7:
                pad_l = 7 if start_x == 0 else 0
                pad_r = 7 if start_x != 0 else 0
                subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
                s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
                dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')
                subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
                subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
                min_clear_images_per_date = np.pad(min_clear_images_per_date, ((pad_u, pad_d), (0, 0)), 'reflect')
            
            # Concatenate the DEM and Sentinel 1 data
            subtile_all = np.zeros((args.length + 1, SIZE + 14, SIZE + 14, 17), dtype = np.float32)
            subtile_all[:-1, ..., :10] = subtile[..., :10]
            subtile_all[:-1, ..., 11:13] = s1_subtile
            subtile_all[:-1, ..., 13:] = subtile[..., 10:]
            subtile_evi = np.copy(subtile_median_s2[..., 7:-7, 7:-7, 10]).squeeze()

            subtile_all[:, ..., 10] = dem_subtile.repeat(args.length + 1, axis = 0)
            subtile_all[-1, ..., :10] = subtile_median_s2[..., :10]
            subtile_all[-1, ..., 11:13] = subtile_median_s1
            subtile_all[-1, ..., 13:] = subtile_median_s2[..., 10:]
            max_cc = np.max(_hollstein_cld(subtile_all), axis = 0)[7:-7, 7:-7]
            bright_surface = identify_bright_bare_surfaces(subtile_all)
            # Create the output folders for the subtile predictions
            output_folder = "/".join(output.split("/")[:-1])
            if not os.path.exists(os.path.realpath(output_folder)):
                os.makedirs(os.path.realpath(output_folder))
            
            assert subtile_all.shape[1] >= (SIZE - 14), f"subtile shape is {subtile_all.shape}"
            assert subtile_all.shape[0] == (args.length + 1), f"subtile shape is {subtile_all.shape}"

            no_images = True if len(dates_tile) < 2 else no_images   
            if no_images:
                print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data"
                    f" {np.percentile(min_clear_images_per_date, 10)} clear images")
                preds = np.full((SIZE, SIZE), 255)
            else:
                subtile_all = normalize_subtile(subtile_all)
                preds = predict_subtile(subtile_all, sess, predict_logits, SIZE)
                time2 = time.time()
                subtile_time = np.around(time2 - time1, 1)
                
                if args.gen_feats:
                    latefeats = predict_subtile(subtile_all, sess, predict_latefeats, SIZE)[..., :32]
                    earlyfeats = predict_subtile(subtile_all, sess, predict_earlyfeats, SIZE)[..., :32]
                    #earlyfeats = earlyfeats.repeat(4, axis = 0).repeat(4, axis = 1)
                    #earlyfeats = earlyfeats[1:-1, 1:-1]
                    earlyfeats = float_to_int16(earlyfeats)
                    latefeats = float_to_int16(latefeats)
                    feats_path = f'{args.local_path}{str(x)}/{str(y)}/feats/'
                    output_feats = f"{feats_path}{str(folder_y)}/{str(folder_x)}.npy"
                    output_folder = "/".join(output_feats.split("/")[:-1])
                    if not os.path.exists(os.path.realpath(output_folder)):
                        os.makedirs(os.path.realpath(output_folder))
                        os.makedirs(f'{args.local_path}{str(x)}/{str(y)}/raw/feats/')
                    if not os.path.exists(f'{args.local_path}{str(x)}/{str(y)}/ard/'):
                        os.makedirs(f'{args.local_path}{str(x)}/{str(y)}/ard/')
                    #np.save(output_feats, earlyfeats)   
                    np.save(output_feats, np.concatenate([earlyfeats,latefeats], axis = -1))
                    print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                    f"for: {dates_tile}, {np.percentile(min_clear_images_per_date, 10)} clear images"
                    f" in {subtile_time} seconds, {np.mean(preds)}")

            # Go back through and remove predictions where there are no cloud-free images
            min_clear_images_per_date = min_clear_images_per_date[6:-6, 6:-6]
            no_images = min_clear_images_per_date < 1
            struct2 = generate_binary_structure(2, 2)
            no_images = 1 - binary_dilation(1 - no_images, structure = struct2, iterations = 6)
            no_images = binary_dilation(no_images,  structure = struct2, iterations = 6)

            if SIZE == 158:
                #no_images = np.pad(no_images, ((1, 1), (1, 1)))
                no_images = np.reshape(no_images, ((4, 40, 4, 40)))
                no_images = np.sum(no_images, axis = (1, 3))
                no_images = no_images > (40*40) * 0.25 # 0.10
                no_images = no_images.repeat(40, axis = 0).repeat(40, axis = 1)
                no_images = no_images[1:-1, 1:-1]
                preds[no_images] = 255.
            if SIZE == 142:
                #no_images = np.pad(no_images, ((1, 1), (1, 1)))
                no_images = np.reshape(no_images, ((9, 16, 9, 16)))
                no_images = np.sum(no_images, axis = (1, 3))
                no_images = no_images > (16*16) * 0.75 # 0.10
                no_images = no_images.repeat(16, axis = 0).repeat(16, axis = 1)
                no_images = no_images[1:-1, 1:-1]
                preds[no_images] = 255.

            # Occasionally, extremely out of bounds data can be falsely identified as tree cover due to
            # Sentinel-1 artifacts. If the data is under the 0.5% of EVI, and is not >80% prob of
            # tree cover, we attenuate the values since this is almost always a Sentinel-1 artifact.
            #med_evi = np.minimum(med_evi, 0.25)
            #evi_flag = subtile_evi < med_evi

            preds = preds * bright_surface
            preds = np.around(preds, 3)
            preds = preds.astype(np.float32)
            np.save(output, preds)

#####################################################
############# SUBTILE -> TILE PRED XFER #############
#####################################################

def fspecial_gauss(size: int, sigma: int) -> np.ndarray:
    """Function to mimic the 'fspecial' gaussian MATLAB function

        Parameters:
         size (int): size of square guassian kernel
         sigma (float): diameter of the kernel
    
        Returns:
         g (np.ndarray): gaussian kernel from [0, 1]
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g

def calc_overlap(idx, tile):
    subtile = tile[..., idx]
    others = np.delete(tile, idx, -1)
    
    others = others[~np.isnan(subtile)].reshape((SIZE, SIZE, tile.shape[-1] - 1))
    remove = np.argwhere(np.sum(np.isnan(others), axis = (0, 1)) == (SIZE*SIZE)).flatten()
    others = bn.nanmean(np.delete(others, remove, -1), axis = -1)
    subtile = subtile[~np.isnan(subtile)].reshape((SIZE, SIZE))
    
    return bn.nanmean(abs(others - subtile))


def load_mosaic_predictions(out_folder: str, depth) -> np.ndarray:
    """
    Loads the .npy subtile files in an output folder and mosaics the overlapping predictions
    to return a single .npy file of tree cover for the 6x6 km tile
    Additionally, applies post-processing threshold rules and implements no-data flag of 255
    
        Parameters:
         out_folder (os.Path): location of the prediction .npy files 
    
        Returns:
         predictions (np.ndarray): 6 x 6 km tree cover data as a uint8 from 0-100 w/ 255 no-data flag
    """
    
    # Generate the tiling schema
    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
    max_x = np.max(x_tiles) + SIZE
    for x_tile in x_tiles:
        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        if len(y_tiles) > 0:
            max_y = np.max(y_tiles) + SIZE
        
    def _load_partial(out_folder, start, end, x_tiles = x_tiles, y_tiles = y_tiles):
        # For predictions, load up a (SIZE, SIZE) array, * by 100
        # For features, load up a (SIZE, SIZE, 64) array and iterate thru 8 at a time
        #     due to memory constraints.
        # 
        MULT = 100

        n = end - start
        
        predictions = np.full((n, max_x, max_y, len(x_tiles) * len(y_tiles)),
                              np.nan, dtype = np.float32)
        mults = np.full((1, max_x, max_y, len(x_tiles) * len(y_tiles)),
                        0, dtype = np.float32)
        i = 0
        for x_tile in x_tiles:
            y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    if prediction.shape[0] == 216:
                        GAUSS = 46
                    elif prediction.shape[0] > 200:
                        GUASS = 44
                    if prediction.shape[0] == 190:
                        GAUSS = 36
                    if prediction.shape[0] <= 150:
                        GAUSS = 30
                    if prediction.shape[0] <= 100:
                    	GAUSS = 20
                    else:
                        GAUSS = 36
                    GAUSS = 36
                    #GAUSS = 46 if prediction.shape[0] > 150 else 5
                    if n > 1:
                        prediction = prediction[..., start:end]
                    else:
                        prediction[prediction < 255] = prediction[prediction < 255] * MULT
                    if (np.sum(prediction) < SIZE*SIZE*255) or depth > 1:
                        prediction = (prediction).T.astype(np.float32)
                        predictions[:, x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = prediction
                        fspecial_i = fspecial_gauss(SIZE, GAUSS)
                        if depth == 1:
                            fspecial_i[prediction > 100] = 0.
                        mults[:, x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = fspecial_i # or 44

                    i += 1
            
        if depth > 1:
            predictions = predictions.astype(np.float32)
            mults = mults / np.sum(mults, axis = -1)[..., np.newaxis]
            predictions = np.nansum(predictions * mults, axis = -1)
            predictions = np.int16(predictions)
            return predictions
        else:
            mults = mults.squeeze()
            predictions = predictions.astype(np.float32)
            predictions = predictions.squeeze()

            ratios = np.zeros((predictions.shape[-1]), dtype = np.float32)
            mults[np.isnan(predictions)] = 0.
            try:
                for i in range(predictions.shape[-1]):
                    ratios[i] = calc_overlap(i, predictions)
                multipliers = np.median(ratios) / ratios
                multipliers[multipliers > 1.5] = 1.5
                for i in range(predictions.shape[-1]):
                    mults[..., i] *= multipliers[i]
            except:
                print("Skipping the weighted average due to cloud cover")

            predictions[predictions > 100] = np.nan
            
            mults = mults / np.sum(mults, axis = -1)[..., np.newaxis]

            out = np.copy(predictions)
            out = np.sum(np.isnan(out), axis = (2))
            n_preds = predictions.shape[-1]
            predictions = np.nansum(predictions * mults, axis = -1)
            predictions[out == n_preds] = np.nan
            predictions[np.isnan(predictions)] = 255.
            predictions = predictions.astype(np.uint8)

            predictions[predictions <= .15*MULT] = 0.        
            predictions[predictions > 100] = 255.
        
            return predictions
    
    output = np.full((depth, max_x, max_y),
                              0., dtype = np.int16)
    if depth == 1:
        output = _load_partial(out_folder, 1, 2)
    else:
        iters = np.arange(0, depth, 8)
        for i in iters:
            preds = _load_partial(out_folder, i, i + 8)
            output[i:i + 8] = preds
    if depth == 1:
        no_images = output == 255
        struct2 = generate_binary_structure(2, 2)
        no_images = binary_dilation(no_images,  structure = struct2, iterations = 10)
        output[no_images] = 255
    return output
     
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument(
        "--local_path", dest = 'local_path', default = '../project-monitoring/tiles/'
    )

    parser.add_argument(
        "--predict_model_path",
        dest = 'predict_model_path',
        #default = '../models/tf2-new/'
        #default = '../models/tf2-nov6-44-master/',
        default = '../models/tf2-ard/'
        #default = '../models/tml-2023-new/'
        #default = '../models/172-ttc-dec2023-3/'
    )

    #parser.add_argument(
    #    "--predict_model_path2",
    #    dest = 'predict_model_path2',
    #    default = '../models/224-tml-asasasa/'
    #)
    #parser.add_argument(
    ##    "--gap_model_path",
    #    dest = 'gap_model_path',
    #    default = '../models/182-gap-sept/'
    #)
    parser.add_argument(
        "--superresolve_model_path",
        dest = 'superresolve_model_path',
        default = '../models/supres/nov-40k-swir/'
    )
    parser.add_argument(
        "--db_path", dest = "db_path", default = "process_area_2022.csv"
    )
    parser.add_argument("--ul_flag", dest = "ul_flag", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--no-cleanup", dest = "no_cleanup", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--year", dest = "year", default = 2020)
    parser.add_argument("--n_tiles", dest = "n_tiles", default = None)
    parser.add_argument("--x", dest = "x", default = None)
    parser.add_argument("--y", dest = "y", default = None)
    parser.add_argument("--reprocess", dest = "reprocess", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--redownload_s1", dest = "redownload_s1", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--redownload", dest = "redownload", default = False, type=str2bool, nargs='?',
                        const=True)
    #parser.add_argument("--model", dest = "model", default = "temporal")
    #parser.add_argument("--is_savannah", dest = "is_savannah", default = False)
    parser.add_argument("--gen_feats", dest = "gen_feats", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--gen_composite", dest = "gen_composite", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--process", dest = "process", default = True, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--snow", dest = "snow", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--length", dest = "length", default = 4)
    parser.add_argument("--start", dest = 'start', default = 0)
    parser.add_argument("--end", dest = "end", default = 100000)
    parser.add_argument("--make_training_data", dest = "make_training_data", default = False, type=str2bool, nargs='?',
                        const=True)
    args = parser.parse_args()
    args.local_path = args.local_path + str(args.year) + "/"

    print(f'Country: {args.country} \n'
          f'Local path: {args.local_path} \n'
          f'Predict model path: {args.predict_model_path} \n'
          f'Superrresolve model path: {args.superresolve_model_path} \n'
          f'DB path: {args.db_path} \n'
          f'S3 Bucket: {args.s3_bucket} \n'
          f'YAML path: {args.yaml_path} \n'
          f'Current dir: {os.getcwd()} \n'
          f'N tiles to download: {args.n_tiles} \n'
          f'no-cleanup: {args.no_cleanup} \n'
          f'Year: {args.year} \n'
          f'X: {args.x} \n'
          f'Y: {args.y} \n'
          f'Reprocess: {args.reprocess} \n'
          f'Redownload: {args.redownload} \n' ### Change the name of this flag
          f'Redownload s1: {args.redownload_s1} \n'
          #f'Model: {args.model} \n'
          f'gen_feats: {args.gen_feats} \n'
          f'snow removal: {args.snow} \n'
          f'length: {args.length} \n'
          f'start: {args.start} \n'
          f'end: {args.end} \n'
          )

    args.year = int(args.year)
    args.length = int(args.length)

    if os.path.exists(args.yaml_path):
        with open(args.yaml_path, 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            SHUB_SECRET = key['shub_secret']
            SHUB_KEY = key['shub_id']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']
        print(f"Successfully loaded key from {args.yaml_path}")
        uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET, overwrite = True)
        shconfig = SHConfig()
        shconfig.instance_id = API_KEY
        shconfig.sh_client_id = SHUB_KEY
        shconfig.sh_client_secret = SHUB_SECRET
    else:
        raise Exception(f"The API keys do not exist in {args.yaml_path}")

    if os.path.exists(args.db_path) or args.db_path[:4] == 's3:/':
        data = args.db_path
        if args.db_path[:4] == 's3:/':
            print(f"Downloading database from {args.db_path}")
            data = args.db_path.split("/")[-1]
            bucket = args.db_path.split("/")[2]
            download_single_file(args.db_path, data, AWSKEY, AWSSECRET, bucket)
        data = pd.read_csv(data)
        data = data[data['country'] == args.country]
        data = data.reset_index(drop = True)
        #data = data.sample(frac=1).reset_index(drop=True)
        n_to_process = len(data)
        print(f"There are {len(data)} tiles for {args.country}, approx {len(data) * 3.6}K ha")
    else:
        raise Exception(f"The database does not exist at {args.db_path}")

    # Lots of code here to load two tensorflow graphs at once
    superresolve_graph_def = tf.compat.v1.GraphDef()
    predict_graph_def = tf.compat.v1.GraphDef()

    if os.path.exists(args.superresolve_model_path):
        print(f"Loading model from {args.superresolve_model_path}")
        superresolve_file = tf.io.gfile.GFile(args.superresolve_model_path + "superresolve_graph.pb", 'rb')
        superresolve_graph_def.ParseFromString(superresolve_file.read())
        superresolve_graph = tf.import_graph_def(superresolve_graph_def, name='superresolve')
        superresolve_sess = tf.compat.v1.Session(graph=superresolve_graph)
        superresolve_logits = superresolve_sess.graph.get_tensor_by_name("superresolve/Add_2:0")
        superresolve_inp = superresolve_sess.graph.get_tensor_by_name("superresolve/Placeholder:0")
        superresolve_inp_bilinear = superresolve_sess.graph.get_tensor_by_name("superresolve/Placeholder_1:0")
    else:
        raise Exception(f"The model path {args.superresolve_model_path} does not exist")

    if os.path.exists(args.predict_model_path):
        print(f"Loading model from {args.predict_model_path}")
        print(f"predict_graph-{str(SIZE+14)}.pb")
        predict_file = tf.io.gfile.GFile(args.predict_model_path + f"predict_graph-{str(SIZE+14)}.pb", 'rb')
        predict_graph_def.ParseFromString(predict_file.read())
        predict_graph = tf.import_graph_def(predict_graph_def, name='predict')
        predict_sess = tf.compat.v1.Session(graph=predict_graph)
        if args.length == 12:
            predict_latefeats = predict_sess.graph.get_tensor_by_name(f"predict/csse_out_mul/mul:0") 
            predict_earlyfeats = predict_sess.graph.get_tensor_by_name(f"predict/gru_drop/drop_block2d/cond/Merge:0")
            predict_logits = 'predict/conv2d_13/Sigmoid:0'
            print("Predict_earlyfeats", predict_earlyfeats)
        else:
            #predict_earlyfeats = predict_sess.graph.get_tensor_by_name(f"predict/IdentityN_7:0")
            #predict_latefeats = predict_sess.graph.get_tensor_by_name(f"predict/out_conv/out/ws_conv2d_7/Conv2D:0")
        #predict_logits = "predict/cropping2d_3/strided_slice:0" #'predict/conv2d_13/Sigmoid:0'
        #predict_height = 'predict/cropping2d_3/strided_slice:0'
        #predict_logits = 'predict/conv2d_5/Sigmoid:0'
        #predict_logits = 'predict/conv2d_13/Sigmoid:0'
            predict_logits = 'predict/conv2d/Sigmoid:0'
        print(f"predict logits: {predict_logits}")
        predict_logits = predict_sess.graph.get_tensor_by_name(predict_logits) 
        #predict_height = predict_sess.graph.get_tensor_by_name(predict_height) 
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")

    # Normalization mins and maxes for the prediction input
    min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 
               0.013351644159609368, 0.01965362020294499, 0.014229037918669413, 
               0.015289539940489814, 0.011993591210803388, 0.008239871824216068,
               0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
               -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]

    max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 
               0.6027466239414053, 0.5650263218127718, 0.5747005416952773,
               0.5933928435187305, 0.6034943160143434, 0.7472037842374304,
               0.7000076295109483, 
               0.4,
               0.948334642387533, 
               0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
               0.7545951919107605, 0.7602693339366691]

    if args.predict_model_path == "../models/224-2023-new/":
        max_all[10] = 0.509269855
        SIZE = 216
    
    n = 0
    exception_counter = 0

    try:
        data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
        data['X_tile'] = pd.to_numeric(data['X_tile'])
        data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
        data['Y_tile'] = pd.to_numeric(data['Y_tile'])
    except Exception as e:
        print(f"Ran into {str(e)} error")
        time.sleep(1)

    if args.x and args.y:
        x = args.x
        y = args.y
        data = data[data['Y_tile'] == int(y)]
        data = data[data['X_tile'] == int(x)]
        data = data.reset_index(drop = True)
        x = str(int(x))
        y = str(int(y))
    for index, row in data.iterrows():
        if np.logical_and(index >= int(args.start), index < int(args.end)):
            x = str(int(row['X_tile']))
            y = str(int(row['Y_tile']))
            to_process = True
            if args.make_training_data == True:
                PX_x = row['X_px']
                PX_y = row['Y_px']
                print(f"Training plot centroid: {PX_x, PX_y}")
                PLOTID = str(row['plot_id'])
                print(PLOTID)
                #to_process = True if int(x) > 0 else False

            x = x[:-2] if ".0" in x else x
            y = y[:-2] if ".0" in y else y
            bbx = None
            year = args.year
            dates = (f'{str(args.year - 1)}-11-15' , f'{str(args.year + 1)}-02-15')
            dates_sentinel_1 = (f'{str(args.year)}-01-01' , f'{str(args.year)}-12-31')
            days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
            starting_days = np.cumsum(days_per_month)

            # Check to see whether the tile exists locally or on s3
            path_to_tile = f'{args.local_path}/{str(x)}/{str(y)}/'
            s3_path_to_tile = f'{str(year)}/tiles/{str(x)}/{str(y)}/'        
            processed = file_in_local_or_s3(path_to_tile,
                                            s3_path_to_tile, 
                                            AWSKEY, AWSSECRET, 
                                            args.s3_bucket)

            if WRITE_TEMP_TIFS:
                TEMP_FOLDER = f"{os.getcwd()}/{str(x)}{str(y)}/"
                if not os.path.exists(TEMP_FOLDER):
                    print('making ', TEMP_FOLDER)
                    os.mkdir(TEMP_FOLDER)
            
            # If the tile does not exist, go ahead and download/process/upload it
            if ((not processed) or args.reprocess == True) and (to_process == True):
                try:
                  bbox = None
                  if not bbx:
                      data2 = data.copy()
                      data2 = data2[data2['Y_tile'] == int(y)]
                      data2 = data2[data2['X_tile'] == int(x)]
                      data2 = data2.reset_index(drop = True)
                      x = str(int(x))
                      y = str(int(y))
                      x = x[:-2] if ".0" in x else x
                      y = y[:-2] if ".0" in y else y
                      if args.make_training_data == True:
                        #print(PLOTID, type(PLOTID), data2['plot_id'])
                        data2 = data.copy()
                        print(len(data2))
                        data2 = data2[data2['plot_id'] == int(PLOTID)]
                        data2 = data2.reset_index(drop = True)
                        print(data2)
                      initial_bbx = [data2['X'][0], data2['Y'][0], data2['X'][0], data2['Y'][0]]
                      bbx = make_bbox(initial_bbx, expansion = 300/30)
                      expansion = 300
                      if args.make_training_data == True:
                            if np.logical_and(int(x) != 0, int(y) != 0):
                                print("Not zeros")
                                initial_train_bbx = [data2['long'][0], data2['lat'][0], data2['long'][0], data2['lat'][0]]
                                train_bbx = make_bbox(initial_train_bbx, expansion = (38/30) / 1.03)
                            else:
                                print("Zeros")
                                initial_train_bbx = [data2['X'][0], data2['Y'][0], data2['X'][0], data2['Y'][0]]
                                train_bbx = make_bbox(initial_train_bbx, expansion = (38/30) / 1.03)
                                bbx = make_bbox(initial_bbx, expansion = 200/30)
                                expansion = 200
                      else:
                        train_bbx = None
                      
                      lat = data2['Y'][0]
                      print("The latitude is", lat)
                  time0 = time.time()
                  print(args.redownload, type(args.redownload))
                  if (args.redownload == False) or not processed:
                      time1 = time.time()
                      bbx, n_images = download_tile(x = x,
                                                    y = y, 
                                                    data = data, 
                                                    api_key = shconfig, 
                                                    year = args.year,
                                                    initial_bbx = initial_bbx,
                                                    expansion = expansion)
                      if os.path.exists(f'{args.local_path}{str(x)}/{str(y)}/processed/'):
                          shutil.rmtree(f'{args.local_path}{str(x)}/{str(y)}/processed/')
                      time2 = time.time()
                      print(f"Finished downloading imagery in {np.around(time2 - time0, 1)} seconds")
                  else:
                      bbx = make_bbox(initial_bbx, expansion = 300/30)
                      download_raw_tile((x, y), args.local_path, "raw")
                      if os.path.exists(f'{args.local_path}{str(x)}/{str(y)}/processed/'):
                          shutil.rmtree(f'{args.local_path}{str(x)}/{str(y)}/processed/')
                      
                      folder = f"{args.local_path}{str(x)}/{str(y)}/"
                      tile_idx = f'{str(x)}X{str(y)}Y'
                      s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
                      s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
                      s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
                      try:
                          print(os.listdir(f'{folder}raw/s2_20/'))
                          size = hkl.load(s2_20_file)
                          size = size.shape[1:3]
                      except:
                          bbx, n_images = download_tile(x = x,
                                                    y = y, 
                                                    data = data, 
                                                    api_key = shconfig, 
                                                    year = args.year)
                          size = hkl.load(s2_20_file)
                          size = size.shape[1:3]
                      if args.redownload_s1 == True:
                          download_s1_tile(data = data, 
                           bbx = bbx,
                           api_key = shconfig,
                           year = year, 
                           dates_sentinel_1 = dates_sentinel_1, 
                           size = size, 
                           s1_file = s1_file, 
                           s1_dates_file = s1_dates_file)
                      time2 = time.time()
                      n_images = 10
                  if n_images > 2:
                      s2, dates, interp, s1, dem, cloudshad, snow = process_tile(x = x, 
                                                                           y = y, 
                                                                           data = data, 
                                                                           local_path = args.local_path, 
                                                                           bbx = bbx,
                                                                           make_shadow = True)
                      s2[..., :10] = superresolve_large_tile(s2[..., :10], superresolve_sess)
                      time1 = time.time()
                      process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, bbx, SIZE, train_bbx)
                      TEST = False
                      if TEST:
                        if not os.path.exists(os.path.realpath(f"{path_to_tile}ard/")):
                            os.makedirs(os.path.realpath(f"{path_to_tile}ard/"))
                        s2med = np.median(s2, axis = 0)
                        s1med = np.median(s1, axis = 0)
                        ard_median = np.concatenate([s2med, dem[..., np.newaxis], s1med], axis = -1)
                        print(f"ARD is {ard_median.shape} shape")
                        hkl.dump(ard_median,
                                  f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl", 
                                  mode='w',
                                  compression='gzip')
                        key = f'{str(year)}/ard/{x}/{y}/{str(x)}X{str(y)}Y_ard.hkl'
                        uploader.upload(bucket = args.s3_bucket, key = key, 
                          file = f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
                      if args.process is True:
                          predictions = load_mosaic_predictions(path_to_tile + "processed/", depth = 1)
                          if args.gen_feats:
                              features = load_mosaic_predictions(path_to_tile + "feats/", depth = 64)
                              if not os.path.exists(os.path.realpath(f"{path_to_tile}raw/feats/")):
                                  os.makedirs(os.path.realpath(f"{path_to_tile}raw/feats/"))
                              if not os.path.exists(os.path.realpath(f"{path_to_tile}ard/")):
                                  os.makedirs(os.path.realpath(f"{path_to_tile}ard/"))
                              predictions = np.int16(predictions)
                              features = np.concatenate([predictions[np.newaxis], features], axis = 0)

                              s2med = np.median(s2, axis = 0)
                              s1med = np.median(s1, axis = 0)
                              ard_median = np.concatenate([s2med, dem[..., np.newaxis], s1med], axis = -1)
                              print(f"ARD is {ard_median.shape} shape")

                              hkl.dump(ard_median,
                                  f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl", 
                                  mode='w',
                                  compression='gzip')

                              print(f"Features are {features.shape} shape")
                              hkl.dump(features,
                                  f"{path_to_tile}raw/feats/{str(x)}X{str(y)}Y_feats.hkl", 
                                  mode='w',
                                  compression='gzip')

                              key = f'{str(year)}/ard/{x}/{y}/{str(x)}X{str(y)}Y_ard.hkl'
                              uploader.upload(bucket = args.s3_bucket, key = key, 
                                  file = f"{path_to_tile}ard/{str(x)}X{str(y)}Y_ard.hkl")
                          #if args.make_training_data:
                            # At the equator, a tile is 618 height 618 width
                            #predsnew = np.copy(predictions).T
                            #y_train = predsnew[PX_y - 38:PX_y + 38, PX_x - 38:PX_x + 38]
                            #outname = f"train-ard-y/{str(PLOTID)}.npy"
                            #np.save(f"train-ard-y/{str(PLOTID)}.npy", y_train*2.5)
                            #key = f'train-ard-y/{str(PLOTID)}.npy'
                            #uploader.upload(bucket = args.s3_bucket, key = key, file = outname)
                     
                          file = write_tif(predictions, bbx, x, y, path_to_tile)
                          key = f'{str(year)}/tiles/{x}/{y}/{str(x)}X{str(y)}Y_FINAL.tif'
                          uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                          
                      if args.ul_flag:
                        upload_raw_processed_s3(path_to_tile, x, y, uploader, year, args.no_cleanup)
                      if int(x) == 0:
                        print(f"DELETING {path_to_tile}")
                        shutil.rmtree(path_to_tile)

                      print(f"Finished {n}/{n_to_process} in {np.around(time.time() - time0, 1)}"
                           f" seconds, total of {exception_counter} exceptions")
                      n += 1
                      predictions = None
                      s2 = None
                      dates = None
                      interp = None
                      s1 = None
                      dem = None
                
                except Exception as e:
                    exception_counter += 1
                    #path_to_tile = f'{args.local_path}/{str(x)}/{str(y)}/'
                    #shutil.rmtree(path_to_tile)
                    print(f"Ran into {str(e)} error, skipping {x}/{y}/")
                    traceback.print_exc()
                    s2 = None
                    dates = None
                    interp = None
                    s1 = None
                    dem = None
                    time.sleep(10 + (exception_counter * 5))
                    continue
                
            else:
                print(f'Skipping {x}, {y} as it is done')

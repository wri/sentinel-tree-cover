import pandas as pd
import numpy as np
from random import shuffle
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
from sentinelhub.config import SHConfig
import logging
import datetime
import os
import yaml
from sentinelhub import DataSource
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from skimage.transform import resize
from sentinelhub import CustomUrlParam
import multiprocessing
import math
import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc
import hickle as hkl
from tqdm import tnrange, tqdm_notebook
import boto3
from typing import Tuple, List
import warnings
from scipy.ndimage import median_filter
import time
import copy
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from glob import glob
import rasterio
from rasterio.transform import from_origin

from preprocessing import slope
from preprocessing import indices
from downloading.utils import tile_window, calculate_and_save_best_images, calculate_proximal_steps
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from tof import tof_downloading
from tof.tof_downloading import to_int16, to_float32
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder
from preprocessing.indices import evi, bi, msavi2, grndvi

#tf.config.optimizer.set_jit(True)
tf.disable_v2_behavior()

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
SIZE = 216


def superresolve_tile(arr: np.ndarray, sess) -> np.ndarray:
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
    to_resolve = np.pad(arr, ((0, 0), (4, 4), (4, 4), (0, 0)), 'reflect')

    bilinear = to_resolve[..., 4:]
    resolved = sess.run([superresolve_logits],
                 feed_dict={superresolve_inp: to_resolve,
                            superresolve_inp_bilinear: bilinear})[0]
    resolved = resolved[:, 4:-4, 4:-4, :]
    arr[..., 4:] = resolved
    return arr


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
    wsize = 110
    step = 110
    x_range = [x for x in range(0, arr.shape[1] - (wsize), step)] + [arr.shape[1] - wsize]
    y_range = [x for x in range(0, arr.shape[2] - (wsize), step)] + [arr.shape[2] - wsize]
    x_end = np.copy(arr[:, x_range[-1]:, ...])
    y_end = np.copy(arr[:, :, y_range[-1]:, ...])
    print(f"There are {len(x_range)*len(y_range)} tiles to supres")
    time1 = time.time()
    for x in x_range:
        for y in y_range:
            if x != x_range[-1] and y != y_range[-1]:
                to_resolve = arr[:, x:x+wsize, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = superresolve_tile(to_resolve, sess)
            # The end x and y subtiles need to be done separately
            # So that a partially resolved tile isnt served as input
            elif x == x_range[-1]:
                to_resolve = x_end[:, :, y:y+wsize, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = superresolve_tile(to_resolve, sess)
            elif y != y_range[-1]:
                to_resolve = y_end[:, x:x+wsize, :, ...]
                arr[:, x:x+wsize, y:y+wsize, ...] = superresolve_tile(to_resolve, sess)

    time2 = time.time()
    print(f"Finished superresolve in {np.around(time2 - time1, 1)} seconds")
    return arr


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


def download_s1_tile(data: np.ndarray, bbx: list, api_key, year: int,
                     dates_sentinel_1: list, size: tuple,
                     s1_file: str, s1_dates_file: str) -> None:
    """Downloads the sentinel 1 data for a tile

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
    s1_layer = tof_downloading.identify_s1_layer((data['Y'][0], data['X'][0]))
    s1, s1_dates = tof_downloading.download_sentinel_1_composite(bbx,
                                       layer = s1_layer,
                                       api_key = api_key,
                                       year = year,
                                       dates = dates_sentinel_1,
                                       size = size,
                                       )

    if s1.shape[0] == 0: # If the first attempt receives no images, swap orbit
        s1_layer = "SENT_DESC" if s1_layer == "SENT" else "SENT_DESC"
        print(f'Switching to {s1_layer}')
        s1, s1_dates = tof_downloading.download_sentinel_1_composite(bbx,
                                           layer = s1_layer,
                                           api_key = api_key,
                                           year = year,
                                           dates = dates_sentinel_1,
                                           size = size,
                                           )
                                           #config = api_key)
    # Convert s1 to monthly mosaics, and write to disk
    s1 = tof_downloading.process_sentinel_1_tile(s1, s1_dates)
    hkl.dump(to_int16(s1), s1_file, mode='w', compression='gzip')
    hkl.dump(s1_dates, s1_dates_file, mode='w', compression='gzip')


def download_tile(x: int, y: int, data: pd.DataFrame, api_key, year) -> None:
    """Downloads the data for an input x, y tile centroid
       including:
        - Clouds
        - Cloud shadows
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
    data = data[data['Y_tile'] == int(y)]
    data = data[data['X_tile'] == int(x)]
    data = data.reset_index(drop = True)
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    initial_bbx = [data['X'][0], data['Y'][0], data['X'][0], data['Y'][0]]

    cloud_bbx = make_bbox(initial_bbx, expansion = 3000/30)
    bbx = make_bbox(initial_bbx, expansion = 300/30)
    dem_bbx = make_bbox(initial_bbx, expansion = 301/30)

    folder = f"{args.local_path}{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'

    make_output_and_temp_folders(folder)
    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'

    if not (os.path.exists(clouds_file)):
        print(f"Downloading {clouds_file}")

        # Identify images with <30% cloud cover
        cloud_probs, clean_steps, image_dates = tof_downloading.identify_clouds_big_bbx(cloud_bbx = cloud_bbx,
                                                            shadow_bbx = cloud_bbx,
                                                            dates = dates,
                                                            api_key = api_key,
                                                            year = year)
        cloud_shadows = np.mean(cloud_probs, axis = (1, 2)) + np.mean(shadows, axis = (1, 2))
        to_remove = [int(x) for x in np.argwhere(cloud_shadows > 0.75)]
        if len(to_remove) > 0:
            image_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)

        # Remove contiguous dates that are sunny, to reduce IO needs
        cloud_percent = np.mean(cloud_probs > 0.4, axis = (1, 2))
        #cloud_percent = np.mean(cloud_probs, axis = (1, 2)
        to_remove = cloud_removal.subset_contiguous_sunny_dates(image_dates,
                                                               cloud_percent)
        if len(to_remove) > 0:
            clean_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
        else:
            clean_dates = image_dates

        if len(clean_dates) >= 11:
            clean_dates = np.delete(clean_dates, 5)
            cloud_probs = np.delete(cloud_probs, 5, 0)

        _ = cloud_removal.print_dates(
            clean_dates, np.mean(cloud_probs, axis = (1, 2))
        )
        print(f"Overall using {len(clean_dates)}/{len(clean_dates)+len(to_remove)} steps")

        hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
        hkl.dump(clean_dates, clean_steps_file, mode='w', compression='gzip')

    if not (os.path.exists(s2_10_file)):
        print(f"Downloading {s2_10_file}")
        clean_steps = list(hkl.load(clean_steps_file))
        cloud_probs = hkl.load(clouds_file)
        s2_10, s2_20, s2_dates = tof_downloading.download_sentinel_2_new(bbx,
                                                     clean_steps = clean_steps,
                                                     api_key = api_key, dates = dates,
                                                     year = year, maxclouds = 1.0)

        # Ensure that L2A, L1C derived products have exact matching dates
        # As sometimes the L1C data has more dates than L2A if processing bug from provider
        print(f"Clouds {cloud_probs.shape},"
              f" S2, {s2_10.shape}, S2d, {s2_dates.shape}")
        to_remove_clouds = [i for i, val in enumerate(clean_steps) if val not in s2_dates]
        to_remove_dates = [val for i, val in enumerate(clean_steps) if val not in s2_dates]

        hkl.dump(to_int16(s2_10), s2_10_file, mode='w', compression='gzip')
        hkl.dump(to_int16(s2_20), s2_20_file, mode='w', compression='gzip')
        hkl.dump(s2_dates, s2_dates_file, mode='w', compression='gzip')
        size = s2_20.shape[1:3]

    if not (os.path.exists(s1_file)):
        download_s1_tile(data = data,
                         bbx = bbx,
                         api_key = api_key,
                         year = year,
                         dates_sentinel_1 = dates_sentinel_1,
                         size = size,
                         s1_file = s1_file,
                         s1_dates_file = s1_dates_file)

    if not os.path.exists(dem_file):
        print(f'Downloading {dem_file}')
        dem = tof_downloading.download_dem(dem_bbx, api_key = api_key)
        hkl.dump(dem, dem_file, mode='w', compression='gzip')

    return bbx


def adjust_shape(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    print(f"Input array shape: {arr.shape}")
    if len(arr.shape) == 3:
        arr = arr[:, :, :, np.newaxis]

    if len(arr.shape) == 2:
        arr = arr[np.newaxis, :, :, np.newaxis]

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

    arr = arr.squeeze()
    print(f"Output array shape: {arr.shape}")
    return arr


def process_tile(x: int, y: int, data: pd.DataFrame,
                 local_path: str, make_shadow: bool = False) -> np.ndarray:
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

    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    folder = f"{local_path}{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'

    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
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
    s1 = hkl.load(s1_file)
    s1 = s1 / 65535
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)

    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))

    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)
    image_dates = hkl.load(s2_dates_file)

    # Ensure arrays are the same dims
    width = s2_20.shape[1] * 2
    height = s2_20.shape[2] * 2
    s1 = adjust_shape(s1, width, height)
    s2_10 = adjust_shape(s2_10, width, height)
    dem = adjust_shape(dem, width, height)

    print(f'Clouds: {clouds.shape}, \n'
          f'S1: {s1.shape} \nS2: {s2_10.shape}, {s2_20.shape} \nDEM: {dem.shape}')

    # The 20m bands must be bilinearly upsampled to 10m as input to superresolve_tile
    if len(s2_10.shape) == 3:
        s2_10 = s2_10[np.newaxis]
    if len(s2_20.shape) == 3:
        s2_20 = s2_20[np.newaxis]

    sentinel2 = np.empty((s2_10.shape[0], width, height, 10), np.float32)
    sentinel2[..., :4] = s2_10
    for band in range(4):
        for step in range(sentinel2.shape[0]):
            sentinel2[step, ..., band + 4] = resize(
                s2_20[step,..., band], (width, height), 1
            )

    for band in range(4, 6):
        for step in range(sentinel2.shape[0]):
            mid = s2_20[step,..., band]
            if (mid.shape[0] % 2 == 0) and (mid.shape[1] % 2) == 0:
                mid = mid.reshape(mid.shape[0] // 2, 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, ..., band + 4] = resize(mid, (width, height), 1)
            if mid.shape[0] %2 != 0 and mid.shape[1] %2 != 0:
                mid_misaligned_x = mid[0, :]
                mid_misaligned_y = mid[:, 0]
                mid = mid[1:, 1:].reshape(
                    np.int(np.floor(mid.shape[0] / 2)), 2,
                    np.int(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, 1:, band + 4] = resize(mid, (width - 1, height - 1), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned_x.repeat(2)
                sentinel2[step, :, 0, band + 4] = mid_misaligned_y.repeat(2)
            elif mid.shape[0] % 2 != 0:
                mid_misaligned = mid[0, :]
                mid = mid[1:].reshape(np.int(np.floor(mid.shape[0] / 2)), 2, mid.shape[1] // 2, 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, 1:, :, band + 4] = resize(mid, (width - 1, height), 1)
                sentinel2[step, 0, :, band + 4] = mid_misaligned.repeat(2)
            elif mid.shape[1] % 2 != 0:
                mid_misaligned = mid[:, 0]
                mid = mid[:, 1:]
                mid = mid.reshape(mid.shape[0] // 2, 2, np.int(np.floor(mid.shape[1] / 2)), 2)
                mid = np.mean(mid, axis = (1, 3))
                sentinel2[step, :, 1:, band + 4] = resize(mid, (width, height - 1), 1)
                sentinel2[step, :, 0, band + 4] = mid_misaligned.repeat(2)

    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)
    if len(missing_px) > 0:
        print(f"Removing {missing_px} dates due to missing data")
        clouds = np.delete(clouds, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)

    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)
    #np.save("before_shadow_interp,npy", sentinel2[:, :222, :222, 0])
    if make_shadow:
        time1 = time.time()
        cloudshad = cloud_removal.remove_missed_clouds(sentinel2)
        #np.save("cloudshad.npy", cloudshad)
        sentinel2, interp = cloud_removal.remove_cloud_and_shadows(
            sentinel2, cloudshad, cloudshad, image_dates, wsize = 8, step = 8, thresh = 8
        )
        #np.save("interp_bef.npy", interp)
        time2 = time.time()
        print(f"Cloud/shadow interp:{np.around(time2 - time1, 1)} seconds, "
            f" {100*np.sum(interp > 0)/np.prod(interp.shape)}%")
       # np.save("after.npy", sentinel2)
        #np.save("interp.npy", interp)
    else:
        interp = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )
        cloudshad = np.zeros(
            (sentinel2.shape[0], sentinel2.shape[1], sentinel2.shape[2]), dtype = np.float32
        )

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    return sentinel2, image_dates, interp, s1, dem, cloudshad

def normalize_first_last_quarter(arr, dates):

    """
    dates_first_third = np.argwhere(dates < 90)
    dates_second_third = np.argwhere(np.logical_and(dates >= 90, dates < 180))
    dates_last_third = np.argwhere(np.logical_and(dates >= 180, dates < 270))
    dates_last_third2 = np.argwhere(dates > 270)

    arr[dates_first_third] = np.mean(arr[dates_first_third], axis = 0)
    arr[dates_second_third] = np.mean(arr[dates_second_third], axis = 0)
    arr[dates_last_third] = np.mean(arr[dates_last_third], axis = 0)
    arr[dates_last_third2] = np.mean(arr[dates_last_third2], axis = 0)

    dates[dates_first_third] = 45
    dates[dates_second_third] = 135
    dates[dates_last_third] = 225
    dates[dates_last_third2] = 315
    """

    dates_first_quarter = np.argwhere(np.logical_and(dates < 90, dates > -30))
    if len(dates_first_quarter) > 0:
        dates_first_quarter = np.argwhere(np.logical_and(dates < 90, dates > -30))
        arr[0]  = np.mean(arr[dates_first_quarter], axis = 0)

    if len(np.argwhere(dates > 270)) > 0:
        dates_last_quarter = np.argwhere(dates > 270)
        arr[-1] = np.mean(arr[dates_last_quarter], axis = 0)
    #print(np.mean(arr[-1, ..., 3]))
    return arr, dates

def fit_upper_curve(arr, dates):
    upper_curve = np.zeros_like(arr)

    for image in range(arr.shape[0]):
        date = dates[image]
        distances = abs(date - dates)
        bimonthly_window = np.argwhere(np.logical_and(
            distances >= 0, distances <= 60)).flatten()

        candidate = arr[image]
        candidate_min = np.amin(arr[bimonthly_window], axis = 0)
        candidate_max = np.amin(arr[bimonthly_window], axis = 0)
        med = np.median(arr[bimonthly_window], axis = 0)

        candidate[candidate == candidate_min] = med[candidate == candidate_min]
        candidate[candidate == candidate_max] = med[candidate == candidate_max]
        upper_curve[image] = candidate
    return upper_curve



def rolling_mean(arr, dates):
    arr = np.array(arr)
    # go through each date and say
    # if it is the image before a gap > 90 days
    # then carry forward the median of 3 images, rather than one image
    # otherwise, let the algorithm carry forward just that one image
    if arr.shape[0] <= 4:
        mean_arr = np.median(arr, axis = 0)
        arr[0] = mean_arr
        arr[-1] = mean_arr
        return arr



    if arr.shape[0] > 4:

        #rolling_average = np.zeros_like(arr)
        for image in range(arr.shape[0]):
            date = dates[image]
            distances = abs(date - dates)
            bimonthly_window = np.argwhere(np.logical_and(
                distances >= 0, distances <= 60)).flatten()
            distances = ((365 - distances) / 365) ** 3
            print(f"Taking the rolling average of {bimonthly_window} for {image}")
            #if i == 0 or i == mean_arr.shape[0] - 1:
            #    mean_arr[i] = np.median(array_to_mean, axis = 0)
            #else:
            arr[image] = np.average(
                arr[bimonthly_window], axis = 0, weights = distances[bimonthly_window]
            )
           # rolling_average[image] = np.mean(arr[bimonthly_window], axis = 0)

        return arr
        #return arr
    """
    if arr.shape[0] > 4:
        mean_arr = np.zeros_like(arr)
        start = np.arange(0, arr.shape[0] - 2, 1)
        start = np.concatenate([
            np.array([0]),
            start,
            np.full((1,), arr.shape[0] - 2)
        ])
        end = start + 2
        i = 0
        for s, e in zip(start, end):
            distances = dates[s:e]
            distances = abs(dates[i] - distances)
            distances = ((365 - distances) / 365) ** 3
            print(distances, dates[s:e], dates[i])
            array_to_mean = arr[s:e]
            #if i == 0 or i == mean_arr.shape[0] - 1:
            #    mean_arr[i] = np.median(array_to_mean, axis = 0)
            #else:
            mean_arr[i] = np.average(array_to_mean, axis = 0, weights = distances)
            i += 1
        return mean_arr
    """


#def rolling_mean(arr):
#    if arr.shape[0] >= 3:
#        arr[0] = np.median(arr[:3], axis = 0)
#        arr[-1] = np.median(arr[-3:], axis = 0)
#    return arr


def process_subtiles(x: int, y: int, s2: np.ndarray = None,
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None,) -> None:
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
        gap_sess (tf.Session): tensorflow session to use for median predicitons

       Returns:
        None
    '''
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    s2 = interpolation.interpolate_na_vals(s2)
    s2 = np.float32(s2)

    # The tiles_folder references the folder names (w/o boundaries)
    # While the tiles_array references the arrays themselves (w/ boudnaries)
    gap_x = int(np.ceil((s1.shape[1] - SIZE) / 4))
    gap_y = int(np.ceil((s1.shape[2] - SIZE) / 4))
    tiles_folder_x = np.hstack([np.arange(0, s1.shape[1] - SIZE, gap_x), np.array(s1.shape[1] - SIZE)])
    tiles_folder_y = np.hstack([np.arange(0, s1.shape[2] - SIZE, gap_y), np.array(s1.shape[2] - SIZE)])
    print(f'There are: {len(tiles_folder_x) * len(tiles_folder_y)} subtiles')

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
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'

    gap_between_years = False
    t = 0
    sm = Smoother(lmbd = 100, size = 24, nbands = 10, dimx = SIZE + 14, dimy = SIZE + 14, outsize = 12)
    # Iterate over each subitle and prepare it for processing and generate predictions
    for t in range(len(tiles_folder)):
        time1 = time.time()
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]

        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[0], tile_folder[1]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]
        subset = np.copy(s2[:, start_x:end_x, start_y:end_y, :])
        interp_tile = interp[:, start_x:end_x, start_y:end_y]
        dates_tile = np.copy(dates)
        dem_subtile = dem[np.newaxis, start_x:end_x, start_y:end_y]

        time1 = time.time()
        missing_px = interpolation.id_missing_px(subset, 10)

        if len(missing_px) > 0:
            print(np.sum(subset == 0, axis = (1, 2, 3)))
            print(np.sum(subset >= 1, axis = (1, 2, 3)))
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)
            interp_tile = np.delete(interp_tile, missing_px, 0)
            print(f"Removing {len(missing_px)} missing images, leaving {len(dates_tile)} / {len(dates)}")

        if np.sum(subset == 0) > 0:
            subset[subset == 0.] = np.tile(
                np.median(subset, axis = 0)[np.newaxis], (subset.shape[0], 1, 1, 1)
            )[subset == 0]
        if np.sum(subset == 1) > 0:
            subset[subset == 1.] = np.tile(
                np.median(subset, axis = 0)[np.newaxis], (subset.shape[0], 1, 1, 1)
            )[subset == 1]

        missing_px = interpolation.id_missing_px(subset, 100)
        if len(missing_px) > 0:
            print(np.sum(subset == 0, axis = (1, 2, 3)))
            print(np.sum(subset >= 1, axis = (1, 2, 3)))
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)
            interp_tile = np.delete(interp_tile, missing_px, 0)
            print(f"Removing {len(missing_px)} missing images, leaving {len(dates_tile)} / {len(dates)}")
        time2 = time.time()
        print(f"Interpolation in {time2 - time1} seconds")

        #np.save("before.npy", subset)
        subset = cloud_removal.adjust_interpolated_groups(subset, interp_tile)

        #np.save("adjusted.npy", subset)

        #np.save("interp.npy", interp_tile)
        min_clear_images_per_date = np.sum(interp_tile == 0, axis = (0))
        no_images = False
        if np.percentile(min_clear_images_per_date, 50) < 2 or np.percentile(min_clear_images_per_date, 25) < 1:
            no_images = True

        to_remove = np.argwhere(np.sum(np.isnan(subset), axis = (1, 2, 3)) > 0).flatten()
        if len(to_remove) > 0:
            print(f"Removing {to_remove} NA dates")
            dates_tile = np.delete(dates_tile, to_remove)
            subset = np.delete(subset, to_remove, 0)
            interp_tile = np.delete(interp_tile, to_remove, 0)

        # Transition (n, 160, 160, ...) array to (72, 160, 160, ...)
        subtile_copy = np.copy(subset)
        subtile_median = np.median(subtile_copy, axis = 0)
        subtile_median = subtile_median[np.newaxis]

        subtile, dates_tile = normalize_first_last_quarter(subset, dates_tile)
        #np.save("after_mean.npy", subset)
        time1 = time.time()
        try:
            subtile, max_distance = calculate_and_save_best_images(subset, dates_tile)
            #np.save("grid_left.npy", subtile)
            time2 = time.time()
            print(f"Calc and save in {time2 - time1} seconds, {np.sum(np.isnan(subtile))}")

        except:
            # If there are no images for the tile, just make them zeros
            # So that they will be picked up by the no-data flag
            print("Skipping because of no images")
            no_images = True
            subtile = np.zeros((36, end_x-start_x, end_y - start_y, 10), dtype = np.float32)
            dates_tile = [0,]

        output = f"{path}{str(folder_y)}/{str(folder_x)}.npy"
        s1_subtile = s1[:, start_x:end_x, start_y:end_y, :]

        # Pad the corner / edge subtiles within each tile
        if subtile.shape[2] == SIZE + 7:
            pad_u = 7 if start_y == 0 else 0
            pad_d = 7 if start_y != 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (0, 0), (pad_u, pad_d)), 'reflect')
            subtile_median = np.pad(subtile_median, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')

        if subtile.shape[1] == SIZE + 7:
            pad_l = 7 if start_x == 0 else 0
            pad_r = 7 if start_x != 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')
            subtile_median = np.pad(subtile_median, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')

        # Interpolate (whittaker smooth) the array and superresolve 20m to 10m
        time1 = time.time()
        subtile = sm.interpolate_array(subtile)
        time2 = time.time()
        print(f"Smoothing in {time2 - time1} seconds, {np.sum(np.isnan(subtile))}")
        #np.save("smooth.npy", subtile)

        # Concatenate the DEM and Sentinel 1 data
        time1 = time.time()
        subtile_all = np.empty((13, SIZE + 14, SIZE + 14, 13), dtype = np.float32)
        subtile_all[:-1, ..., :10] = subtile
        subtile_all[:, ..., 10] = dem_subtile.repeat(13, axis = 0)
        subtile_all[:-1, ..., 11:] = s1_subtile
        subtile_all[-1, ..., :10] = subtile_median
        subtile_all[-1, ..., 11:] = np.median(s1_subtile, axis = (0))
        time2 = time.time()
        print(f"Concat in {time2 - time1} seconds")

        # Create the output folders for the subtile predictions
        output_folder = "/".join(output.split("/")[:-1])
        if not os.path.exists(os.path.realpath(output_folder)):
            os.makedirs(os.path.realpath(output_folder))

        subtile_all = np.clip(subtile_all, 0, 1)
        assert subtile_all.shape[1] >= 145, f"subtile shape is {subtile_all.shape}"
        assert subtile_all.shape[0] == 13, f"subtile shape is {subtile_all.shape}"

        no_images = True if len(dates_tile) < 2 else no_images
        if no_images:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data"
                f" {np.percentile(min_clear_images_per_date, 10)} clear images")
            preds = np.full((SIZE, SIZE), 255)
        else:
            preds = predict_subtile(subtile_all, sess)
            #np.save("feats.npy", feats)
            time2 = time.time()
            subtile_time = np.around(time2 - time1, 1)
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                f"for: {dates_tile}, {max_distance} max dist, {np.percentile(min_clear_images_per_date, 10)} clear images"
                f" in {subtile_time} seconds")
        np.save(output, preds)


def convert_to_db(x: np.ndarray, min_db: int) -> np.ndarray:
    """ Converts unitless backscatter coefficient
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


def predict_subtile(subtile: np.ndarray, sess: "tf.Sess") -> np.ndarray:
    """ Runs temporal (convGRU + UNET) predictions on a (12, 174, 174, 13) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
         sess (tf.Session): tensorflow session for prediction

        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    """
    if np.sum(subtile) > 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.

        time1 = time.time()
        indices = np.empty((13, subtile.shape[1], subtile.shape[2], 17))
        indices[:, ..., :13] = subtile
        indices[:, ..., 13] = evi(subtile)
        indices[:, ...,  14] = bi(subtile)
        indices[:, ...,  15] = msavi2(subtile)
        indices[:, ...,  16] = grndvi(subtile)
        time2 = time.time()
        print(f"Indices in {time2 - time1} seconds")

        subtile = indices
        subtile = subtile.astype(np.float32)
        subtile = np.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)

        #np.save("tilebottom.npy", subtile)

        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        time1 = time.time()
        preds = sess.run(predict_logits,
                              feed_dict={predict_inp:batch_x,
                                         predict_length:lengths})

        #features = sess.run(feature_extraction,
        #                      feed_dict={predict_inp:batch_x,
        #                                 predict_length:lengths})
        #features = features.squeeze()
        #print(features.shape)
        #np.save("features.npy", features)
        #print("saving features")
        preds = preds.squeeze()
        preds = preds[1:-1, 1:-1]
        time2 = time.time()
        print(f"Preds in {time2 - time1} seconds")

    else:
        preds = np.full((SIZE, SIZE), 255)
        #features = np.full((SIZE, SIZE), 255)

    return preds#, features


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
                    mults[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, 44) # or 44
                i += 1

    predictions = predictions.astype(np.float32)
    predictions_range = np.nanmax(predictions, axis=-1) - np.nanmin(predictions, axis=-1)
    mean_certain_pred = np.nanmean(predictions[predictions_range < 50])
    mean_uncertain_pred = np.nanmean(predictions[predictions_range > 50])

    overpredict = True if (mean_uncertain_pred - mean_certain_pred) > 0 else False
    underpredict = True if not overpredict else False

    mults = mults / np.sum(mults, axis = -1)[..., np.newaxis]
    predictions[predictions > 100] = np.nan
    out = np.copy(predictions)
    out = np.sum(np.isnan(out), axis = (2))
    n_preds = predictions.shape[-1]
    predictions = np.nansum(predictions * mults, axis = -1)
    predictions[out == n_preds] = np.nan
    predictions[np.isnan(predictions)] = 255.
    predictions = predictions.astype(np.uint8)

    """
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
    """
    predictions[predictions <= .20*100] = 0.
    predictions[predictions > 100] = 255.
    return predictions


def download_raw_tile(tile_idx: tuple, local_path: str,
                      subfolder: str = "raw") -> None:
    x = tile_idx[0]
    y = tile_idx[1]

    path_to_tile = f'{local_path}{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/{subfolder}/{str(x)}/{str(y)}/'
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/224-temporal-jan/')
    parser.add_argument("--gap_model_path", dest = 'gap_model_path', default = '../models/182-gap-sept/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/nov-40k-swir/')
    parser.add_argument("--db_path", dest = "db_path", default = "processing_area_june_28.csv")
    parser.add_argument("--ul_flag", dest = "ul_flag", default = False)
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--year", dest = "year", default = 2020)
    parser.add_argument("--n_tiles", dest = "n_tiles", default = None)
    parser.add_argument("--x", dest = "x", default = None)
    parser.add_argument("--y", dest = "y", default = None)
    parser.add_argument("--reprocess", dest = "reprocess", default = False)
    parser.add_argument("--redownload", dest = "redownload", default = False)
    parser.add_argument("--model", dest = "model", default = "temporal")
    args = parser.parse_args()

    print(f'Country: {args.country} \n'
          f'Local path: {args.local_path} \n'
          f'Predict model path: {args.predict_model_path} \n'
          f'Gap model path: {args.gap_model_path} \n'
          f'Superrresolve model path: {args.superresolve_model_path} \n'
          f'DB path: {args.db_path} \n'
          f'S3 Bucket: {args.s3_bucket} \n'
          f'YAML path: {args.yaml_path} \n'
          f'Current dir: {os.getcwd()} \n'
          f'N tiles to download: {args.n_tiles} \n'
          f'Year: {args.year} \n'
          f'X: {args.x} \n'
          f'Y: {args.y} \n'
          f'Reprocess: {args.reprocess} \n'
          f'Redownload: {args.redownload} \n'
          f'Model: {args.model} \n'
          )

    args.year = int(args.year)

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

    if os.path.exists(args.db_path):
        data = pd.read_csv(args.db_path)
        data = data[data['country'] == args.country]
        data = data.reset_index(drop = True)
        data = data.sample(frac=1).reset_index(drop=True)
        print(f"There are {len(data)} tiles for {args.country}")
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
        predict_file = tf.io.gfile.GFile(args.predict_model_path + "predict_graph.pb", 'rb')
        predict_graph_def.ParseFromString(predict_file.read())
        predict_graph = tf.import_graph_def(predict_graph_def, name='predict')
        predict_sess = tf.compat.v1.Session(graph=predict_graph)
        predict_logits = predict_sess.graph.get_tensor_by_name(f"predict/conv2d_13/Sigmoid:0")
        #print([n.name for n in predict_sess.graph.as_graph_def().node])
        feature_extraction = predict_sess.graph.get_tensor_by_name(f"predict/csse_out_mul/mul:0")
        print(feature_extraction.shape)
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")


    # Normalization mins and maxes for the prediction input
    min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 0.013351644159609368, 0.01965362020294499,
               0.014229037918669413, 0.015289539940489814, 0.011993591210803388, 0.008239871824216068, 0.006546120393682765,
               0.0, 0.0, 0.0, -0.1409399364817101, -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]
    max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 0.6027466239414053, 0.5650263218127718,
               0.5747005416952773, 0.5933928435187305, 0.6034943160143434, 0.7472037842374304, 0.7000076295109483,
               0.509269855802243, 0.948334642387533, 0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
               0.7545951919107605, 0.7602693339366691]

    min_all = np.array(min_all)
    max_all = np.array(max_all)
    min_all = np.broadcast_to(min_all, (1, SIZE + 14, SIZE + 14, 17)).astype(np.float32)
    max_all = np.broadcast_to(max_all, (1, SIZE + 14, SIZE + 14, 17)).astype(np.float32)
    midrange = (max_all + min_all) / 2
    midrange = midrange.astype(np.float32)
    rng = max_all - min_all
    rng = rng.astype(np.float32)
    n = 0

    if args.x and args.y:
        x = args.x
        y = args.y
        data = data[data['Y_tile'] == int(y)]
        data = data[data['X_tile'] == int(x)]
        data = data.reset_index(drop = True)
        x = str(int(x))
        y = str(int(y))
    for index, row in data.iterrows():
        x = str(int(row['X_tile']))
        y = str(int(row['Y_tile']))
        x = x[:-2] if ".0" in x else x
        y = y[:-2] if ".0" in y else y
        bbx = None
        year = args.year
        dates = (f'{str(args.year - 1)}-11-15' , f'{str(args.year + 1)}-02-15')
        dates_sentinel_1 = (f'{str(args.year)}-01-01' , f'{str(args.year)}-12-31')
        days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
        starting_days = np.cumsum(days_per_month)

        # Check to see whether the tile exists locally or on s3
        path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/'
        s3_path_to_tile = f'2020/tiles/{str(x)}/{str(y)}/'
        processed = file_in_local_or_s3(path_to_tile,
                                        s3_path_to_tile,
                                        AWSKEY, AWSSECRET,
                                        args.s3_bucket)

        # If the tile does not exist, go ahead and download/process/upload it
        if not processed or args.reprocess:
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
                    initial_bbx = [data2['X'][0], data2['Y'][0], data2['X'][0], data2['Y'][0]]
                    bbx = make_bbox(initial_bbx, expansion = 300/30)
                time0 = time.time()
                if not args.redownload:
                    time1 = time.time()
                    bbx = download_tile(x = x, y = y, data = data, api_key = shconfig, year = args.year)
                    time2 = time.time()
                    print(f"Finished downloading imagery in {np.around(time2 - time0, 1)} seconds")
                else:
                    download_raw_tile((x, y), args.local_path, "raw")
                    folder = f"{args.local_path}{str(x)}/{str(y)}/"
                    tile_idx = f'{str(x)}X{str(y)}Y'
                    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
                    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
                    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
                    size = hkl.load(s2_20_file)
                    size = size.shape[1:3]

                s2, dates, interp, s1, dem, cloudshad = process_tile(x = x, y = y, data = data, local_path = args.local_path, make_shadow = True)
                s2 = superresolve_large_tile(s2, superresolve_sess)

                time1 = time.time()
                process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess)
                time2 = time.time()
                print(f"Finished processing subtiles in {np.around(time2 - time1, 1)} seconds")

                time1 = time.time()
                predictions = load_mosaic_predictions(path_to_tile + "processed/")
                time2 = time.time()
                print(f"Finished making tif in {np.around(time2 - time1, 1)} seconds")


                file = write_tif(predictions, bbx, x, y, path_to_tile)
                key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_FINAL.tif'
                uploader.upload(bucket = args.s3_bucket, key = key, file = file)
                if args.ul_flag:
                    upload_raw_processed_s3(path_to_tile, x, y, uploader)
                time2 = time.time()
                print(f"Finished {n} in {np.around(time2 - time0, 1)} seconds")
                n += 1
            except Exception as e:
                print(f"Ran into {str(e)} error, skipping {x}/{y}/")
                time.sleep(10)
                continue
        else:
            print(f'Skipping {x}, {y} as it is done')

import pandas as pd
import numpy as np
from random import shuffle
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
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
import tensorflow as tf
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

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

SIZE = 168


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


def id_iqr_outliers(arr):
    if arr.shape[0] > 6:
        lower_qr = np.percentile(arr, 25, axis = 0)
        upper_qr = np.percentile(arr, 75, axis = 0)
        iqr = (upper_qr - lower_qr) * 2
        lower_thresh = lower_qr - iqr
        upper_thresh = upper_qr + iqr        
        return lower_thresh, upper_thresh
    else:
        return None, None

        

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
        cloud_probs, shadows, _, image_dates, shadow_img = tof_downloading.identify_clouds(cloud_bbx = bbx, 
                                                            shadow_bbx = bbx,
                                                            dates = dates,
                                                            api_key = api_key, 
                                                            year = year)
        cloud_shadows = np.mean(cloud_probs, axis = (1, 2)) + np.mean(shadows, axis = (1, 2))
        to_remove = [int(x) for x in np.argwhere(cloud_shadows > 0.5)]
        if len(to_remove) > 0:
            print(f"Removing {len(to_remove)} timesteps with > 0.5 cloud/shadow extent")
            image_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)

        to_remove, _ = cloud_removal.calculate_cloud_steps(cloud_probs, image_dates)

        # Remove cloudy images
        if len(to_remove) > 0:
            clean_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)
        else:
            clean_dates = image_dates
        
        cloud_shadows = np.mean(cloud_probs, axis = (1, 2)) + np.mean(shadows, axis = (1, 2))
        # Remove contiguous dates that are sunny, to reduce IO needs
        to_remove = cloud_removal.subset_contiguous_sunny_dates(clean_dates,
                                                               cloud_shadows)
        # Remove the cloudiest date if at least 15 images
        n_remaining = (len(clean_dates) - len(to_remove))
        cloud_shadows[to_remove] = 0.
        if n_remaining >= 13 or np.max(cloud_shadows) >= 0.20:
            cloud_shadows[to_remove] = 0.
            max_cloud = int(np.argmax(cloud_shadows))
            print(f"Removing cloudiest date: {max_cloud}, {cloud_shadows[max_cloud]}")
            to_remove.append(max_cloud)

        if len(to_remove) > 0:
            clean_dates = np.delete(clean_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)
            cloud_shadows = np.delete(cloud_shadows, to_remove, 0)

        if len(clean_dates) >= 13:
            max_cloud = int(np.argmax(cloud_shadows))
            print(f"Removing cloudiest date: {max_cloud}, {cloud_shadows[max_cloud]}")
            clean_dates = np.delete(clean_dates, max_cloud)
            cloud_probs = np.delete(cloud_probs, max_cloud, 0)
            shadows = np.delete(shadows, max_cloud, 0)
            cloud_shadows = np.delete(cloud_shadows, max_cloud, 0)

        cloud_removal.print_dates(clean_dates, cloud_shadows)

        print(f"Overall using {len(clean_dates)}/{len(clean_dates)+len(to_remove)} steps")
        print(clean_dates)

        hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
        hkl.dump(shadows, shadows_file, mode='w', compression='gzip')
        hkl.dump(clean_dates, clean_steps_file, mode='w', compression='gzip')
            
    if not (os.path.exists(s2_10_file)):
        print(f"Downloading {s2_10_file}")
        clean_steps = list(hkl.load(clean_steps_file))
        cloud_probs = hkl.load(clouds_file)
        shadows = hkl.load(shadows_file)    
        s2_10, s2_20, s2_dates = tof_downloading.download_sentinel_2(bbx,
                                                     clean_steps = clean_steps,
                                                     api_key = api_key, dates = dates,
                                                     year = year)

        # Ensure that L2A, L1C derived products have exact matching dates
        # As sometimes the L1C data has more dates than L2A if processing bug from provider
        print(f"Shadows {shadows.shape}, clouds {cloud_probs.shape},"
              f" S2, {s2_10.shape}, S2d, {s2_dates.shape}")
        to_remove_clouds = [i for i, val in enumerate(clean_steps) if val not in s2_dates]
        to_remove_dates = [val for i, val in enumerate(clean_steps) if val not in s2_dates]
        if len(to_remove_clouds) >= 1:
            print(f"Removing {to_remove_dates} from clouds because not in S2")
            cloud_probs = np.delete(cloud_probs, to_remove_clouds, 0)
            shadows = np.delete(shadows, to_remove_clouds, 0)
            hkl.dump(cloud_probs, clouds_file, mode='w', compression='gzip')
            hkl.dump(shadows, shadows_file, mode='w', compression='gzip')

        assert cloud_probs.shape[0] == s2_10.shape[0], "There is a date mismatch"
        hkl.dump(to_int16(s2_10), s2_10_file, mode='w', compression='gzip')
        hkl.dump(to_int16(s2_20), s2_20_file, mode='w', compression='gzip')
        hkl.dump(s2_dates, s2_dates_file, mode='w', compression='gzip')
            
    if not (os.path.exists(s1_file)):
        print(f"Downloading {s1_file}")
        s1_layer = tof_downloading.identify_s1_layer((data['Y'][0], data['X'][0]))
        s1, s1_dates = tof_downloading.download_sentinel_1(bbx,
                                           layer = s1_layer,
                                           api_key = api_key,
                                           year = year,
                                           dates = dates_sentinel_1)
        if s1.shape[0] == 0: # If the first attempt receives no images, swap orbit
            s1_layer = "SENT_DESC" if s1_layer == "SENT" else "SENT"
            print(f'Switching to {s1_layer}')
            s1, s1_dates = tof_downloading.download_sentinel_1(bbx,
                                               layer = s1_layer,
                                               api_key = api_key,
                                               year = year,
                                               dates = dates_sentinel_1)
        # Convert s1 to monthly mosaics, and write to disk
        s1 = tof_downloading.process_sentinel_1_tile(s1, s1_dates)
        hkl.dump(to_int16(s1), s1_file, mode='w', compression='gzip')
        hkl.dump(s1_dates, s1_dates_file, mode='w', compression='gzip')
                
    if not os.path.exists(dem_file):
        print(f'Downloading {dem_file}')
        dem = tof_downloading.download_dem(dem_bbx, api_key = API_KEY)
        hkl.dump(dem, dem_file, mode='w', compression='gzip')
    return bbx


def process_tile(x: int, y: int, data: pd.DataFrame, local_path) -> np.ndarray:
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
    shadows = hkl.load(shadows_file)
    s1 = hkl.load(s1_file)

    # The S1 data here needs to be bilinearly upsampled as it is in training time! 
    s1 = s1.reshape((s1.shape[0], s1.shape[1] // 2, 2, s1.shape[2] // 2, 2, 2))
    s1 = np.mean(s1, (2, 4))
    s1 = resize(s1, (s1.shape[0], s1.shape[1] * 2, s1.shape[2] * 2, 2), order = 1)
    s1 = s1 / 65535
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)
    
    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))


    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)
    image_dates = hkl.load(s2_dates_file)
    
    # The below code is somewhat ugly, but it is geared to ensure that the
    # Different data sources are all the same shape, as they are downloaded
    # with varying resolutions (10m, 20m, 60m, 160m)
    width = s2_10.shape[1]
    height = s2_20.shape[2] * 2
    
    if clouds.shape[1] < width:
        pad_amt =  (width - clouds.shape[1]) // 2
        clouds = np.pad(clouds, ((0, 0), (pad_amt, pad_amt), (0,0)), 'edge')
        
    if shadows.shape[1] < width:
        pad_amt =  (width - shadows.shape[1]) // 2
        shadows = np.pad(shadows, ((0, 0), (pad_amt, pad_amt), (0,0)), 'edge')
        
    if dem.shape[0] < width:
        pad_amt =  (width - dem.shape[0]) // 2
        dem = np.pad(dem, ((pad_amt, pad_amt), (0, 0)), 'edge')
        
    if s2_10.shape[2] < height:
        pad_amt =  (height - s2_10.shape[2]) / 2
        if pad_amt % 2 == 0:
            pad_amt = int(pad_amt)
            s2_10 = np.pad(s2_10, ((0, 0), (0, 0), (pad_amt, pad_amt), (0,0)), 'edge')
        else:
            s2_10 = np.pad(s2_10, ((0, 0), (0, 0), (0, int(pad_amt * 2)), (0,0)), 'edge')
    
    if s2_10.shape[2] > height:
        pad_amt =  abs(height - s2_10.shape[2])
        s2_10 = s2_10[:, :, :-pad_amt, :]
        print(s2_10.shape)
       
    if dem.shape[1] < height:
        pad_amt =  (height - dem.shape[1]) / 2
        if pad_amt % 2 == 0:
            pad_amt = int(pad_amt)
            dem = np.pad(dem, ((0, 0), (pad_amt, pad_amt)), 'edge')
        else:
            dem = np.pad(dem, ( (0, 0), (0, int(pad_amt * 2))), 'edge')
            
    if dem.shape[1] > height:
        pad_amt =  abs(height - dem.shape[1])
        dem = dem[:, :-pad_amt]
        
    print(f'Clouds: {clouds.shape}, \nShadows: {shadows.shape} \n'
          f'S1: {s1.shape} \nS2: {s2_10.shape}, {s2_20.shape} \nDEM: {dem.shape}')
            
    # The 20m bands must be bilinearly upsampled to 10m as input to superresolve_tile
    #! TODO: Parallelize this function such that
         # sentinel2 = np.reshape(sentinel2, sentinel2.shape[0]*sentinel2.shape[-1], width, height)
         # parallel_apply_along_axis(resize, sentinel2, 0)
         # sentinel2 = np.reshape(sentinel2, ...)
    sentinel2 = np.empty((s2_10.shape[0], width, height, 10))
    sentinel2[..., :4] = s2_10
    for band in range(6):
        for time in range(sentinel2.shape[0]):
            sentinel2[time, ..., band + 4] = resize(s2_20[time,..., band], (width, height), 1)


    lower_thresh, upper_thresh = id_iqr_outliers(sentinel2)
    if lower_thresh is not None and upper_thresh is not None:
        above = np.sum(sentinel2 > upper_thresh, axis = (1, 2))
        below = np.sum(sentinel2 < lower_thresh, axis = (1, 2))
        probs = above + below
        n_bands_outlier = (np.sum(probs > (0.5 * sentinel2.shape[1] * sentinel2.shape[2]), axis = (1)))
        print(n_bands_outlier)
        to_remove = np.argwhere(n_bands_outlier >= 1)
        if len(to_remove) > 0:
            print(f"Removing {to_remove} dates due to IQR threshold")
            clouds = np.delete(clouds, to_remove, axis = 0)
            shadows = np.delete(shadows, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)


    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)

    if len(missing_px) > 0:
        print(f"Removing {missing_px} dates due to missing data")
        clouds = np.delete(clouds, missing_px, axis = 0)
        shadows = np.delete(shadows, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)

    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)

    # interpolate cloud and cloud shadows linearly
    sentinel2, interp = cloud_removal.remove_cloud_and_shadows(sentinel2, clouds, shadows, image_dates)
    to_remove_interp = np.argwhere(np.sum(interp, axis = (1, 2)) > (sentinel2.shape[1] * sentinel2.shape[2] * 0.5) ).flatten()

    if len(to_remove_interp > 0):
        print(f"Removing: {to_remove_interp}")
        sentinel2 = np.delete(sentinel2, to_remove_interp, 0)
        image_dates = np.delete(image_dates, to_remove_interp)
        interp = np.delete(interp, to_remove_interp, 0)

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    return sentinel2, image_dates, interp, s1, dem
    

def process_subtiles(x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None,
                       gap_sess = None) -> None:
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
    sm = Smoother(lmbd = 800, size = 72, nbands = 10, dim = SIZE + 14)
    n_median = 0
    median_thresh = 5
    # Iterate over each subitle and prepare it for processing and generate predictions
    while t < len(tiles_folder):
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]
        t += 1
        
        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[0], tile_folder[1]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]
        subset = s2[:, start_x:end_x, start_y:end_y, :]
        interp_tile = interp[:, start_x:end_x, start_y:end_y]
        interp_tile_sum = np.sum(interp_tile, axis = (1, 2))
        dates_tile = np.copy(dates)
        dem_subtile = dem[np.newaxis, start_x:end_x, start_y:end_y]

        # Remove dates with >25% interpolation
        to_remove = np.argwhere(interp_tile_sum > ((SIZE*SIZE) / 4)).flatten()
        
        if len(to_remove) > 0: 
            dates_tile = np.delete(dates_tile, to_remove)
            subset = np.delete(subset, to_remove, 0)

        # Remove dates with >1% missing data
        missing_px = interpolation.id_missing_px(subset, 100)
        if len(missing_px) > 0:
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)
            print(f"Removing {len(missing_px)} missing images, leaving {len(dates_tile)} / {len(dates)}")

        # Remove dates with high likelihood of missed cloud or shadow (false negatives)
        to_remove = cloud_removal.remove_missed_clouds(subset)
        if len(to_remove) > 0:
            subset = np.delete(subset, to_remove, axis = 0)
            dates_tile = np.delete(dates_tile, to_remove)
            print(f"Removing {to_remove} missed clouds, leaving {len(dates_tile)} / {len(dates)}")

        # Transition (n, 160, 160, ...) array to (72, 160, 160, ...)
        no_images = False
        try:
            subtile, max_distance = calculate_and_save_best_images(subset, dates_tile)
        except:
            # If there are no images for the tile, just make them zeros
            # So that they will be picked up by the no-data flag
            print("Skipping because of no images")
            no_images = True
            subtile = np.zeros((72, end_x-start_x, end_y - start_y, 10))
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

        if subtile.shape[1] == SIZE + 7:
            pad_l = 7 if start_x == 0 else 0
            pad_r = 7 if start_x != 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')

        # Interpolate (whittaker smooth) the array and superresolve 20m to 10m
        subtile = sm.interpolate_array(subtile)
        #subtile_stc = make_stc(subtile, superresolve_sess, dem_subtile, s1_subtile)
        subtile_s2 = superresolve_tile(subtile, sess = superresolve_sess)

        # Concatenate the DEM and Sentinel 1 data
        subtile = np.empty((12, SIZE + 14, SIZE + 14, 13))
        subtile[..., :10] = subtile_s2
        subtile[..., 10] = dem_subtile.repeat(12, axis = 0)
        subtile[..., 11:] = s1_subtile
        
        # Create the output folders for the subtile predictions
        output_folder = "/".join(output.split("/")[:-1])
        if not os.path.exists(os.path.realpath(output_folder)):
            os.makedirs(os.path.realpath(output_folder))
        
        subtile = np.clip(subtile, 0, 1)
        assert subtile.shape[1] >= 145, f"subtile shape is {subtile.shape}"
        assert subtile.shape[0] == 12, f"subtile shape is {subtile.shape}"

        # Select between temporal and median models for prediction, based on simple logic:
        # If the first image is after June 15 or the last image is before July 15
        # or the maximum gap is >270 days or < 5 images --- then do median, otherwise temporal
        no_images = True if len(dates_tile) < 3 else no_images
        if no_images:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data")
            preds = np.full((SIZE, SIZE), 255)
        else:
            #! TODO: or if start - end is > some % threshold difference for EVI suggesting deforestation
            if dates_tile[0] >= 150 or dates_tile[-1] <= 215 or max_distance > 265 or args.model == "median" or len(dates_tile) < 5:
                if args.model != 'time':
                    n_median += 1
                    print(f"There are {n_median}/{median_thresh} medians in tile")
                    if not gap_between_years and n_median >= median_thresh:
                        if t > n_median:
                            print("Restarting the predictions with median")
                            t = 0 if t > 1 else t
                        gap_between_years = True
            #if len(dates_tile) == 3:
            #    print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
            #        f" median, {max_distance} max dist")
            #    preds = predict_stc(subtile, gap_sess)
            if (len(dates_tile) < 5 or gap_between_years) or args.model == "median":
                # Then run the median prediction
                print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                    f" median, {max_distance} max dist")
                preds = predict_gap(subtile, gap_sess)
            else:
                # Otherwise run the non-median prediction
                print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                    f" time series, {max_distance} max dist")
                preds = predict_subtile(subtile, sess)
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
 

def predict_subtile(subtile, sess) -> np.ndarray:
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

        indices = np.empty((13, subtile.shape[1], subtile.shape[2], 17))
        indices[:12, ..., :13] = subtile
        indices[:12, ..., 13] = evi(subtile)
        indices[:12, ...,  14] = bi(subtile)
        indices[:12, ...,  15] = msavi2(subtile)
        indices[:12, ...,  16] = grndvi(subtile)
        indices[-1] = np.median(indices[:12], axis = 0)

        subtile = indices
        subtile = subtile.astype(np.float32)
        subtile = np.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        
        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        preds = sess.run(predict_logits,
                              feed_dict={predict_inp:batch_x, 
                                         predict_length:lengths})
        preds = preds.squeeze()
        preds = preds[1:-1, 1:-1]
        
    else:
        preds = np.full((SIZE, SIZE), 255)
    
    return preds


def predict_gap(subtile, sess) -> np.ndarray:
    """ Runs non-temporal predictions on a (12, 174, 174, 13) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
                               that will be median aggregated for model input
         sess (tf.Session): tensorflow session for prediction
    
        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    """
    
    if np.sum(subtile) > 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.
        
        indices = np.empty((13, subtile.shape[1], subtile.shape[2], 17))
        indices[:12, ..., :13] = subtile
        indices[:12, ..., 13] = evi(subtile)
        indices[:12, ...,  14] = bi(subtile)
        indices[:12, ...,  15] = msavi2(subtile)
        indices[:12, ...,  16] = grndvi(subtile)
        indices[-1] = np.median(indices[:12], axis = 0)

        subtile = indices
        subtile = subtile.astype(np.float32)
        subtile = np.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        subtile = subtile[-1]


        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        preds = sess.run(gap_logits,
                              feed_dict={gap_inp:batch_x})

        preds = preds.squeeze()
        preds = preds[1:-1, 1:-1]
    else:
        preds = np.full((SIZE, SIZE), 255)
    
    return preds

def make_stc(subtile, sess, s1_subtile, dem_subtile):

    b2 = subtile[..., 0]
    b3 = subtile[..., 1]
    b4 = subtile[..., 2]
    b8 = subtile[..., 3]
    b8a = subtile[..., 7]
    b11 = subtile[..., 8]
    b12 = subtile[..., 9]

    mndwi = (b3 - b11) / (b3 + b11)
    ndvi = (b8 - b4) / (b8 + b4)
    tcb = ((0.3029 * b2) + (0.2786 * b3) + (0.47 * b4) + (0.5599 * b8a) +\
    (0.508 * b11) + (0.1872 * b12))

    stc = np.nan((SIZE, SIZE, 10))

    first_criteria = np.logical_and(np.mean(mndwi, axis = 0) < -0.55, (ndvi[ndvi_argmax] - np.mean(ndvi, axis = 0)) < 0.05)
    second_criteria = np.logical_and(np.mean(ndvi, axis = 0) < -0.3, (np.mean(mndwi, axis = 0) - np.min(ndvi, axis = 0)) < 0.05)
    third_criteria = np.logical_and(np.mean(ndvi, axis = 0) > 0.6, np.mean(tcb, axis = 0) < 0.45)
    fourth_criteria = np.mean(ndvi, axis = 0) < - 0.2

    # pixels with the max NDVI
    # argmax of ndvi for each pixel
    # argmax 

    stc[first_criteria] = maxndvi
    stc[np.logical_and(second_criteria, np.isnan(stc))] = maxndwi
    stc[np.logical_and(third_criteria, np.isnan(stc))] = maxndvi
    stc[np.logical_and(fourth_criteria, np.isnan(stc))] = maxndwi
    stc[np.isnan(stc)] = maxndvi
    stc = superresolve(stc, sess)
    stc_tile = np.empty((SIZE, SIZE, 13))
    stc_tile[..., :10] = stc
    stc_tile[..., 10] = dem_subtile.repeat(12, axis = 0)
    subtile[..., 11:] = s1_subtile
    return stc



def predict_stc(subtile, sess) -> np.ndarray:
    """ Runs non-temporal predictions on a (12, 174, 174, 13) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
                               that will be median aggregated for model input
         sess (tf.Session): tensorflow session for prediction
    
        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    """
    
    if np.sum(subtile) > 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.
        
        indices = np.empty((subtile.shape[1], subtile.shape[2], 17))
        indices[..., :13] = subtile
        indices[..., 13] = evi(subtile)
        indices[...,  14] = bi(subtile)
        indices[...,  15] = msavi2(subtile)
        indices[...,  16] = grndvi(subtile)

        subtile = indices
        subtile = subtile.astype(np.float32)
        subtile = np.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        subtile = subtile[-1]


        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        preds = sess.run(gap_logits,
                              feed_dict={gap_inp:batch_x})

        preds = preds.squeeze()
        preds = preds[1:-1, 1:-1]
    else:
        preds = np.full((SIZE, SIZE), 255)
    
    return preds


def fspecial_gauss(size, sigma):
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
    print(predictions.shape)
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
                    mults[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, 35)
                i += 1

    predictions = predictions.astype(np.float32)

    predictions_range = np.nanmax(predictions, axis=-1) - np.nanmin(predictions, axis=-1)
    mean_certain_pred = np.nanmean(predictions[predictions_range < 50])
    mean_uncertain_pred = np.nanmean(predictions[predictions_range > 50])
    
    overpredict = True if (mean_uncertain_pred - mean_certain_pred) > 0 else False
    underpredict = True if not overpredict else False
    
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
    #predictions = np.around(predictions / 20, 0) * 20
    predictions[predictions > 100] = 255.
    return predictions


def download_raw_tile(tile_idx, local_path, subfolder = "raw"):
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
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/182-temporal-aug/')
    parser.add_argument("--gap_model_path", dest = 'gap_model_path', default = '../models/182-gap-sept/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/')
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
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']
        print(f"Successfully loaded key from {args.yaml_path}")
        uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET, overwrite = True)
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
    gap_graph_def = tf.compat.v1.GraphDef()

    if os.path.exists(args.superresolve_model_path):
        print(f"Loading model from {args.superresolve_model_path}")
        superresolve_file = tf.io.gfile.GFile(args.superresolve_model_path + "superresolve_graph.pb", 'rb')
        superresolve_graph_def.ParseFromString(superresolve_file.read())
        superresolve_graph = tf.import_graph_def(superresolve_graph_def, name='superresolve')
        superresolve_sess = tf.compat.v1.Session(graph=superresolve_graph)
        superresolve_logits = superresolve_sess.graph.get_tensor_by_name("superresolve/Add_6:0")
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
        predict_logits = predict_sess.graph.get_tensor_by_name(f"predict/conv2d_8/Sigmoid:0")            
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")

    if os.path.exists(args.gap_model_path):
        print(f"Loading gap model from {args.gap_model_path}")
        gap_file = tf.io.gfile.GFile(args.gap_model_path + "gap_graph.pb", 'rb')
        gap_graph_def.ParseFromString(gap_file.read())
        gap_graph = tf.import_graph_def(gap_graph_def, name='gap')
        gap_sess = tf.compat.v1.Session(graph=gap_graph)
        gap_logits = gap_sess.graph.get_tensor_by_name(f"gap/conv2d_13/Sigmoid:0")             # CONV2d_8 is master model
        gap_inp = gap_sess.graph.get_tensor_by_name("gap/Placeholder:0")
    else:
        raise Exception(f"The model path {args.gap_model_path} does not exist")

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
    min_all = np.broadcast_to(min_all, (13, SIZE + 14, SIZE + 14, 17)).astype(np.float32)
    max_all = np.broadcast_to(max_all, (13, SIZE + 14, SIZE + 14, 17)).astype(np.float32)
    midrange = (max_all + min_all) / 2
    midrange = midrange.astype(np.float32)
    rng = max_all - min_all
    rng = rng.astype(np.float32)
    n = 0

    # If downloading an individually indexed tile, go ahead and execute this code block
    if args.x and args.y:
        print(f"Downloading an individual tile: {args.x}X{args.y}Y")
        x = args.x
        y = args.y

        
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
        if processed or args.reprocess:
            if args.n_tiles:
                below = n <= int(args.n_tiles)
            else:
                below = True
            if below:
                bbx = None
                time1 = time.time()
                bbx = download_tile(x = x, y = y, data = data, api_key = API_KEY, year = args.year)
                s2, dates, interp, s1, dem = process_tile(x = x, y = y, data = data, local_path = args.local_path)
                process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, gap_sess)
                predictions = load_mosaic_predictions(path_to_tile + "processed/")
                if not bbx:
                    data = data[data['Y_tile'] == int(y)]
                    data = data[data['X_tile'] == int(x)]
                    data = data.reset_index(drop = True)
                    x = str(int(x))
                    y = str(int(y))
                    x = x[:-2] if ".0" in x else x
                    y = y[:-2] if ".0" in y else y
                        
                    initial_bbx = [data['X'][0], data['Y'][0], data['X'][0], data['Y'][0]]
                    bbx = make_bbox(initial_bbx, expansion = 300/30)

                file = write_tif(predictions, bbx, x, y, path_to_tile)
                key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_FINAL.tif'
                uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                if args.ul_flag:
                    upload_raw_processed_s3(path_to_tile, x, y, uploader)
                time2 = time.time()
                print(f"Finished {n} in {np.around(time2 - time1, 1)} seconds")
                n += 1
    # If downloading all tiles for a country, go ahead and execute this code block
    else:
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
                if args.n_tiles:
                    below = n <= int(args.n_tiles)
                else:
                    below = True
                if below:
                    try:
                        time1 = time.time()
                        if not args.redownload:
                            bbx = download_tile(x = x, y = y, data = data, api_key = API_KEY, year = args.year)
                        else:
                            download_raw_tile((x, y), args.local_path, "raw")

                        s2, dates, interp, s1, dem = process_tile(x = x, y = y, data = data, local_path = args.local_path)
                        process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, gap_sess)
                        predictions = load_mosaic_predictions(path_to_tile + "processed/")
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

                        file = write_tif(predictions, bbx, x, y, path_to_tile)
                        key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_FINAL.tif'
                        uploader.upload(bucket = args.s3_bucket, key = key, file = file)
                        if args.ul_flag:
                            upload_raw_processed_s3(path_to_tile, x, y, uploader)
                        time2 = time.time()
                        print(f"Finished {n} in {np.around(time2 - time1, 1)} seconds")
                        n += 1
                    except Exception as e:
                       print(f"Ran into {str(e)} error, skipping {x}/{y}/")
                       time.sleep(10)
                       continue
            else:
                print(f'Skipping {x}, {y} as it is done')

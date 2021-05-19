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
import geopandas
from tqdm import tnrange, tqdm_notebook
import boto3
from pyproj import Proj, transform
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
from preprocessing.whittaker_smoother import Smoother
from tof import tof_downloading
from tof.tof_downloading import to_int16, to_float32
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles
from models import utils
from preprocessing.indices import evi, bi, msavi2, grndvi

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    

def download_tile(x: int, y: int, data: pd.DataFrame, api_key) -> None:
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
        cloud_probs, shadows, _, image_dates, _ = tof_downloading.identify_clouds(bbox = bbx,
                                                            dates = dates,
                                                            api_key = api_key, 
                                                            year = 2020)

        to_remove, _ = cloud_removal.calculate_cloud_steps(cloud_probs, image_dates)

        # Remove cloudy images
        if len(to_remove) > 0:
            clean_dates = np.delete(image_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)
        else:
            clean_dates = image_dates
        
        # Remove contiguous dates that are sunny, to reduce IO needs
        to_remove = cloud_removal.subset_contiguous_sunny_dates(clean_dates)
        if len(to_remove) > 0:
            clean_dates = np.delete(clean_dates, to_remove)
            cloud_probs = np.delete(cloud_probs, to_remove, 0)
            shadows = np.delete(shadows, to_remove, 0)

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
                                                     year = 2020)

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
                                           year = 2020,
                                           dates = dates_sentinel_1)
        if s1.shape[0] == 0: # If the first attempt receives no images, swap orbit
            s1_layer = "SENT_DESC" if s1_layer == "SENT" else "SENT"
            print(f'Switching to {s1_layer}')
            s1, s1_dates = tof_downloading.download_sentinel_1(bbx,
                                               layer = s1_layer,
                                               api_key = api_key,
                                               year = 2020,
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
    

def id_missing_px(sentinel2: np.ndarray, thresh: int = 11) -> np.ndarray:
    """
    Identifies missing (NA) values in a sentinel 2 array 
    Writes the raw data to the output/x/y folder as .hkl structure

    Parameters:
         sentinel2 (np.ndarray): multitemporal sentinel 2 array
         thresh (int): denominator for threshold (missing < 1 / thresh)

        Returns:
         missing_images (np.ndarray): (N,) array of time steps to remove
                                      due to missing imagery

    """
    missing_images_0 = np.sum(sentinel2[..., :10] == 0.0, axis = (1, 2, 3))
    missing_images_p = np.sum(sentinel2[..., :10] >= 1., axis = (1, 2, 3))
    missing_images = missing_images_0 + missing_images_p
    
    missing_images = np.argwhere(missing_images >= (sentinel2.shape[1]**2) / thresh).flatten()
    return missing_images


def process_tile(x: int, y: int, data: pd.DataFrame) -> np.ndarray:
    """
    Processes raw data structure (in temp/raw/*) to processed data structure
    (in temp/processed/*) including:
        - aligning shapes of different data sources (clouds / shadows / s1 / s2 / dem)
        - superresolution of 20m to 10m with bilinear upsampling
        - removing clouds and shadows

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
            
    folder = f"{args.local_path}{str(x)}/{str(y)}/"
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

    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))
    dem = hkl.load(dem_file)
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
    #! TODO: Investigate whether this can be vectorized rather than a nested for loop
    sentinel2 = np.empty((s2_10.shape[0], width, height, 10))
    sentinel2[..., :4] = s2_10
    for band in range(6):
        for time in range(sentinel2.shape[0]):
            sentinel2[time, ..., band + 4] = resize(s2_20[time,..., band], (width, height), 1)

    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    missing_px = id_missing_px(sentinel2, 3)
    if len(missing_px) > 0:
        print(f"Removing {missing_px} dates due to missing data")
        clouds = np.delete(clouds, missing_px, axis = 0)
        shadows = np.delete(shadows, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)

    # interpolate cloud and cloud shadows linearly
    x, interp = cloud_removal.remove_cloud_and_shadows(sentinel2, clouds, shadows, image_dates) 

    dem_i = np.tile(dem[np.newaxis, :, :, np.newaxis], (x.shape[0], 1, 1, 1))
    dem_i = dem_i / 90
    x = np.concatenate([x, dem_i], axis = -1)
    x = np.clip(x, 0, 1)
    return x, image_dates, interp, s1
            

def interpolate_na_vals(s2: np.ndarray) -> np.ndarray:
    '''Interpolates NA values with closest time steps, to deal with
       the small potential for NA values in calculating indices
    
    #! TODO: Investigate whether this can be vectorized

    '''
    for x_loc in range(s2.shape[1]):
        for y_loc in range(s2.shape[2]):
            n_na = np.sum(np.isnan(s2[:, x_loc, y_loc, :]), axis = 1)
            for date in range(s2.shape[0]):
                if n_na.flatten()[date] > 0:
                    before, after = calculate_proximal_steps(date, np.argwhere(n_na == 0))
                    s2[date, x_loc, y_loc, :] = ((s2[date + before, x_loc, y_loc] + 
                                                 s2[date + after, x_loc, y_loc]) / 2)
    numb_na = np.sum(np.isnan(s2), axis = (1, 2, 3))
    if np.sum(numb_na) > 0:
        print(f"There are {numb_na} NA values")
    return s2
    

def process_subtiles(x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None,
                       sess = None) -> None:
    '''Wrapper function to interpolate clouds and temporal gaps, superresolve tiles,
       calculate relevant indices, and save predicted tree cover as a .npy
       
       Parameters:
        coord (tuple)
        step_x (int):
        step_y (int):
        folder (str):

       Returns:
        None
    '''
    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y
    
    s2 = interpolate_na_vals(s2)

    # The tiles_folder references the folder names (w/o boundaries)
    # While the tiles_array references the arrays themselves (w/ boudnaries)
    tiles_folder = tile_window(s1.shape[2], s1.shape[1], window_size = 140)
    tiles_array = tof_downloading.make_overlapping_windows(tiles_folder)
    
    
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/processed/',
                  tiles_folder)
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    for t in range(len(tiles_folder)):
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]
        
        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[0], tile_folder[1]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]
        subset = s2[:, start_x:end_x, start_y:end_y, :]
        interp_tile = interp[:, start_x:end_x, start_y:end_y]
        interp_tile = np.sum(interp_tile, axis = (1, 2))
        
        dates_tile = np.copy(dates)
        to_remove = np.argwhere(interp_tile > ((150*150) / 4)).flatten()
        if len(to_remove) > 0:
            dates_tile = np.delete(dates_tile, to_remove)
            subset = np.delete(subset, to_remove, 0)
            print(f"Removing {to_remove} interp, leaving {len(dates_tile)} / {len(dates)}")

        missing_px = id_missing_px(subset)
        if len(missing_px) > 0:
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)

        to_remove = cloud_removal.remove_missed_clouds(subset)
        if len(to_remove) > 0:
            subset = np.delete(subset, to_remove, axis = 0)
            dates_tile = np.delete(dates_tile, to_remove)
        try:
            subtile, _ = calculate_and_save_best_images(subset, dates_tile)
        except:
            # If there are no images for the tile, just make them zeros
            # So that they will be picked up by the no-data flag
            subtile = np.zeros((72, end_x-start_x, end_y - start_y, 11))
            dates_tile = [0,]
        output = f"{path}{str(folder_y)}/{str(folder_x)}.npy"
        s1_subtile = s1[:, start_x:end_x, start_y:end_y, :]
        
        if subtile.shape[2] == 147: 
            pad_u = 7 if start_y == 0 else 0
            pad_d = 7 if start_y != 0 else 0

            subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
        if subtile.shape[1] == 147:
            pad_l = 7 if start_x == 0 else 0
            pad_r = 7 if start_x != 0 else 0
   
            subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')

        dem = subtile[..., -1]
        sm = Smoother(lmbd = 800, size = subtile.shape[0], nbands = 10, dim = subtile.shape[1])
        subtile = sm.interpolate_array(subtile[..., :-1])
        subtile = superresolve_tile(subtile, sess = superresolve_sess)
        
        subtile = np.concatenate([subtile, dem[:12, :, :, np.newaxis]], axis = -1)
        subtile = np.concatenate([subtile,  s1_subtile], axis = -1)
        subtile[..., -2:] = subtile[..., -2:] / 65535
        
        output_folder = "/".join(output.split("/")[:-1])
        if not os.path.exists(os.path.realpath(output_folder)):
            os.makedirs(os.path.realpath(output_folder))
        
        subtile = np.clip(subtile, 0, 1)

        assert subtile.shape[1] >= 145, f"subtile shape is {subtile.shape}"
        assert subtile.shape[0] == 12, f"subtile shape is {subtile.shape}"

        # If the first image is after May 1 or the last image is before September 1
        # Then we cannot make predictions... although we try to "wrap around" and linearly
        # interpolate to the other side of the year, this effectively doubles the temporal gap
        # e.g. a date[0] = 120 and date[-1] = 245 is a (45 + 120) + (410 - 245) = 330 day gap
        # from the POV of the model

        gap_between_years = False
        if dates_tile[0] >= 150 or dates_tile[-1] <= 215:
            gap_between_years = True

        if len(dates_tile) < 4 or gap_between_years:
            subtile = np.zeros_like(subtile)

        preds = predict_subtile(subtile, sess)
        np.save(output, preds)
        print(f"Writing {output}")


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
 

def predict_subtile(x, sess) -> np.ndarray:
    """ - Loads sentinel 1, 2, dem images
        - Calculates remote sensing indices
        - Normalizes data
        - Calculates smooth window predictions
        - Mosaics predictions
        - Returns predictions for subtile
    """
    
    if np.sum(x) > 0:
        if not isinstance(x.flat[0], np.floating):
            assert np.max(x) > 1
            x = x / 65535.

        #! TODO: Convert the next few code blocks
        # to a preprocessing_fn function
        x[..., -1] = convert_to_db(x[..., -1], 22)
        x[..., -2] = convert_to_db(x[..., -2], 22)
        
        indices = np.empty((12, x.shape[1], x.shape[2], 4))
        indices[..., 0] = evi(x)
        indices[..., 1] = bi(x)
        indices[..., 2] = msavi2(x)
        indices[..., 3] = grndvi(x)
        x = np.concatenate([x, indices], axis = -1)

        med = np.median(x, axis = 0)
        med = med[np.newaxis, :, :, :]
        x = np.concatenate([x, med], axis = 0)

        filtered = median_filter(x[0, :, :, 10], size = 5)
        x[:, :, :, 10] = np.stack([filtered] * x.shape[0])
        x = np.clip(x, min_all, max_all)
        x = (x - midrange) / (rng / 2)
        
        #x = tile_images(x)
        #batch_x = np.stack(x)
        batch_x = x[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        preds = sess.run(predict_logits,
                              feed_dict={predict_inp:batch_x, 
                                         predict_length:lengths})
        stacked = preds.squeeze()
        stacked = stacked[1:-1, 1:-1]
        """
        stacked = np.full((140, 140, 4), 255.)
        stacked[:90, :90, 0] = preds[0].squeeze()
        stacked[-90:, :90, 1] = preds[2].squeeze()
        stacked[:90, -90:, 2] = preds[1].squeeze()
        stacked[-90:, -90:, 3] = preds[3].squeeze()
        stacked[stacked == 255] = np.nan
        stacked = np.nanmean(stacked, axis = -1).astype(np.float32)
        """
        
    else:
        stacked = np.full((140, 140), 255)
    
    return stacked


def tile_images(arr: np.ndarray) -> list:
    """ Converts a 142x142 array to a 289, 24, 24 array
        
        Parameters:
         arr (np.ndaray): (142, 142) float array
    
        Returns:
         images (list): 
    """

    # Normal
    images = []
    for x_offset, cval in enumerate([x for x in range(0, 70, 50)]):
        for y_offset, rval in enumerate([x for x in range(0, 70, 50)]):
            min_x = np.max([cval - 0, 0])
            max_x = np.min([cval + 104, 154])
            min_y = np.max([rval - 0, 0])
            max_y = np.min([rval + 104, 154])
            subs = arr[:, min_x:max_x, min_y:max_y]
            images.append(subs)
    return images


def load_mosaic_predictions(out_folder: str) -> np.ndarray:
    """
    Loads the .npy subtile files in an output folder and mosaics the overlapping predictions
    to return a single .npy file of tree cover for the 6x6 km tile

    Additionally, applies post-processing threshold rules and implements no-data flag of 255
    """
    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
    max_x = np.max(x_tiles) + 140

    for x_tile in x_tiles:
        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        max_y = np.max(y_tiles) + 140

    predictions = np.full((max_x, max_y), 0, dtype = np.uint8)
    x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
    for x_tile in x_tiles:
        y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        for y_tile in y_tiles:
            output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
            if os.path.exists(output_file):
                prediction = np.load(output_file)
                prediction = (prediction * 100).T.astype(np.uint8)
                predictions_tile = predictions[x_tile: x_tile+140, y_tile:y_tile + 140]

                if np.max(prediction) <= 100:
                    existing_predictions = predictions_tile[np.logical_and(predictions_tile != 0, predictions_tile <= 100)] 
                    current_predictions = prediction[np.logical_and(predictions_tile != 0, predictions_tile <= 100)]
                    if current_predictions.shape[0] > 0:
                        # Require colocation. Here we can have a lower threshold as
                        # the likelihood of false positives is much higher
                        current_predictions[(current_predictions - existing_predictions) > 50] = np.min([current_predictions, existing_predictions])
                        existing_predictions[(existing_predictions - current_predictions) > 50] = np.min([current_predictions, existing_predictions])
                        existing_predictions = (current_predictions + existing_predictions) / 2
         
                    predictions_tile[predictions_tile == 0] = prediction[predictions_tile == 0]
                else:
                    predictions[(x_tile): (x_tile+140),
                           y_tile:y_tile + 140] = prediction
                
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

    predictions[predictions <= .25*100] = 0.        
    predictions = np.around(predictions / 20, 0) * 20
    predictions[predictions > 100] = 255.
    return predictions
     

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/master-154/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/')
    parser.add_argument("--db_path", dest = "db_path", default = "processing_area_may_18.csv")
    parser.add_argument("--ul_flag", dest = "ul_flag", default = False)
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--n_tiles", dest = "n_tiles", default = None)
    parser.add_argument("--x", dest = "x", default = None)
    parser.add_argument("--y", dest = "y", default = None)
    args = parser.parse_args()

    print(f'Country: {args.country} \n'
          f'Local path: {args.local_path} \n'
          f'Predict model path: {args.predict_model_path} \n'
          f'Superrresolve model path: {args.superresolve_model_path} \n'
          f'DB path: {args.db_path} \n'
          f'S3 Bucket: {args.s3_bucket} \n'
          f'YAML path: {args.yaml_path} \n'
          f'Current dir: {os.getcwd()} \n'
          f'N tiles to download: {args.n_tiles} \n'
          f'X: {args.x} \n'
          f'Y: {args.y} \n')

    if os.path.exists(args.yaml_path):
        with open(args.yaml_path, 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']
        print(f"Successfully loaded key from {args.yaml_path}")
        uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
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

    # Normalization mins and maxes for the prediction input
    min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 0.013351644159609368, 0.01965362020294499, 0.014229037918669413, 0.015289539940489814, 0.011993591210803388, 0.008239871824216068, 0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101, -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]
    max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 0.6027466239414053, 0.5650263218127718, 0.5747005416952773, 0.5933928435187305, 0.6034943160143434, 0.7472037842374304, 0.7000076295109483, 0.509269855802243, 0.948334642387533, 0.6729257769285485, 0.8177635298774327, 0.35768999002433816, 0.7545951919107605, 0.7602693339366691]

    min_all = np.array(min_all)
    max_all = np.array(max_all)
    min_all = np.broadcast_to(min_all, (13, 154, 154, 17))
    max_all = np.broadcast_to(max_all, (13, 154, 154, 17))
    midrange = (max_all + min_all) / 2
    rng = max_all - min_all

    # For generating the subtile mosaicing
    SIZE = 10
    SIZE_N = SIZE*SIZE
    SIZE_UR = (SIZE - 1) * (SIZE - 1)
    n = 0

    # If downloading an individually indexed tile, go ahead and execute this code block
    if args.x and args.y:
        print(f"Downloading an individual tile: {args.x}X{args.y}Y")
        x = args.x
        y = args.y

        year = 2020
        dates = (f'{str(year - 1)}-11-15' , f'{str(year + 1)}-02-15')
        dates_sentinel_1 = (f'{str(year)}-01-01' , f'{str(year)}-12-31')
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
        if not processed:
            if args.n_tiles:
                below = n <= int(args.n_tiles)
            else:
                below = True
            if below:
                bbx = None
                time1 = time.time()
                bbx = download_tile(x = x, y = y, data = data, api_key = API_KEY)
                s2, dates, interp, s1 = process_tile(x = x, y = y, data = data)
                process_subtiles(x, y, s2, dates, interp, s1, predict_sess)
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
                key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_POST.tif'
                uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                if args.ul_flag:
                    upload_raw_processed_s3(path_to_tile, x, y)
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
            year = 2020
            dates = (f'{str(year - 1)}-11-15' , f'{str(year + 1)}-02-15')
            dates_sentinel_1 = (f'{str(year)}-01-01' , f'{str(year)}-12-31')
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
            if not processed:
                if args.n_tiles:
                    below = n <= int(args.n_tiles)
                else:
                    below = True
                if below:
                    try:
                        time1 = time.time()
                        bbx = download_tile(x = x, y = y, data = data, api_key = API_KEY)
                        s2, dates, interp, s1 = process_tile(x = x, y = y, data = data)
                        process_subtiles(x, y, s2, dates, interp, s1, predict_sess)
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
                        key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_POST.tif'
                        uploader.upload(bucket = args.s3_bucket, key = key, file = file)
                        if args.ul_flag:
                            upload_raw_processed_s3(path_to_tile, x, y)
                        time2 = time.time()
                        print(f"Finished {n} in {np.around(time2 - time1, 1)} seconds")
                        n += 1
                    except Exception as e:
                        print(f"Ran into {str(e)} error, skipping {x}/{y}/")
                        continue
            else:
                print(f'Skipping {x}, {y} as it is done')

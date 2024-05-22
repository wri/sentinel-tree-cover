import pandas as pd
import numpy as np
import os
import yaml
from scipy.sparse.linalg import splu
from skimage.transform import resize
import hickle as hkl
import boto3
from scipy.ndimage import median_filter
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
#import tensorflow as tf
from glob import glob
import rasterio
from rasterio.transform import from_origin
import time
import re
import shutil
import bottleneck as bn
import gc
import copy

from sys import getsizeof
from preprocessing import slope
from downloading.utils import calculate_and_save_best_images
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from tof.tof_downloading import to_int16, to_float32
from downloading.io import download_ard_file
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder
from preprocessing.indices import evi, bi, msavi2, grndvi
from download_and_predict_job import process_tile, make_bbox, convert_to_db, deal_w_missing_px
from download_and_predict_job import fspecial_gauss, make_and_smooth_indices, write_ard_to_tif, str2bool
from download_and_predict_job import float_to_int16
from tof import tof_downloading


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

LEN = 12

def make_bbox_rect(initial_bbx: list, expansionx: int = 10, expansiony: int = 10) -> list:
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
    bbx[0] -= expansionx * multiplier
    bbx[1] -= expansiony * multiplier
    bbx[2] += expansionx * multiplier
    bbx[3] += expansiony * multiplier
    return bbx

def download_raw_tile(tile_idx, local_path, subfolder = "raw"):
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


def split_fn(item, form):
    if form == 'tile':
        overlap_left = (item.shape[2]) - (SIZE // 2) - 7
        tiles_x = overlap_left + 7
        item = item[:, :, overlap_left:]
    if form == 'neighbor':
        overlap_right = (SIZE // 2) + 7
        item = item[:, :, :overlap_right]
        tiles_x = None
    return item, tiles_x


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape


def split_to_border(s2, interp, s1, dem, fname, edge):
    if edge == "right":
        if fname == "tile":
            s1, tiles_x = split_fn(s1, fname)
        else:
            s1, _ = split_fn(s1, fname)
        interp, _ = split_fn(interp, fname)
        s2, _ = split_fn(s2, fname)
        dem, _ = split_fn(dem[np.newaxis], fname)
    print(s1.shape)
    return s2, interp, s1, dem.squeeze(), _


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
    resolved = sess.run(superresolve_logits, 
                 feed_dict={superresolve_inp: to_resolve,
                            superresolve_inp_bilinear: bilinear})
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
    wsize = 125
    step = 125
    x_range = [x for x in range(0, arr.shape[1] - (wsize), step)] + [arr.shape[1] - wsize]
    y_range = [x for x in range(0, arr.shape[2] - (wsize), step)] + [arr.shape[2] - wsize]
    x_end = np.copy(arr[:, x_range[-1]:, ...])
    y_end = np.copy(arr[:, :, y_range[-1]:, ...])
    print(f"There are {len(x_range)*len(y_range)} tiles to supres")
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
    return arr


def predict_subtile(subtile: np.ndarray, sess: "tf.Sess", op: "tf.Tensor") -> np.ndarray:
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
    size = 216
    if np.sum(subtile) != 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.

        time1 = time.time()
        subtile = np.core.umath.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), LEN)
        time2 = time.time()
        print(f"Other prep in {time2 - time1} seconds")

        time1 = time.time()
        preds = sess.run(op, feed_dict={predict_inp:batch_x, 
                                         predict_length:lengths})
        
        preds = preds.squeeze()
        clip = (preds.shape[0] - size) // 2
        if clip > 0:
            preds = preds[clip:-clip, clip:-clip]
        #preds = preds[1:-1, 1:-1]
        time2 = time.time()
        print(f"The preds for: {op} are {preds.shape}")
        print(f"Preds in {time2 - time1} seconds")
        
    else:
        print("Empty subtile, skipping")
        preds = np.full((SIZE, SIZE), 255)
    
    return preds


def check_if_processed(tile_idx, local_path, year):

    x = tile_idx[0]
    y = tile_idx[1]
    path_to_tile = f'{local_path}{str(x)}/{str(y)}/'
    s3_path_to_tile = f'{str(year)}/tiles/{str(x)}/{str(y)}/'
    processed = file_in_local_or_s3(path_to_tile,
                                    s3_path_to_tile, 
                                    AWSKEY, AWSSECRET, 
                                    args.s3_bucket)
    return processed


def align_dates(tile_date, neighb_date):
    # RESEGMENT
    # Add a 7 day grace period b/w dates
    differences_tile = [np.min(abs(a - np.array([neighb_date]))) for a in tile_date]
    differences_neighb = [np.min(abs(a - np.array([tile_date]))) for a in neighb_date]
    duplicate_dates = np.argwhere(np.diff(tile_date, prepend = 0) == 0).flatten()
    duplicate_neighb = np.argwhere(np.diff(neighb_date, prepend = 0) == 0).flatten()

    to_rm_tile = [idx for idx, diff in enumerate(differences_tile) if diff > 1]
    to_rm_tile = list(to_rm_tile) + list(duplicate_dates)
    to_rm_neighb = [idx for idx, diff in enumerate(differences_neighb) if diff > 1]
    to_rm_neighb = to_rm_neighb + list(duplicate_neighb)
    n_to_rm = len(to_rm_tile) + len(to_rm_neighb)
    min_images_left = np.minimum(
        len(tile_date) - len(to_rm_tile),
        len(neighb_date) - len(to_rm_neighb)
    )
    print(f"{len(to_rm_tile) + len(to_rm_neighb)} dates are mismatched,"
          f" leaving a minimum of {min_images_left}")
    return to_rm_tile, to_rm_neighb, min_images_left


def check_n_tiles(x, y):
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    n_tiles_x = len([x for x in os.listdir(path) if x.isnumeric()]) - 1
    n_tiles_y = len([x for x in os.listdir(path + "0/") if x[-4:] == ".npy"]) - 1
    return n_tiles_x, n_tiles_y


def make_tiles_right_neighb(tiles_folder_x, tiles_folder_y):
    windows = cartesian(tiles_folder_x, tiles_folder_y)
    _size = 230 - 14
    win_sizes = np.full_like(windows, _size + 7)
    tiles_folder = np.hstack([windows, win_sizes])
    tiles_folder = np.sort(tiles_folder, axis = 0)
    tiles_folder[:, 1] = np.tile(np.unique(tiles_folder[:, 1]), 
        int(len(tiles_folder[:, 1]) / len(np.unique(tiles_folder[:, 1]))))
    tiles_array = np.copy(tiles_folder)
    tiles_array[1:, 1] -= 7
    tiles_array[:, 0] = 0.
    tiles_array[:, 2] = _size + 14.
    tiles_array[:, 3] = _size + 7.
    tiles_array[1:-1, 3] += 7
    print(tiles_array, tiles_folder)
    return tiles_array, tiles_folder


def align_subtile_histograms(array) -> np.ndarray:
    # RESEGMENT
    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    left_water = _water_ndwi(
        np.median(array[:, :, (SIZE + 14) // 2:], axis = 0))

    left_water = left_water >= 0.1
    print(f'{np.mean(left_water)}% of the left is water')

    right_water = _water_ndwi(
        np.median(array[:, :, :(SIZE + 14) // 2], axis = 0))
    right_water = right_water >= 0.1
    print(f'{np.mean(right_water)}% of the right is water')

    # Why is std_ref calculated separately?
    for time in range(array.shape[0]):

        # Identify all of the areas that are, and aren't interpolated
        left = array[time, :, (SIZE + 14) // 2:]
        right = array[time, :, :(SIZE + 14) // 2]
        array_time = array[time]

        # And calculate their means and standard deviation per band
        std_right = np.nanstd(right[~right_water], axis = (0))
        std_left = np.nanstd(left[~left_water], axis = (0))
        std_ref = (std_right + std_left) / 2
        
        mean_right = np.nanmean(right[~right_water], axis = (0))
        mean_left = np.nanmean(left[~left_water], axis = (0))
        mean_ref = (mean_right + mean_left) / 2
        
        std_mult_left = (std_left / std_ref)
        addition_left = (mean_left - (mean_ref * (std_mult_left)))  
        
        std_mult_right = (std_right / std_ref)
        addition_right = (mean_right - (mean_ref * (std_mult_right)))
        
        before = abs(np.roll(array[time], 1, axis = 1) - array[time])
        before = np.mean(before[:, (SIZE // 2) + 7, :], axis = (0, 1))
        
        candidate = np.copy(array[time])
        candidate[:, :(SIZE + 14) // 2, :] = candidate[:, :(SIZE + 14) // 2, :] * std_mult_left + addition_left
        candidate[:, (SIZE + 14) // 2:, :] = (
                candidate[:, (SIZE + 14) // 2:, :] * std_mult_right + addition_right)
            
        after = abs(np.roll(candidate, 1, axis = 1) - candidate)
        after = np.mean(after[:, (SIZE // 2) + 7, :], axis = (0, 1))
        
        if after < before:
            array[time, :, :(SIZE + 14) // 2, :] = (
                    array[time, :, :(SIZE + 14) // 2, :] * std_mult_left + addition_left
            )

            array[time, :, (SIZE + 14) // 2:, :] = (
                    array[time, :, (SIZE + 14) // 2:, :] * std_mult_right + addition_right
            )

    return array


def adjust_predictions(preds, ref):
    # RESEGMENT
    std_src = bn.nanstd(preds)
    std_ref = bn.nanstd(ref)
    mean_src = bn.nanmean(preds)
    mean_ref = bn.nanmean(ref)
    std_mult = (std_ref / std_src)

    addition = (mean_ref - (mean_src * (std_mult)))
    preds = preds * std_mult + addition
    preds = np.clip(preds, 0, 1)
    return preds


def process_subtiles(x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None,
                       gap_sess = None, tiles_folder = None, tiles_array = None,
                       right_all = None,
                       left_all = None,
                       hist_align = True,
                       min_clear_images_per_date = None) -> None:
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
    
    time1 = time.time()
    s2 = interpolation.interpolate_na_vals(s2)
    s2 = np.float32(s2)
    s2_median = np.median(s2, axis = 0)[np.newaxis]
    s1_median = np.median(s1, axis = 0)[np.newaxis]


    time2 = time.time()
    print(f"Finished interpolate subtile in {np.around(time2 - time1, 1)} seconds")
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/processed/',
                  tiles_folder)
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    path_neighbor = f'{args.local_path}{str(int(x) + 1)}/{str(y)}/processed/'

    if LEN == 4:
        print('mking quarterly')
        s2 = np.reshape(s2, (4, 3, s2.shape[1], s2.shape[2], s2.shape[3]))
        s2 = np.median(s2, axis = 1, overwrite_input = True)
        s1 = np.reshape(s1, (4, 3, s1.shape[1], s1.shape[2], s1.shape[3]))
        s1 = np.median(s1, axis = 1, overwrite_input = True)
        #np.save("s2.npy", s2[..., :3])

    gap_between_years = False
    t = 0
    # Iterate over each subitle and prepare it for processing and generate predictions
    print(f"Tiles folder: {tiles_folder}")
    for t in range(len(tiles_folder)):
        _size = 230 - 14
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]

        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[1], tile_folder[0]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]

        subset = np.copy(s2[:, start_y:end_y, start_x:end_x, :])
        subtile_median_s2 = np.copy(s2_median[:, start_y:end_y, start_x:end_x, :])
        subtile_median_s1 = s1_median[:, start_y:end_y, start_x:end_x, :]
        interp_tile = interp[:, start_y:end_y, start_x:end_x]
        interp_tile_sum = np.sum(interp_tile, axis = (1, 2))
        min_clear_tile = min_clear_images_per_date[start_y:end_y, start_x:end_x]
        dates_tile = np.copy(dates)
        dem_subtile = dem[np.newaxis, start_y:end_y, start_x:end_x]
        s1_subtile = s1[:, start_y:end_y, start_x:end_x, :]
        output = f"{path}/right{str(folder_y)}/{str(folder_x)}.npy"
        output2 = f"{path_neighbor}{str(0)}/left{str(folder_x)}.npy"
        print(f"There are only {np.min(min_clear_images_per_date)} clear images")
        no_images = False
        #if np.percentile(min_clear_images_per_date, 25) < 1:
        #    no_images = True

        to_remove = np.argwhere(np.sum(np.isnan(subset), axis = (1, 2, 3)) > 0).flatten()
        if len(to_remove) > 0: 
            print(f"Removing {to_remove} NA dates")
            dates_tile = np.delete(dates_tile, to_remove)
            subset = np.delete(subset, to_remove, 0)
            interp_tile = np.delete(interp_tile, to_remove, 0)
        
        subtile = subset

        if hist_align:
            print("Aligning for real")
            subset = align_subtile_histograms(subset)
            subtile_median_s2 = align_subtile_histograms(subtile_median_s2)
       
        # Pad the corner / edge subtiles within each tile
        if subtile.shape[2] == _size + 7: 
            pad_u = 7 if start_y != 0 else 0
            pad_d = 7 if start_y == 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (0, 0), (pad_u, pad_d)), 'reflect')
            subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            min_clear_tile = np.pad(min_clear_tile, ((0, 0), (pad_u, pad_d)), 'reflect')


        if subtile.shape[1] == _size + 7:
            pad_l = 7 if start_y == 0 else 0
            pad_r = 7 if start_y != 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')
            subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            min_clear_tile = np.pad(min_clear_tile, ((pad_l, pad_r), (0, 0)), 'reflect')

        subtile_s2 = subtile
        # Concatenate the DEM and Sentinel 1 data
        subtile = np.empty((LEN + 1, _size + 14, _size + 14, 17), dtype = np.float32)
        subtile[:-1, ..., :10] = subtile_s2[..., :10]
        subtile[:-1, ..., 11:13] = s1_subtile
        subtile[:-1, ..., 13:] = subtile_s2[..., 10:]

        subtile[:, ..., 10] = dem_subtile.repeat(LEN + 1, axis = 0)
        
        subtile[-1, ..., :10] = subtile_median_s2[..., :10]
        subtile[-1, ..., 11:13] = subtile_median_s1
        subtile[-1, ..., 13:] = subtile_median_s2[..., 10:]

        
        # Create the output folders for the subtile predictions
        output_folder = "/".join(output.split("/")[:-1])
        if not os.path.exists(os.path.realpath(output_folder)):
            os.makedirs(os.path.realpath(output_folder))

        output_folder = "/".join(output2.split("/")[:-1])
        if not os.path.exists(os.path.realpath(output_folder)):
            os.makedirs(os.path.realpath(output_folder))
        
        
        assert subtile.shape[1] >= 145, f"subtile shape is {subtile.shape}"
        assert subtile.shape[0] == LEN + 1, f"subtile shape is {subtile.shape}"

        # Select between temporal and median models for prediction, based on simple logic:
        # If the first image is after June 15 or the last image is before July 15
        # or the maximum gap is >270 days or < 5 images --- then do median, otherwise temporal
        no_images = True if len(dates_tile) < 2 else no_images
        print(f"Subtile shape is {subtile.shape}")
        if no_images:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data")
            preds = np.full((_size, _size), 255)
        else:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                f"for: {dates_tile}")
            preds = predict_subtile(subtile, sess, predict_logits)
            print(f"Preds shape is {preds.shape}")

            if args.gen_feats:
                latefeats = predict_subtile(subtile, sess, predict_latefeats)[..., :32]
                earlyfeats = predict_subtile(subtile, sess, predict_earlyfeats)[..., :32]
                #earlyfeats = earlyfeats.repeat(4, axis = 0).repeat(4, axis = 1)
                #earlyfeats = earlyfeats[1:-1, 1:-1]
                earlyfeats = float_to_int16(earlyfeats)
                latefeats = float_to_int16(latefeats)
                feats_path = f'{args.local_path}{str(x)}/{str(y)}/feats/'
                output_feats = f"{feats_path}{str(folder_y)}/{str(folder_x)}_middle.npy"
                output_folder = "/".join(output_feats.split("/")[:-1])
                if not os.path.exists(os.path.realpath(output_folder)):
                    os.makedirs(os.path.realpath(output_folder))
                if not os.path.exists(os.path.realpath(f'{args.local_path}{str(x)}/{str(y)}/raw/feats/')):
                    os.makedirs(f'{args.local_path}{str(x)}/{str(y)}/raw/feats/')
                if not os.path.exists(f'{args.local_path}{str(x)}/{str(y)}/ard/'):
                    os.makedirs(f'{args.local_path}{str(x)}/{str(y)}/ard/')
                np.save(output_feats, np.concatenate([earlyfeats,latefeats], axis = -1))
                print(f"{str(folder_y)}/{str(folder_x)}_middle: {len(dates_tile)} / {len(dates)} dates,"
                f"for: {dates_tile}, {np.percentile(min_clear_images_per_date, 10)} clear images"
                f", {np.mean(preds)}")
        

def preprocess_tile(arr, dates, interp, clm, fname, dem, bbx):
    time1 = time.time()

    missing_px = interpolation.id_missing_px(arr, 20)
    if len(missing_px) > 0:
        dates = np.delete(dates, missing_px)
        arr = np.delete(arr, missing_px, 0)
        #interp = np.delete(interp, missing_px, 0)
        print(f"Removing {len(missing_px)} missing images")
    time2 = time.time()
    print(f"Finished missing px in preprocess_tile in {np.around(time2 - time1, 1)} seconds")

        # Remove dates with high likelihood of missed cloud or shadow (false negatives)
    # This is done here because make_shadow = False in process_tile
    cld, fcps = cloud_removal.identify_clouds_shadows(arr, dem, bbx)
    if clm is not None:
        if len(missing_px) > 0:
            print(f"Deleting {missing_px} from cloud mask")
            clm = np.delete(clm, missing_px, 0)
        try:
            clm[fcps] = 0.
            print("CLM", np.mean(clm, axis = (1, 2)))
            cld = np.maximum(clm, cld)
        except:
            print("There is a date mismatch between clm and cld, continuing")
        print(cld.shape)
        
    print(np.mean(cld, axis = (1, 2)))


    interp = cloud_removal.id_areas_to_interp(
            arr, cld, cld, dates, fcps
    )
    print(np.mean(interp == 1, axis = (1, 2)))
    to_remove = np.argwhere(np.mean(interp == 1, axis = (1, 2)) > 0.95)
    if len(to_remove) > 0:
        cld = np.delete(cld, to_remove, axis = 0)
        dates = np.delete(dates, to_remove)
        interp = np.delete(interp, to_remove, axis = 0)
        arr = np.delete(arr, to_remove, axis = 0)
        cld, fcps = cloud_removal.identify_clouds_shadows(arr, dem, bbx)

    arr, interp2, to_remove = cloud_removal.remove_cloud_and_shadows(arr, 
        cld, 
        cld, 
        dates, 
        pfcps = fcps,
        sentinel1 = None,
        mosaic = None)

    print(np.mean(interp2, axis = (1, 2)))
    #interp = np.maximum(interp, interp2)
    
    return arr, interp2, dates


def check_if_artifact(tile, neighb):
    # RESEGMENT
    right = neighb[:, :3]
    left = tile[:, -3:]

    right_mean = bn.nanmean(neighb[:, :3])
    left_mean = bn.nanmean(tile[:, -3:])
    right = neighb[:,  0]#bn.nanmean(neighb[:, :3], axis = 1)
    right = np.pad(right, (10 - (right.shape[0] % 10)) // 2, constant_values = np.nan)
    right = np.reshape(right, (right.shape[0] // 10, 10))
    right = bn.nanmean(right, axis = 1)    
    
    left = tile[:, -1]#bn.nanmean(tile[:, -3:], axis = 1)
    left = np.pad(left, (10 - (left.shape[0] % 10)) // 2, constant_values =  np.nan)
    left = np.reshape(left, (left.shape[0] // 10, 10))
    left = bn.nanmean(left, axis = 1)

    fraction_diff = bn.nanmean(abs(right - left) > 20) #was 15
    fraction_diff_2 = bn.nanmean(abs(right - left) > 12.5) # was 10
    fraction_diff_left = bn.nanmean(abs(right[:15] - left[:15]) > 17.5)# was 10
    fraction_diff_right = bn.nanmean(abs(right[-15:] - left[-15:]) > 17.5) # was 10
    left_right_diff = abs(right_mean - left_mean)

    other0 = left_right_diff > 6 # was 5

    other = fraction_diff_2 > 0.5
    other = np.logical_and(other, (left_right_diff > 1) ) # wasa 6

    other2 = (fraction_diff > 0.3) or (fraction_diff_left > 0.5) or (fraction_diff_right > 0.5) # was 0.2
    other2 = np.logical_and(other2, (left_right_diff > 1) )

    print(x, y, left_right_diff, fraction_diff, fraction_diff_right, fraction_diff_left, other0, other, other2)
    if other0 or other or other2:
        return 1
    else:
        return 0
        

def load_tif(tile_id, local_path):
    dir_i = f"{local_path}/{tile_id[0]}/{tile_id[1]}/"
    tifs = []
    is_smooth_y = 0
    if os.path.exists(dir_i): 

        files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
        final_files = [x for x in files if "_FINAL" in x]
        post_files = [x for x in files if "_POST" in x]
        
        smooth_files = [file for file in files if "_SMOOTH" in file]
        smooth_x = [file for file in files if "_SMOOTH_X" in file]
        smooth_y = [file for file in files if "_SMOOTH_Y" in file]
        smooth_xy = [file for file in files if "_SMOOTH_XY" in file]
        
        smooth_files = [file for file in smooth_files if os.path.splitext(file)[-1] == '.tif']
        #smooth_files = []
        if len(smooth_files) > 0:
            if len(smooth_xy) > 0:
                files = smooth_xy
                is_smooth_y = 1
            elif len(smooth_x) > 0:
                files = smooth_x
            elif len(smooth_y) > 0:
                files = smooth_y
                is_smooth_y = 1

        elif len(final_files) > 0:
            files = final_files
        else:
            files = post_files

        for file in files:
           tifs.append(os.path.join(dir_i, file))

    tifs = tifs[0]
    print(tifs)
    tifs = rasterio.open(tifs).read(1)
    return tifs, is_smooth_y


def concatenate_s2_files(s2, s2_neighb):
    # RESEGMENT
    s2_diff = s2.shape[2] - s2_neighb.shape[2]
    if s2_diff > 0:
        s2 = s2[:, :, s2_diff // 2: -(s2_diff // 2), :]
        
    if s2_diff < 0:
        s2_neighb = s2_neighb[:, :, - (s2_diff // 2) : (s2_diff // 2)]
    return s2, s2_neighb


def load_dates(tile_x, tile_y, local_path):
    # RESEGMENT
    tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
    folder = f"{local_path}{str(tile_x)}/{str(tile_y)}/"
    return hkl.load(f'{folder}raw/misc/s2_dates_{tile_idx}.hkl')


def regularize_and_smooth(arr, dates):
    # RESEGMENT
    sm = Smoother(lmbd = 100, size = 24, nbands = 2, dimx = arr.shape[1], dimy = arr.shape[2])
    # Perhaps do 2-band windows separately here to clear up RAM
    js = np.arange(0, 10, 2)
    ks = js + 2
    n_images = arr.shape[0]
    if n_images < 12:
        empty_images = np.zeros((12 - n_images, arr.shape[1], arr.shape[2], arr.shape[3]), dtype = np.float32)
        arr = np.concatenate([arr, empty_images], axis = 0)
    for j, k in zip(js, ks):
        try:
            arr_i, _ = calculate_and_save_best_images(arr[:n_images, ..., j:k], dates)
        except:
            no_images = True
        arr[:12, ..., j:k] = sm.interpolate_array(arr_i)

    arr = arr[:12]

    return arr

def update_ard_tiles(x, y, m, bbx, neighb_bbx):
    # Step 1 check if ARD exists
    # Step 2 update the 

    left_s3_path = f'2020/ard/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_ard.hkl'
    right_s3_path = f'2020/ard/{str(int(x) + 1)}/{str(y)}/{str(int(x) + 1)}X{str(y)}Y_ard.hkl'
    left_local_path = f'{str(x)}X{str(y)}Y_ard.hkl'
    right_local_path = f'{str(int(x) + 1)}X{str(y)}Y_ard.hkl'

    download_ard_file(left_s3_path,
        left_local_path,
        AWSKEY, AWSSECRET, 'tof-output')
    download_ard_file(right_s3_path,
        right_local_path,
        AWSKEY, AWSSECRET, 'tof-output')

    l = hkl.load(left_local_path)[..., :13]
    r = hkl.load(right_local_path)[..., :13]
    #m = np.load(middle)
    inp_mid_shape = m.shape[1]
    out_mid_shape = l.shape[1]
    middle_adjust = (inp_mid_shape - out_mid_shape) // 2
    m = m[:, middle_adjust:-middle_adjust, :]
    half = m.shape[1] // 2
    lsize = l.shape[1] - half
    rsize = lsize + half + half
    print(l.shape, r.shape, m.shape)
    img = np.concatenate([l, r], axis = 1)


    sums = np.zeros((img.shape[0], img.shape[1]), dtype = np.float32)
    sums[:, :l.shape[0] // 2] = 1
    sums[:, l.shape[0] // 2:(l.shape[0] // 2)+half] += (1 - (np.arange(0, half, 1) / half))
    sums[:, (l.shape[0] // 2)+half:(l.shape[0] // 2)+half+half] += ((np.arange(0, half, 1) / half))
    sums[:, -(r.shape[0] // 2):] = 1.
    sums = sums[..., np.newaxis]
    sumsright = 1 - sums
    img[..., :10] = img[..., :10] * sums
    print(m.shape, img[:, lsize:rsize, :10].shape)
    img[:, lsize:rsize, :10] += (m[..., :10] * (1 - sums[:, lsize:rsize]))
    #img[:, lsize:rsize, :10] /= 2,# * (1 - sums[:, lsize:rsize]))
    leftfile = img[:, :l.shape[1]]
    rightfile = img[:, -r.shape[1]:]
    hkl.dump(leftfile, left_local_path, mode='w', compression='gzip')
    hkl.dump(rightfile, right_local_path,  mode='w', compression='gzip')
    uploader.upload(bucket = 'tof-output', key = right_s3_path, file = right_local_path)
    uploader.upload(bucket = 'tof-output', key = left_s3_path, file = left_local_path)
    write_ard_to_tif(leftfile[..., :3], bbx,
                                left_local_path[:-4], "")
    write_ard_to_tif(rightfile[..., :3], neighb_bbx,
                                right_local_path[:-4], "")
    return img


def resegment_border(tile_x, tile_y, edge, local_path, bbx, neighb_bbx, min_dates, initial_bbx):
    # RESEGMENT
    processed = check_if_processed((tile_x, tile_y), local_path, args.year)
    neighbor_id = [str(int(tile_x) + 1), tile_y]

    processed_neighbor = check_if_processed(neighbor_id, local_path, args.year)
    if processed_neighbor:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['X_tile'] == int(neighbor_id[0])]
        data_temp = data_temp[data_temp['Y_tile'] == int(neighbor_id[1])]
        processed_neighbor = True if len(data_temp) > 0 else False

    if processed and processed_neighbor:
        print(f"Downloading {tile_x}, {tile_y}")
        download_raw_tile((tile_x, tile_y), local_path, "tiles")
        download_raw_tile(neighbor_id, local_path, "tiles")
        tile_tif, smooth_y_tile = load_tif((tile_x, tile_y), local_path)
        if type(tile_tif) is not np.ndarray:
            print("Skipping because one of the TIFS doesnt exist")
            return 0, None, None, 0, 0

        tile_tif = tile_tif.astype(np.float32)
        neighbor_tif, smooth_y_neighb = load_tif(neighbor_id, local_path)
        neighbor_tif = neighbor_tif.astype(np.float32)
        neighbor_tif[neighbor_tif > 100] = np.nan
        tile_tif[tile_tif > 100] = np.nan
        
        diff_for_compare = np.mean(abs(np.mean(neighbor_tif[:, :8], axis = 1) - 
                               np.mean(tile_tif[:, -8:], axis = 1)))
        
        artifact = check_if_artifact(tile_tif, neighbor_tif)
        right_all = np.nanmean(neighbor_tif[:, :(SIZE // 2)], axis = 1)
        left_all = np.nanmean(tile_tif[:, -(SIZE // 2):], axis = 1)
        print(f"The left median is {np.mean(left_all)} and the right median is {np.mean(right_all)}")

        if args.resmooth:
            print(f"Before: {artifact}")
            artifact = artifact if (smooth_y_tile + smooth_y_neighb) > 0 else 0
            print(f"After: {artifact}")

        if artifact == 1 or args.process_all:
            
            download_raw_tile((tile_x, tile_y), local_path, "processed")
            download_raw_tile((tile_x, tile_y), local_path, "raw")

            if edge == "right":
                print(f"Downloading {neighbor_id}")
                download_raw_tile(neighbor_id, local_path, "raw")
                download_raw_tile(neighbor_id, local_path, "processed")
        else:
            print("The tiles are pretty close, skipping")
            return 0, None, None, 0, 0
    else:
        print("One of the tiles isn't processed, skipping.")
        return 0, None, None, 0, 0

    print("Checking the date overlaps")
    dates = load_dates(tile_x, tile_y, local_path)
    #dates = list(np.sort(np.unique(np.array(dates))))
    dates_neighb = load_dates(neighbor_id[0], neighbor_id[1], local_path)

    #dates_neighb = list(np.sort(np.unique(np.array(dates_neighb))))
    to_rm_tile, to_rm_neighb, min_images = align_dates(dates, dates_neighb)
    print(f"There are {min_images} overlapping")

    if min_images >= 3:
        print("Preprocessing the tiles together")
        print("Loading and splitting the tile")
        s2, dates, interp, s1, dem, _, _ = process_tile(tile_x, tile_y, data, local_path, bbx)
        s2_shape = s2.shape[1:-1]
        #n_tiles_x, n_tiles_y = check_n_tiles(tile_x, tile_y)
        tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
        cloudmask_path = f"{local_path}{str(tile_x)}/{str(tile_y)}/raw/clouds/cloudmask_{tile_idx}.hkl"
        s2, interp, s1, dem, tiles_folder_x = split_to_border(s2, interp, s1, dem, "tile", edge)

        print("Loading and splitting the neighbor tile")
        s2_neighb, dates_neighb, interp_neighb, s1_neighb, dem_neighb, _, _ = \
            process_tile(neighbor_id[0], neighbor_id[1], data, args.local_path, neighb_bbx)
        s2_neighb_shape = s2_neighb.shape[1:-1]
        tile_idx = f'{str(neighbor_id[0])}X{str(neighbor_id[1])}Y'
        s2_neighb, interp_neighb, s1_neighb, dem_neighb, _ = \
            split_to_border(s2_neighb, interp_neighb, s1_neighb, dem_neighb, "neighbor", edge)
        
        print("After loading and splitting")
        print(s2.shape, s2_neighb.shape)
        print(dates, dates_neighb)
        tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
        cloudmask_path = f"{local_path}{str(tile_x)}/{str(tile_y)}/raw/clouds/cloudmask_{tile_idx}.hkl"
        if os.path.exists(cloudmask_path):
            clm = hkl.load(cloudmask_path).repeat(2, axis = 1).repeat(2, axis = 2)
            clm, _ = split_fn(clm, 'tile')
        else: 
            clm = None

        tile_idx = f'{str(neighbor_id[0])}X{str(neighbor_id[1])}Y'
        cloudmask_path_n = f"{local_path}{str(neighbor_id[0])}/{str(neighbor_id[1])}/raw/clouds/cloudmask_{tile_idx}.hkl"
        if os.path.exists(cloudmask_path_n):
            clm_neighb = hkl.load(cloudmask_path_n).repeat(2, axis = 1).repeat(2, axis = 2)
            clm_neighb, _ = split_fn(clm_neighb, 'neighbor')
        else:
            clm_neighb = None

        # Align dates
        to_rm_tile, to_rm_neighb, min_images = align_dates(dates, dates_neighb)
        if len(to_rm_tile) > 0:
            print(f"Removing: {to_rm_tile} from left")
            s2 = np.delete(s2, to_rm_tile, 0)
            dates = np.delete(dates, to_rm_tile)
            if clm is not None:
                if clm.shape[0] > 0:
                    clm = np.delete(clm, to_rm_tile, 0)
        if len(to_rm_neighb) > 0:
            print(f"Removing: {to_rm_neighb} from right")
            s2_neighb = np.delete(s2_neighb, to_rm_neighb, 0)
            dates_neighb = np.delete(dates_neighb, to_rm_neighb)
            if clm_neighb is not None:
                if clm_neighb.shape[0] > 0:
                    clm_neighb = np.delete(clm_neighb, to_rm_neighb, 0)

        if (clm is not None) and (clm_neighb is not None):
            try:
                clm = np.concatenate([clm_neighb, clm], axis = 2)
                clm = np.float32(clm)
                clm[np.isnan(clm)] = 0.
            except:
                clm = None

        print("Concatenating S2!")
        print(s2.shape, s2_neighb.shape)
        print(dates, dates_neighb)
        s2_diff = s2.shape[2] - s2_neighb.shape[2]
        s2 = s2[:, :, s2_diff // 2: -(s2_diff // 2), :] if s2_diff > 0 else s2
        s2_neighb = s2_neighb = s2_neighb[:, :, - (s2_diff // 2) : (s2_diff // 2)] if s2_diff < 0 else s2_neighb
        dem_diff = dem.shape[1] - dem_neighb.shape[1]
        if dem_diff > 0:
            dem = dem[:, dem_diff // 2: -(dem_diff // 2)]
        if dem_diff < 0:
            dem_neighb = dem_neighb[:, -(dem_diff // 2) : (dem_diff // 2)]

        s2_all = np.empty((s2.shape[0], s2.shape[1], s2.shape[2] + s2_neighb.shape[2], s2.shape[-1]), dtype = np.float32)
        s2_all[:, :, :s2.shape[2], :] = s2
        s2_all[:, :, s2.shape[2]:, :] = s2_neighb
        s2 = s2_all
        s2_neighb = None
        #s2 = np.concatenate([s2, s2_neighb], axis = 2)
        dem = np.concatenate([dem, dem_neighb], axis = 1)

        print("Removing clouds, shared tile")     
        s2, interp, dates = preprocess_tile(s2, dates, None, clm, "tile", dem, bbx)
        s2, dates, interp = deal_w_missing_px(s2, dates, interp)
        # this is important! Otherwise hist_align will happen.
        dates_neighb = dates
        indices = make_and_smooth_indices(s2, dates)
        s2 = regularize_and_smooth(s2, dates)
        min_clear_images_per_date = np.sum(interp != 1, axis = 0)

    elif min_images < 3:
        time1 = time.time()
        print("Separate tile preprocessing")
        print("Loading and processing the tile")
        s2, dates, interp, s1, dem, _, _ = process_tile(tile_x, tile_y, data, local_path, bbx)
        s2_shape = s2.shape[1:-1]
        #n_tiles_x, n_tiles_y = check_n_tiles(tile_x, tile_y)
        tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
        cloudmask_path = f"{local_path}{str(tile_x)}/{str(tile_y)}/raw/clouds/cloudmask_{tile_idx}.hkl"
        if os.path.exists(cloudmask_path):
            clm = hkl.load(cloudmask_path).repeat(2, axis = 1).repeat(2, axis = 2)
        else:
            clm = None
        s2, interp, dates = preprocess_tile(s2, dates, interp, clm, "tile", dem, bbx)

        print("Splitting the tile to border")
        s2, interp, s1, dem, tiles_folder_x = split_to_border(s2, interp, s1, dem, "tile", edge)
       
        print("Loading and processing the neighbor tile")
        s2_neighb, dates_neighb, interp_neighb, s1_neighb, dem_neighb, _, _ = \
            process_tile(neighbor_id[0], neighbor_id[1], data, args.local_path, neighb_bbx)
        s2_neighb_shape = s2_neighb.shape[1:-1]
        tile_idx = f'{str(neighbor_id[0])}X{str(neighbor_id[1])}Y'
        cloudmask_path = f"{local_path}{str(neighbor_id[0])}/{str(neighbor_id[1])}/raw/clouds/cloudmask_{tile_idx}.hkl"
        if os.path.exists(cloudmask_path):
            clm = hkl.load(cloudmask_path).repeat(2, axis = 1).repeat(2, axis = 2)
        else:
            clm = None
        s2_neighb, interp_neighb, dates_neighb = preprocess_tile(
            s2_neighb, dates_neighb, interp_neighb, clm, 'neighbor', dem_neighb, neighb_bbx)

        print("Splitting the neighbor tile to border")
        s2_neighb, interp_neighb, s1_neighb, dem_neighb, _ = \
            split_to_border(s2_neighb, interp_neighb, s1_neighb, dem_neighb, "neighbor", edge)

        print("Aligning the dates")
        to_rm_tile, to_rm_neighb, min_images = align_dates(dates, dates_neighb)
        min_clear_images_per_date = np.concatenate([np.sum(interp[..., -(SIZE + 14)//2:] != 1, axis = (0)),
                                                   np.sum(interp_neighb[..., :(SIZE + 14)//2] != 1, axis = (0))], axis = 1)
        print(min_clear_images_per_date.shape)
        print(to_rm_tile, to_rm_neighb)
        if min_images >= min_dates:
            if len(to_rm_tile) > 0:
                s2 = np.delete(s2, to_rm_tile, 0)
                interp = np.delete(interp, to_rm_tile, 0)
                dates = np.delete(dates, to_rm_tile)
            if len(to_rm_neighb) > 0:
                s2_neighb = np.delete(s2_neighb, to_rm_neighb, 0)
                interp_neighb = np.delete(interp_neighb, to_rm_neighb, 0)
                dates_neighb = np.delete(dates_neighb, to_rm_neighb)
            hist_align = False
        else:
            hist_align = True
        print(dates, dates_neighb)

        s2, dates, interp = deal_w_missing_px(s2, dates, interp)
        s2_indices = make_and_smooth_indices(s2, dates)
        
    
        time1 = time.time()
        s2 = regularize_and_smooth(s2, dates)
        s2_neighb, dates_neighb, interp_neighb = deal_w_missing_px(s2_neighb, dates_neighb, interp_neighb)
        neighb_indices = make_and_smooth_indices(s2_neighb, dates_neighb)
        smneighb = Smoother(lmbd = 100, size = 24, nbands = 10, dimx = s2_neighb.shape[1], dimy = s2_neighb.shape[2])
        s2_neighb = regularize_and_smooth(s2_neighb, dates_neighb)
        time2 = time.time()
        print(f"Finished smoothing in {np.around(time2 - time1, 1)} seconds")

        print("Concatenating the files")
        if s1.shape[0] != s1_neighb.shape[0]:
            if s1.shape[0] == 12:
                if s1_neighb.shape[0] == 6:
                    s1_indices = [0, 2, 4, 6, 8, 10]
                    s1 = s1[s1_indices]
                if s1_neighb.shape[0] == 4:
                    s1_indices = [0, 3, 6, 9]
                    s1 = s1[s1_indices]

            elif s1_neighb.shape[0] == 12:
                if s1.shape[0] == 6:
                    s1_indices = [0, 2, 4, 6, 8, 10]
                    s1_neighb = s1_neighb[s1_indices]
                if s1.shape[0] == 4:
                    s1_indices = [0, 3, 6, 9]
                    s1_neighb = s1_neighb[s1_indices]

            elif s1.shape[0] == 6:
                if s1_neighb.shape[0] == 4:
                    s1_indices = [0, 1, 3, 5]
                    s1 = s1[s1_indices]

            elif s1_neighb.shape[0] == 6:
                if s1.shape[0] == 4:
                    s1_indices = [0, 1, 3, 5]
                    s1_neighb = s1_neighb[s1_indices]

        time1 = time.time()
        s2_all = np.empty((s2.shape[0], s2.shape[1], s2.shape[2] + s2_neighb.shape[2], s2.shape[-1]), dtype = np.float32)
        s2_all[:, :, :s2.shape[2], :] = s2
        s2_all[:, :, s2.shape[2]:, :] = s2_neighb
        s2 = s2_all
        s2_neighb = None

        indices = np.concatenate([s2_indices, neighb_indices], axis = 2)
        dem = np.concatenate([dem, dem_neighb], axis = 1)
        interp = np.concatenate([interp[:interp_neighb.shape[0]],
                                 interp_neighb[:interp.shape[0]]], axis = 2)
        time2 = time.time()
        print(f"Finished concat tile in {np.around(time2 - time1, 1)} seconds")

    s2_neighb = None
    dem_neighb = None
    interp_neighb = None
    #s1_neighb = None
    neighb_indices = None
    s2_indices = None

    print("Continuing with the shared processing pipeline")
    print("Concatenating the files")
    s1 = np.concatenate([s1, s1_neighb], axis = 2)

    time1 = time.time()
    print(initial_bbx)
    initial_bbx[0] += (300/30) * (1/360)
    initial_bbx[2] += (300/30) * (1/360)
    print(initial_bbx)
    bbx_middle = make_bbox_rect(initial_bbx, expansionx = (342/30) / 1.03, expansiony = 300/30)
    print(bbx_middle)
    s2 = superresolve_large_tile(s2, superresolve_sess)
    out = np.empty((s2.shape[0], s2.shape[1], s2.shape[2], 14), dtype = np.float32)
    out[..., :10] = s2
    out[..., 10:] = indices
    s2 = out
    time2 = time.time()
    print(f"Finished superresolve tile in {np.around(time2 - time1, 1)} seconds")
    print(f"the sentinel-2 data that goes out is of: {out.shape} shape")
    _ = update_ard_tiles(tile_x, tile_y, np.median(s2, axis = 0), bbx, neighb_bbx)
    #np.save('test_example.npy', np.median(s2, axis = 0))

    n_tiles_x, n_tiles_y = check_n_tiles(tile_x, tile_y)
    print(f"There are {n_tiles_y} y tiles")
    gap_y = int(np.ceil((s1.shape[2] - 216) / 5))
    gap_x = int(np.ceil((s1.shape[1] - 216) / 5 ))
    tiles_folder_y = np.hstack([np.arange(0, s1.shape[2] - 216, gap_y), np.array(s1.shape[1] - 216)])
    tiles_folder_x = np.hstack([np.arange(0, s1.shape[1] - 216, gap_x), np.array(s1.shape[1] - 216)])
    #tiles_array, tiles_folder = make_tiles_right_neighb(tiles_folder_x, tiles_folder_y)

    def cartesian(*arrays):
        mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
        dim = len(mesh)  # number of dimensions
        elements = mesh[0].size  # number of elements, any index will do
        flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
        reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
        return reshape

    windows = cartesian(tiles_folder_x, tiles_folder_y)
    win_sizes = np.full_like(windows, 216)
    tiles_folder = np.hstack([windows, win_sizes])
    tiles_folder = np.sort(tiles_folder, axis = 0)
    tiles_folder[:, 1] = np.tile(np.unique(tiles_folder[:, 1]), 
        int(len(tiles_folder[:, 1]) / len(np.unique(tiles_folder[:, 1]))))
    tiles_array = tof_downloading.make_overlapping_windows(tiles_folder, diff = 7)

    if not np.array_equal(np.array(dates), np.array(dates_neighb)):
        hist_align = True
        print("Aligning histogram")
    else:
        hist_align = False
        print("No align needed")

    right_all = neighbor_tif[:, :(SIZE // 2)]
    left_all = tile_tif[:, -(SIZE // 2):]

    process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, gap_sess, tiles_folder, tiles_array,
        right_all, left_all, hist_align, min_clear_images_per_date)

    return 1, s2_shape, s2_neighb_shape, diff_for_compare, min_images


def adjust_resegment(res, mults, n):
    sum_normal_mults = np.sum(mults[..., :n], axis = -1)
    return res * np.maximum(sum_normal_mults, 1.)


def mosaic_subtiles(preds, mults, na, kind, left, right, up, down):
    na = np.tile(na, (1, 1, preds.shape[-1]))
    preds[na > 0] = np.nan

    mults[np.isnan(preds)] = 0.
    mults = mults / np.sum(mults, axis = -1)[..., np.newaxis]
    preds = np.nansum(preds * mults, axis = -1)
    m = (np.arange(0, 300, 1) / 300)
    m = m ** 1.33
    border = np.tile(m, (SIZE // 2, 1))
    mult_arr = (np.arange(0, SIZE // 2, 1) / (SIZE // 2))[:, np.newaxis]
    mult_arr = np.ones((SIZE // 2, preds.shape[1])) * mult_arr
    mult_arr = mult_arr ** 1.2
    
    left = True if left is not None else False
    right = True if right is not None else False
    up = True if up is not None else False
    down = True if down is not None else False
    print(left, right, up, down)

    if kind == 'n':
        m = fspecial_gauss(preds.shape[0], preds.shape[0] / 5.25)
        m = resize(m, (preds.shape[0], preds.shape[1]), order = 1)
    if kind == 'r':
        m = np.copy(mult_arr)
        if up:
            m[:, :300] *= border
        if down:
            m[:, -300:] *= np.fliplr(border)
        m = resize(m, (SIZE // 2, preds.shape[1]), 1)
        zero_arr = np.zeros((preds.shape[0] - (SIZE // 2), preds.shape[1]))
        m = np.concatenate([np.zeros((preds.shape[0] - (SIZE // 2), preds.shape[1])), m], axis = 0)
        m = resize(m, (preds.shape[0], preds.shape[1]), order = 1)
    if kind == 'l':
        m = np.copy(mult_arr)
        m = np.flipud(m)
        if up:
            m[:, :300] *= border
        if down:
            m[:, -300:] *= np.fliplr(border)
        m = resize(m, (SIZE // 2, preds.shape[1]), 1)
        zero_arr = np.zeros((preds.shape[0] - (SIZE // 2), preds.shape[1]))
        m = np.concatenate([m, zero_arr], axis = 0)
        m = resize(m, (preds.shape[0], preds.shape[1]), order = 1)
    if kind == 'u':
        m = np.copy(mult_arr)
        m = np.flipud(m)
        if left:
            m[:, :300] *= border
        if right:
            m[:, -300:] *= np.fliplr(border)
        zero_arr = np.zeros((preds.shape[0] - (SIZE // 2), preds.shape[1]))
        m = resize(m, (SIZE // 2, preds.shape[1]), 1)
        m = np.concatenate([m, zero_arr], axis = 0)
        m = m.T
        m = resize(m, (preds.shape[0], preds.shape[1]), order = 1)
    if kind == 'd':
        m = np.copy(mult_arr)
        if right:
            m[:, :300] *= border
        if left:
            m[:, -300:] *= np.fliplr(border)
        m = resize(m, (SIZE // 2, preds.shape[1]), 1)
        zero_arr = np.zeros((preds.shape[0] - (SIZE // 2), preds.shape[1]))
        m = np.concatenate([zero_arr, m], axis = 0)
        m = np.flipud(m.T)
        m = resize(m, (preds.shape[0], preds.shape[1]), order = 1)
    m[np.isnan(preds)] = 0.
    return preds, m


def recreate_resegmented_tifs(out_folder: str, shape) -> np.ndarray:
    """
    Loads the .npy subtile files in an output folder and mosaics the overlapping predictions
    to return a single .npy file of tree cover for the 6x6 km tile
    Additionally, applies post-processing threshold rules and implements no-data flag of 255
    
        Parameters:
         out_folder (os.Path): location of the prediction .npy files 
    
        Returns:
         predictions (np.ndarray): 6 x 6 km tree cover data as a uint8 from 0-100 w/ 255 no-data flag
    """
    from skimage.transform import resize
    print(f"Recreating: {out_folder}")
    n_up = len(glob(out_folder + "*/up*.npy"))
    n_down = len(glob(out_folder + "*/down*.npy"))
    n_left = len(glob(out_folder + "*/left*.npy"))
    n_right = len(glob(out_folder + "right*/*.npy"))

    right = [x for x in os.listdir(out_folder) if 'right' in x]
                      
    x_tiles = [x for x in os.listdir(out_folder) if 'right' not in x]
    x_tiles = [x for x in x_tiles if '.DS' not in x]
    x_tiles = [x for x in x_tiles if len(os.listdir(out_folder + "/" + x)) > 0]
    x_tiles = [int(x) for x in x_tiles]

    max_x = np.max(x_tiles) + SIZE
    for x_tile in x_tiles:
        y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        y_tiles = [y for y in y_tiles if 'left' not in y]
        y_tiles = [y for y in y_tiles if 'down' not in y]
        y_tiles = [int(y[:-4]) for y in y_tiles if 'up' not in y]
        
    n_tiles = len(glob(out_folder + "*/*.npy"))
    n_border = n_up + n_down + n_left + n_right
    predictions_n = np.full((shape[1], shape[0], n_tiles - n_border), np.nan, dtype = np.float32)
    mults_n = np.full((shape[1], shape[0], n_tiles - n_border), 0, dtype = np.float32)
    predictions_l = None
    predictions_r = None
    predictions_u = None
    predictions_d = None

    i = 0

    sum_normal = np.zeros((shape[1], shape[0]))
    sum_normal_na = np.zeros((shape[1], shape[0]))
    sum_resegment = np.zeros((shape[1], shape[0]))
    sum_resegment_na = np.zeros((shape[1], shape[0]))

    small_tile = False
    for x_tile in x_tiles:
        y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        y_tiles = [y for y in y_tiles if 'left' not in y]
        y_tiles = [y for y in y_tiles if 'down' not in y]
        y_tiles = [int(y[:-4]) for y in y_tiles if 'up' not in y]
        for y_tile in y_tiles:
            output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
            if os.path.exists(output_file):
                prediction = np.load(output_file)
                size_y = prediction.shape[0]
                size_x = prediction.shape[1]
                subtile_size = np.maximum(size_x, size_y)
                if np.sum(prediction) < size_x*size_y*255:
                    prediction = (prediction * 100).T.astype(np.float32)
                    if subtile_size == 208 or subtile_size == 216:
                        fspecial_size = 44
                    elif subtile_size == 348:
                        fspecial_size = 85
                    elif subtile_size == 412:
                        fspecial_size = 95
                    elif subtile_size == 168:
                        fspecial_size = 38
                    else:
                        fspecial_size = 28
                        small_tile = True
                    if (x_tile + size_x - 1) < shape[1] and (y_tile + size_y- 1) < shape[0]:
                        predictions_n[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = prediction
                        fspecial_i = fspecial_gauss(subtile_size, fspecial_size)
                        fspecial_i[prediction > 100] = 0.
                        mults_n[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = fspecial_i

                        counter = np.ones_like(prediction)
                        sum_normal[x_tile: x_tile+size_x, y_tile:y_tile + size_y] += counter
                        counter[prediction <= 100] = 0.
                        sum_normal_na[x_tile: x_tile+size_x, y_tile:y_tile + size_y,] += counter
                    else:
                        print(f"Skipping {x_tile, y_tile} because of {predictions_n.shape}")
                    i += 1
                
    # LEFT BLOCK
    if n_left > 0:
        i = 0
        predictions_l = np.full((shape[1], shape[0], n_left), np.nan, dtype = np.float32)
        mults_l = np.full((shape[1], shape[0], n_left), 0, dtype = np.float32)
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[4:-4]) for y in y_tiles if 'left' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/left" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    size_y = prediction.shape[0]
                    size_x = prediction.shape[1] // 2
                    subtile_size = np.maximum(size_x * 2, size_y)
                    if subtile_size == 208 or subtile_size == 216:
                        fspecial_size = 44
                    elif subtile_size == 348:
                        fspecial_size = 85
                    elif subtile_size == 412:
                        fspecial_size = 95
                    elif subtile_size == 588 or subtile_size >= 620:
                        fspecial_size = 150# if not small_tile else 100
                    else:
                        fspecial_size = 28
                    fspecial_i = fspecial_gauss(subtile_size, fspecial_size)[size_x:, :]
                    fspecial_i = resize(fspecial_i, (size_x, size_y), order = 1)
                    if np.sum(prediction) < size_x*size_y*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[size_x:, :]
                        predictions_l[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = prediction
                        preds_to_adj = predictions_l[x_tile: x_tile+size_x, y_tile:y_tile + size_y, :]
                        preds_to_adj[prediction > 100] = 255.
                        fspecial_i[prediction > 100] = 0.
                        mults_l[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = fspecial_i

                        counter = np.ones_like(prediction)
                        sum_resegment[x_tile: x_tile+size_x, y_tile:y_tile + size_y] += counter
                        counter[prediction <= 100] = 0.
                        sum_resegment_na[x_tile: x_tile+size_x, y_tile:y_tile + size_y,] += counter
                    i += 1
                    
    # RIGHT BLOCK
    if n_right > 0:
        i = 0
        predictions_r = np.full((shape[1], shape[0], n_right), np.nan, dtype = np.float32)
        mults_r = np.full((shape[1], shape[0], n_right), 0, dtype = np.float32)
        for x_tile in right:
            x_tile_name = x_tile
            x_tile = int(x_tile[5:])
            y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile_name) + "/") if '.DS' not in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile_name) + "/" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    size_y = prediction.shape[0]
                    size_x = prediction.shape[1] // 2
                    subtile_size = np.maximum(size_x * 2, size_y)
                    if subtile_size == 208 or subtile_size == 216:
                        fspecial_size = 44
                    elif subtile_size == 348:
                        fspecial_size = 85
                    elif subtile_size == 412:
                        fspecial_size = 95
                    elif subtile_size == 588 or subtile_size >= 620:
                        fspecial_size = 150# if not small_tile else 100
                    else:
                        fspecial_size = 28
                    fspecial_i = fspecial_gauss(subtile_size, fspecial_size)[:size_x, :]
                    fspecial_i = resize(fspecial_i, (size_x, size_y), order = 1)

                    if np.sum(prediction) < size_x*size_y*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:size_x, :]
                        predictions_r[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = prediction
                        preds_to_adj = predictions_r[x_tile: x_tile+size_x, y_tile:y_tile + size_y, :]
                        preds_to_adj[prediction > 100] = 255.
                        fspecial_i[prediction > 100] = 0.
                        mults_r[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = fspecial_i

                        counter = np.ones_like(prediction)
                        sum_resegment[x_tile: x_tile+size_x, y_tile:y_tile + size_y] += counter
                        counter[prediction <= 100] = 0.
                        sum_resegment_na[x_tile: x_tile+size_x, y_tile:y_tile + size_y,] += counter
                    i += 1
                    
    if n_up > 0:
        i = 0
        predictions_u = np.full((shape[1], shape[0], n_up), np.nan, dtype = np.float32)
        mults_u = np.full((shape[1], shape[0], n_up), 0, dtype = np.float32)
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[2:-4]) for y in y_tiles if 'up' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/up" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    size_y = prediction.shape[0] // 2
                    size_x = prediction.shape[1]
                    subtile_size = np.maximum(size_x, size_y * 2)
                    if subtile_size == 208 or subtile_size == 216:
                        fspecial_size = 44
                    elif subtile_size == 348:
                        fspecial_size = 85
                    elif subtile_size == 412:
                        fspecial_size = 95
                    elif subtile_size == 588 or subtile_size >= 620:
                        fspecial_size = 150# if not small_tile else 100
                    else:
                        fspecial_size = 28
                    fspecial_i = fspecial_gauss(subtile_size, fspecial_size)[:, size_y:]
                    fspecial_i = resize(fspecial_i, (size_x, size_y), order = 1)
                    if np.sum(prediction) < size_x*size_y*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:, size_y:]
                        predictions_u[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = prediction
                        preds_to_adj = predictions_u[x_tile: x_tile+size_x, y_tile:y_tile + size_y, :]
                        preds_to_adj[prediction > 100] = 255.
                        fspecial_i[prediction > 100] = 0.
                        mults_u[x_tile: x_tile+ size_x, y_tile:y_tile + size_y, i] = fspecial_i

                        counter = np.ones_like(prediction)
                        sum_resegment[x_tile: x_tile+size_x, y_tile:y_tile + size_y] += counter
                        counter[prediction <= 100] = 0.
                        sum_resegment_na[x_tile: x_tile+size_x, y_tile:y_tile + size_y,] += counter
                    i += 1
                    
    if n_down > 0:
        i = 0
        predictions_d = np.full((shape[1], shape[0], n_down), np.nan, dtype = np.float32)
        mults_d = np.full((shape[1], shape[0], n_down), 0, dtype = np.float32)
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[4:-4]) for y in y_tiles if 'down' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/down" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    size_y = prediction.shape[0] // 2
                    size_x = prediction.shape[1]
                    subtile_size = np.maximum(size_x, size_y * 2)
                    if subtile_size == 208 or subtile_size == 216:
                        fspecial_size = 44
                    elif subtile_size == 348:
                        fspecial_size = 85
                    elif subtile_size == 412:
                        fspecial_size = 95
                    elif subtile_size == 588 or subtile_size >= 620:
                        fspecial_size = 150# if not small_tile else 100
                    else:
                        fspecial_size = 28
                    fspecial_i = fspecial_gauss(subtile_size, fspecial_size)[:, :size_y]
                    fspecial_i = resize(fspecial_i, (size_x, size_y), order = 1)
                    if np.sum(prediction) < size_x*size_y*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:, :size_y]
                        predictions_d[x_tile: x_tile+size_x, y_tile:y_tile + size_y, i] = prediction
                        preds_to_adj = predictions_d[x_tile: x_tile+size_x, y_tile:y_tile + size_y, :]
                        preds_to_adj[prediction > 100] = 255.
                        fspecial_i[prediction > 100] = 0.
                        mults_d[x_tile: x_tile+ size_x, y_tile:y_tile + size_y, i] = fspecial_i

                        counter = np.ones_like(prediction)
                        sum_resegment[x_tile: x_tile+size_x, y_tile:y_tile + size_y] += counter
                        counter[prediction <= 100] = 0.
                        sum_resegment_na[x_tile: x_tile+size_x, y_tile:y_tile + size_y,] += counter
                    i += 1
                    
    isnanpreds = np.zeros_like(sum_resegment)
    # Non resegmented areas with no normal predictions are NA
    isnanpreds[(sum_resegment == 0) * ((sum_normal - sum_normal_na) == 0)] = 1.

    # Resegmented areas, without BOTH normal and ALL resegmented are NA
    isnanpreds[(sum_resegment > 0) * (np.logical_or(((sum_normal - sum_normal_na) == 0),
                                              (sum_resegment_na > 0)))] = 1.
    isnanpreds = isnanpreds[..., np.newaxis]
    preds_n, mults_n = mosaic_subtiles(predictions_n, mults_n, isnanpreds, "n",
                                       predictions_l, predictions_r, 
                                       predictions_u, predictions_d)
    if predictions_r is not None:
        preds_r, mults_r = mosaic_subtiles(predictions_r, mults_r,
                                           isnanpreds, "r", 
                                           predictions_l, predictions_r, 
                                           predictions_u, predictions_d)
        mults_r[np.sum(~np.isnan(predictions_r), axis = -1) == 0] = 0.
    else:
        preds_r = np.zeros_like(preds_n)
        mults_r = np.zeros_like(preds_n)
    if predictions_l is not None:
        preds_l, mults_l = mosaic_subtiles(predictions_l, mults_l, isnanpreds, "l",
                                           predictions_l, predictions_r, 
                                           predictions_u, predictions_d)
        mults_l[np.sum(~np.isnan(predictions_l), axis = -1) == 0] = 0.
    else:
        preds_l = np.zeros_like(preds_n)
        mults_l = np.zeros_like(preds_n)
    if predictions_u is not None:
        preds_u, mults_u = mosaic_subtiles(predictions_u, mults_u, isnanpreds, 'u',
                                           predictions_l, predictions_r, 
                                           predictions_u, predictions_d)
        mults_u[np.sum(~np.isnan(predictions_u), axis = -1) == 0] = 0.
    else:
        preds_u = np.zeros_like(preds_n)
        mults_u = np.zeros_like(preds_n)
    if predictions_d is not None:
        preds_d, mults_d = mosaic_subtiles(predictions_d, mults_d, isnanpreds, "d",
                                           predictions_l, predictions_r, 
                                           predictions_u, predictions_d)
        mults_d[np.sum(~np.isnan(predictions_d), axis = -1) == 0] = 0.
    else:
        preds_d = np.zeros_like(preds_n)
        mults_d = np.zeros_like(preds_n)
        
    sums = (mults_l + mults_r + mults_u + mults_d + mults_n)
    preds = (preds_l * (mults_l / sums)) + (preds_d * (mults_d / sums))
    preds = preds + (preds_r * (mults_r / sums)) + (preds_u * (mults_u / sums)) 
    preds = preds + (preds_n * (mults_n / sums))
    preds[np.isnan(preds)] = 255.
    preds[isnanpreds.squeeze() == 1.] = 255.
    
    return preds, sums


def cleanup(path_to_tile, path_to_right, delete = True, upload = True):

    for file in glob(path_to_right + "processed/*/left*"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'{str(args.year)}/processed/{str(int(x) + 1)}/{str(y)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for file in glob(path_to_tile + "processed/right*/*.npy"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'{str(args.year)}/processed/{x}/{y}/' + internal_folder
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for folder in glob(path_to_tile + "processed/*"):
        for file in os.listdir(folder):
            if delete:
                os.remove(folder + "/" + file)

    for folder in glob(path_to_tile + "raw/*"):
        for file in os.listdir(folder):
            if delete:
                os.remove(folder + "/" + file)

    for file in os.listdir(folder):
        if delete:
            os.remove(folder + "/" + file)

    return None


def cleanup_row_or_col(idx, current_idx, local_path):
    if int(idx) < current_idx:
        print("Emptying the working directory")
        current_idx = int(idx)
        try:
            shutil.rmtree(local_path)
            os.makedirs(local_path)
        except Exception as e:
            print(f"Ran into {str(e)}")
    return current_idx
        

if __name__ == "__main__":
    SIZE = 670
    SIZE_Y = 206

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/retrain-combined-ca-220-684/')
    parser.add_argument("--gap_model_path", dest = 'gap_model_path', default = '../models/182-gap-sept/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/nov-40k-swir/')
    parser.add_argument("--db_path", dest = "db_path", default = "process_area_2022.csv")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--start_y", dest = "start_y", default = 10000)
    parser.add_argument("--process_all", dest = "process_all", default = False)
    parser.add_argument("--resmooth", dest = "resmooth", default = False)
    parser.add_argument("--year", dest = "year", default = 2020)
    parser.add_argument("--length", dest = "length", default = 4)
    parser.add_argument("--gen_feats", dest = "gen_feats", default = False, type=str2bool, nargs='?',
                        const=True)
    args = parser.parse_args()

    print(f'Country: {args.country} \n'
          f'Local path: {args.local_path} \n'
          f'Predict model path: {args.predict_model_path} \n'
          f'Superrresolve model path: {args.superresolve_model_path} \n'
          f'DB path: {args.db_path} \n'
          f'S3 Bucket: {args.s3_bucket} \n'
          f'YAML path: {args.yaml_path} \n'
          f'Current dir: {os.getcwd()} \n'
          f'Year: {args.year} \n'
          #f'Model: {args.model} \n'
          f'gen_feats: {args.gen_feats} \n'
          )

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

    superresolve_graph_def = tf.compat.v1.GraphDef()
    predict_graph_def = tf.compat.v1.GraphDef()
    gap_graph_def = tf.compat.v1.GraphDef()

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
        #if args.length == 12:
        predict_latefeats = predict_sess.graph.get_tensor_by_name(f"predict/csse_out_mul/mul:0") 
        predict_earlyfeats = predict_sess.graph.get_tensor_by_name(f"predict/gru_drop/drop_block2d/cond/Merge:0")    
        print(f'Late feats is: {predict_latefeats}')
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
        print(f"The input shape is: {predict_inp.shape}")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")

    gap_file = None
    gap_graph = None
    gap_sess = None
    gap_logits = None
    gap_inp = None

    # Normalization mins and maxes for the prediction input
    min_all = [0.006576638437476157, 0.0162050812542916, 0.010040436408026246, 
               0.013351644159609368, 0.01965362020294499, 0.014229037918669413, 
               0.015289539940489814, 0.011993591210803388, 0.008239871824216068,
               0.006546120393682765, 0.0, 0.0, 0.0, -0.1409399364817101,
               -0.4973397113668104, -0.09731556326714398, -0.7193834232943873]

    max_all = [0.2691233691920348, 0.3740291447318227, 0.5171435111009385, 
               0.6027466239414053, 0.5650263218127718, 0.5747005416952773,
               0.5933928435187305, 0.6034943160143434, 0.7472037842374304,
               0.7000076295109483, 0.509269855802243, 0.948334642387533, 
               0.6729257769285485, 0.8177635298774327, 0.35768999002433816,
               0.7545951919107605, 0.7602693339366691]

    min_all = np.array(min_all)
    max_all = np.array(max_all)
    min_all = np.broadcast_to(min_all, (1, 1, 1, 17)).astype(np.float32)
    max_all = np.broadcast_to(max_all, (1, 1, 1, 17)).astype(np.float32)

    midrange = (max_all + min_all) / 2
    midrange = midrange.astype(np.float32)
    rng = max_all - min_all
    rng = rng.astype(np.float32)

    if os.path.exists(args.db_path):
        data = pd.read_csv(args.db_path)
        data = data[data['country'] == args.country]
        data = data.reset_index(drop = True)
        print(f"There are {len(data)} tiles for {args.country}")
    else:
        raise Exception(f"The database does not exist at {args.db_path}")

    try:
        data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
        data['X_tile'] = pd.to_numeric(data['X_tile'])
        data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
        data['Y_tile'] = pd.to_numeric(data['Y_tile'])
    except Exception as e:
        print(f"Ran into {str(e)} error")
        time.sleep(1)
    
    data['X_tile'] = data['X_tile'].astype(int)
    data['Y_tile'] = data['Y_tile'].astype(int)
    data = data.sort_values(['Y_tile', 'X_tile'], ascending=[False, True])
    print(len(data))
    
    current_y = 10000
    for index, row in data.iterrows(): # We want to sort this by the X so that it goes from left to right
        time1 = time.time()
        x = str(int(row['X_tile']))
        y = str(int(row['Y_tile']))
        x = x[:-2] if ".0" in x else x
        y = y[:-2] if ".0" in y else y

        current_y = cleanup_row_or_col(idx = y,
                            current_idx = current_y,
                            local_path = args.local_path)

        if int(y) < int(args.start_y):
            path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/'
            path_to_right = f'{args.local_path}{str(int(x) + 1)}/{str(y)}/'
            print(path_to_tile, path_to_right)

            initial_bbx = [row['X'], row['Y'], row['X'], row['Y']]
            bbx = make_bbox(initial_bbx, expansion = 300/30)
            print(initial_bbx)
            print(data['X_tile'][index], data['Y_tile'][index])
            data_neighb = data.copy()
            neighb_bbx = None
            print(int(x) + 1, int(y))
            try:
                data_neighb = data_neighb[data_neighb['X_tile'] == int(x) + 1]
                data_neighb = data_neighb[data_neighb['Y_tile'] == int(y)]
                data_neighb = data_neighb.reset_index()
                neighb = [data_neighb['X'][0], data_neighb['Y'][0], data_neighb['X'][0], data_neighb['Y'][0]]
                print(data_neighb['X_tile'][0], data_neighb['Y_tile'][0])
                neighb_bbx = make_bbox(neighb, expansion = 300/30)
                print(neighb_bbx)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Ran into {str(e)}")
            #try:
            time1 = time.time()
            finished, s2_shape, s2_neighb_shape, diff, min_images = resegment_border(x, y, "right", args.local_path, bbx, neighb_bbx, 2, initial_bbx)
            time2 = time.time()
            print(f"Finished the predictions in: {np.around(time2 - time1, 1)} seconds")
            #except KeyboardInterrupt:
            #    break
            
            #except Exception as e:
            #    print(f"Ran into {str(e)}")
            #    finished = 0
            #    s2_shape = (0, 0)

            if finished == 1:
                try:
                    predictions_left, _ = recreate_resegmented_tifs(path_to_tile + "processed/", s2_shape)
                    predictions_right, _ = recreate_resegmented_tifs(path_to_right + "processed/", s2_neighb_shape)

                    right = predictions_right[:8, :].astype(np.float32)
                    left = predictions_left[-8:, :].astype(np.float32)
                    right[right == 255] = np.nan
                    left[left == 255] = np.nan
                    right_mean = np.nanmean(right, axis = 0)
                    left_mean = np.nanmean(left, axis = 0)
                    smooth_diff = np.nanmean(abs(right_mean - left_mean))
                    diff = 100 if np.isnan(diff) else diff
                    print(f"Before smooth: {diff}, after smooth: {smooth_diff}")

                    if (smooth_diff > 5 and min_images == 2):
                        print(f"Smooth diff {smooth_diff}, 2 min images, trying with histogram matching")
                        # If there were only 2 overlapping images, and the resegment didnt help
                        # Test out doing histogram alignment, if it is better, use it
                        predictions_left_original = np.copy(predictions_left)
                        predictions_right_original = np.copy(predictions_right)

                        finished, s2_shape, s2_neighb_shape, diff, min_images  = resegment_border(x, y, "right", args.local_path, bbx, neighb_bbx, 3)
                        predictions_left, _ = recreate_resegmented_tifs(path_to_tile + "processed/", s2_shape)
                        predictions_right, _ = recreate_resegmented_tifs(path_to_right + "processed/", s2_neighb_shape)

                        right = predictions_right[:8, :].astype(np.float32)
                        left = predictions_left[-8:, :].astype(np.float32)
                        right[right == 255] = np.nan
                        left[left == 255] = np.nan
                        right_mean = np.nanmean(right, axis = 0)
                        left_mean = np.nanmean(left, axis = 0)
                        smooth_diff_new = np.nanmean(abs(right_mean - left_mean))
                        if smooth_diff_new > smooth_diff:
                            # If the histogram matched version is worse than the 2 image version
                            # Then use the 2 image version, otherwise use histogram version
                            predictions_left = predictions_left_original
                            predictions_right = predictions_right_original

                    if smooth_diff < (diff + 20) or np.isnan(smooth_diff):

                        if os.path.exists(f"{path_to_tile}/{str(x)}X{str(y)}Y_SMOOTH_XY.tif"):
                            suffix = "_SMOOTH_XY"
                        elif os.path.exists(f"{path_to_tile}/{str(x)}X{str(y)}Y_SMOOTH_Y.tif"):
                            suffix = "_SMOOTH_XY"
                        else:
                            suffix = "_SMOOTH_X"

                        file = write_tif(predictions_left, bbx, x, y, path_to_tile, suffix)
                        key = f'{str(args.year)}/tiles/{x}/{y}/{str(x)}X{str(y)}Y{suffix}.tif'
                        uploader.upload(bucket = args.s3_bucket, key = key, file = file)


                        if os.path.exists(f"{path_to_right}/{str(int(x) + 1)}X{str(int(y))}Y_SMOOTH_XY.tif"):
                            suffix = "_SMOOTH_XY"
                        elif os.path.exists(f"{path_to_right}/{str(int(x) + 1)}X{str(int(y))}Y_SMOOTH_Y.tif"):
                            suffix = "_SMOOTH_XY"
                        else:
                            suffix = "_SMOOTH_X"

                        file = write_tif(predictions_right, neighb_bbx, str(int(x) + 1), y, path_to_right, suffix)
                        key = f'{str(args.year)}/tiles/{str(int(x) + 1)}/{y}/{str(int(x) + 1)}X{str(y)}Y{suffix}.tif'
                        uploader.upload(bucket = args.s3_bucket, key = key, file = file)
                        # dELETE FALSE
                        cleanup(path_to_tile, path_to_right, delete = True, upload = True)
                        time2 = time.time()
                        print(f"Finished resegment in {np.around(time2 - time1, 1)} seconds")
                    else:
                        continue
                        cleanup(path_to_tile, path_to_right, delete = True, upload = False)
   
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Ran into {str(e)}")
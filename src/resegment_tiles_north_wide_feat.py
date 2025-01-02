import pandas as pd
import numpy as np
import os
import yaml
from scipy.sparse.linalg import splu
from skimage.transform import resize
import hickle as hkl
import boto3
from scipy.ndimage import median_filter
from glob import glob
import rasterio
from rasterio.transform import from_origin
import shutil
import bottleneck as bn
import gc
import sys
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()

from preprocessing import slope
from downloading.utils import calculate_and_save_best_images
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from tof.tof_downloading import to_int16, to_float32
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder, download_ard_file
from preprocessing.indices import evi, bi, msavi2, grndvi
from download_and_predict_job import process_tile, make_bbox, convert_to_db, deal_w_missing_px
from download_and_predict_job import fspecial_gauss, make_and_smooth_indices, write_ard_to_tif, str2bool
from download_and_predict_job import float_to_int16
from resegment_tiles_wide_feat import load_mosaic_feats
from tof import tof_downloading

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#tf.disable_v2_behavior()

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
        overlap_top = (SIZE // 2) + 7
        tiles_y = None
        item = item[:, :overlap_top, :]
    if form == 'neighbor':
        overlap_bottom = (SIZE // 2) + 7
        tiles_y = item.shape[1] - (SIZE // 2)
        print("TILES Y", tiles_y)
        item = item[:, -overlap_bottom:, :]
    return item, tiles_y


def cartesian(*arrays):
    mesh = np.meshgrid(*arrays)  # standard numpy meshgrid
    dim = len(mesh)  # number of dimensions
    elements = mesh[0].size  # number of elements, any index will do
    flat = np.concatenate(mesh).ravel()  # flatten the whole meshgrid
    reshape = np.reshape(flat, (dim, elements)).T  # reshape and transpose
    return reshape


def split_to_border(s2, interp, s1, dem, fname, edge):
    if edge == "up":
        if fname == "tile":
            s1, _ = split_fn(s1, fname)
        else:
            s1, tiles_y = split_fn(s1, fname)
        interp, _ = split_fn(interp, fname)
        s2, _ = split_fn(s2, fname)
        dem, _ = split_fn(dem[np.newaxis], fname)
    if fname == "tile":
        return s2, interp, s1, dem.squeeze(), _
    else:
        return s2, interp, s1, dem.squeeze(), tiles_y


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


def predict_subtile(subtile: np.ndarray, sess: "tf.Sess",
    op: 'tf.Tensor') -> np.ndarray:
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

        subtile = np.core.umath.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), LEN)

        preds = sess.run(op,
                              feed_dict={predict_inp:batch_x,
                                         predict_length:lengths})
        
        preds = preds.squeeze()
        clip = (preds.shape[0] - size) // 2
        if clip > 0:
            preds = preds[clip:-clip, clip:-clip]
        #preds = preds[1:-1, 1:-1]

    else:
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


def make_tiles_right_neighb(tiles_folder_x, tiles_folder_y):
    windows = cartesian(tiles_folder_x, tiles_folder_y)
    _size = 230 - 14
    win_sizes = np.full_like(windows, _size + 7)
    tiles_folder = np.hstack([windows, win_sizes])
    tiles_folder = np.sort(tiles_folder, axis = 0)
    tiles_folder[:, 1] = np.tile(np.unique(tiles_folder[:, 1]),
        int(len(tiles_folder[:, 1]) / len(np.unique(tiles_folder[:, 1]))))
    tiles_array = np.copy(tiles_folder)
    tiles_array[1:, 0] -= 7

    tiles_array[:, 1] = 0.
    tiles_array[:, 3] = _size + 14.
    tiles_array[:, 2] = _size + 7.
    tiles_array[1:-1, 2] += 7
    return tiles_array, tiles_folder


def check_n_tiles(x, y):
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    n_tiles_x = len([x for x in os.listdir(path) if x.isnumeric()]) - 1
    n_tiles_y = len([x for x in os.listdir(path + "0/") if x[-4:] == ".npy"]) - 1
    return n_tiles_x, n_tiles_y


def align_subtile_histograms(array) -> np.ndarray:

    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])
        
    left_water = _water_ndwi(
        np.median(array[:, (SIZE + 14) // 2:], axis = 0))
    left_water = left_water >= 0.1
    print(f'{np.mean(left_water)}% of the left is water')

    right_water = _water_ndwi(
        np.median(array[:, :(SIZE + 14) // 2], axis = 0))
    right_water = right_water >= 0.1
    print(f'{np.mean(right_water)}% of the right is water')

    for time in range(array.shape[0]):

        # Identify all of the areas that are, and aren't interpolated
        left = array[time, (SIZE + 14) // 2:]
        right = array[time, :(SIZE + 14) // 2]

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

        before = abs(np.roll(array[time], 1, axis = 0) - array[time])
        before = np.mean(before[(SIZE // 2) + 7], axis = (0, 1))
        print("before", before)

        candidate = np.copy(array[time])
        
        candidate[:(SIZE + 14) // 2] = (
                candidate[:(SIZE + 14) // 2] * std_mult_left + addition_left
        )

        candidate[(SIZE + 14) // 2:] = (
                candidate[(SIZE + 14) // 2:] * std_mult_right + addition_right
        )

        after = abs(np.roll(candidate, 1, axis = 0) - candidate)
        after = np.mean(after[(SIZE // 2) + 7], axis = (0, 1))
        print("after", after)

        if after < before:
            array[time] = candidate

    return array


def adjust_predictions(preds, ref):

    std_src = bn.nanstd(preds)
    std_ref = bn.nanstd(ref)
    mean_src = bn.nanmean(preds)
    mean_ref = bn.nanmean(ref)
    std_mult = (std_ref / std_src)

    addition = (mean_ref - (mean_src * (std_mult)))
    print("adjusting prediction", std_mult, addition, mean_src, mean_ref)
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

    s2 = interpolation.interpolate_na_vals(s2)
    s2_median = np.median(s2, axis = 0)[np.newaxis]
    s1_median = np.median(s1, axis = 0)[np.newaxis]
    
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/processed/',
                  tiles_folder)
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    path_neighbor = f'{args.local_path}{str(x)}/{str(int(y) + 1)}/processed/'

    if LEN == 4:
        s2 = np.reshape(s2, (4, 3, s2.shape[1], s2.shape[2], s2.shape[3]))
        s2 = np.median(s2, axis = 1, overwrite_input = True)
        s1 = np.reshape(s1, (4, 3, s1.shape[1], s1.shape[2], s1.shape[3]))
        s1 = np.median(s1, axis = 1, overwrite_input = True)
        #np.save("s2.npy", s2[..., :3])

    gap_between_years = False
    t = 0
    # Iterate over each subitle and prepare it for processing and generate predictions
    while t < len(tiles_folder):
        _size = 230 - 14
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]
        t += 1

        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[1], tile_folder[0]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]

        subset = s2[:, start_y:end_y, start_x:end_x, :]
        print(f"Subsetting s2 {s2.shape} to {subset.shape}, {start_y}-{end_y}, {start_x} - {end_x}")
        subtile_median_s2 = s2_median[:, start_y:end_y, start_x:end_x, :]
        subtile_median_s1 = s1_median[:, start_y:end_y, start_x:end_x, :]
        interp_tile = interp[:, start_y:end_y, start_x:end_x]
        interp_tile_sum = np.sum(interp_tile, axis = (1, 2))
        min_clear_tile = min_clear_images_per_date[start_y:end_y, start_x:end_x]
        dates_tile = np.copy(dates)
        dem_subtile = dem[np.newaxis, start_y:end_y, start_x:end_x]
        s1_subtile = s1[:, start_y:end_y, start_x:end_x, :]
        output = f"{path}{str(folder_y)}/up{str(0)}.npy"
        output2 = f"{path_neighbor}/{str(folder_y)}/down{str(folder_x)}.npy"

        #min_clear_images_per_date = np.sum(interp_tile == 0, axis = (0))
        no_images_subtile = False
        #if np.percentile(min_clear_images_per_date, 25) < 1:
        #    no_images_subtile = True

        subset[np.isnan(subset)] = np.median(subset[np.isnan(subset)], axis = 0)
        to_remove = np.argwhere(np.sum(np.isnan(subset), axis = (1, 2, 3)) > 10000).flatten()
        if len(to_remove) > 0:
            print(f"Removing {to_remove} NA dates")
            dates_tile = np.delete(dates_tile, to_remove)
            subset = np.delete(subset, to_remove, 0)
            interp_tile = np.delete(interp_tile, to_remove, 0)

        subtile = subset

        # Pad the corner / edge subtiles within each tile
        if subtile.shape[2] == _size + 7:
            pad_d = 7 if start_x != 0 else 0
            pad_u = 7 if start_x == 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (0, 0), (pad_u, pad_d)), 'reflect')
            subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            min_clear_tile = np.pad(min_clear_tile, ((0, 0), (pad_u, pad_d)), 'reflect')

        if subtile.shape[1] == _size + 7:
            print("PADDING!!")
            pad_l = 7 if start_y == 0 else 0
            pad_r = 7 if start_y != 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')
            subtile_median_s2 = np.pad(subtile_median_s2, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            subtile_median_s1 = np.pad(subtile_median_s1, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
            min_clear_tile = np.pad(min_clear_tile, ((pad_l, pad_r), (0, 0)), 'reflect')

        subtile_s2 = subtile
        #subtile_s2 = superresolve_tile(subtile, sess = superresolve_sess)

        # Concatenate the DEM and Sentinel 1 data
        subtile = np.zeros((LEN + 1, _size + 14, _size + 14, 17), dtype = np.float32)
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

        if hist_align:
            print("Actually doing it tho")
            subtile = align_subtile_histograms(subtile)
        assert subtile.shape[1] >= 145, f"subtile shape is {subtile.shape}"
        assert subtile.shape[0] == (LEN + 1), f"subtile shape is {subtile.shape}"

        # Select between temporal and median models for prediction, based on simple logic:
        # If the first image is after June 15 or the last image is before July 15
        # or the maximum gap is >270 days or < 5 images --- then do median, otherwise temporal
        no_images_subtile = True if len(dates_tile) < 2 else no_images_subtile
        if no_images_subtile:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data")
            preds = np.full((_size, _size), 255)
        else:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                f" time series")
            preds = predict_subtile(subtile, sess, predict_logits)

        if args.gen_feats:
            pred = predict_subtile(subtile, sess, predict_logits)
            pred = np.int16(np.around(pred * 100, 0))
            latefeats = predict_subtile(subtile, sess, predict_latefeats)[..., :32]
            earlyfeats = predict_subtile(subtile, sess, predict_earlyfeats)[..., :32]
            print(f'Late feats: {latefeats.shape}, subtile: {subtile.shape}')
            print(f'Early feats: {earlyfeats.shape}, subtile: {subtile.shape}')
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
            np.save(output_feats, np.concatenate([pred[..., np.newaxis], earlyfeats,latefeats], axis = -1))
            print(f"{str(folder_y)}/{str(folder_x)}_middle: {len(dates_tile)} / {len(dates)} dates,"
            f"for: {dates_tile}, {np.percentile(min_clear_images_per_date, 10)} clear images"
            f", {np.mean(preds)}")

def preprocess_tile(arr, dates, interp, clm, fname, dem, bbx):
    
    print(np.sum(np.isnan(arr), axis = (1, 2, 3)))
    missing_px = interpolation.id_missing_px(arr, 5)
    if len(missing_px) > 0:
        dates = np.delete(dates, missing_px)
        arr = np.delete(arr, missing_px, 0)
        #interp = np.delete(interp, missing_px, 0)
        print(f"Removing {len(missing_px)} missing images")

    # Remove dates with high likelihood of missed cloud or shadow (false negatives)
    cld, fcps = cloud_removal.identify_clouds_shadows(arr, dem, bbx)
    if clm is not None:
        #print("CLM", np.mean(clm, axis = (1, 2)))
        try:
            if len(missing_px) > 0:
                print(f"Deleting {missing_px} from cloud mask")
                clm = np.delete(clm, missing_px, 0)
                clm[fcps] = 0.
            cld = np.maximum(clm, cld)
        except:
            print("There is a date mismatch between clm and cld, continuing")
        print(cld.shape)
        

    interp = cloud_removal.id_areas_to_interp(
            arr, cld, cld, dates, fcps
    )
    print(np.sum(np.isnan(arr), axis = (1, 2, 3)))
    print("TO REMOVE INTERP", np.mean(interp == 1, axis = (1, 2)))
    to_remove = np.argwhere(np.mean(interp == 1, axis = (1, 2)) > 0.95)

    if len(to_remove) > 0:
        print(f"DELETING {to_remove}")
        cld = np.delete(cld, to_remove, axis = 0)
        dates = np.delete(dates, to_remove)
        interp = np.delete(interp, to_remove, axis = 0)
        arr = np.delete(arr, to_remove, axis = 0)
        cld, fcps = cloud_removal.identify_clouds_shadows(arr, dem, bbx)
        print(np.mean(cld, axis = (1, 2)))

    print(interp.dtype, cld.dtype, fcps.dtype)
    print("CLD", np.mean(cld, axis = (1, 2)))
    arr, interp2, to_remove = cloud_removal.remove_cloud_and_shadows(arr, 
        cld,
        cld,
        dates,
        pfcps = fcps,
        sentinel1 = None,
        mosaic = None)
    #interp = np.maximum(interp, interp2)
    print(np.sum(np.isnan(arr), axis = (1, 2, 3)))
    return arr, interp2, dates


def load_tif(tile_id, local_path):
    dir_i = f"{local_path}/{tile_id[0]}/{tile_id[1]}/"
    tifs = []
    smooth = 0
    if os.path.exists(dir_i):

        processed = [file for file in os.listdir(dir_i)  if "SMOOTH" in file]
        if len(processed) > 0:
            smooth = 1

        files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
        final_files = [x for x in files if "_FINAL" in x]
        post_files = [x for x in files if "_POST" in x]

        smooth_files = [file for file in files if "_SMOOTH" in file]
        smooth_x = [file for file in files if "_SMOOTH_X" in file]
        smooth_x = [file for file in smooth_x if "_SMOOTH_XY" not in file]
        smooth_y = [file for file in files if "_SMOOTH_Y" in file]
        smooth_xy = [file for file in files if "_SMOOTH_XY" in file]

        smooth_files = [file for file in smooth_files if os.path.splitext(file)[-1] == '.tif']
        if len(smooth_files) > 0:
            #if len(smooth_files) > 1:
            if len(smooth_xy) > 0:
                files = smooth_xy
            elif len(smooth_x) > 0:
                files = smooth_x
            elif len(smooth_y) > 0:
                files = smooth_y

        elif len(final_files) > 0:
            files = final_files
        else:
            files = post_files

        for file in files:
           tifs.append(os.path.join(dir_i, file))

    tifs = tifs[0]
    print(tifs)
    tifs = rasterio.open(tifs).read(1)
    return tifs, smooth


def make_binned_tif(arr):
    arr[np.logical_and(arr >= 95, arr <= 100)] = 100.
    arr[np.logical_and(arr >= 85, arr < 95)] = 90.
    arr[np.logical_and(arr >= 75, arr < 85)] = 80.
    arr[np.logical_and(arr >= 65, arr < 75)] = 70.
    arr[np.logical_and(arr >= 55, arr < 65)] = 60.
    arr[np.logical_and(arr >= 45, arr < 55)] = 50.
    arr[np.logical_and(arr >= 35, arr < 45)] = 40.
    arr[np.logical_and(arr >= 30, arr < 35)] = 30.
    arr[arr < 25] = 0.
    arr[np.logical_and(arr >= 25, arr < 30)] = 20.
    return arr


def check_if_artifact(tile, neighb):
    #tile = make_binned_tif(tile)
    #neighb = make_binned_tif(neighb)
    right = neighb[-3:]
    left = tile[:3]

    right_mean = bn.nanmean(neighb[-3:])
    left_mean = bn.nanmean(tile[:3])

    right = neighb[-1]
    left = tile[0]

    right = right[:left.shape[0]]
    left = left[:right.shape[0]]
    right = np.pad(right, (10 - (right.shape[0] % 10)) // 2, constant_values = np.nan)
    right = np.reshape(right, (right.shape[0] // 10, 10))
    right = bn.nanmean(right, axis = 1)    
    
    left = np.pad(left, (10 - (left.shape[0] % 10)) // 2, constant_values =  np.nan)
    left = np.reshape(left, (left.shape[0] // 10, 10))
    left = bn.nanmean(left, axis = 1)

    fraction_diff = bn.nanmean(abs(right - left) > 25) #normally 25
    fraction_diff_left = bn.nanmean(abs(right[:15] - left[:15]) > 15)
    fraction_diff_right = bn.nanmean(abs(right[-15:] - left[-15:]) > 15)
    fraction_diff_2 = bn.nanmean(abs(right - left) > 15)
    left_right_diff = abs(right_mean - left_mean)

    other0 = left_right_diff > 7

    other = fraction_diff_2 > 0.6
    other = np.logical_and(other, (left_right_diff > 0) ) # normally 6
    other2 = (fraction_diff > 0.25) or ((fraction_diff_left > 0.5) or (fraction_diff_right > 0.5))
    #other2 = np.logical_and(other2, (left_right_diff > 1) ) # normally 3

    print(x, y, left_right_diff, fraction_diff, other0, other, other2)
    if other0 or other or other2:
        #print("BAD")
        return 1
    else:
        return 0


def concatenate_s2_files(s2, s2_neighb):
    s2_diff = s2.shape[2] - s2_neighb.shape[2]
    if s2_diff > 0:
        s2 = s2[:, :, s2_diff // 2: -(s2_diff // 2), :]
        
    if s2_diff < 0:
        s2_neighb = s2_neighb[:, :, - (s2_diff // 2) : (s2_diff // 2)]
    return s2, s2_neighb


def load_dates(tile_x, tile_y, local_path):
    tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
    folder = f"{local_path}{str(tile_x)}/{str(tile_y)}/"
    return hkl.load(f'{folder}raw/misc/s2_dates_{tile_idx}.hkl')


def regularize_and_smooth(arr, dates):
    sm = Smoother(lmbd = 100, size = 24, nbands = 2, dimx = arr.shape[1], dimy = arr.shape[2])
    # Perhaps do 2-band windows separately here to clear up RAM
    js = np.arange(0, 10, 2)
    ks = js + 2
    n_images = arr.shape[0]
    if n_images < 12:
        empty_images = np.zeros((12 - n_images, arr.shape[1], arr.shape[2], arr.shape[3]), dtype = np.float32)
        arr = np.concatenate([arr, empty_images], axis = 0)
    for j, k in zip(js, ks):
        print(j, k)
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

    right_s3_path = f'2020/ard/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_ard.hkl'
    left_s3_path = f'2020/ard/{str(int(x))}/{str(int(y) + 1)}/{str(int(x))}X{str(int(y) + 1)}Y_ard.hkl'
    right_local_path = f'{str(x)}X{str(y)}Y_ard.hkl'
    left_local_path = f'{str(int(x))}X{str(int(y) + 1)}Y_ard.hkl'

    download_ard_file(left_s3_path,
        left_local_path,
        AWSKEY, AWSSECRET, 'tof-output')
    download_ard_file(right_s3_path,
        right_local_path,
        AWSKEY, AWSSECRET, 'tof-output')
    l = hkl.load(left_local_path)[..., :13]
    r = hkl.load(right_local_path)[..., :13]
    #m = np.load(middle)[..., :13]
    img = np.concatenate([l, r], axis = 0)

    inp_mid_shape = m.shape[0]
    out_mid_shape = l.shape[0]
    middle_adjust = (inp_mid_shape - out_mid_shape) // 2
    m = m[middle_adjust:-middle_adjust, :]    

    half = m.shape[0] // 2
    lsize = l.shape[0] - (half)
    rsize = lsize + half + half
    
    sums = np.zeros((img.shape[0], img.shape[1]), dtype = np.float32)
    sums[:l.shape[0] // 2] = 1
    sums[l.shape[0] // 2:(l.shape[0] // 2)+half, :] += (1 - (np.arange(0, half, 1) / half))[:, np.newaxis]
    sums[(l.shape[0] // 2)+half:(l.shape[0] // 2)+half+half] += ((np.arange(0, half, 1) / half))[:, np.newaxis]
    sums[-(r.shape[0] // 2):] = 1.
    #sums = sums[..., np.newaxis]

    sumsright = 1 - sums
    img[..., :10] = img[..., :10] * sums[..., np.newaxis]
    img[lsize:rsize, :, :10] += (m[..., :10]* (1 - sums[lsize:rsize][..., np.newaxis]))
    leftfile = img[:l.shape[0]]
    rightfile = img[-r.shape[0]:]
    hkl.dump(leftfile, left_local_path, mode='w', compression='gzip')
    hkl.dump(rightfile, right_local_path,  mode='w', compression='gzip')
    uploader.upload(bucket = 'tof-output', key = right_s3_path, file = right_local_path)
    uploader.upload(bucket = 'tof-output', key = left_s3_path, file = left_local_path)
    write_ard_to_tif(leftfile[..., :3], neighb_bbx,
                                left_local_path[:-4] + "_y", "")
    write_ard_to_tif(rightfile[..., :3], bbx,
                                right_local_path[:-4] + "_y", "")
    
    return None


def resegment_border(tile_x, tile_y, edge, local_path, bbx, neighb_bbx, min_dates, initial_bbx):
    print("WTF")
    no_images = False
    processed = check_if_processed((tile_x, tile_y), local_path, args.year)
    neighbor_id = [tile_x, str(int(tile_y)+ 1 )]

    processed_neighbor = check_if_processed(neighbor_id, local_path, args.year)
    if processed_neighbor:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['X_tile'] == int(neighbor_id[0])]
        data_temp = data_temp[data_temp['Y_tile'] == int(neighbor_id[1])]
        processed_neighbor = True if len(data_temp) > 0 else False
    print(processed, processed_neighbor, "PRINGTING SHIT")
    if processed and processed_neighbor:
        download_raw_tile((tile_x, tile_y), local_path, "tiles")
        download_raw_tile(neighbor_id, local_path, "tiles")
        tile_tif, _ = load_tif((tile_x, tile_y), local_path)
        #print(type(tile_tif))
        if type(tile_tif) is not np.ndarray:
            print("Skipping because one of the TIFS doesnt exist")
            return 0, None, None, 0, 3

        tile_tif = tile_tif.astype(np.float32)
        neighbor_tif, smooth = load_tif(neighbor_id, local_path)
        neighbor_tif = neighbor_tif.astype(np.float32)
        neighbor_tif[neighbor_tif > 100] = np.nan
        tile_tif[tile_tif > 100] = np.nan

        artifact = check_if_artifact(tile_tif, neighbor_tif)
        right_mean = np.nanmean(neighbor_tif[-6:])
        left_mean = np.nanmean(tile_tif[:6])
        right = np.nanmean(neighbor_tif[-6:], axis = 0)
        left = np.nanmean(tile_tif[:6], axis = 0)
        right = right[:left.shape[0]]
        left = left[:right.shape[0]]

        right_all = np.mean(neighbor_tif[-(SIZE // 2):], axis = 0)
        left_all = np.mean(tile_tif[:(SIZE // 2)], axis = 0)

        left_right_diff = abs(right_mean - left_mean)
        fraction_diff = np.nanmean(abs(right - left) > 25)

        if artifact == 1 or args.process_all:
            download_raw_tile((tile_x, tile_y), local_path, "processed")
            download_raw_tile((tile_x, tile_y), local_path, "raw")

            print(f"Downloading {neighbor_id}")
            download_raw_tile(neighbor_id, local_path, "raw")
            download_raw_tile(neighbor_id, local_path, "processed")
        else:
            print("The tiles are pretty close, skipping")
            return 0, None, None, 0, 3
    else:
        print("One of the tiles isn't processed, skipping.")
        return 0, None, None, 0, 3

    print("Checking the date overlaps")
    dates = load_dates(tile_x, tile_y, local_path)
    dates_neighb = load_dates(neighbor_id[0], neighbor_id[1], local_path)
    to_rm_tile, to_rm_neighb, min_images = align_dates(dates, dates_neighb)
    print(f"There are {min_images} overlapping")

    if min_images >= 6:
        print("Preprocessing the tiles together")
        print("Loading and splitting the tile")
        s2, dates, interp, s1, dem, _, _ = process_tile(tile_x, tile_y, data, local_path, bbx)
        s2_shape = s2.shape[1:-1]
        #n_tiles_x, n_tiles_y = check_n_tiles(tile_x, tile_y)
        tile_idx = f'{str(tile_x)}X{str(tile_y)}Y'
        s2, interp, s1, dem, _ = split_to_border(s2, interp, s1, dem, "tile", edge)

        print("Loading and splitting the neighbor tile")
        s2_neighb, dates_neighb, interp_neighb, s1_neighb, dem_neighb, _, _ = \
            process_tile(neighbor_id[0], neighbor_id[1], data, args.local_path, neighb_bbx)
        s2_neighb_shape = s2_neighb.shape[1:-1]
        tile_idx = f'{str(neighbor_id[0])}X{str(neighbor_id[1])}Y'
        s2_neighb, interp_neighb, s1_neighb, dem_neighb, tiles_folder_y = \
            split_to_border(s2_neighb, interp_neighb, s1_neighb, dem_neighb, "neighbor", edge)

        # S2Cloudless CLM maps
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
            s2 = np.delete(s2, to_rm_tile, 0)
            dates = np.delete(dates, to_rm_tile)
            if clm is not None:
                if clm.shape[0] > 0:
                    clm = np.delete(clm,  to_rm_tile, 0)
        
        if len(to_rm_neighb) > 0:
            s2_neighb = np.delete(s2_neighb, to_rm_neighb, 0)
            dates_neighb = np.delete(dates_neighb, to_rm_neighb)
            if clm_neighb is not None:
                if clm.shape[0] > 0:
                    clm_neighb = np.delete(clm_neighb, to_rm_neighb, 0)

        if (clm is not None) and (clm_neighb is not None):
            try:
                clm = np.concatenate([clm_neighb, clm], axis = 1)
                clm = np.float32(clm)
                clm[np.isnan(clm)] = 0.
            except:
                clm = None

        print("Concatenating S2!")
        s2_diff = s2.shape[2] - s2_neighb.shape[2]
        s2 = s2[:, :, s2_diff // 2: -(s2_diff // 2), :] if s2_diff > 0 else s2
        s2_neighb = s2_neighb = s2_neighb[:, :, - (s2_diff // 2) : (s2_diff // 2)] if s2_diff < 0 else s2_neighb
        dem_diff = dem.shape[1] - dem_neighb.shape[1]
        if dem_diff > 0:
            dem = dem[:, dem_diff // 2: -(dem_diff // 2)]
        if dem_diff < 0:
            dem_neighb = dem_neighb[:, -(dem_diff // 2) : (dem_diff // 2)]

        s2_all = np.empty((s2.shape[0], s2.shape[1] + s2_neighb.shape[1], s2.shape[2], s2.shape[-1]), dtype = np.float32)
        s2_all[:, :s2.shape[1],:, :] = s2_neighb
        s2_all[:, s2.shape[1]:, :, :] = s2
        s2 = s2_all
        s2_neighb = None
        

        dem = np.concatenate([dem_neighb, dem], axis = 0)
    
        print("Removing clouds, shared tile")

        initial_bbx[0] += (300/30) * (1/360)
        initial_bbx[2] += (300/30) * (1/360)

        bbx_middle = make_bbox(initial_bbx, expansion = 300/30)
        
        s2 = np.float32(s2)
        s2, interp, dates = preprocess_tile(s2, dates, None, clm, "tile", dem, bbx)
        print(np.sum(np.isnan(s2)))
        s2, dates, interp = deal_w_missing_px(s2, dates, interp)
        print(s2.dtype, interp.dtype, dem.dtype, s1.dtype, s1_neighb.dtype)

        # this is important! Otherwise hist_align will happen.
        dates_neighb = dates
        min_clear_images_per_date = np.sum(interp != 1, axis = 0)
        indices = make_and_smooth_indices(s2, dates)
        print(s2.dtype, indices.dtype, interp.dtype, dem.dtype, s1.dtype, s1_neighb.dtype)
        s2 = regularize_and_smooth(s2, dates)

    elif min_images < 6:
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
        s2, interp, s1, dem, _ = split_to_border(s2, interp, s1, dem, "tile", edge)

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
        s2_neighb, interp_neighb, s1_neighb, dem_neighb, tiles_folder_y = \
            split_to_border(s2_neighb, interp_neighb, s1_neighb, dem_neighb, "neighbor", edge)

        print("Aligning the dates between the tiles")
        min_clear_images_per_date = np.concatenate([np.sum(interp[:, :(SIZE + 14)//2, :] != 1, axis = (0)),
                                                   np.sum(interp_neighb[:, -(SIZE + 14)//2:, :] != 1, axis = (0))], axis = 0)
        to_rm_tile, to_rm_neighb, min_images = align_dates(dates, dates_neighb)
        if min_images >= min_dates: #(len(to_rm_tile) <= 3 and len(to_rm_neighb) <= 3) or
            if len(to_rm_tile) > 0:
                s2 = np.delete(s2, to_rm_tile, 0)
                interp = np.delete(interp, to_rm_tile, 0)
                dates = np.delete(dates, to_rm_tile)
            if len(to_rm_neighb) > 0:
                s2_neighb = np.delete(s2_neighb, to_rm_neighb, 0)
                interp_neighb = np.delete(interp_neighb, to_rm_neighb, 0)
                dates_neighb = np.delete(dates_neighb, to_rm_neighb)
        else:
            n_images = np.minimum(len(dates), len(dates_neighb))
            if n_images < 2:
                no_images = True

        s2, dates, interp = deal_w_missing_px(s2, dates, interp)
        s2_neighb, dates_neighb, interp_neighb = deal_w_missing_px(s2_neighb, dates_neighb, interp_neighb)

        s2_indices = make_and_smooth_indices(s2, dates)
        print(s2_indices.dtype)
        neighb_indices = make_and_smooth_indices(s2_neighb, dates_neighb)
        print(neighb_indices.dtype)
        s2 = regularize_and_smooth(s2, dates)
        s2_neighb = regularize_and_smooth(s2_neighb, dates_neighb)

        s2_diff = s2.shape[2] - s2_neighb.shape[2]
        if s2_diff > 0:
            s2 = s2[:, :, s2_diff // 2: -(s2_diff // 2), :]
            s2_indices = s2_indices[:, :, s2_diff // 2: -(s2_diff // 2), :]
            
        if s2_diff < 0:
            s2_neighb = s2_neighb[:, :, - (s2_diff // 2) : (s2_diff // 2)]
            neighb_indices = neighb_indices[:, :, - (s2_diff // 2) : (s2_diff // 2)]

        interp_diff = interp.shape[2] - interp_neighb.shape[2]
        if interp_diff > 0:
            interp = interp[:, :, interp_diff // 2: -(interp_diff // 2)]
        if interp_diff < 0:
            interp_neighb = interp_neighb[:, :, -(interp_diff // 2) : (interp_diff // 2)]

        dem_diff = dem.shape[1] - dem_neighb.shape[1]
        if dem_diff > 0:
            dem = dem[:, dem_diff // 2: -(dem_diff // 2)]
        if interp_diff < 0:
            dem_neighb = dem_neighb[:, -(dem_diff // 2) : (dem_diff // 2)]

        s2_all = np.empty((s2.shape[0], s2.shape[1] + s2_neighb.shape[1], s2.shape[2], s2.shape[-1]))
        s2_all[:, :s2.shape[1],:, :] = s2_neighb
        s2_all[:, s2.shape[1]:, :, :] = s2
        s2 = s2_all
        s2_neighb = None
        interp = np.concatenate([interp[:interp_neighb.shape[0]],
                                 interp_neighb[:interp.shape[0]]], axis = 1)
        indices = np.concatenate([neighb_indices, s2_indices], axis = 1)
        dem = np.concatenate([dem_neighb, dem], axis = 0)
        
    print("Continuing with the shared processing pipeline")
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

    s1_diff = s1.shape[2] - s1_neighb.shape[2]
    if s1_diff > 0:
        s1 = s1[:, :, s1_diff // 2: -(s1_diff // 2), :]
    if s1_diff < 0:
        s1_neighb = s1_neighb[:, :, - (s1_diff // 2) :(s1_diff // 2)]

    s2 = superresolve_large_tile(s2, superresolve_sess)
    s1 = np.concatenate([s1_neighb, s1], axis = 1)

    print(f"The sentinel 1 shape is: {s1.shape}")
    #gap_x = int(np.ceil((s1.shape[2] - SIZE_X) / 3))
    #tiles_folder_x = np.hstack([np.arange(0, s1.shape[2] - SIZE_X, gap_x), np.array(s1.shape[2] - SIZE_X)])
    gap_y = int(np.ceil((s1.shape[1] - 216) / 5))
    gap_x = int(np.ceil((s1.shape[2] - 216) / 5 ))
    tiles_folder_y = np.hstack([np.arange(0, s1.shape[1] - 216, gap_y), np.array(s1.shape[1] - 216)])
    tiles_folder_x = np.hstack([np.arange(0, s1.shape[2] - 216, gap_x), np.array(s1.shape[2] - 216)])
    
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
    print("LOOK HERE")
    print(tiles_folder)
    print(tiles_array)

    
    
    s2_neighb, s1_neighb = None, None
    s2_indices, neighb_indices = None, None

    out = np.zeros(
        (s2.shape[0], s2.shape[1], s2.shape[2], 14), dtype = np.float32
    )
    out[..., :10] = s2
    out[..., 10:] = indices
    s2 = out
    print("Updating the ARD Tiles")
    print(f"The subtile shape is {s2.shape}")
    _ = update_ard_tiles(tile_x, tile_y, np.median(s2, axis = 0), bbx, neighb_bbx)
    #np.save('test_example.npy', np.median(s2, axis = 0))

    #tiles_array, tiles_folder = make_tiles_right_neighb(tiles_folder_x, tiles_folder_y)

    if not np.array_equal(np.array(dates), np.array(dates_neighb)):
        hist_align = True
        print("Aligning histogram")
    else:
        hist_align = False
        print("No align needed")

    right_all = neighbor_tif[-(SIZE // 2):]
    left_all = tile_tif[:(SIZE // 2)]

    print("starting process_subtiles")
    process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, gap_sess, tiles_folder, tiles_array,
        right_all, left_all, hist_align, min_clear_images_per_date, )

    return 1, s2_shape, s2_neighb_shape, left_right_diff, min_images


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
    m = m ** 1.5
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
        #m[m > 0.5] = 0.5
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


def combine_resegmented_feats(x, y, m, indices, base_path = '../project-monitoring/tiles/'):
    # Step 1 check if ARD exists
    # Step 2 update the 

    left_local_path = f'{base_path}{str(x)}/{str(y)}/raw/feats/{str(x)}X{str(y)}Y_feats.hkl'
    right_local_path = f'{base_path}{str(int(x))}/{str(int(y) + 1)}/raw/feats/{str(int(x))}X{str(int(y) + 1)}Y_feats.hkl'
    
    path_to_left = f'{str(args.year)}/raw/{str(x)}/{str(y)}/'
    path_to_right = f'{str(args.year)}/raw/{str(x)}/{str(int(y) + 1)}/'
    left_s3_path = f"{path_to_left}raw/feats/{str(x)}X{str(y)}Y_feats.hkl"
    right_s3_path = f"{path_to_right}raw/feats/{str(x)}X{str(int(y) + 1)}Y_feats.hkl"

    print("Loading original features")
    print(f"The s3 path proto is: {left_s3_path}")
    
    l = hkl.load(left_local_path) / 32768
    r = hkl.load(right_local_path) / 32768
    m = m / 32768
    l = l.T
    r = r.T
    m = m.T    
    inp_mid_shape = m.shape[0]
    out_mid_shape = l.shape[0]
    middle_adjust = (inp_mid_shape - out_mid_shape) // 2
    m = m[middle_adjust:-middle_adjust, :]
    half = m.shape[0] // 2
    lsize = ((l.shape[0] + r.shape[0]) - (half * 2)) // 2
    lsize = lsize - 7
    rsize = lsize + half + half
    print("Concatenating three feature tiles ", l.shape, r.shape, m.shape)
    img = np.concatenate([r, l], axis = 0)
    print("Concatenated three feature tiles ", img.shape)
    sums = np.zeros((img.shape[0], img.shape[1]), dtype = np.float32)
    sums[:l.shape[1] // 2] = 1
    sums[l.shape[1] // 2:(l.shape[1] // 2)+half] += (1 - (np.arange(0, half, 1) / half))[:, np.newaxis]
    sums[(l.shape[1] // 2)+half:(l.shape[1] // 2)+half+half] += ((np.arange(0, half, 1) / half))[:, np.newaxis]
    rightend = (l.shape[1] // 2)+half+half
    sums[rightend:] = 1.
    sums = sums[..., np.newaxis]
    sums = np.repeat(sums[..., np.newaxis], 65, axis = -1).squeeze()
    print(f"The sums are: {sums.shape}")
    print(f"The img shape is: {img.shape}")
    non_indices = [x for x in np.arange(0, 65) if x not in indices]
    #print(f"The non-indices are: {non_indices}")
    #sums[..., non_indices] = 1.
    img = img * sums
    #img[..., 1:] = img[..., 1:] * sums[..., 1:]
    print(f"The image is: {img.shape}, the M is {m.shape}, the sums is {sums.shape}")
    img[lsize:rsize, :, 0:] += (m * (1 - sums[lsize:rsize, :, 0:])) # should be 0 for non-adj feats
    print("Blended three feature tiles")

    img = img.T
    img = np.int16(img * 32768)
    print(img.shape)
    _right = img[:, :, :img.shape[2] // 2]
    _left = img[:, :, -img.shape[2] // 2:]
    
    print("Saving the updated feature tiles and continuing ", _left.shape, _right.shape)
    print(f"The left features are saved to: {left_local_path}")
    print(f"The right features are saved to: {right_local_path}")
    hkl.dump(_left, left_local_path, mode='w', compression='gzip')
    hkl.dump(_right, right_local_path,  mode='w', compression='gzip')
    uploader.upload(bucket = 'tof-output', key = right_s3_path, file = right_local_path)
    uploader.upload(bucket = 'tof-output', key = left_s3_path, file = left_local_path)
    return img, sums


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
    #mults_n = np.clip(1 - sums, 0, 1)
    #sums += mults_n
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
        key = f'{str(args.year)}/processed/{str(x)}/{str(int(y) + 1)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for file in glob(path_to_tile + "processed/*/up*"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'{str(args.year)}/processed/{str(x)}/{str(y)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for file in glob(path_to_right + "processed/*/down*"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'{str(args.year)}/processed/{str(x)}/{str(int(y) + 1)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for folder in glob(path_to_tile + "processed/right*/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            internal_folder = folder[len(path_to_right):]
            print(internal_folder)
            key = f'{str(args.year)}/processed/{x}/{y}/' + internal_folder
            if upload:
                uploader.upload(bucket = 'tof-output', key = key, file = _file)

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
    SIZE = 618
    SIZE_X = 206

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/tf2-ard-left/')
    parser.add_argument("--gap_model_path", dest = 'gap_model_path', default = '../models/182-gap-sept/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/nov-40k-swir/')
    parser.add_argument("--db_path", dest = "db_path", default = "process_area_2022.csv")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--start_id", dest = "start_id", default = 0)
    parser.add_argument("--process_all", dest = "process_all", default = False)
    parser.add_argument("--start_x", dest = "start_x", default = 10000)
    parser.add_argument("--year", dest = "year", default = 2020)
    parser.add_argument("--snow", dest = "snow", default = False)
    parser.add_argument("--length", dest = "length", default = 4)
    parser.add_argument("--gen_feats", dest = "gen_feats", default = False, type=str2bool, nargs='?',
                        const=True)
    parser.add_argument("--feat_ids", dest = 'feat_ids',
        default = [ 3, 14, 15, 55, 27, 18, 44,  5, 11, 12,  2, 38, 45, 25, 26, 61, 48, 29, 53,
          4, 46, 41, 63,  9, 43, 28, 59, 39, 64, 52, 13, 58])
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
        #predict_logits = predict_sess.graph.get_tensor_by_name(f"predict/conv2d/Sigmoid:0")
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        print(predict_inp.shape)
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")

    gap_file = None
    gap_graph_def = None
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

    data['X_tile'] = data['X_tile'].astype(int)
    data['Y_tile'] = data['Y_tile'].astype(int)
    data = data.sort_values(['X_tile', 'Y_tile'], ascending=[False, True])



    print(len(data))
    current_x = 10000
    n = 0
    for index, row in data[10:].iterrows(): # We want to sort this by the X so that it goes from left to right
        print("WATWAT")
        if index > int(args.start_id):
            x = str(int(row['X_tile']))
            y = str(int(row['Y_tile']))
            x = x[:-2] if ".0" in x else x
            y = y[:-2] if ".0" in y else y

            #current_x = cleanup_row_or_col(idx = x,
            #    current_idx = current_x,
            #    local_path = args.local_path)
            print("WAT")
            if int(x) < int(args.start_x):
                path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/'
                path_to_right = f'{args.local_path}{str(x)}/{str(int(y) + 1)}/'

                print(path_to_tile, path_to_right, n)

                initial_bbx = [row['X'], row['Y'], row['X'], row['Y']]
                bbx = make_bbox(initial_bbx, expansion = 300/30)
                print(bbx)

                data_neighb = data.copy()
                neighb_bbx = None

                try:
                    data_neighb = data_neighb[data_neighb['X_tile'] == int(x)]
                    data_neighb = data_neighb[data_neighb['Y_tile'] == int(y) + 1]
                    data_neighb = data_neighb.reset_index()
                    neighb = [data_neighb['X'][0], data_neighb['Y'][0], data_neighb['X'][0], data_neighb['Y'][0]]
                    #print(data_neighb['X_tile'][0], data_neighb['Y_tile'][0])
                    neighb_bbx = make_bbox(neighb, expansion = 300/30)
                except KeyboardInterrupt:
                    break
                except:
                    print("One of the tiles doesnt exist, skipping")
                    continue
                try:
                    finished, s2_shape, s2_neighb_shape, diff, min_images  = resegment_border(x, y, "up", args.local_path,  bbx, neighb_bbx, 2, initial_bbx)
                except KeyboardInterrupt:
                        break
                except Exception as e:
                    print(f"PROCESS, Ran into {str(e)}")
                    finished = 0
                    s2_shape = (0, 0)
                    s2_neighb_shape = (0, 0)
                if finished == 1:
                    #try:
                    # This recreates the TTC tiles
                    # So is disabled here
                    #predictions_left, _ = recreate_resegmented_tifs(path_to_tile + "processed/", s2_shape)
                    #predictions_right, _ = recreate_resegmented_tifs(path_to_right + "processed/", s2_neighb_shape)

                    # This recreates the TTC features
                    middle = load_mosaic_feats(path_to_tile + "feats/", depth = 65)
                    np.save("middle.npy", middle)
                    img, sums = combine_resegmented_feats(x, y, middle, indices = args.feat_ids)
                    # Upload the left file
                    # Upload the right file

                    # Generate the top tile
                    if os.path.exists(f"{path_to_tile}/{str(x)}X{str(y)}Y_SMOOTH_XY.tif"):
                        suffix = "_SMOOTH_XY"
                    elif os.path.exists(f"{path_to_tile}/{str(x)}X{str(y)}Y_SMOOTH_Y.tif"):
                        suffix = "_SMOOTH_XY"
                    else:
                        suffix = "_SMOOTH_X"

                    #file = write_tif(predictions_left, bbx, x, y, path_to_tile, suffix)
                    #key = f'{str(args.year)}/tiles/{x}/{y}/{str(x)}X{str(y)}Y{suffix}.tif'
                    #uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                    # Generate the right tile
                    if os.path.exists(f"{path_to_right}/{str(x)}X{str(int(y) + 1)}Y_SMOOTH_XY.tif"):
                        suffix = "_SMOOTH_XY"
                    elif os.path.exists(f"{path_to_right}/{str(x)}X{str(int(y) + 1)}Y_SMOOTH_Y.tif"):
                        suffix = "_SMOOTH_XY"
                    else:
                        suffix = "_SMOOTH_X"

                    #file = write_tif(predictions_right, neighb_bbx, str(int(x) + 1), y, path_to_right, suffix)
                    #key = f'{str(args.year)}/tiles/{str(int(x) + 1)}/{y}/{str(int(x) + 1)}X{str(y)}Y{suffix}.tif'
                    #uploader.upload(bucket = args.s3_bucket, key = key, file = file)
                    
                    # Cleanup
                    #cleanup(path_to_tile, path_to_right, delete = True, upload = True)
                    #print(f"Finished resegment in {np.around(time2 - time1, 1)} seconds")
                    

                    #except KeyboardInterrupt:
                    #    break
                    #except Exception as e:
                    #    print(f"POSTPROCESS: Ran into {str(e)}")
                        
                    #n += 1
                
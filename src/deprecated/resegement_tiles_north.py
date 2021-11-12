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
from glob import glob
import rasterio
from rasterio.transform import from_origin

from preprocessing import slope
from downloading.utils import calculate_and_save_best_images
from preprocessing import cloud_removal
from preprocessing import interpolation
from preprocessing.whittaker_smoother import Smoother
from tof.tof_downloading import to_int16, to_float32
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder
from preprocessing.indices import evi, bi, msavi2, grndvi
from download_and_predict_job import process_tile, make_bbox, convert_to_db
from download_and_predict_job import id_iqr_outliers, fspecial_gauss, rolling_mean

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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
    print(item.shape)
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
    print(s1.shape)

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

"""
def predict_gap(subtile, sess) -> np.ndarray:
     Runs non-temporal predictions on a (12, 174, 174, 13) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
                               that will be median aggregated for model input
         sess (tf.Session): tensorflow session for prediction
    
        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    
    
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
"""

def check_if_processed(tile_idx, local_path):

    x = tile_idx[0]
    y = tile_idx[1]
    path_to_tile = f'{local_path}{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/tiles/{str(x)}/{str(y)}/'
    processed = file_in_local_or_s3(path_to_tile,
                                    s3_path_to_tile, 
                                    AWSKEY, AWSSECRET, 
                                    args.s3_bucket)
    return processed


def align_dates(tile_date, neighb_date):
    to_rm_tile = [idx for idx, date in enumerate(tile_date) if date not in neighb_date]
    to_rm_neighb = [idx for idx, date in enumerate(neighb_date) if date not in tile_date]
    return to_rm_tile, to_rm_neighb


def make_tiles_right_neighb(tiles_folder_x, tiles_folder_y):
    windows = cartesian(tiles_folder_x, tiles_folder_y)
    win_sizes = np.full_like(windows, SIZE + 7)
    tiles_folder = np.hstack([windows, win_sizes])
    tiles_folder = np.sort(tiles_folder, axis = 0)
    tiles_folder[:, 1] = np.tile(np.unique(tiles_folder[:, 1]), 
        int(len(tiles_folder[:, 1]) / len(np.unique(tiles_folder[:, 1]))))
    tiles_array = np.copy(tiles_folder)
    tiles_array[1:, 0] -= 7
    tiles_array[1:-1, 2] += 7
    tiles_array[:, 1] = 0.
    tiles_array[:, 3] = 182.
    print(tiles_array)
    print(tiles_folder)
    return tiles_array, tiles_folder


def process_subtiles(x: int, y: int, s2: np.ndarray = None, 
                       dates: np.ndarray = None,
                       interp: np.ndarray = None, s1 = None, dem = None,
                       sess = None,
                       gap_sess = None, tiles_folder = None, tiles_array = None) -> None:
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
    
    make_subtiles(f'{args.local_path}{str(x)}/{str(y)}/processed/',
                  tiles_folder)
    path = f'{args.local_path}{str(x)}/{str(y)}/processed/'
    path_neighbor = f'{args.local_path}{str(x)}/{str(int(y) + 1)}/processed/'

    gap_between_years = False
    t = 0
    sm = Smoother(lmbd = 150, size = 36, nbands = 10, dim = SIZE + 14)
    n_median = 0
    median_thresh = 5
    # Iterate over each subitle and prepare it for processing and generate predictions
    while t < len(tiles_folder):
        tile_folder = tiles_folder[t]
        tile_array = tiles_array[t]
        t += 1
        
        start_x, start_y = tile_array[0], tile_array[1]
        folder_x, folder_y = tile_folder[1], tile_folder[0]
        end_x = start_x + tile_array[2]
        end_y = start_y + tile_array[3]
        subset = s2[:, start_y:end_y, start_x:end_x, :]
        interp_tile = interp[:, start_y:end_y, start_x:end_x]
        interp_tile_sum = np.sum(interp_tile, axis = (1, 2))
        dates_tile = np.copy(dates)
        dem_subtile = dem[np.newaxis, start_y:end_y, start_x:end_x]

        missing_px = interpolation.id_missing_px(subset, 10)

        if len(missing_px) > 0:
            print(np.sum(subset == 0, axis = (1, 2)))
            print(np.sum(subset >= 1, axis = (1, 2)))
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)
            interp_tile = np.delete(interp_tile, missing_px, 0)
            print(f"Removing {len(missing_px)} missing images, leaving {len(dates_tile)} / {len(dates)}")

        subset[subset == 0.] = np.tile(np.median(subset, axis = 0)[np.newaxis], (subset.shape[0], 1, 1, 1))[subset == 0]

        missing_px = interpolation.id_missing_px(subset, 100)
        if len(missing_px) > 0:
            print(np.sum(subset == 0, axis = (1, 2)))
            print(np.sum(subset >= 1, axis = (1, 2)))
            dates_tile = np.delete(dates_tile, missing_px)
            subset = np.delete(subset, missing_px, 0)
            interp_tile = np.delete(interp_tile, missing_px, 0)
            print(f"Removing {len(missing_px)} missing images, leaving {len(dates_tile)} / {len(dates)}")

        # Remove dates with high likelihood of missed cloud or shadow (false negatives)
        cld = cloud_removal.remove_missed_clouds(subset)
        subset, interp2 = cloud_removal.remove_cloud_and_shadows(subset, cld, cld, dates_tile, wsize = 20, step = 10, thresh = 100)
        interp_tile = np.maximum(interp_tile, interp2)
        subset = cloud_removal.adjust_interpolated_areas(subset, interp_tile)

        subset = rolling_mean(subset)

        # Transition (n, 160, 160, ...) array to (72, 160, 160, ...)
        no_images = False
        try:
            subtile, max_distance = calculate_and_save_best_images(subset, dates_tile)
        except:
            # If there are no images for the tile, just make them zeros
            # So that they will be picked up by the no-data flag
            print("Skipping because of no images")
            no_images = True
            subtile = np.zeros((36, end_y-start_y, end_x - start_x, 10))
            dates_tile = [0,]
        output = f"{path}{str(folder_y)}/up{str(0)}.npy"
        output2 = f"{path_neighbor}/{str(folder_y)}/down{str(folder_x)}.npy"
        s1_subtile = s1[:, start_y:end_y, start_x:end_x, :]
        print(subtile.shape)
        # Pad the corner / edge subtiles within each tile
        if subtile.shape[2] == SIZE + 7: 
            pad_d = 7 if start_x != 0 else 0
            pad_u = 7 if start_x == 0 else 0
            subtile = np.pad(subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            s1_subtile = np.pad(s1_subtile, ((0, 0,), (0, 0), (pad_u, pad_d), (0, 0)), 'reflect')
            dem_subtile = np.pad(dem_subtile, ((0, 0,), (0, 0), (pad_u, pad_d)), 'reflect')

        #if subtile.shape[1] == SIZE + 7:
        #    pad_l = 7 if start_x == 0 else 0
        #    pad_r = 7 if start_x != 0 else 0
        #    subtile = np.pad(subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
        #    s1_subtile = np.pad(s1_subtile, ((0, 0,), (pad_l, pad_r), (0, 0), (0, 0)), 'reflect')
        #    dem_subtile = np.pad(dem_subtile, ((0, 0,), (pad_l, pad_r), (0, 0)), 'reflect')
        print(subtile.shape)
        # Interpolate (whittaker smooth) the array and superresolve 20m to 10m
        subtile = sm.interpolate_array(subtile)
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
        no_images = True if len(dates_tile) < 2 else no_images
        if no_images:
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates -- no data")
            preds = np.full((SIZE, SIZE), 255)
        else:
            """
            #! TODO: or if start - end is > some % threshold difference for EVI suggesting deforestation
            if dates_tile[0] >= 150 or dates_tile[-1] <= 215 or max_distance > 265 or len(dates_tile) < 5:
                n_median += 1
                print(f"There are {n_median}/{median_thresh} medians in tile")
                if not gap_between_years and n_median >= median_thresh:
                    print("Restarting the predictions with median")
                    t = 0 if t > 1 else t
                    gap_between_years = True

            if len(dates_tile) < 5 or gap_between_years:
                # Then run the median prediction
                print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                    f" median, {max_distance} max dist")
                preds = predict_gap(subtile, gap_sess)
            else:
            """
                # Otherwise run the non-median prediction
            print(f"{str(folder_y)}/{str(folder_x)}: {len(dates_tile)} / {len(dates)} dates,"
                f" time series, {max_distance} max dist")
            preds = predict_subtile(subtile, sess)
        np.save(output, preds)
        np.save(output2, preds)
        print(output2)
        print(output)


def load_tif(tile_id, local_path):
    dir_i = f"{local_path}/{tile_id[0]}/{tile_id[1]}/"
    tifs = []
    smooth = 0
    if os.path.exists(dir_i): 

        processed = [file for file in os.listdir(dir_i)  if "SMOOTH" in file]
        if len(processed) > 0:
            smooth = 1

        files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
        smooth_files = [x for x in files if "_SMOOTH" in x]
        final_files = [x for x in files if "_FINAL" in x]
        post_files = [x for x in files if "_POST" in x]
        if len(smooth_files) > 0:
            files = smooth_files
        elif len(final_files) > 0:
            files = final_files
        else:
            files = post_files

        for file in files:
           tifs.append(os.path.join(dir_i, file))
    
    tifs = tifs[0]
    tifs = rasterio.open(tifs).read(1)
    return tifs, smooth


def resegment_border(tile_x, tile_y, edge, local_path):

    processed = check_if_processed((tile_x, tile_y), local_path)
    neighbor_id = [tile_x, str(int(tile_y)+ 1 )]
    print(neighbor_id)

    processed_neighbor = check_if_processed(neighbor_id, local_path)
    if processed_neighbor:
        data_temp = data.copy()
        data_temp = data_temp[data_temp['X_tile'] == int(neighbor_id[0])]
        data_temp = data_temp[data_temp['Y_tile'] == int(neighbor_id[1])]
        processed_neighbor = True if len(data_temp) > 0 else False
        #rocessed_neighbor = True

    if processed and processed_neighbor:
        print(f"Downloading {tile_x}, {tile_y}")
        download_raw_tile((tile_x, tile_y), local_path, "tiles")
        download_raw_tile(neighbor_id, local_path, "tiles")
        tile_tif, _ = load_tif((tile_x, tile_y), local_path)
        if type(tile_tif) is not np.ndarray:
            print("Skipping because one of the TIFS doesnt exist")
            return 0, None, None

        tile_tif = tile_tif.astype(np.float32)
        neighbor_tif, smooth = load_tif(neighbor_id, local_path)
        #if smooth == 1:
        #    print("Skipping because already smoothed")
        #    return 0
        neighbor_tif = neighbor_tif.astype(np.float32)
        neighbor_tif[neighbor_tif > 100] = np.nan
        tile_tif[tile_tif > 100] = np.nan
        right_mean = np.nanmean(neighbor_tif[-1, :])
        left_mean = np.nanmean(tile_tif[0, :])
        print(right_mean, left_mean)

        if abs(right_mean - left_mean) > 9:
            
            download_raw_tile((tile_x, tile_y), local_path, "processed")
            test_subtile = np.load(f"{local_path}/{tile_x}/{tile_y}/processed/0/0.npy")
            print(test_subtile.shape)
            if test_subtile.shape[0] != SIZE:
                print("Skipping cause of subtile size")
                return 0, None, None
            download_raw_tile((tile_x, tile_y), local_path, "raw")

            if edge == "up":
                print(f"Downloading {neighbor_id}")
                download_raw_tile(neighbor_id, local_path, "raw")
                download_raw_tile(neighbor_id, local_path, "processed")
                test_subtile = np.load(f"{local_path}/{neighbor_id[0]}/{neighbor_id[1]}/processed/0/0.npy")
                print(test_subtile.shape)
                if test_subtile.shape[0] != SIZE:
                    print("Skipping cause of subtile size")
                    return 0, None, None
        else:
            print("The tiles are pretty close, skipping")
            return 0, None, None
    else:
        print("One of the tiles isn't processed, skipping.")
        return 0, None, None
        
    print("Loading and processing the tile")
    s2, dates, interp, s1, dem = process_tile(tile_x, tile_y, data, local_path)
    s2_shape = s2.shape[1:-1]

    gap_x = int(np.ceil((s1.shape[2] - SIZE) / 4))
    tiles_folder_x = np.hstack([np.arange(0, s1.shape[2] - SIZE, gap_x), np.array(s1.shape[2] - SIZE)]) 
    print(tiles_folder_x)

    print("Splitting the tile to border")
    s2, interp, s1, dem, _ = split_to_border(s2, interp, s1, dem, "tile", edge)
    
    print("Loading and processing the neighbor tile")
    s2_neighb, dates_neighb, interp_neighb, s1_neighb, dem_neighb = \
        process_tile(neighbor_id[0], neighbor_id[1], data, args.local_path)
    s2_neighb_shape = s2_neighb.shape[1:-1]

    print("Splitting the neighbor tile to border")
    s2_neighb, interp_neighb, s1_neighb, dem_neighb, tiles_folder_y = \
        split_to_border(s2_neighb, interp_neighb, s1_neighb, dem_neighb, "neighbor", edge)


    print("Aligning the dates between the tiles")
    to_rm_tile, to_rm_neighb = align_dates(dates, dates_neighb)
    if ((len(dates) - len(to_rm_tile)) < 6) or ((len(dates_neighb) - len(to_rm_neighb)) < 6):
        dates = dates[:len(dates_neighb)]
        s2 = s2[:len(dates_neighb)]
        interp = interp[:len(dates_neighb)]

        dates_neighb = dates_neighb[:len(dates)]
        s2_neighb = s2_neighb[:len(dates)]
        interp_neighb = interp_neighb[:len(dates)]

    else:
        if len(to_rm_tile) > 0:
            s2 = np.delete(s2, to_rm_tile, 0)
            interp = np.delete(interp, to_rm_tile, 0)
            dates = np.delete(dates, to_rm_tile)

        if len(to_rm_neighb) > 0:
            s2_neighb = np.delete(s2_neighb, to_rm_neighb, 0)
            interp_neighb = np.delete(interp_neighb, to_rm_neighb, 0)
            dates_neighb = np.delete(dates_neighb, to_rm_neighb)
        if len(dates_neighb) == 0 or len(dates) == 0:
            return 0, None, None

    if edge == "up":
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

        print(s1.shape, s1_neighb.shape)

        s2 = np.concatenate([s2_neighb, s2], axis = 1)
        s1 = np.concatenate([s1_neighb, s1], axis = 1)
        dem = np.concatenate([dem_neighb, dem], axis = 0)
        interp = np.concatenate([interp_neighb, interp], axis = 1)
        print(tiles_folder_x)
        print(tiles_folder_y)
        tiles_array, tiles_folder = make_tiles_right_neighb(tiles_folder_x, tiles_folder_y)
        process_subtiles(x, y, s2, dates, interp, s1, dem, predict_sess, gap_sess, tiles_folder, tiles_array)

    return 1, s2_shape, s2_neighb_shape

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
        if len(y_tiles) > 1:
            max_y = np.max(y_tiles) + SIZE
        
    n_tiles = len(glob(out_folder + "*/*.npy"))
    n_border = n_up + n_down + n_left + n_right
    predictions = np.full((shape[1], shape[0], n_tiles), np.nan, dtype = np.float32)
    mults = np.full((shape[1], shape[0], n_tiles), 0, dtype = np.float32)
    print(predictions.shape)
    i = 0

    for x_tile in x_tiles:
        y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
        y_tiles = [y for y in y_tiles if 'left' not in y]
        y_tiles = [y for y in y_tiles if 'down' not in y]
        y_tiles = [int(y[:-4]) for y in y_tiles if 'up' not in y]
        for y_tile in y_tiles:
            output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
            if os.path.exists(output_file):
                prediction = np.load(output_file)
                if prediction.shape[0] == SIZE:
                    if np.sum(prediction) < SIZE*SIZE*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        if (x_tile + SIZE - 1) < shape[1] and (y_tile + SIZE - 1) < shape[0]:
                            predictions[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = prediction
                            mults[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, 28)
                        else:
                            print(f"Skipping {x_tile, y_tile} because of {predictions.shape}")
                    i += 1
                
    # LEFT BLOCK
    if n_left > 0:
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[4:-4]) for y in y_tiles if 'left' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/left" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    if np.sum(prediction) < SIZE*SIZE*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[84:, :]
                        predictions[x_tile: x_tile+SIZE // 2, y_tile:y_tile + SIZE, i] = prediction
                        mults[x_tile: x_tile+ SIZE // 2, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, 35)[84:, :]
                    i += 1
                    
    # RIGHT BLOCK
    if n_right > 0:
        for x_tile in right:
            x_tile_name = x_tile
            x_tile = int(x_tile[5:])
            y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile_name) + "/") if '.DS' not in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile_name) + "/" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    if np.sum(prediction) < SIZE*SIZE*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:84, :]
                        predictions[x_tile: x_tile+SIZE // 2, y_tile:y_tile + SIZE, i] = prediction
                        mults[x_tile: x_tile+ SIZE // 2, y_tile:y_tile + SIZE, i] = fspecial_gauss(SIZE, 35)[:84, :]
                    i += 1
                    
    if n_up > 0:
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[2:-4]) for y in y_tiles if 'up' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/up" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    if np.sum(prediction) < SIZE*SIZE*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:, 84:]
                        predictions[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE // 2, i] = prediction
                        mults[x_tile: x_tile+ SIZE, y_tile:y_tile + SIZE // 2, i] = fspecial_gauss(SIZE, 35)[:, 84:]
                    i += 1
                    
    if n_down > 0:
        for x_tile in x_tiles:
            y_tiles = [y for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
            y_tiles = [int(y[4:-4]) for y in y_tiles if 'down' in y]
            for y_tile in y_tiles:
                output_file = out_folder + str(x_tile) + "/down" + str(y_tile) + ".npy"
                if os.path.exists(output_file):
                    prediction = np.load(output_file)
                    if np.sum(prediction) < SIZE*SIZE*255:
                        prediction = (prediction * 100).T.astype(np.float32)
                        prediction = prediction[:, :84]
                        predictions[x_tile: x_tile+SIZE, y_tile:y_tile + SIZE // 2, i] = prediction
                        mults[x_tile: x_tile+ SIZE, y_tile:y_tile + SIZE // 2, i] = fspecial_gauss(SIZE, 35)[:, :84]
                    i += 1

    predictions = predictions.astype(np.float32)

    predictions_range = np.nanmax(predictions, axis=-1) - np.nanmin(predictions, axis=-1)
    mean_certain_pred = np.nanmean(predictions[predictions_range < 50])
    mean_uncertain_pred = np.nanmean(predictions[predictions_range > 50])
    
    overpredict = True if (mean_uncertain_pred - mean_certain_pred) > 0 else False
    underpredict = True if not overpredict else False
    
    for i in range(predictions.shape[-1] - n_border):
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
    
    return predictions, mults

def cleanup(path_to_tile, path_to_right, delete = True, upload = True):

    for file in glob(path_to_right + "processed/*/left*"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'2020/processed/{str(x)}/{str(int(y) + 1)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for file in glob(path_to_tile + "processed/*/up*"):
        internal_folder = file[len(path_to_tile):]
        print(internal_folder)
        key = f'2020/processed/{str(x)}/{str(y)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for file in glob(path_to_right + "processed/*/down*"):
        internal_folder = file[len(path_to_tile):] 
        print(internal_folder)
        key = f'2020/processed/{str(x)}/{str(int(y) + 1)}/{internal_folder}'
        if upload:
            uploader.upload(bucket = 'tof-output', key = key, file = file)

    for folder in glob(path_to_tile + "processed/right*/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            internal_folder = folder[len(path_to_right):]
            print(internal_folder)
            key = f'2020/processed/{x}/{y}/' + internal_folder
            if upload:
                uploader.upload(bucket = 'tof-output', key = key, file = _file)
            #if delete:
            #    os.remove(_file)

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

if __name__ == "__main__":
    SIZE = 168

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--predict_model_path", dest = 'predict_model_path', default = '../models/182-temporal-oct-regularized/')
    parser.add_argument("--gap_model_path", dest = 'gap_model_path', default = '../models/182-gap-sept/')
    parser.add_argument("--superresolve_model_path", dest = 'superresolve_model_path', default = '../models/supres/')
    parser.add_argument("--db_path", dest = "db_path", default = "processing_area_june_28.csv")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    parser.add_argument("--start_id", dest = "start_id", default = 0)
    args = parser.parse_args()

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
    #gap_graph_def = tf.compat.v1.GraphDef()

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
        predict_logits = predict_sess.graph.get_tensor_by_name(f"predict/conv2d_13/Sigmoid:0")            
        predict_inp = predict_sess.graph.get_tensor_by_name("predict/Placeholder:0")
        predict_length = predict_sess.graph.get_tensor_by_name("predict/PlaceholderWithDefault:0")
    else:
        raise Exception(f"The model path {args.predict_model_path} does not exist")
    """
    if os.path.exists(args.gap_model_path):
        print(f"Loading gap model from {args.gap_model_path}")
        gap_file = tf.io.gfile.GFile(args.gap_model_path + "gap_graph.pb", 'rb')
        gap_graph_def.ParseFromString(gap_file.read())
        gap_graph = tf.import_graph_def(gap_graph_def, name='gap')
        gap_sess = tf.compat.v1.Session(graph=gap_graph)
        gap_logits = gap_sess.graph.get_tensor_by_name(f"gap/conv2d_13/Sigmoid:0")
        gap_inp = gap_sess.graph.get_tensor_by_name("gap/Placeholder:0")
    else:
        raise Exception(f"The model path {args.gap_model_path} does not exist")
    """
    gap_file = None
    gap_graph_def = None
    gap_graph = None
    gap_sess = None
    gap_logits = None
    gap_inp = None
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

    if os.path.exists(args.db_path):
        data = pd.read_csv(args.db_path)
        data = data[data['country'] == args.country]
        data = data.reset_index(drop = True)
        print(f"There are {len(data)} tiles for {args.country}")
    else:
        raise Exception(f"The database does not exist at {args.db_path}")

    data['X_tile'] = data['X_tile'].astype(int)
    data['Y_tile'] = data['Y_tile'].astype(int)
    data = data.sort_values(['X_tile', 'Y_tile'], ascending=[False, True])

    print(data.head(5))
    print(len(data))
    
    for index, row in data.iterrows(): # We want to sort this by the X so that it goes from left to right
        if index > int(args.start_id):
            x = str(int(row['X_tile']))
            y = str(int(row['Y_tile']))
            x = x[:-2] if ".0" in x else x
            y = y[:-2] if ".0" in y else y
            
            
            path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/'
            path_to_right = f'{args.local_path}{str(x)}/{str(int(y) + 1)}/'

            print(path_to_tile, path_to_right)

            initial_bbx = [row['X'], row['Y'], row['X'], row['Y']]
            bbx = make_bbox(initial_bbx, expansion = 300/30)

            print(data['X_tile'][index], data['Y_tile'][index])
            data_neighb = data.copy()
            neighb_bbx = None

            print(int(x), int(y) + 1)
            try:
                data_neighb = data_neighb[data_neighb['X_tile'] == int(x)]
                data_neighb = data_neighb[data_neighb['Y_tile'] == int(y) + 1]
                data_neighb = data_neighb.reset_index()
                neighb = [data_neighb['X'][0], data_neighb['Y'][0], data_neighb['X'][0], data_neighb['Y'][0]]
                print(data_neighb['X_tile'][0], data_neighb['Y_tile'][0])
                neighb_bbx = make_bbox(neighb, expansion = 300/30)
            except KeyboardInterrupt:
                break
            except:
                print("One of the tiles doesnt exist, skipping")
                continue
            try:
                finished, s2_shape, s2_neighb_shape  = resegment_border(x, y, "up", args.local_path)
            except KeyboardInterrupt:
                    break
            except Exception as e:
                print(f"Ran into {str(e)}")
                finished = 0
                s2_shape = (0, 0)
                s2_neighb_shape = (0, 0)
            #finished = 1
            if finished == 1:
                try:
                    predictions, _ = recreate_resegmented_tifs(path_to_tile + "processed/", s2_shape)
                    file = write_tif(predictions, bbx, x, y, path_to_tile, "_SMOOTH")
                    key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_SMOOTH.tif'
                    uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                    predictions, _ = recreate_resegmented_tifs(path_to_right + "processed/", s2_neighb_shape)
                    file = write_tif(predictions, neighb_bbx, x, str(int(y) + 1), path_to_right, "_SMOOTH")
                    key = f'2020/tiles/{x}/{str(int(y) + 1)}/{x}X{str(int(y) + 1)}Y_SMOOTH.tif'
                    uploader.upload(bucket = args.s3_bucket, key = key, file = file)

                    cleanup(path_to_tile, path_to_right, delete = True, upload = True)
                except KeyboardInterrupt:
                    break
                except Exception as e:
                    print(f"Ran into {str(e)}")
    
    """
    x = 1657
    y = 1173

    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    data_neighb = data.copy()
    data = data[data['Y_tile'] == int(y)]
    print(len(data))
    data = data[data['X_tile'] == int(x)]
    print(len(data))
    data = data.reset_index(drop = True)
    
    
    data_neighb = data_neighb[data_neighb['Y_tile'] == int(y) + 1]
    data_neighb = data_neighb[data_neighb['X_tile'] == int(x)]
    data_neighb = data_neighb.reset_index(drop = True)

    path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/'
    path_to_right = f'{args.local_path}{str(x)}/{str(int(y) + 1)}/'

    print(path_to_tile, path_to_right)

    initial_bbx = [data['X'][0], data['Y'][0], data['X'][0], data['Y'][0]]
    bbx = make_bbox(initial_bbx, expansion = 300/30)
   
    neighb_bbx = [data_neighb['X'][0], data_neighb['Y'][0], data_neighb['X'][0], data_neighb['Y'][0]]
    neighb_bbx = make_bbox(neighb_bbx, expansion = 300/30)
    
    try:
    
        finished = resegment_border(x, y, "up", args.local_path)
    except:
        print("Ran into an error!")
        finished = 0
    #finished = 1
    if finished == 1:
        try:
            predictions, _ = recreate_resegmented_tifs(path_to_tile + "processed/")
            file = write_tif(predictions, bbx, x, y, path_to_tile, "_SMOOTH")
            key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_SMOOTH.tif'
            uploader.upload(bucket = args.s3_bucket, key = key, file = file)

            predictions, _ = recreate_resegmented_tifs(path_to_right + "processed/")
            file = write_tif(predictions, neighb_bbx, x, str(int(y) + 1), path_to_right, "_SMOOTH")
            key = f'2020/tiles/{x}/{str(int(y) + 1)}/{x}X{str(int(y) + 1)}Y_SMOOTH.tif'
            uploader.upload(bucket = args.s3_bucket, key = key, file = file)

            cleanup(path_to_tile, path_to_right, delete = True, upload = True)
        except:
            continue
    """
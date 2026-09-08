import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import rasterio as rs
import hickle as hkl
from scipy import ndimage
from scipy.ndimage import median_filter, maximum_filter, percentile_filter
import yaml
import boto3
import itertools
import zipfile
import os
import psutil
import copy
import platform
import time
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from change import change
import shutil
from datetime import datetime, timezone, timedelta
import gc
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

DRIVE = 'John'
START_YEAR = 2017
END_YEAR = 2026  # exclusive: years 2017..END_YEAR-1 (e.g. 2025)
# If True, download TTC tile data from s3://tof-output/YEAR/tiles/ before loading; if False, use only what's on disk
REFRESH_TTC = False
TTC_BASE = f'/Volumes/{DRIVE}'  # local base for tof-output-{YEAR}/{x}/{y}/
S3_BUCKET_TTC = 'tof-output'
VERBOSE = True   # False = fewer prints (faster I/O)
GC_EVERY_N_TILES = 5  # gc.collect() every N tiles (0 = every tile)

def _log(*args, **kwargs):
    if VERBOSE:
        print(*args, **kwargs)

def days_since_creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        cdate = stat.st_birthtime
        past = datetime.now(tz = timezone.utc) - datetime.fromtimestamp(cdate, tz=timezone.utc)
        return past.days

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

# Passed memprofile
def validate_ard(n_imgs_per_year, ard, dates, start_year=START_YEAR):
    # Compares between-year and within-year NDMI values
    # To look for huge shifts that could mean that
    # The ARD data for a specific year is problematic
    total_imgs = 0
    annual_ndmis = []
    for i in n_imgs_per_year:
        start = total_imgs
        end = total_imgs + i
        if i > 0:
            annual_ndmi = np.mean(ard[start:end])
            annual_ndmis.append(annual_ndmi)
            total_imgs += i
        else:
            annual_ndmis.append(np.nan)
    nans, x = nan_helper(annual_ndmis)
    annual_ndmis = np.array(annual_ndmis)
    if sum(nans) > 0:
        l = np.interp(x(nans), x(~nans), annual_ndmis[~nans])
        annual_ndmis[nans]= np.interp(x(nans), x(~nans), annual_ndmis[~nans])
    annual_ndmi_diff = np.diff(annual_ndmis)
    abs_diffs = np.abs(annual_ndmi_diff)
    sum_abs = np.sum(abs_diffs)
    n_d = len(annual_ndmi_diff)
    outliers = []
    if n_d > 1:
        for i in range(n_d):
            mean_others = (sum_abs - abs_diffs[i]) / (n_d - 1)
            if mean_others > 0:
                outlier_ratio = annual_ndmi_diff[i] / mean_others
                if outlier_ratio >= 3 and i == 0:
                    outliers.append(i)
    _log(annual_ndmis)
    return outliers

# Passed memprofile
def validate_gain(gain, potential_loss, fs):
    # Removes gain events where the tree cover shows
    # Tree -> No Tree -> Tree but no loss event is identified
    # This predicates the gain on loss if there is rotation
    for i in range(gain.shape[0]):
        if i != 0:
            candidate = (np.min(fs[i - 1: i + 1], axis = 0) - fs[i + 1]) > 45
        else:
            candidate = (fs[i] - fs[i + 1]) > 45
        candidate = candidate * (fs[i + 1] <= 35) 
        potential_loss[i] = change.remove_nonoverlapping_events(candidate, 
            potential_loss[i], 2)

    for i in range(gain.shape[0]):
        gaini = gain[i]
        lossi = potential_loss[i]
        # i = 0 = 2018 = fs[1]
        #early = np.clip(i, 0, 10)
        early_years = fs[:i]
        
        later_years = fs[i + 2:]
        gainedareas = gaini > 0

        if len(early_years.shape) == 2:
            early_years = early_years[np.newaxis]

        was_trees_before = np.sum(np.logical_and(early_years >= 70,
                                    early_years <= 100), axis = 0) > 0
        if early_years.shape[0] > 1:
            max_diff = np.diff(early_years, axis = 0)
            max_diff = np.min(max_diff, axis = 0)
            was_trees_before *= (max_diff <= -50)

        if i > 0:
            no_prior_loss = (np.sum(potential_loss[:i] > 0, axis = 0) == 0)
        else:
            no_prior_loss = np.ones_like(potential_loss[0])
        no_later_loss = (np.sum(potential_loss[i:] > 0, axis = 0) == 0)
        was_notrees_after = np.sum(later_years < 30, axis = 0) > 0

        # If there was trees before the gain, but no loss event
        # Or if there is non trees after the gain, but no loss event
        # Then the gain is false positive
        #if i != 0:
        bad_gain_before = (was_trees_before * no_prior_loss)
        #else:
            #bad_gain_before = np.zeros_like(was_trees_before)
        if (i + 1) != gain.shape[0]:
            bad_gain_after = (was_notrees_after * no_later_loss)
        else:
            bad_gain_after = np.zeros_like(was_trees_before)
        gain[i][np.logical_or(bad_gain_before > 0, bad_gain_after > 0)] = 0
    return gain

# Passed memprofile
def remove_unstable_loss(year, med, fs, nans, start_year=START_YEAR, n_years=None, year_index=None):
    # If the loss year is start_year+1, then there is only 1 image before
    # If there is increase in tree cover for both of two years after a loss event
    # but no gain/rotation event is detected, then remove the loss
    if n_years is None:
        n_years = nans.shape[0]
    yi = year_index if year_index is not None else (year - start_year)  # index into fs/nans

    def _id_lgl(year_idx, fs, gain):
        second_largest_loss = np.partition(np.diff(fs, axis=0), 1, axis=0)[1]
        largest_gain = np.max(np.diff(fs, axis=0)[:year_idx + 1], axis=0)
        second_largest_loss[second_largest_loss >= -30] = 0
        largest_gain[largest_gain <= 40] = 0.
        largest_gain = largest_gain / 2
        largest_gain[gain > 0] = 0.
        loss_clip = np.maximum(largest_gain, second_largest_loss * -1)
        loss_clip[loss_clip > 40] = 40
        return loss_clip

    gain = np.logical_or(
        np.logical_and(med >= 150, med <= 160),
        np.logical_and(med >= 101, med <= 105)
    )
    ttc_year = fs[yi]
    loss_year = med == (year - 1817)
    thresh = 60
    if yi > 0 and yi < n_years - 1 and (year > start_year + 1 and year < start_year + 5):
        # middle years: 2+ images before and 2+ after
        next_year = np.mean(fs[yi + 1:min(yi + 3, n_years)], axis=0)
        unstable_loss = (next_year > thresh) * (ttc_year < 40) * loss_year
        no_img_lossyear = binary_dilation(nans[yi] == 1, iterations=15)
        if yi >= 1:
            no_img_lossyear = np.logical_or(no_img_lossyear, binary_dilation(nans[yi - 1] == 1, iterations=15))
        if yi + 1 < n_years:
            no_img_lossyear = np.logical_or(no_img_lossyear, binary_dilation(nans[yi + 1] == 1, iterations=15))
    elif yi == 1 or (year_index is None and year == start_year + 1):
        # first loss year in array: only 1 image before
        next_year = np.mean(fs[yi + 1:], axis=0)
        unstable_loss = (next_year > 50) * (ttc_year < 50) * loss_year
        no_img_lossyear = binary_dilation(nans[yi] == 1, iterations=15)
        if yi >= 1:
            no_img_lossyear = np.logical_or(no_img_lossyear, binary_dilation(nans[yi - 1] == 1, iterations=15))
        if yi + 1 < n_years:
            no_img_lossyear = np.logical_or(no_img_lossyear, binary_dilation(nans[yi + 1] == 1, iterations=15))
    else:
        # later years (e.g. start_year + 5 and beyond)
        no_img_lossyear = binary_dilation(nans[yi] == 1, iterations=30)
        if yi >= 1:
            no_img_lossyear = np.logical_or(no_img_lossyear, binary_dilation(nans[yi - 1] == 1, iterations=30))
        unstable_loss = no_img_lossyear
    
    #np.save("fs.npy", fs)
    if np.mean(gain) > 0 and np.mean(gain == 0) > 0:
        mean_tc = np.nanmean(fs[:, gain == 0], axis = (1))
    else:
        mean_tc = np.nanmean(fs, axis = (1, 2))
    max_increase = np.nanmax(np.diff(mean_tc))
    max_decrease = np.nanmin(np.diff(mean_tc))
    ov_mean_tc = np.nanmean(mean_tc)

    # If there has previously been a decrease in tree cover, or a non-gain increase
    # Then the loss threshold is incremented accordingly
    # prior years: all years before current (loss) year
    prior_notree = np.sum(fs[:yi] < 30, axis=0) >= 1 if yi > 0 else np.zeros_like(ttc_year, dtype=bool)
    prior_gain = np.max(fs[:yi], axis=0) - np.min(fs[:yi], axis=0) if yi > 0 else np.zeros_like(ttc_year)
    #prior_gain = np.max(np.diff(fs[:year - 2015], axis = 0), 0) >= 50
    prior_notree *= (gain == 0)
    prior_gain = (prior_gain >= 40) * (gain == 0)
    prior_notree = np.logical_or(prior_notree, prior_gain)
    unstable_loss = np.maximum(unstable_loss, prior_notree)
    #print(f"The overall mean tc is {ov_mean_tc}, {max_increase}, {max_decrease}")
    """
    bad_flag = False
    if ov_mean_tc < 20:
        if max_increase > 5 and max_decrease < -5:
            bad_flag = True
    elif ov_mean_tc < 30:
        if max_increase > 8 and max_decrease < -8:
            bad_flag = True
    elif ov_mean_tc < 50:
        if max_increase > 12 and max_decrease < -12:
            bad_flag = True
    else:
        if max_increase > 20 and max_decrease < -20:
            bad_flag = True
    if bad_flag:
        if ov_mean_tc < 70:
            print(f"The loss mask before is: {np.mean(unstable_loss)}")
            loss_mask = np.min(np.diff(fs, axis = 0), axis = 0) > -60
            second_largest_loss = np.partition(np.diff(fs, axis = 0), 1, axis = 0)[1]
            loss_mask = np.logical_or(loss_mask, second_largest_loss < -30)
            print(unstable_loss.shape, unstable_loss.dtype, loss_mask.shape, loss_mask.dtype)
            unstable_loss = np.maximum(unstable_loss, loss_mask)
            print(f"The loss mask after is: {np.mean(unstable_loss)}")
    """
    return unstable_loss, no_img_lossyear


def download_ttc_tile_from_s3(x, y, awskey, awssecret, ttc_base=None, years=None, s3_client=None):
    """
    Download TTC tile data from s3://tof-output/YEAR/tiles/{x}/{y}/ to local
    ttc_base/tof-output-{YEAR}/{x}/{y}/ for each year. Creates dirs as needed.
    Pass s3_client to reuse one client (faster when processing many tiles).
    """
    if ttc_base is None:
        ttc_base = TTC_BASE
    if years is None:
        years = range(START_YEAR, END_YEAR)
    conn = s3_client if s3_client is not None else boto3.client('s3')
    x, y = str(int(x)), str(int(y))
    for year in years:
        s3_prefix = f"{year}/tiles/{x}/{y}/"
        local_dir = os.path.join(ttc_base, f"tof-output-{year}", x, y)
        os.makedirs(local_dir, exist_ok=True)
        try:
            paginator = conn.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=S3_BUCKET_TTC, Prefix=s3_prefix):
                for obj in page.get('Contents', []):
                    key = obj['Key']
                    if key.endswith('/'):
                        continue
                    rel = os.path.relpath(key, s3_prefix)
                    target = os.path.join(local_dir, rel)
                    os.makedirs(os.path.dirname(target), exist_ok=True)
                    conn.download_file(S3_BUCKET_TTC, key, target)
            # if we listed nothing, no error but nothing written
        except Exception as e:
            print(f"  TTC S3 download {year}/{x}/{y}: {e}")


def load_ttc_tiles(x, y, ttc_base=None):
    if ttc_base is None:
        ttc_base = TTC_BASE

    def _load_file(dir_i):
        try:
            all_files = os.listdir(dir_i)
        except OSError:
            return None
        smooth = [f for f in all_files if "_SMOOTH" in f and os.path.splitext(f)[-1] == ".tif"]
        smooth = None
        if smooth:
            smooth_xy = [f for f in smooth if "_SMOOTH_XY" in f]
            smooth_x = [f for f in smooth if "_SMOOTH_X" in f]
            smooth_y = [f for f in smooth if "_SMOOTH_Y" in f]
            files = smooth_xy or smooth_x or smooth_y or smooth
        else:
            files = [f for f in all_files if "_FINAL" in f and f.endswith(".tif")]
        if not files:
            return None
        return os.path.join(dir_i, files[0])

    def _load_one_year(i):
        key = 'f' + str(i)[-2:]
        dir_i = os.path.join(ttc_base, f'tof-output-{i}', str(x), str(y))
        fpath = _load_file(dir_i)
        if fpath is None:
            return i, None
        try:
            with rs.open(fpath) as arr:
                fx = arr.read(1).astype(np.float32)[np.newaxis]
            return i, (key, fx, fpath)
        except Exception:
            return i, None

    data = {}
    for i in range(START_YEAR, END_YEAR):
        data['f' + str(i)[-2:]] = np.zeros((3, 3))
    n_workers = min(END_YEAR - START_YEAR, 8)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(_load_one_year, i): i for i in range(START_YEAR, END_YEAR)}
        for fut in as_completed(futures):
            i, result = fut.result()
            if result is not None:
                key, fx, fpath = result
                data[key] = fx
                if VERBOSE:
                    _log(f"{i} processed {days_since_creation_date(fpath)} days ago")

    list_of_files = list(data.values())
    # Resize all tiles to a common shape so we can concatenate (years may have different grid sizes)
    valid_shapes = [x.shape[1:] for x in list_of_files if x.shape[0] != 3]
    if not valid_shapes:
        raise ValueError(f"No valid TTC data for tile {x},{y}")
    target_h = min(s[0] for s in valid_shapes)
    target_w = min(s[1] for s in valid_shapes)
    target_shape = (target_h, target_w)
    resized = []
    for i, arr in enumerate(list_of_files):
        if arr.shape[0] == 3:
            resized.append(arr)
        else:
            if arr.shape[1:] != target_shape:
                arr = resize(np.ascontiguousarray(arr), (1, target_h, target_w), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
            else:
                arr = np.ascontiguousarray(arr.copy())
            resized.append(arr)
    list_of_files = resized

    # Fill placeholders (3,3) from nearest resized year so all arrays are (1, target_h, target_w)
    valid_idx = [i for i in range(len(list_of_files)) if list_of_files[i].shape[0] != 3]
    for i in range(len(list_of_files)):
        if list_of_files[i].shape[0] == 3:
            if i == 0:
                print(f"{START_YEAR} does not exist")
            # nearest valid index (prefer next, then previous)
            j = min(valid_idx, key=lambda j: (abs(j - i), j))
            list_of_files[i] = list_of_files[j].copy()

    valid_shape = target_shape
    n_valid_years = np.zeros(valid_shape)
    nans = np.zeros((len(list_of_files), valid_shape[0], valid_shape[1]), dtype=np.float32)
    try:
        for i in range(len(list_of_files)):
            if list_of_files[i].shape[0] != 3:
                nans[i] = list_of_files[i] == 255
    except Exception:
        print(f"Skipping {str(x)}, {str(y)}")
        raise

    fs = np.concatenate(list_of_files, axis=0)
    fs = np.float32(fs)
    _log(f"  TTC: {fs.shape[0]} years, grid {fs.shape[1]}x{fs.shape[2]}")
    #fs = 100 * (fs - 15) / 85
    fs[fs < 0] = 0.
    fs[fs < 20] = 0.
    
    n_valid_years[:] = np.sum((fs != 255) & ~np.isnan(fs), axis=0)
    for i in range(0, fs.shape[0]):
        if i == 0:
            isnan = np.logical_or(np.isnan(fs[i]), fs[i] >= 255)
            fs[i, isnan] = fs[i + 1, isnan]
        elif i == (fs.shape[0] - 1):
            isnan = np.logical_or(np.isnan(fs[i]), fs[i] >= 255)
            fs[i, isnan] = fs[i - 1, isnan]
        else:
            isnan = np.logical_or(np.isnan(fs[i]), fs[i] >= 255)
            isnannext = np.logical_or(np.isnan(fs[i + 1]), fs[i + 1] >= 255)
            isnanbefore = np.logical_or(np.isnan(fs[i - 1]), fs[i - 1] >= 255)
            isnan = isnan * isnannext * isnanbefore
            fs[i, isnan] = (fs[i - 1, isnan] + fs[i + 1, isnan]) / 2
    
    n_years = fs.shape[0]
    stable_n = min(6, n_years)  # require at least 6 years of stable when available
    stable = np.sum(np.logical_and(fs >= 40, fs <= 100), axis=0) >= stable_n
    stable = binary_erosion(stable)
    _log(f"  Stable pixels: {np.sum(stable)}")
    notree = np.sum(fs < 50, axis=0) == n_years
    notree = binary_erosion(notree)
    #np.save('notree.npy', notree)
    fs = change.temporal_filter(fs)
    changemap = None
    return fs, changemap, stable, notree, n_valid_years, nans


def validate_patch_gain(fs, gain, loss):
    #! Deprecated
    gain = gain == 5
    Zlabeled,Nlabels = ndimage.measurements.label(gain)
    for i in range(Nlabels):
        was_loss = np.mean(loss[Zlabeled == i] > 0.1)
        if not was_loss:
            prior_treecover = np.mean(fs[:4, Zlabeled == i], axis = 1)
            #if np.min(np.diff(prior_treecover)) < -30:
                #print(f"Possible problem, {np.sum(Zlabeled == i)}")
            #else:
            #    print(f"{prior_treecover}, {np.sum(Zlabeled == i)}")

country = 'Rwanda'
local_path = '../project-monitoring/tiles/'
output_path = f'/Volumes/John/change-new/{country.replace(" ", "")}/'
country = country.title()
print(country)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Change detection job (TTC + ARD)")
    parser.add_argument(
        "--refresh_ttc",
        action="store_true",
        default=None,
        help="Download TTC tiles from s3://tof-output/YEAR/tiles/ before loading (overrides REFRESH_TTC)",
    )
    parser.add_argument(
        "--no_refresh_ttc",
        action="store_true",
        help="Use only existing TTC files on disk (default)",
    )
    args, _ = parser.parse_known_args()
    if args.refresh_ttc:
        REFRESH_TTC = True
    if args.no_refresh_ttc:
        REFRESH_TTC = False

    # Use START_YEAR..END_YEAR for ARD/TTC year range
    change.YEARS = list(range(START_YEAR, END_YEAR))
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open("../config.yaml", 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        SHUB_SECRET = key['shub_secret']
        SHUB_KEY = key['shub_id']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']

    s3_client = None
    if REFRESH_TTC:
        s3_client = boto3.client('s3')

    data = pd.read_csv("process_area_2022.csv")
    data = pd.read_csv('rwanda-tiles.csv')#"process_area_2022.csv")
    #data = pd.read_csv('maharashtra.csv')
    #data = pd.read_csv("santacruz.csv")
    data = data[data['country'] == country]
    try:
        # Allow numeric or string columns (e.g. "2233" or 2233)
        for col in ('X_tile', 'Y_tile'):
            if col in data.columns:
                ser = data[col].astype(str).str.extract(r'(\d+)', expand=False)
                data[col] = pd.to_numeric(ser)
    except Exception as e:
        print(f"Ran into {str(e)} error")
        traceback.print_exc()
        time.sleep(1)

    #data = data[3000:]
    x = 2335
    y = 972
    #data = data[data['Y_tile'] == int(y)]
    #data = data[data['X_tile'] == int(x)]
    #data = data.sample(frac=1).reset_index(drop=True)


    #data = data.sort_values(by=['Y_tile'])
    #data = data.iloc[::-1]
    data = data.reset_index(drop = True)
    #data = data[:1500]
    x = str(int(x))
    y = str(int(y))

    #for i in [0]:
    for i, val in data[0:].iterrows():
        x = val.X_tile
        y = val.Y_tile
        suffix = 'CHANGENEW_bigall4-may'
        fname = f"{output_path}{str(x)}X{str(y)}Y{suffix}.tif"
        if os.path.exists(fname):
            print(i, fname, " exists")
        else:
            try:
                print(f"\n--- Tile {x}, {y} ---")
                if REFRESH_TTC:
                    _log("  Downloading TTC from s3://tof-output/YEAR/tiles/ ...")
                    download_ttc_tile_from_s3(x, y, AWSKEY, AWSSECRET, s3_client=s3_client)
                fs, changemap, stable, notree, n_valid_years, nans = load_ttc_tiles(x, y)
                _log("  TTC mean % and adjustment by year:")
                # Vectorized: same formula (mean of year-to-year diffs, averaged with next when available)
                n_y = fs.shape[0]
                diffs = np.diff(fs.astype(np.float32), axis=0)
                diff_means = np.mean(diffs, axis=(1, 2))
                adjustments = [0.0]
                for i in range(1, n_y):
                    if i < n_y - 1:
                        adj = (diff_means[i - 1] + diff_means[i]) * 0.5
                    else:
                        adj = float(diff_means[i - 1])
                    adjustments.append(adj)
                    _log(f"    {START_YEAR + i}: {np.mean(fs[i]):.2f}% mean TC, adj={adj:.2f}")
                change.download_and_unzip_data(x, y, local_path, AWSKEY, AWSSECRET)
                _log("The data has been downloaded")
                bbx = change.tile_bbx(x, y, data) 

                # Load the separate ARD files (years come from change.YEARS = range(START_YEAR, END_YEAR))
                list_of_files, list_of_dates, dem = change.load_all_ard(x, y, local_path)
                _log("The data has been loaded")
                dem = median_filter(dem, size=9)
                dem = resize(dem, (n_valid_years.shape), 0)

                # Identify which years have valid data; derive N_YEARS and actual calendar years
                years_with_data = [i for i, val in enumerate(list_of_files) if val.shape[1] != 3]
                list_of_files = [val for i, val in enumerate(list_of_files) if i in years_with_data]
                list_of_dates = [val for i, val in enumerate(list_of_dates) if i in years_with_data]
                N_YEARS = len(years_with_data)
                actual_years = np.array([START_YEAR + idx for idx in years_with_data], dtype=np.int32)
                MAX_YEAR = int(actual_years.max()) if N_YEARS else START_YEAR
                _log(f"  ARD: years {actual_years.tolist()}, N_YEARS={N_YEARS}, range {int(actual_years.min())}-{MAX_YEAR}")
                if MAX_YEAR >= 2024:
                    _log(f"  (includes 2024, 2025 in processing)")

                # Resize ARD arrays to the TTC grid (contiguous input can be faster)
                ttc_grid = n_valid_years.shape
                resized = []
                for arr in list_of_files:
                    if arr.shape[1:] != ttc_grid:
                        arr = resize(np.ascontiguousarray(arr), (arr.shape[0], ttc_grid[0], ttc_grid[1]), order=1, preserve_range=True, anti_aliasing=True).astype(np.float32)
                    resized.append(arr)
                list_of_files = resized

                n_imgs_per_year = np.zeros((N_YEARS,), dtype=np.int32)
                _log("  ARD per year (year: n_images, shape):")
                for i, val in enumerate(list_of_files):
                    if val.shape[1] != 3:
                        n_imgs_per_year[i] = val.shape[0]
                    _log(f"    {actual_years[i]}: {n_imgs_per_year[i]} imgs, {val.shape}")

                ard = np.concatenate(list_of_files, axis=0)
                dates = np.concatenate(list_of_dates)
                #np.save("dates.npy", dates)

                # Validate the L2A imagery for 2017, which can be wrong due to 
                # Sensor calibration in 2017
                # Look for >= 3 median change difference in 2017 -> 2018
                # Since the L2A images for Q1/Q2 2017 can be suspect in some areas
                # And can mean that a bad baseline is set
                outliers = validate_ard(n_imgs_per_year, ard, dates)
                if len(outliers) > 0:
                    _log(f"Removing {START_YEAR} as an outlier")
                    ims_second_year = ard[n_imgs_per_year[1]:n_imgs_per_year[2]]
                    ard[:n_imgs_per_year[0]] = np.median(ims_second_year, axis=(0))[np.newaxis]
                    fs[0] = np.mean(fs[0:2], axis=0)

                kde = None
                if (len(years_with_data) > 3) and np.sum(stable) > 100:
                    _log("  --- Change detection (gain/loss) ---")
                    # Create the Kernel Density Estimates based on the stable tree pixels
                    # Assume that with 2%, so 7000 samples, we can get a good KDE
                    # With 2000 samples we need it to be in the 200, which is 2.8 isntead of 10
                    # so the divider is (stable / 7000)
                    multiplier = np.clip(np.sum(stable) / 8000, 0.33, 1)
                    kde, kde10, kde_expected, kde2, percentiles = change.make_all_kde(ard, stable, maxpx = 15000, multiplier = 1)
                    #else:
                    # If not enough reference pixels, use the stable non-tree and invert all the calculations
                        #kde, kde10, kde_expected, kde2, percentiles, percentiles = change.make_all_kde(ard, notree)
                    gain = np.zeros((N_YEARS-1, ard.shape[1], ard.shape[2]), dtype = np.float32)
                    loss = np.zeros((N_YEARS-1, ard.shape[1], ard.shape[2]), dtype = np.float32)
                    ndmiloss = np.zeros((N_YEARS-1, ard.shape[1], ard.shape[2]), dtype = np.float32)

                    # For each year, identify the KDE NDMI gain/loss
                    # And assign a year to the values
                    for i in range(N_YEARS-1):
                        if np.sum(stable) < (600*600*.02):
                            lower = np.clip(i - 2, 0, i)
                            upper = i + 1 if i > 0 else i + 2
                            n_years = upper - lower
                            stable_twoyear = np.sum(np.logical_and(fs[lower:upper] >= 40, fs[lower:upper] <= 100), axis = 0) >= n_years # 40
                            stable_twoyear = binary_erosion(stable_twoyear)
                            _log(f"    Stable pixels for {actual_years[i + 1]}: {np.sum(stable_twoyear)}")
                            kde_win, kde10_win, kde_expected_win, kde2_win, percentiles = change.make_all_kde(ard, stable_twoyear, maxpx = 20000)
                            loss[i], ndmiloss[i] = change.identify_loss_in_year(kde2_win, kde_win, kde_expected_win, kde2_win, dates, actual_years[i + 1])
                        # Can only detect gain if there is at least 1% stable pixels
                        gain[i] = change.identify_gain_in_year(kde, kde10, kde_expected, dates, actual_years[i + 1]) * (i + 2)
                        # Can detect loss with the two-year KDE values where <2% stable
                        if np.sum(stable) >= (600*600*.02):
                            loss[i], ndmiloss[i] = change.identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, actual_years[i + 1]) 
                        loss[i] *= (i + 2)
                        ndmiloss[i] *= (i + 2)

                    # Predicate the gain on loss if there is a NT -> T -> NT
                    potential_loss = np.copy(loss)
                    gain = validate_gain(gain, potential_loss, fs)

                    # Fuzzy set matching btwn NDMI gain/loss and subraction gain/loss
                    #if kde is not None:

                    # TODO!: THIS ONE DOES NOT WORK
                    gain, loss = change.adjust_loss_gain(gain, loss, ndmiloss, fs, dates, adjustments, N_YEARS, max_year=MAX_YEAR, actual_years=actual_years)
                    #gain, loss = change.adjust_loss_gain(gain, loss, ndmiloss, fs, kde, kde10, kde_expected, kde2, dates)
                    #else:

                    rotational = np.logical_and(gain > 0, loss > 0)

                    # Rule-based cleanup of KDE gain based on
                    # Trends in the KDE vs time graph
                    befores = np.zeros((N_YEARS,))
                    afters = np.zeros((N_YEARS,))
                    #if np.sum(stable) > (600*600*.01):
                    movingavg = np.copy(percentiles).reshape((percentiles.shape[0], percentiles.shape[1] * percentiles.shape[2]))
                    movingavg = np.apply_along_axis(change.moving_average, 0, movingavg, 5)
                    movingavg = np.reshape(movingavg, (percentiles.shape[0]-4,percentiles.shape[1], percentiles.shape[2]))
                    

                    cfs_flat = change.calc_reference_change(movingavg, 0, 50, notree, dem)
                    cfs_hill = change.calc_reference_change(movingavg, 10, 50, notree, dem)
                    cfs_steep = change.calc_reference_change(movingavg, 20, 50, notree, dem)
                    cfs_trees = change.calc_tree_change(movingavg, 5, stable, dem)
                    cfs_trees10 = change.calc_tree_change(movingavg, 10, stable, dem)
                    befores = []
                    _log("  Gain fraction by year (before filter):")
                    for i in range(1, N_YEARS):
                        frac = np.mean(gain == i)
                        _log(f"    {actual_years[i]}: {frac:.6f}")
                        befores.append(frac)

                    modifier = 0.
                    n_stable = int(np.sum(stable))
                    if n_stable < 6000:
                        modifier += 0.025
                    if n_stable < 4000:
                        modifier += 0.025
                    if n_stable < 2000:
                        modifier += 0.025
                    if n_stable < 1000:
                        modifier += 0.025
                    if n_stable < 500:
                        modifier += 0.05
                    if n_stable < 250:
                        modifier += 0.05
                    if n_stable < 100:
                        modifier += 0.05
                    _log(f"  Modifier: {modifier}")
                    gainpx, Zlabeled, additional_gain, gaindates = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                            cfs_trees, cfs_trees10, notree, dem, dates, n_imgs_per_year, modifier)

                    gaindatesarr = np.zeros_like(gain)
                    for idx, date in zip(gainpx, gaindates):
                        gaindatesarr[Zlabeled == idx] = date
                    #np.save("gaindatesarr.npy", gaindatesarr)
                    gain[~np.isin(Zlabeled, gainpx)] = 0.
                    gain = np.maximum(gain, additional_gain)
                    afters = []
                    _log("  Gain fraction by year (after filter):")
                    for i in range(1, N_YEARS):
                        frac = np.mean(gain == i)
                        _log(f"    {actual_years[i]}: {frac:.6f}")
                        afters.append(frac)
                    befores_arr = np.array(befores)
                    afters_arr = np.array(afters)
                    with np.errstate(divide="ignore", invalid="ignore"):
                        ratio = np.where(befores_arr > 0, afters_arr / befores_arr, np.nan)
                    total_before = np.nansum(befores_arr)
                    total_after = np.nansum(afters_arr)
                    _log(f"  Ratio of gain remaining: {ratio}")
                    if total_before > 0:
                        _log(f"  Before: {total_before:.4f}, After: {total_after:.4f}, Change: {total_after / total_before:.4f}")
                    ratio = ratio * (np.array(befores) > 0.02)
                    ratio_flaglow = np.logical_and(ratio > 0, ratio < 0.33)
                    ratio_flaghigh = np.logical_and(ratio > 0, ratio < 0.1)
                    ratio_flaglow = np.nansum(ratio_flaglow[3:] > 0)
                    ratio_flaghigh = np.nansum(ratio_flaghigh > 0)
                    ratio_flagveryhigh = np.nanmax(np.array(befores) - np.array(afters)) > 0.15
                    absolute_flag = np.nanmax(np.array(befores) - np.array(afters)) > 0.05
                    #ratio_flaghigh = np.logical_or(ratio_flaghigh, (befores[-1] / total_before) > 0.8)
                    _log("  Flags: ratio_very_high={}, ratio_high={}, ratio_low={}, absolute={}".format(
                        ratio_flagveryhigh, ratio_flaghigh, ratio_flaglow, absolute_flag))
                    if ratio_flagveryhigh:
                        gainpx, Zlabeled, additional_gain, gaindates = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                                cfs_trees, cfs_trees10, notree, dem, dates, n_imgs_per_year, modifier + 0.2)
                        gain[~np.isin(Zlabeled, gainpx)] = 0.
                        gain = np.maximum(gain, additional_gain)
                    elif ratio_flaghigh:
                        gainpx, Zlabeled, additional_gain, gaindates = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                                cfs_trees, cfs_trees10, notree, dem, dates, n_imgs_per_year, modifier + 0.1)
                        gain[~np.isin(Zlabeled, gainpx)] = 0.
                        gain = np.maximum(gain, additional_gain)
                    elif ratio_flaglow or absolute_flag:
                        gainpx, Zlabeled, additional_gain, gaindates = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                                cfs_trees, cfs_trees10, notree, dem, dates, n_imgs_per_year, modifier + 0.05)
                        gain[~np.isin(Zlabeled, gainpx)] = 0.
                        gain = np.maximum(gain, additional_gain)
                    afters = []
                    for i in range(1, N_YEARS):
                        afters.append(np.mean(gain == i))
                    afters = np.array(afters)
                    _log(f"  Gain after 2nd filter: total={np.sum(afters):.4f} | by year: {dict(zip(actual_years[1:].tolist(), np.round(afters, 6).tolist()))}")
                    _log(f"  Memory: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2:.0f} MB")

                    # If more than 80% of gain is removed, or 10% of the total plot
                    # Then we're likely in an area that has false positive gain
                    # E.g. a dry forest.
                    # Then add 0.05 to the gain requirement
                    # Or even better, regenerate KDE with the removed gain as stable??
                    #validate_patch_gain(fs, gain2, loss2)
                    
                    rotational = np.logical_and(gain > 0, loss > 0)
                    med = np.median(fs, axis = 0)
                    med[gain > 0] = (gain[gain > 0] + 100)
                    med[loss > 0] = (loss[loss > 0] + 200)
                    rotational = np.logical_and(gain > 0, loss > 0)
                    remove_rot = False
                    if remove_rot:
                        med[rotational] = np.median(fs, axis = 0)[rotational]
                    else:
                        med[np.logical_and(rotational, gain > loss)] = 150.
                        med[np.logical_and(rotational, loss > gain)] = 160.
                    fs[(np.median(fs, axis = 0) > 100)[np.newaxis].repeat(fs.shape[0], axis = 0)] = 255.
                    #np.save("fs.npy", fs)
                    #np.save("med.npy", med)
                    # If there is no tree -> tree -> no tree, and no gain event
                    # Then we can't say there is a loss event, because why would it be more likely
                    # For the loss to be true than for the gain to be true? 
                    for yi, year in enumerate(actual_years):
                        unstable_loss, noimg = remove_unstable_loss(
                            year, med, fs, nans, start_year=START_YEAR, n_years=N_YEARS, year_index=yi
                        )
                        unstable_loss[gain > 0] = 0.
                        loss_flag = np.logical_or(unstable_loss, noimg)
                        loss_flag = loss_flag * (med == (year - 1817))
                        med[loss_flag] = np.median(fs, axis=0)[loss_flag]

                    lte2_data = binary_dilation(n_valid_years <= 2, iterations=50)
                    is_oob = np.logical_and(med > 110, med < 150)
                    med[is_oob] = np.median(fs, axis=0)[is_oob]
                    med[lte2_data] = np.median(fs, axis=0)[lte2_data]

                    # If the most recent expected year (e.g. 2025) is missing, don't assign change
                    # to the terminal year — we can't confirm it without the following year.
                    if actual_years[-1] < END_YEAR - 1:
                        terminal_year = int(actual_years[-1])
                        last_year_med = terminal_year - 1817
                        mask_terminal = (med == last_year_med)
                        n_reset = np.sum(mask_terminal)
                        if n_reset > 0:
                            med[mask_terminal] = np.median(fs, axis=0)[mask_terminal]
                            _log(f"  Most recent year ({END_YEAR - 1}) missing: reset {n_reset} px change in terminal year {terminal_year} to median (no change)")
                else:
                    med = np.median(fs, axis = 0)
                change.write_tif(med, bbx, x, y, output_path, suffix = suffix)
                try:
                    loss_date = np.argmin(np.diff(kde, axis = 0), axis = 0)
                    loss_date[loss == 0] = 0.
                    np.save("lossdate.npy", loss_date)
                except:
                    continue
                

                if GC_EVERY_N_TILES == 0 or (i % GC_EVERY_N_TILES == 0):
                    gc.collect()
                for year in actual_years.tolist():
                    shutil.rmtree(f"{local_path}/{str(year)}/{str(x)}/{str(y)}/")
                    
            except Exception as e:
                gc.collect()
                print(f"Ran into {str(e)} error")
                traceback.print_exc()
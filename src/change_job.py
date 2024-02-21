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

DRIVE = 'John'

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
def validate_ard(n_imgs_per_year, ard, dates):
    total_imgs = 0
    annual_ndmis = []
    year = 2017
    for i in n_imgs_per_year:
        start = total_imgs
        end = total_imgs + i
        if i > 0:
            annual_ndmi = np.mean(ard[start:end])
            annual_ndmis.append(annual_ndmi)
            total_imgs += i
        else:
            annual_ndmis.append(np.nan)
        year += 1
    nans, x= nan_helper(annual_ndmis)
    annual_ndmis = np.array(annual_ndmis)
    if sum(nans) > 0:
        l = np.interp(x(nans), x(~nans), annual_ndmis[~nans])
        annual_ndmis[nans]= np.interp(x(nans), x(~nans), annual_ndmis[~nans])
    annual_ndmi_diff = np.diff(annual_ndmis)
    year = 2017
    outliers = []
    for i in range(len(n_imgs_per_year) - 1):
        other_diffs = np.copy(annual_ndmi_diff)
        other_diffs = np.delete(other_diffs, i)
        mean_others = np.mean(np.abs(other_diffs))
        outlier_ratio = annual_ndmi_diff[i] / mean_others
        print(year, outlier_ratio)
        if outlier_ratio >= 3 and i == 0:
            outliers.append(i)
        year += 1
    print(annual_ndmis)
    return outliers

# Passed memprofile
def validate_gain(gain, potential_loss, fs):
    # Remove notrees -> trees -> no trees, -> trees
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
def remove_unstable_loss(year, med, fs):
    # If there is increase in tree cover for both of two years after a loss event
    # But no gain/rotation event is detected, then remove the loss
    # Also -- should require loss events to be > 500m away from
    # No image predictions
    ttc_year = fs[year - 2017]
    loss_year = med == (year - 1817)
    if year == 2021:
        thresh = 60
    else:
        thresh = 60
    if year < 2022:
        next_year = np.min(fs[year - 2016:year+2-2016], axis = 0)
        unstable_loss = (next_year > thresh) * (ttc_year < 40) * loss_year
        no_img_lossyear = binary_dilation(ttc_year == 255, iterations = 20) * loss_year
        no_img_before = binary_dilation(fs[year - 2018] == 255, iterations = 20) * loss_year
        no_img_lossyear = np.logical_or(no_img_lossyear, no_img_before)
        unstable_loss = no_img_lossyear#np.logical_or(unstable_loss, no_img_lossyear)
    else:
        no_img_2022 = binary_dilation(fs[year - 2017] == 255, iterations = 20) * loss_year
        no_img_2021 = binary_dilation(fs[year - 2018] == 255, iterations = 20) * loss_year
        unstable_loss = np.logical_or(no_img_2022, no_img_2021)
    return unstable_loss


def load_ttc_tiles(x, y):
    f20_path = f'/Volumes/{DRIVE}/tof-output-2020/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'

    try:
        fpath = f'/Volumes/{DRIVE}/tof-output-2017/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        arr = rs.open(fpath)
        f17 = arr.read(1).astype(np.float32)[np.newaxis]
        arr.close()
        print(f"2017 processed {days_since_creation_date(fpath)} days ago")
    except:
        f17 = np.zeros((3, 3))
    try:
        fpath = f'/Volumes/{DRIVE}/tof-output-2018/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        arr = rs.open(fpath)
        f18 = arr.read(1).astype(np.float32)[np.newaxis]
        arr.close()
        print(f"2018 processed {days_since_creation_date(fpath)} days ago, {f18.shape}")
    except:
        f18 = np.zeros((3, 3))
    try:
        fpath = f'/Volumes/{DRIVE}/tof-output-2019/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        arr = rs.open(fpath)
        f19 = arr.read(1).astype(np.float32)[np.newaxis]
        arr.close()
        print(f"2019 processed {days_since_creation_date(fpath)} days ago, {f19.shape}")
    except:
        f19 = np.zeros((3, 3))

    if os.path.exists(f20_path):
        fpath = f'/Volumes/{DRIVE}/tof-output-2020/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        arr = rs.open(fpath)
        f20 = arr.read(1).astype(np.float32)[np.newaxis]
        arr.close()
        print(f"2020 processed {days_since_creation_date(fpath)} days ago")
        f20 = f20 if days_since_creation_date(fpath) < 100 else f19
    else:
        f20 = np.zeros((3, 3))
    try:
        fpath = f'/Volumes/{DRIVE}/tof-output-2021/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        f21 = rs.open(fpath).read(1).astype(np.float32)[np.newaxis]
        print(f"2021 processed {days_since_creation_date(fpath)} days ago")
    except:
        f21 = np.zeros((3, 3))
    try:
        fpath = f'/Volumes/{DRIVE}/tof-output-2022/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'
        f22 = rs.open(fpath).read(1).astype(np.float32)[np.newaxis]
        print(f"2022 processed {days_since_creation_date(fpath)} days ago")
    except:
        f22 = np.zeros((3, 3))

    list_of_files = [f17, f18, f19, f20, f21, f22]
    # get the shape
    # make a numb_years_valid file
    # return numb_years_valid
    valid_shape = [x.shape[1:] for x in list_of_files if x.shape[0] != 3][0]
    n_valid_years = np.zeros(valid_shape)
    nans = np.zeros((6, valid_shape[0], valid_shape[1]), dtype = np.float32)
    try:
        for i in range(len(list_of_files)):
            if list_of_files[i].shape[0] == 3:
                if i == 0:
                    print("17 does not exist")
                    list_of_files[i] = f18 if f18.shape[0] != 3 else f19
                    nans[0] = 1.
                elif i == len(list_of_files) - 1:
                    list_of_files[i] = list_of_files[i - 1]
                    nans[i] = 1
                else:
                    next_img = list_of_files[i + 1].shape[0] != 3
                    prev_img = list_of_files[i - 1].shape[0] != 3
                    if (next_img and prev_img):
                        list_of_files[i] = (list_of_files[i - 1] + list_of_files[i + 1]) / 2
                    elif next_img:
                        list_of_files[i] = list_of_files[i + 1]
                    else:
                        list_of_files[i] = list_of_files[i - 1]
                    nans[i] = 1
            else:
                nans[i] = list_of_files[i] == 255
    except:
        print(f"Skipping {str(x)}, {str(y)}")
        #list_of_files[i] = np.zeros((3, 3))

    fs = np.concatenate(list_of_files, axis = 0) # , f22
    fs = np.float32(fs)
    #fs = 100 * (fs - 15) / 85
    fs[fs < 0] = 0.
    fs[fs < 20] = 0.
    
    for i in range(0, fs.shape[0]):
        n_valid_years[np.logical_and(fs[i] != 255, ~np.isnan(fs[i]))] += 1
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
    
    stable = np.sum(np.logical_and(fs > 40, fs <= 100), axis = 0) == 6 # 40
    stable = binary_erosion(stable)
    notree = np.sum(fs < 50, axis = 0) == 6 # 30
    notree = binary_erosion(notree)
    #np.save('notree.npy', notree)
    fs = change.temporal_filter(fs)
    changemap = None
    return fs, changemap, stable, notree, n_valid_years, nans


def validate_patch_gain(fs, gain, loss):
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

year = 2019
country = 'humid'
local_path = '../project-monitoring/tiles/'
output_path = f'/Volumes/John/change/{country.replace(" ", "")}/'
country = country.title()
print(country)

if __name__ == '__main__':
    import argparse

    with open("../config.yaml", 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        SHUB_SECRET = key['shub_secret']
        SHUB_KEY = key['shub_id']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']

    data = pd.read_csv("centam_humid.csv")
    data = data[data['country'] == 'A']
    try:
        data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
        data['X_tile'] = pd.to_numeric(data['X_tile'])
        data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
        data['Y_tile'] = pd.to_numeric(data['Y_tile'])
    except Exception as e:
        print(f"Ran into {str(e)} error")
        time.sleep(1)

    #data = data[230:]
    x = 468
    y = 1161
    data = data[data['Y_tile'] == int(y)]
    data = data[data['X_tile'] == int(x)]
    #data = data.sample(frac=1).reset_index(drop=True)
    #data = data.iloc[::-1]
    data = data.reset_index(drop = True)
    x = str(int(x))
    y = str(int(y))

    #for i in [0]:
    for i, val in data.iterrows():
        x = val.X_tile
        y = val.Y_tile
        suffix = 'CHANGENEW_bigall4-may'
        fname = f"{output_path}{str(x)}X{str(y)}Y{suffix}.tif"
        if os.path.exists(fname):
            #os.remove(fname)
            print(fname, " exists")
        else:
            try:
                print(f"STARTING {x}, {y}")
                
                # Open all the TTC data
                fs, changemap, stable, notree, n_valid_years, nans = load_ttc_tiles(x, y)
                #np.save("stable.npy", stable)
                change.download_and_unzip_data(x, y, local_path, AWSKEY, AWSSECRET)
                bbx = change.tile_bbx(x, y, data)

                # Load the separate ARD files
                a17, a18, a19, a20, a21, a22, d17, d18, d19, d20, d21, d22, dem = change.load_all_ard(x, y, local_path)

                ard_path = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/'
                dem = median_filter(dem, size = 9)
                dem = resize(dem, (n_valid_years.shape), 0)

                # Reload the data, for some reason?
                #fs, changemap, stable, notree, n_valid_years, nans = load_ttc_tiles(x, y)
                list_of_files = [a17, a18, a19, a20, a21, a22]
                list_of_dates = [d17, d18, d19, d20, d21, d22]
                years_with_data = [i for i, val in enumerate(list_of_files) if val.shape[1] != 3]
                list_of_files = [val for i, val in enumerate(list_of_files) if i in years_with_data]
                list_of_dates = [val for i, val in enumerate(list_of_dates) if i in years_with_data]
                print(f"ARD data: {years_with_data}")
                n_imgs_per_year = np.zeros((6, ), dtype = np.int32)
                for i, val in enumerate(list_of_files):
                    if val.shape[1] != 3:
                        n_imgs_per_year[i] = val.shape[0]

                ard = np.concatenate(list_of_files, axis = 0)
                dates = np.concatenate(list_of_dates)

                # Look for >= 3 median change difference in 2017 -> 2018
                # Since the L2A images for Q1/Q2 2017 can be suspect in some areas
                # And can mean that a bad baseline is set
                outliers = validate_ard(n_imgs_per_year, ard, dates)
                if len(outliers) > 0:
                    print("Removing 2017 as an outlier")
                    ims2018 = ard[n_imgs_per_year[1]:n_imgs_per_year[2]]
                    ard[:n_imgs_per_year[0]] = np.median(ims2018, axis = (0))[np.newaxis]
                    fs[0] = np.mean(fs[0:2], axis = 0)

                kde, kde10, kde_expected, kde2, percentiles = change.make_all_kde(ard, stable)
                gain = np.zeros((5, ard.shape[1], ard.shape[2]), dtype = np.float32)
                loss = np.zeros((5, ard.shape[1], ard.shape[2]), dtype = np.float32)
                ndmiloss = np.zeros((5, ard.shape[1], ard.shape[2]), dtype = np.float32)
                print('after KDE', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                for i in range(5):
                    gain[i] = change.identify_gain_in_year(kde, kde10, kde_expected, dates, 2017 + i + 1) * (i + 2)
                    loss[i], ndmiloss[i] = change.identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, 2017 + i + 1) 
                    loss[i] *= (i + 2)
                    ndmiloss[i] *= (i + 2)

                potential_loss = np.copy(loss)
                #np.save('gain2.npy', gain)
                gain = validate_gain(gain, potential_loss, fs)
                print('before adjust', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                gain, loss = change.adjust_loss_gain(gain, loss, ndmiloss, fs, kde, kde10, kde_expected, kde2, dates)
                
                print('after adjust', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                #np.save('gain2-2.npy', gain2)
                
                rotational = np.logical_and(gain > 0, loss > 0)

                movingavg = np.copy(percentiles).reshape((percentiles.shape[0], percentiles.shape[1] * percentiles.shape[2]))
                movingavg = np.apply_along_axis(change.moving_average, 0, movingavg, 5)
                movingavg = np.reshape(movingavg, (percentiles.shape[0]-4,percentiles.shape[1], percentiles.shape[2]))
                #np.save('percentiles.npy', percentiles)
                # Because our minimum mapping unit is 10 px
                #for i in range(movingavg.shape[0]):
                 #   movingavg[i] = median_filter(movingavg[i], 3)
                #np.save('movingavg.npy', movingavg)
                

                cfs_flat = change.calc_reference_change(movingavg, 0, 50, notree, dem)
                cfs_hill = change.calc_reference_change(movingavg, 10, 50, notree, dem)
                cfs_steep = change.calc_reference_change(movingavg, 20, 50, notree, dem)
                cfs_trees = change.calc_tree_change(movingavg, 5, stable, dem)
                cfs_trees10 = change.calc_tree_change(movingavg, 10, stable, dem)
                print('before filter', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                befores = []
                for i in range(1, 6):
                    print(f'{i + 2017}: {np.mean(gain == i)}')
                    befores.append(np.mean(gain == i))
                gainpx, Zlabeled, additional_gain = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                        cfs_trees, cfs_trees10, notree, dem, dates, 0)

                gain[~np.isin(Zlabeled, gainpx)] = 0.
                gain = np.maximum(gain, additional_gain)
                print('gain2', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                afters = []
                for i in range(1, 6):
                    print(f'{i + 2017}: {np.mean(gain == i)}')
                    afters.append(np.mean(gain == i))
                ratio = (np.array(afters) / np.array(befores)) 
                total_before = np.sum(np.array(befores))
                total_after = np.sum(np.array(afters))
                print(f'The ratio of gain remaining is {ratio}')
                print(f'Before: {total_before}, After: {total_after}, Change: {total_after / total_before}')
                ratio = ratio * (np.array(befores) > 0.02)
                ratio_flaglow = np.logical_and(ratio > 0, ratio < 0.33)
                ratio_flaghigh = np.logical_and(ratio > 0, ratio < 0.1)
                ratio_flaglow = np.nansum(ratio_flaglow[3:] > 0)
                ratio_flaghigh = np.nansum(ratio_flaghigh > 0)
                ratio_flagveryhigh = np.nanmax(np.array(befores) - np.array(afters)) > 0.15
                absolute_flag = np.nanmax(np.array(befores) - np.array(afters)) > 0.05
                #ratio_flaghigh = np.logical_or(ratio_flaghigh, (befores[-1] / total_before) > 0.8)
                if ratio_flagveryhigh:
                    gainpx, Zlabeled, additional_gain = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                            cfs_trees, cfs_trees10, notree, dem, dates, 0.2)
                    gain[~np.isin(Zlabeled, gainpx)] = 0.
                    gain = np.maximum(gain, additional_gain)
                elif ratio_flaghigh:
                    gainpx, Zlabeled, additional_gain = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                            cfs_trees, cfs_trees10, notree, dem, dates, 0.1)
                    gain[~np.isin(Zlabeled, gainpx)] = 0.
                    gain = np.maximum(gain, additional_gain)
                elif ratio_flaglow or absolute_flag:
                    gainpx, Zlabeled, additional_gain = change.filter_gain_px(gain, loss, percentiles, fs, cfs_flat, cfs_hill, cfs_steep,
                                            cfs_trees, cfs_trees10, notree, dem, dates, 0.05)
                    gain[~np.isin(Zlabeled, gainpx)] = 0.
                    gain = np.maximum(gain, additional_gain)
                afters = []
                for i in range(1, 6):
                    afters.append(np.mean(gain == i))
                afters = np.array(afters)
                print(f"After2: {np.sum(afters)}")
                print('after2', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)

                # If more than 80% of gain is removed, or 10% of the total plot
                # Then we're likely in an area that has false positive gain
                # E.g. a dry forest.
                # Then add 0.05 to the gain requirement
                # Or even better, regenerate KDE with the removed gain as stable??
                #validate_patch_gain(fs, gain2, loss2)
                #change.make_loss_plot(percentiles, loss2, gain2, dates, befores, afters, f"{str(x)}{str(y)}.png")
                
                rotational = np.logical_and(gain > 0, loss > 0)
                med = np.median(fs, axis = 0)
                med[gain > 0] = (gain[gain > 0] + 100)
                med[loss > 0] = (loss[loss > 0] + 200)
                med[np.logical_and(rotational, gain > loss)] = 150.
                med[np.logical_and(rotational, loss > gain)] = 160.
                fs[(np.median(fs, axis = 0) > 100)[np.newaxis].repeat(fs.shape[0], axis = 0)] = 255.
                for i in range(2017, 2023):
                    unstable_loss = remove_unstable_loss(i, med, fs)
                    unstable_loss[gain > 0] = 0.
                    med[unstable_loss] = np.median(fs, axis = 0)[unstable_loss]
                lte2_data = binary_dilation(n_valid_years <= 2, iterations = 50)
                #np.save("lte2data.npy", lte2_data)
                #np.save("med.npy", np.median(fs, axis = 0))
                is_oob = np.logical_and(med > 110, med < 150)
                med[is_oob] = np.median(fs, axis = 0)[is_oob]
                med[lte2_data] = np.median(fs, axis = 0)[lte2_data]

                #for i in range(0, 5):
                #    l = remove_noise(med == (i + 200), 10)
                #    med[med == (i + 200)]
                change.write_tif(med, bbx, x, y, output_path, suffix = suffix)
                print('end', psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2)
                del afters, befores, med, rotational, lte2_data, is_oob, unstable_loss
                del ratio, movingavg, potential_loss, dem, fs, gain, loss
                gc.collect()
                for year in range(2017, 2023):
                    shutil.rmtree(f"{local_path}/{str(year)}/{str(x)}/{str(y)}/")
            except Exception as e:
                gc.collect()
                print(f"Ran into {str(e)} error")
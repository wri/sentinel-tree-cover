import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import rasterio as rs
import hickle as hkl
from scipy.ndimage import median_filter, maximum_filter, percentile_filter
import yaml
import boto3
import itertools
import zipfile
import os
import copy
import time
from skimage.transform import resize
from scipy.ndimage.morphology import binary_dilation, binary_erosion
from change import change
import shutil

DRIVE = 'John'

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
        thresh = 40
    if year < 2022:
        next_year = np.min(fs[year - 2016:year+2-2016], axis = 0)
        unstable_loss = (next_year > thresh) * (ttc_year < 40) * loss_year
        no_img_lossyear = binary_dilation(ttc_year == 255, iterations = 20) * loss_year
        no_img_before = binary_dilation(fs[year - 2018] == 255, iterations = 20) * loss_year
        no_img_lossyear = np.logical_or(no_img_lossyear, no_img_before)
        unstable_loss = np.logical_or(unstable_loss, no_img_lossyear)
    else:
        no_img_2022 = binary_dilation(fs[year - 2017] == 255, iterations = 20) * loss_year
        no_img_2021 = binary_dilation(fs[year - 2018] == 255, iterations = 20) * loss_year
        unstable_loss = np.logical_or(no_img_2022, no_img_2021)
    return unstable_loss


def load_ttc_tiles(x, y):
    f20_path = f'/Volumes/{DRIVE}/tof-output/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif'

    try:
        f17 = rs.open(f'/Volumes/{DRIVE}/tof-output-2017/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif').read(1).astype(np.float32)[np.newaxis]
    except:
        f17 = np.zeros((3, 3))
    try:
        f18 = rs.open(f'/Volumes/{DRIVE}/tof-output-2018/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif').read(1).astype(np.float32)[np.newaxis]
    except:
        f18 = np.zeros((3, 3))
    try:
        f19 = rs.open(f'/Volumes/{DRIVE}/tof-output-2019/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif').read(1).astype(np.float32)[np.newaxis]
    except:
        f19 = np.zeros((3, 3))

    if os.path.exists(f20_path):
        f20 = rs.open(f20_path).read(1).astype(np.float32)[np.newaxis]
    else:
        f20 = np.zeros((3, 3))
    try:
        f21 = rs.open(f'/Volumes/{DRIVE}/tof-output-2021/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif').read(1).astype(np.float32)[np.newaxis]
    except:
        f21 = np.zeros((3, 3))
    try:
        f22 = rs.open(f'/Volumes/{DRIVE}/tof-output-2022/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_FINAL.tif').read(1).astype(np.float32)[np.newaxis]
    except:
        f22 = np.zeros((3, 3))
    list_of_files = [f17, f18, f19, f20, f21, f22]
    # get the shape
    # make a numb_years_valid file
    # return numb_years_valid
    valid_shape = [x.shape[1:] for x in list_of_files if x.shape[0] != 3][0]
    n_valid_years = np.zeros(valid_shape)
    print(n_valid_years.shape, "N VALID YEARS")
    try:
        for i in range(len(list_of_files)):
            if list_of_files[i].shape[0] == 3:
                if i == 0:
                    list_of_files[i] = list_of_files[i + 1]
                if i == len(list_of_files) - 1:
                    list_of_files[i] = list_of_files[i - 1]
                else:
                    list_of_files[i] = (list_of_files[i - 1] + list_of_files[i + 1]) / 2
    except:
        print(f"Skipping {str(x)}, {str(y)}")
        #list_of_files[i] = np.zeros((3, 3))
    for i in list_of_files:
        print(i.shape)
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
        if i == (fs.shape[0] - 1):
            isnan = np.logical_or(np.isnan(fs[i]), fs[i] >= 255)
            fs[i, isnan] = fs[i - 1, isnan]
        else:
            isnan = np.logical_or(np.isnan(fs[i]), fs[i] >= 255)
            isnannext = np.logical_or(np.isnan(fs[i + 1]), fs[i + 1] >= 255)
            isnanbefore = np.logical_or(np.isnan(fs[i - 1]), fs[i - 1] >= 255)
            isnan = isnan * isnannext * isnanbefore
            fs[i, isnan] = (fs[i - 1, isnan] + fs[i + 1, isnan]) / 2
    
    stable = np.sum(fs > 30, axis = 0) == 6
    stable = binary_erosion(stable)
    notree = np.sum(fs < 50, axis = 0) == 6
    notree = binary_erosion(notree)

    fs = change.temporal_filter(fs)
    changemap = None
    return fs, changemap, stable, notree, n_valid_years

year = 2018
local_path = '../project-monitoring/tiles/'
output_path = '../notebooks/development/change/nicaragua/'


if __name__ == '__main__':
    import argparse

    with open("../config.yaml", 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        SHUB_SECRET = key['shub_secret']
        SHUB_KEY = key['shub_id']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']

    data = pd.read_csv("process_area_2022.csv")
    data = data[data['country'] == 'Nicaragua']
    try:
        data['X_tile'] = data['X_tile'].str.extract('(\d+)', expand=False)
        data['X_tile'] = pd.to_numeric(data['X_tile'])
        data['Y_tile'] = data['Y_tile'].str.extract('(\d+)', expand=False)
        data['Y_tile'] = pd.to_numeric(data['Y_tile'])
    except Exception as e:
        print(f"Ran into {str(e)} error")
        time.sleep(1)

    #data = data[230:]
    x = 453
    y = 1236
    #data = data[data['Y_tile'] == int(y)]
    #data = data[data['X_tile'] == int(x)]
    data = data.sample(frac=1).reset_index(drop=True)
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
            print(fname, " exists")
        else:
            try:
                print(f"STARTING {x}, {y}")
                #if np.logical_and(x > 2214, x < 2220):
                    #if np.logical_and(y > 915, y < 920):
                fs, changemap, stable, notree, n_valid_years = load_ttc_tiles(x, y)
                change.download_and_unzip_data(x, y, local_path, AWSKEY, AWSSECRET)
                bbx = change.tile_bbx(x, y, data)
                a17, a18, a19, a20, a21, a22, d17, d18, d19, d20, d21, d22 = change.load_all_ard(x, y, local_path)

                ard_path = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/'
                dem = f'{ard_path}/dem_{str(x)}X{str(y)}Y.hkl'
                dem = hkl.load(dem)
                dem = median_filter(dem, size = 9)
                dem = resize(dem, (a18.shape[1:]), 0)

                fs, changemap, stable, notree, n_valid_years = load_ttc_tiles(x, y)
                print(a17.shape, a18.shape, a19.shape, a20.shape, a21.shape, a22.shape, dem.shape)
                list_of_files = [a17, a18, a19, a20, a21, a22]
                list_of_dates = [d17, d18, d19, d20, d21, d22]
                years_with_data = [i for i, val in enumerate(list_of_files) if val.shape[1] != 3]
                list_of_files = [val for i, val in enumerate(list_of_files) if i in years_with_data]
                list_of_dates = [val for i, val in enumerate(list_of_dates) if i in years_with_data]
                ard = np.concatenate(list_of_files, axis = 0)
                dates = np.concatenate(list_of_dates)

                kde, kde10, kde_expected, kde2, percentiles = change.make_all_kde(ard, stable)
                gain = np.zeros((5, ard.shape[1], ard.shape[2]))
                loss = np.zeros((5, ard.shape[1], ard.shape[2]))
                ndmiloss = np.zeros((5, ard.shape[1], ard.shape[2]))
                for i in range(5):
                    print(2017 + i + 1, i)
                    gain[i] = change.identify_gain_in_year(kde, kde10, kde_expected, dates, 2017 + i + 1) * (i + 2)
                    loss[i], ndmiloss[i] = change.identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, 2017 + i + 1) 
                    loss[i] *= (i + 2)
                    ndmiloss[i] *= (i + 2)

                gain2, loss2 = change.adjust_loss_gain(gain, loss, ndmiloss, fs, kde, kde10, kde_expected, kde2, dates)
                potential_loss = np.max(loss, axis = 0)
                np.save('potential_loss.npy', potential_loss)
                rotational = np.logical_and(gain2 > 0, loss2 > 0)
                med = np.median(fs, axis = 0)
                med[gain2> 0] = (gain2[gain2 > 0] + 100)
                med[loss2 > 0] = (loss2[loss2 > 0] + 200)
                med[np.logical_and(rotational, gain2 > loss2)] = 150.
                med[np.logical_and(rotational, loss2 > gain2)] = 160.

                movingavg = np.copy(percentiles).reshape((percentiles.shape[0], percentiles.shape[1] * percentiles.shape[2]))
                movingavg = np.apply_along_axis(change.moving_average, 0, movingavg, 9)
                movingavg = np.reshape(movingavg, (percentiles.shape[0]-8,percentiles.shape[1], percentiles.shape[2]))

                # Because our minimum mapping unit is 10 px
                for i in range(movingavg.shape[0]):
                    movingavg[i] = median_filter(movingavg[i], 5)

                cfs_flat = change.calc_reference_change(movingavg, 0, 50, notree, dem)
                cfs_hill = change.calc_reference_change(movingavg, 10, 50, notree, dem)
                cfs_steep = change.calc_reference_change(movingavg, 20, 50, notree, dem)
                cfs_trees = change.calc_tree_change(movingavg, 5, stable, dem)
                cfs_trees10 = change.calc_tree_change(movingavg, 10, stable, dem)

                gainpx, Zlabeled, additional_gain = change.filter_gain_px(gain2, loss2, percentiles, cfs_flat, cfs_hill, cfs_steep,
                                        cfs_trees, cfs_trees10, notree, dem, dates)

                gain2[~np.isin(Zlabeled, gainpx)] = 0.
                gain2 = np.maximum(gain2, additional_gain)
                rotational = np.logical_and(gain2 > 0, loss2 > 0)
                med = np.median(fs, axis = 0)
                med[gain2 > 0] = (gain2[gain2 > 0] + 100)
                med[loss2 > 0] = (loss2[loss2 > 0] + 200)
                med[np.logical_and(rotational, gain2 > loss2)] = 150.
                med[np.logical_and(rotational, loss2 > gain2)] = 160.
                fs[(np.median(fs, axis = 0) > 100)[np.newaxis].repeat(fs.shape[0], axis = 0)] = 255.
                for i in range(2017, 2023):
                    unstable_loss = remove_unstable_loss(i, med, fs)
                    med[unstable_loss] = np.median(fs, axis = 0)[unstable_loss]

                print(n_valid_years.shape)
                lte2_data = binary_dilation(n_valid_years <= 2, iterations = 50)
                #np.save("lte2data.npy", lte2_data)
                #np.save("med.npy", np.median(fs, axis = 0))

                print(lte2_data.shape)
                is_oob = np.logical_and(med > 110, med < 150)
                med[is_oob] = np.median(fs, axis = 0)[is_oob]
                med[lte2_data] = np.median(fs, axis = 0)[lte2_data]

                #for i in range(0, 5):
                #    l = remove_noise(med == (i + 200), 10)
                #    med[med == (i + 200)]
                change.write_tif(med, bbx, x, y, output_path, suffix = suffix)
                for year in range(2017, 2023):
                    shutil.rmtree(f"{local_path}/{str(year)}/{str(x)}/{str(y)}/")

            except Exception as e:
                print(f"Ran into {str(e)} error")
import numpy as np
import boto3
import zipfile
import os
import rasterio as rs
import hickle as hkl
import copy
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from skimage.filters.rank import mean
from scipy.ndimage import median_filter
from scipy.stats import gaussian_kde
from scipy.special import ndtr
from scipy.ndimage.morphology import binary_dilation, binary_erosion
import math
from scipy.signal import medfilt
from scipy.ndimage.filters import minimum_filter1d, uniform_filter1d
import seaborn as sns
from matplotlib import pyplot as plt

VERBOSE = False

def download_single_file(s3_file, local_file, apikey, apisecret, bucket):
    '''Downloads a file from s3 to local_file'''
    conn = boto3.client('s3', aws_access_key_id=apikey,
                        aws_secret_access_key=apisecret) 
    #print(f"Starting download of {s3_file} to {local_file} from {bucket}")
    key = "/".join(s3_file.split("/")[3:])
    conn.download_file(bucket, key, local_file)
    

def unzip_to_directory(path, directory):
    '''takes a zip file and unzips it to directory'''
    with zipfile.ZipFile('output.zip', 'r') as zip_ref:
        names = zip_ref.namelist()
        for file in names:
            outfile = file.split("/")[-1]
            with open(directory + outfile, 'wb') as f:
                f.write(zip_ref.read(file))


def download_and_unzip_data(x, y, local_path, awskey, awssecret):
    '''wrapper of above fns to get all TTC tiles for all years for specific
    tile id'''
    year = 2020
    local_path = '../project-monitoring/tiles/'
    
    ard_path = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/'

    for year in [2017, 2018, 2019, 2020, 2021, 2022]:

        local_path = '../project-monitoring/tiles/'
        ard_path = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/'
        if not os.path.exists(ard_path):
            os.makedirs(ard_path)

        s3_file = f's3://tof-output/{str(year)}/change/{str(x)}/{str(y)}/{str(x)}X{str(y)}Y_ard.zip'
        try:
            download_single_file(s3_file, "output.zip", awskey, awssecret, 'tof-output')
            unzip_to_directory('output.zip', ard_path) 
        except:
            f"Error: {year}"
        


def make_bbox(initial_bbx: list, expansion: int = 10) -> list:
    '''recreate the bounding box of a TTC tile'''
    multiplier = 1/360 # Sentinel-2 pixel size in decimal degrees
    bbx = copy.deepcopy(initial_bbx)
    bbx[0]-= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx


def tile_bbx(x, y, data):
    '''Wrapper of make_bbox to connect it with tile database'''
    data2 = data.copy()
    data2 = data2[data2['Y_tile'] == int(y)]
    data2 = data2[data2['X_tile'] == int(x)]
    data2 = data2.reset_index(drop = True)
    initial_bbx = [data2['X'][0], data2['Y'][0], data2['X'][0], data2['Y'][0]]
    bbx = make_bbox(initial_bbx, expansion = 300/30)
    return bbx


def moving_average(a, n=3):
    '''n window moving avg on input a np.ndarray
    reutnrs shape of a.shape[0] - n + 1'''
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def load_ard_and_dates(x, y, year, local_path):
    local_path = '../project-monitoring/tiles/'
    ard_path = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/ard_ndmi.hkl'
    ard_dates = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/ard_dates.npy'
    dem = f'{local_path}/{str(year)}/{str(x)}/{str(y)}/dem_{str(x)}X{str(y)}Y.hkl'
                
    try:
        x = hkl.load(ard_path)
        y = np.load(ard_dates)
        y = ((year - 2017) * 365) + y
        dem = hkl.load(dem)
    except:
        x = np.zeros((3, 3))
        y = np.zeros((3, 3))
        dem = np.zeros((3, 3))
        print(f"{year} does not exist")
    return x, y, dem


def load_all_ard(x, y, local_path):
    '''Loads the NDMI ard and image dates for all years
    and returns them
    #! TODO clean up the return statement'''
    a17, d17, dem1 = load_ard_and_dates(x, y, 2017, local_path)
    a18, d18, dem2 = load_ard_and_dates(x, y, 2018, local_path)
    a19, d19, dem3 = load_ard_and_dates(x, y, 2019, local_path)
    a20, d20, dem4 = load_ard_and_dates(x, y, 2020, local_path)
    a21, d21, dem5 = load_ard_and_dates(x, y, 2021, local_path)
    a22, d22, dem6 = load_ard_and_dates(x, y, 2022, local_path)

    demshape = 3
    dems = [dem1, dem2, dem3, dem4, dem5, dem6]
    i = 0
    dem = dem1
    for i in range(6):
        if dems[i].shape[0] != 3:
            dem = dems[i]
    return a17, a18, a19, a20, a21, a22, d17, d18, d19, d20, d21, d22, dem


def assign_loss_year(loss, fs):
    '''Re-assign loss-year based on the most probable year
    using the TTC data instead of the NDMI data'''
    #try:
        # Nyears - 1, so a (6, X, Y) = (5, X, Y)
        # So a 0 here is 2018 loss, which is a 1
        # And a 3 here is 2021 loss, which is a 4
    max_tree_cover_loss = np.argmin(np.diff(fs, axis = 0), axis = 0) + 1
    for i in range(loss.shape[0]):
        ndmiloss = loss[i] == (i + 1)
        lossi = (loss[i] > 0) * max_tree_cover_loss 
        loss[i] = lossi
    return loss

def assign_gain_year(gain, fs):
    #try:
        # Nyears - 1, so a (6, X, Y) = (5, X, Y)
        # So a 0 here is 2018 loss, which is a 1
        # And a 3 here is 2021 loss, which is a 4
    max_tree_cover_gain = np.argmax(np.diff(fs, axis = 0), axis = 0) + 1
    for i in range(gain.shape[0]):
        gaini = (gain[i] > 0) * gain 
        lossi[lossi]
        loss[i] = lossi
    return loss
    #except:
    #    print('exception')
    #    return loss


def temporal_filter(inp):
    # Remove single-year positive anomalies
    output = np.copy(inp)
    for i in range(1, inp.shape[0] - 1):
        inpi = np.copy(inp[i])
        ismax = inp[i] == np.max(inp[i-1:i+2])
        ismax = ismax + np.isnan(inpi)
        med = np.nanmedian(inp[i-1:i+2], axis = 0)
        inpi[ismax] = med[ismax]
        output[i] = inpi
    return output


def remove_noise(arr, thresh = 15):
    '''Removes patches in arr of size smaller than thresh'''
    Zlabeled,Nlabels = ndimage.measurements.label(arr)
    label_size = [(Zlabeled == label).sum() for label in range(Nlabels + 1)]
    for label,size in enumerate(label_size):
        if size < thresh:
            arr[Zlabeled == label] = 0
    return arr


def identify_anomaly_events(inp, n, shape):
    '''Looks for >= n values out of shape moving window within inp'''
    inp_ = inp == n    
    sums = np.sum(sliding_window_view(inp_, window_shape = (shape, 1, 1)), axis = 3).squeeze()
    sums = np.concatenate([np.zeros_like(sums[0])[np.newaxis],
                           sums,
                           np.zeros_like(sums[0])[np.newaxis]], axis = 0)
    if shape == 5:
        sums = np.concatenate([np.zeros_like(sums[0])[np.newaxis],
                           sums,
                           np.zeros_like(sums[0])[np.newaxis]], axis = 0)

    if shape == 4:
        sums = np.concatenate([np.zeros_like(sums[0])[np.newaxis],
                           sums], axis = 0)
    sums = sums.astype(np.int16)
    return sums


def remove_nonoverlapping_events(candidate, anomaly, thresh = 2):
    '''Removes patches in candidate where the overlap between
    candidate and anomaly is less than 1/thresh'''

    #direct_overlap = candidate * anomaly
    candidate_labels, n = ndimage.measurements.label(candidate)
    for i in range(n):
        candidate_i = candidate_labels == i
        if np.sum(anomaly[candidate_i]) < (np.sum(candidate_i) / thresh):
            if np.sum(anomaly[candidate_i] < 100):
                candidate[candidate_i] = 0.
   # candidate = np.maximum(candidate, direct_overlap)
    return candidate


def prop_overlapping_events(before, current, thresh):
    '''#! TODO: documentation'''
    candidate_labels, n = ndimage.measurements.label(before)
    for i in range(1, n):
        before_i = before == i
        if np.sum(current[before_i]) > (np.sum(before_i > 0) / thresh):
            current[before_i] = 1.
    return current


def identify_outliers(inp):
    '''#! TODO: Documentation'''
    inp_ = inp == 0
    m = np.diff(np.where(np.concatenate(([inp_[0]],
                                         inp_[:-1] != inp_[1:],
                                         [True])))[0])[::2]
    return np.max(m) if m.shape[0] > 0 else 0

### KDE ###

def make_and_analyze_kde_for_one_img(ard, step, ref, multiplier):
    '''Makes the 2.5, 5, 10, and 25% KDE for a single image based on
    ref stable pixels'''
    kde = gaussian_kde(ref[:, step])
    reg_grid = np.arange(-10000, 10000, 20)
    cdf = tuple(ndtr(np.ravel(item - kde.dataset) / kde.factor).mean()
            for item in reg_grid)
    cdf_2_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - (0.025 * multiplier)))]
    cdf_5_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - (0.05 * multiplier)))]
    cdf_10_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - (0.1  / multiplier)))]
    cdf_25_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - 0.25))]
    #cdf_50_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - 0.50))]
    f = ard[step] >= cdf_5_percentile
    m = ard[step] >= cdf_10_percentile
    b = ard[step] >= cdf_25_percentile
    h = ard[step] >= cdf_2_percentile
    
    percentiles = np.zeros_like(ard[step], dtype = np.float32)
    for i in range(0, 100, 5):
        fraction = i / 100
        cdf_percentile = np.array(reg_grid)[np.argmin(abs(np.array(cdf) - fraction))]
        is_greater = ard[step] >= cdf_percentile
        percentiles[is_greater] = fraction
        
    return f, m, b, h, percentiles

def make_all_kde(ard, stable, maxpx = 36000, multiplier = 1):
    '''For all images in ard stack, make 2.5, 5, 10, 25% KDE
    based on stable px'''
    d = ard[:, stable]
    d = d.swapaxes(0, 1)
    # Sample up to 10% of the image (600 * 600) of the stable pixels
    dsamp = np.random.randint(0, d.shape[0], np.minimum(maxpx, d.shape[0]))
    d = d[dsamp]
    percentiles = np.zeros_like(ard, dtype = np.float32)
    kde = np.zeros_like(ard)
    kde10 = np.zeros_like(ard)
    kde2 = np.zeros_like(ard)
    kde_expected = np.zeros_like(ard)
    to_delete = []
    for i in range(ard.shape[0]):
        try:
            kde[i], kde10[i], kde_expected[i], kde2[i], percentiles[i] = make_and_analyze_kde_for_one_img(ard, i, d, multiplier)
        except:
            kde[i], kde10[i], kde_expected[i], kde2[i], percentiles[i] = 0., 0., 0., 0., 0.
            to_delete.append(i)
    if len(to_delete) > 0:
        kde = np.delete(kde, to_delete, 0)
        kde10 = np.delete(kde10, to_delete, 0)
        kde_expected = np.delete(kde_expected, to_delete, 0)
        kde2 = np.delete(kde2, to_delete, 0)
        percentiles = np.delete(percentiles, to_delete, 0)
    return kde, kde10, kde_expected, kde2, percentiles

def check_for_step_change(fs):
    # Gain cannot be described by a step-change in NDMI 
    # Between Y-1 and Y0. Loss can, since it is dramatic,
    # But gain is a slow process. If there is a big shift between
    # Years, this can be explained by ARD differences due to clouds
    # And image availability
    return None


### CANDIDATES ###

def identify_gain_in_year(kde, kde10, kde_expected, dates, year):
    '''Identifies candidate gain in year'''
    if year > 2018:
        negative_anomaly_after = identify_anomaly_events(kde, 0, 2) == 2
        negative_anomaly_prior = identify_anomaly_events(kde, 0, 3) >= 2
        positive_anomaly = identify_anomaly_events(kde10, 1, 5) >= 4
    if year == 2018:
        # For first year of gain, remove lower confidence events by:
           # - Requiring negative anomaly to be < 5%
        negative_anomaly_prior = identify_anomaly_events(kde, 0, 3) == 3
        negative_anomaly_after = identify_anomaly_events(kde, 0, 2) == 2
        positive_anomaly = identify_anomaly_events(kde10, 1, 5) == 5
    #positive_anomaly = identify_anomaly_events(kde10, 1, 3) == 3
    
    img_prior3_start = np.sum(dates <= ((year - 2017 - 3) * 365 ))
    img_prior2_start = np.sum(dates <= ((year - 2017 - 2) * 365 ))
    img_prior_start = np.sum(dates <= ((year - 2017 - 1) * 365 ))
    img_current_start = np.sum(dates <= ((year - 2017) * 365 ))
    img_next_start = np.sum(dates <= ((year - 2017 + 1) * 365 ))
    img_next_end = np.sum(dates <= ((year - 2017 + 2) * 365 ))
    img_next2_end = np.sum(dates <= ((year - 2017 + 3) * 365 ))

    if year == 2017:
        img_current_start = img_next_start

    # Loss event in Y-2, Y-1 or in Y0
    negative_prior = np.sum(negative_anomaly_prior[img_prior3_start:img_next_start], axis = 0) > 0
    
    # Tree event in Y + 1
    positive_after = np.sum(positive_anomaly[img_current_start:img_next2_end], axis = 0) > 0
    # No loss event in Y + 1
    #if year < 2022:
    negative_after = np.sum(negative_anomaly_after[img_next_start:img_next_end], axis = 0) > 0
    #else:
    #    negative_after = np.sum(negative_anomaly_after[img_current_start:img_next_end], axis = 0) > 0
    
    candidate_gain = (negative_prior) * (1 - negative_after) * (positive_after)
    struct = ndimage.generate_binary_structure(2, 1)
    candidate_gain = binary_dilation(1 - (binary_dilation(1 - candidate_gain)))
    return candidate_gain

def identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, year):
    '''Identifies candidate loss in year'''
    # A loss is defined as: 
    # - A medium confidence tree sometime in the prior year (90% confident)
    # - No positive anomaly within 6 images after the negative anomaly
    # - Negative anomaly in Y0 or Y-1, or Y+1 
    # Note that negative anomaly may show up after TTC identifies the loss, as
    # the anomaly needs consistency between many images

    # For an example loss event in 2021, the following checks happen:
    # - A positive anomaly (3 of 5) in Jan 2020 through Dec 2021
    # - A negative anomaly (as per below) in Jan 2020 thru July 2022

    # 2.5% KDE, 2 of 3 images (had been 3 of 5)
    # 5% KDE, 3 of 4 images (had been 4 of 5)
    # 10% KDE, 4 of 5 images (had been 5 of 5)
    # 2022, KDE or 5%, 2 of last 3 images 
    
    positive_anomaly = identify_anomaly_events(kde_expected, 1, 4) >= 3
    positive_anomaly_5 = identify_anomaly_events(kde_expected, 1, 5) == 5
    #positive_anomaly_cf = identify_anomaly_events(kde_expected, 1, 5) >= 2
    #positive_anomaly_cf2 = identify_anomaly_events(kde_expected, 1, 6) >= 2
    #positive_anomaly17 = identify_anomaly_events(kde_expected, 1, 5) >= 3
    negative_anomaly_10 = identify_anomaly_events(kde10, 0, 5) >= 4
    negative_anomaly_5 = identify_anomaly_events(kde, 0, 5) >= 3
    #negative_anomaly_5 *= np.logical_not(positive_anomaly_cf)
    #negative_anomaly_10 *= np.logical_not(positive_anomaly_cf2)
    negative_anomaly_2022 = identify_anomaly_events(kde, 0, 2) >= 2
    negative_anomaly_2 = identify_anomaly_events(kde2, 0, 5) >= 3
    negative_anomaly_2 = np.logical_or(negative_anomaly_2,
        identify_anomaly_events(kde, 0, 5) >= 5)
    #negative_anomaly_2 = identify_anomaly_events(kde, 0, 3) >= 3
    
    img_2prior_start = np.sum(dates <= ((year - 2017 - 2) * 365 ))
    img_1p5prior_start = np.sum(dates <= ((year - 2017 - 1.5) * 365 ))
    img_prior_start = np.sum(dates <= ((year - 2017 - 1) * 365 ))
    img_prior_mid = np.sum(dates <= ((year - 2017 - 0.5) * 365 ))
    img_current_start = np.sum(dates <= ((year - 2017) * 365 ))
    img_next_start = np.sum(dates <= ((year - 2017 + 1) * 365 ))
    img_next_mid = np.sum(dates <= ((year - 2017 + 1.5) * 365 ))
    img_next_end = np.sum(dates <= ((year - 2017 + 2) * 365 ))
    img_next2_end = np.sum(dates <= ((year - 2017 + 3) * 365 ))

    if year == 2018:
        positive_anomaly = positive_anomaly
    if year == 2022:
        negative_anomaly = np.sum(negative_anomaly_2022[-2:], axis = 0) > 0

    # Updating the loss mechanism, we would ideally want:
        # - the positive anomaly to occur before the negative anomaly
        # - the regrowth after the loss to not be quick

    # Medium confidence tree sometime in last 2 years (10%)
    positive_prior = np.sum(positive_anomaly[img_prior_start:img_next_start], axis = 0) > 0
    positive_prior_high =  np.sum(positive_anomaly_5[img_prior_start:img_next_start], axis = 0) > 0
    # Negative anomaly In this year, year before, or year after
    negative_after_5 = np.sum(negative_anomaly_5[img_prior_start:img_next_mid], axis = 0) > 0
    negative_after_10 = np.sum(negative_anomaly_10[img_prior_start:img_next_mid], axis = 0) > 0
    negative_after_2 = np.sum(negative_anomaly_2[img_prior_mid:img_next_mid], axis = 0) > 0
    # Remove loss events followed by a positive anomaly

    #### TODO IF Y+1 is >50 but no gain, then needs to be negative_after_2
    candidate_loss = positive_prior * np.logical_or(negative_after_5, negative_after_10) #* positive_prior #(1 - positive_after)
    #if year == 2022:
         #negative_anomaly = negative_anomaly * np.sum(positive_prior[-10:], axis = 0) > 0
         #candidate_loss = np.logical_or(negative_anomaly, candidate_loss)
    #struct = ndimage.generate_binary_structure(2, 1)
    candidate_loss = median_filter(candidate_loss, 3)
    candidate_loss_ndmi = positive_prior_high * negative_after_2
    candidate_loss_ndmi = median_filter(candidate_loss_ndmi, 3)
    #candidate_loss = binary_dilation(1 - (binary_dilation(1 - candidate_loss)))
    return candidate_loss, candidate_loss_ndmi

### COMBINING TTC CHANGE W CANDIDATEs ###
def adjust_gain_with_ndmi(idx, ff, gain2):
    '''Combines the TTC candidate gain with the candidate
    NDMI gain'''

    # For example, 2020 to 2021 gain
    # If 2021 - (2019 or 2020) is greater than 50
    # And if 2021 and 2022 is greater than 30
    # Then we have a candidate event
   # prior = ff[idx - 1]
    #if idx > 1:
        # e.g. 80 -> 60
        #mean_prior_two = np.mean(ff[idx - 2:idx], axis = 0)
        #min_prior_two = np.min(ff[idx - 2:idx], axis = 0) 

        #prior[ff[idx - 2] >= prior] = min_prior_two[ff[idx - 2] >= prior]
        #prior[ff[idx - 2] < prior] = mean_prior_two[ff[idx - 2] < prior]
    prior = np.clip(idx - 1, 0, idx - 2)
    if prior != 0:
        candidate = (((ff[idx] - np.min(ff[prior:idx], axis = 0)) > 50) *
                        (ff[idx] > 50) * (ff[idx + 1] > 40)) * ((ff[idx + 1] - ff[idx]) > -50)
    else:
        candidate = (((ff[idx] - np.mean(ff[prior:idx], axis = 0)) > 50) *
                        (ff[idx] > 50) * (ff[idx + 1] > 40)) * ((ff[idx + 1] - ff[idx]) > -50)
    candidate = candidate * (ff[idx + 1] <= 100) * (ff[idx] <= 100)
    out = remove_nonoverlapping_events(candidate, np.max(gain2[idx-1:idx], axis = 0), 4) * idx
    #out = out * (np.min(ff, axis = 0) < 40)
    return out

def adjust_loss_with_ndmi(idx, ff, loss2, ndmiloss, adjustment):
    # For example, 2020 to 2021 loss
    # If (2019 or 2020) - 2021 is greater than 50
    # And if 2021 is less than 40, and either 2019 or 2020 is < 40
    # Then we have a candidate event
    print(f"{idx}: {adjustment}")
    base_change = 50
    base_change = base_change - adjustment
    base_change = np.clip(base_change, 40, 80)
    print(f"Adjusted base_change to {base_change}")
    # Expand the possible loss events for very small holes
    loss_year = (ff[idx + 1] < 40).astype(np.float32)
    is_small = np.ones_like(loss_year, dtype = np.float32)
    zlabels ,Nlabels = ndimage.measurements.label(is_small)
    for i in range(Nlabels):
        npx = np.sum(zlabels == i)
        if np.logical_and(npx < 10, npx >= 2):
            is_small -= binary_dilation(zlabels == i, iterations = 1)
    is_small = np.clip(is_small, 0, 1)

    candidate = (np.mean(ff[idx - 1: idx + 1], axis = 0) - (ff[idx + 1]) * is_small) > base_change#((np.max(ff[idx-1: idx + 1], axis = 0) - ff[idx + 1]) > 50)
    candidate = candidate * ((ff[idx + 1] * is_small) <= 40) 
    prior = (np.mean(ff[idx - 1: idx + 1], axis = 0))
    candidate = candidate * (prior >= 60)
    if idx <= 3:
        # Where Y1 to Y2 is loss, but Y3 has trees
        # There needs to be a higher overlap with the NDMI loss
        unstable_candidate = candidate * np.logical_or(ff[idx + 2] >= 50, prior <= 60)
        fp_loss = remove_nonoverlapping_events(unstable_candidate, loss2[idx], 2)
        fp_loss = median_filter(fp_loss, 3)
        #fp_loss = remove_noise(fp_loss, thresh = 10)
        candidate[unstable_candidate] = fp_loss[unstable_candidate]
    out = remove_nonoverlapping_events(candidate, loss2[idx], 4)
    ndmiloss[idx] = (ndmiloss[idx] * np.logical_or(ff[idx] > 80, ff[idx - 1] > 80))
    ndmiloss[idx] = ndmiloss[idx] * (np.min(ff[idx:], axis = 0) < 70)
    ndmiloss[idx] = remove_noise(ndmiloss[idx], thresh = 8)
    out = np.logical_or(out, ndmiloss[idx]) * (idx + 1)
    return out

def remove_unstable_gain(loss, gain, fs):
    # If the gain happens in year Y, where the window is T, NT, T
    # And a loss event is not detected, then remove the gain
    for i in range(loss.shape[0]):
        ttcyear = fs[i + 1]
        gainyear = gain[i]
        priormax = np.zeros_like(fs[0]) if i == 0 else np.logical_and(fs[i-1] > 70, fs[i] < 35)
        gain_prior_trees = (gainyear > 0) * priormax
        nopriorloss = np.zeros_like(fs[0]) if i == 0 else np.sum(loss[:i + 1] > 0, axis = 0, keepdims = True) == 0
        unstable_gain = gain_prior_trees * nopriorloss
        gain[i] *= (1 - unstable_gain.squeeze())
    return gain 


def adjust_loss_gain(gain, loss, ndmiloss, fs, dates, adjustments):

    fs = fs.astype(np.float32)
    ff = temporal_filter(fs)

    loss22 = loss[-1]
    #np.save('loss22.npy', loss22)
    ndmi22 = ndmiloss[-1]
    #loss22, ndmi22 = identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, 2022)
    base_change = 50
    print(f"The adjustment is: {adjustments[-1]}")
    base_change -= adjustments[-1]
    base_change = np.clip(base_change, 40, 80)
    print(f"The base change is: {base_change}")
    candidateloss2022 = ((np.mean(ff[3:5], axis = 0) - ff[5]) >= base_change) * np.logical_or(ff[4] > base_change, ff[3] > base_change) * (ff[5] < 30)
    candndmiloss2022 = (np.min(ff[3:5], axis = 0) > 60) * ((np.min(ff[3:5], axis = 0) - ff[5]) >= 20)
    ndmi22 = remove_nonoverlapping_events(candndmiloss2022, ndmi22, 10)
    loss22 = remove_nonoverlapping_events(candidateloss2022, loss22, 4)
    loss22 = np.logical_or(loss22, ndmi22)
    loss22 = remove_noise(loss22, thresh = 6)
    

    gain18 = ((ff[1] - ff[0]).squeeze() >= 50) * (ff[0] < 30) * (ff[2] > 50)
    gain18 = remove_nonoverlapping_events(gain18, gain[0], 2)
    gain18 = remove_noise(gain18, thresh = 10).squeeze() * 1
    gain18 = np.clip(gain18, 0, 1)

    #loss18, _ = identify_loss_in_year(kde, kde10, kde_expected, kde2, dates, 2018)
    loss18 = loss[0]
    candidateloss2018 = ((ff[0] - ff[1]).squeeze() >= 50) * (ff[0] > 60) * (ff[1] < 40)
    loss18 = remove_nonoverlapping_events(candidateloss2018, loss18, 3)
    loss18 = loss18.squeeze() * 1

    #gain2 = np.copy(gain)
    gain[0] = 0.

    gain[1] = adjust_gain_with_ndmi(2, ff, gain)
    gain[2] = adjust_gain_with_ndmi(3, ff, gain)
    gain[3] = adjust_gain_with_ndmi(4, ff, gain)

    candidate2022 = ((ff[5] - np.min(ff[3:5], axis = 0) >= 50) * (ff[5] > 50))
    candidate2022 = candidate2022 * np.logical_or(ff[4] < 30, ff[3] < 30)
    gain[4] = remove_nonoverlapping_events(candidate2022, np.max(gain[4:5], axis = 0), 4) * 5

    #loss2 = np.copy(loss)
    loss[0] = 0.
    #loss[0] = adjust_loss_with_ndmi(1, ff, loss, ndmiloss)
    loss[1] = adjust_loss_with_ndmi(1, ff, loss, ndmiloss, adjustments[2])
    loss[2] = adjust_loss_with_ndmi(2, ff, loss, ndmiloss, adjustments[3])
    loss[3] = adjust_loss_with_ndmi(3, ff, loss, ndmiloss, adjustments[4])
    loss[4] = loss22 * 5
    #loss[4] = adjust_loss_with_ndmi(4, ff, loss, ndmiloss)
    #loss[4][loss[4] > 0] = 5.
    #loss2 = assign_loss_year(loss2, ff)
    gain = remove_unstable_gain(loss, gain, fs)
    gain[gain == 0] = 255
    gain = np.min(gain, axis = 0)
    gain[gain == 255] = 0.
    gain[gain18 > 0] = gain18[gain18 > 0] * 1
    # Remove_nonoverlaping needs a size and fraction thresh, not just fraction
    #! TODO!!
    #candidateloss2022 = ((np.max(ff[3:5], axis = 0) - ff[5]) >= 50) * np.logical_or(ff[4] > 50, ff[3] > 50) * (ff[5] < 25)
    #loss2[4] = remove_nonoverlapping_events(candidateloss2022, np.max(loss2[4:5], axis = 0), 3) * 5


    loss[loss == 0] = 255
    loss = np.min(loss, axis = 0)
    loss[loss == 255] = 0.
    ### THE 2018 ERROR IS HERE !!!!!
    
    delta1718 = fs[0] - fs[1]
    delta1918 = fs[1] - fs[2]
    print("2018 loss", np.mean(loss18))
    is18loss = delta1718 > delta1918
    is18loss = (delta1718 > 50) * is18loss

    is19loss = delta1918 > delta1718
    is19loss = (delta1918 > 50) * is19loss

    loss[(loss <= 2) * (loss18 > 0) * is18loss] = 1.
    loss[(loss <= 2) * (loss18 > 0) * is19loss] = 2.#loss18[np.logical_and(loss2 == 0, loss18 > 0)] * 1

    ### RULE BASED CLEAN UP ###
    ### The following rules are applied:
    # If there is early loss, but later trees, with no gain, then remove the loss
    # If there is early gain, but later no trees, with no reloss, remove the gain
    # If there is late loss, but early no trees, with no gain, remove the loss
    # If there is late gain, but early trees, with no loss, remove the gain
    # If there is 4 gain or loss events in 6 years, remove the event 
    #     - (e.g. rotation has to be >= 3 year cycle)

    # For loss year 2018-2020, if 2021-2022 have trees,
    # but there is no regain, then remove the loss event
    #loss_noregain = (np.sum(fs[-3:] >= 40, axis = 0) >= 2) *\
    #                   (gain2 <= 3) * (loss2 <= 4) * (loss2 > 0)
    #print("LOSS NO REGAIN", np.mean(loss_noregain))
    # For gain year 2018-2020, if 2021-2022 have no trees,
    # but there is no reloss, then remove the gain event
    #gain_no_reloss = ((np.median(fs[-3:], axis = 0) <= 20) *
    #                   (gain2 <= 4) * (loss2 <= 3)) * (gain2 > 0)

    # For gain year 2021-2022, if 2017 or 2018 have trees, but there is no loss
    # Then remove the gain
    gain_but_previous_trees = (gain >= 4) * (loss == 0) * (np.max(fs[:2], axis = 0) > 70)
    gain_but_previous_trees2 = (gain >= 5) * (loss == 0) * (np.max(fs[:3], axis = 0) > 70)
    gain_but_previous_trees = np.logical_or(gain_but_previous_trees, gain_but_previous_trees2)
    # For loss year 2021-2022, if 2017 or 2018 have no trees, but there is no gain
    # Then remove the loss
    loss_but_previous_notrees = (loss >= 4) * (gain == 0) * (np.min(fs[:2], axis =0) < 30)
    #loss_but_previous_notrees2 = (loss2 >= 5) * (gain2 == 0) * (np.min(fs[:3], axis =0) < 30)
    #loss_but_previous_notrees = np.logical_or(loss_but_previous_notrees, loss_but_previous_notrees2)
    unstable_preds = np.sum(abs(np.diff(fs, axis = 0)) > 40, axis = 0) > 3

    # Either have a loss event, or have the min tree cover be < 40 to detect gain
    loss_or_no_trees = np.logical_or(np.logical_and(loss > 0, loss < 255), np.min(fs, axis = 0) < 30)
    gain = gain * (1 - unstable_preds) #* (1 - gain_but_previous_trees) #loss_or_no_trees * (1 - gain_but_previous_trees) 
    loss = loss * (1 - unstable_preds)#(1 - loss_but_previous_notrees) * (1 - loss_noregain) * (1 - unstable_preds)
    losses = np.copy(loss) > 0
    losses = remove_noise(losses, 5)
    losses[losses > 0] = 1.
    loss = loss * losses 

    gains = np.copy(gain) > 0
    gains = remove_noise(gains, 10)
    gains[gains > 0] = 1.
    gain = gain * gains
    del unstable_preds

    return gain, loss


### REFERENCE CHANGE THRESHOLDS ###
def round_up(x, a):
    return math.ceil(x / a) * a

def round_down(x, a):
    return math.floor(x / a) * a



def calc_reference_change(movingavg, slopemin, slopemax, notree, dem):
    counterfactuals = []
    prior = 0.2
    lowest_change = 0.15
    previous_change = 0.15
    for i in range(0, 60, 5):
        baseline = i / 100
        counterfactual = np.mean(movingavg[:6], axis = 0)
        counterfactual = np.logical_and(notree, np.logical_and(counterfactual >= baseline, counterfactual < baseline + 0.05))
        if np.mean(dem >= slopemin) > 0.05:
            counterfactual = np.logical_and(counterfactual, (dem >= slopemin))#, dem < slopemax))
            counterfactual = np.logical_and(counterfactual, (dem <= slopemax))#, dem < slopemax))
        npx = np.sum(counterfactual)
        if np.sum(counterfactual) > 500:  

            if ((i / 100) < 0.10):# or npx > 2000:
                #print(movingavg.shape)
                #upper_percentile = np.percentile(movingavg[6:-6, counterfactual], 90, axis = 0)
                #counterfactual = np.percentile(upper_percentile, 90)
                counterfactual = np.percentile(movingavg[6:, counterfactual], 95) # 97.5
                #print(i, counterfactual, prior)
            elif slopemin >= 20:
                 counterfactual = np.percentile(movingavg[6:, counterfactual], 95) # 97.5
            else:
                #upper_percentile = np.percentile(movingavg[6:-6, counterfactual], 80, axis = 0)
                #counterfactual = np.percentile(upper_percentile, 90)
                counterfactual = np.percentile(movingavg[6:, counterfactual], 95)

            prior = counterfactual
            change = (counterfactual - baseline)

            #change = np.maximum(change, baseline - 0.20)
        else:
            
            #counterfactual = previous_change + 0.025
            #counterfactual = np.maximum(counterfactual, baseline - 0.05)
            change = previous_change + 0.01#(counterfactual - baseline)
            counterfactual = baseline + change
            #change = np.maximum(change, baseline - 0.10)
            #print(baseline - 0.05)
            #print(i / 100, counterfactual)
        #if i != 0:
        #    counterfactual = np.maximum(counterfactual, prior + 0.05)
        #change = (counterfactual - baseline)
        change = np.clip(change, 0.15, 0.4)
        lowest_change += 0.01
        lowest_change = np.maximum(lowest_change, change)
        change = np.maximum(lowest_change, change)
        if change > (previous_change + 0.01):
            change = (previous_change + 0.01)
        counterfactual = baseline + change
        
        reference = counterfactual #- 0.05
        #print(f"{slopemin}: prev: {previous_change}, Target change from {baseline} is {np.around(change, 2)} to {np.around(reference, 2)}, {npx}")
        counterfactuals.append(counterfactual)
        #if (counterfactual - baseline) > previous_change:
        previous_change = change
        #else:
        #    previous_change += 0.10
    return counterfactuals

def calc_tree_change(movingavg, pct, stable, dem):
    counterfactuals = []
    for i in range(20, 80, 5):
        baseline = i / 100
        counterfactual = np.percentile(movingavg, 80, axis = 0)
        #counterfactual = np.logical_and(stable,
        #                               np.logical_andd())
        counterfactual = np.logical_and(stable, np.logical_and(counterfactual >= baseline, counterfactual < baseline + 0.05))
        counterfactual = np.percentile(np.percentile(
            movingavg[:, counterfactual], 20, axis = 0), pct)
        #counterfactual = np.percentile(movingavg[10:-10, counterfactual], pct)
        #counterfactual = baseline - 0.506
        #counterfactual = np.clip(counterfactual, 0.025, 1.)
        #print(i, i / 100, counterfactual)
        counterfactuals.append(counterfactual)
    return counterfactuals

def calc_threshold_for_notree(maxval, cfs_trees):
    maxval = round_down(maxval, 0.05)
    maxval = np.clip(maxval, 0.2, 0.75)
    thresh = cfs_trees[int(maxval // 0.05) - 3]
    return thresh

def calc_tree_change(movingavg, pct, stable, dem):
    counterfactuals = []
    for i in range(20, 80, 5):
        baseline = i / 100
        if movingavg.shape[0] > 30:
            edges = 6
        elif movingavg.shape[0] > 20:
            edges = 4
        else:
            edges = 2
        high = np.percentile(movingavg[edges:-edges], 90, axis = 0)
        highlocs = np.logical_and(stable, np.logical_and(high >= baseline, high < baseline + 0.05))
        high = high[highlocs]
        low = np.percentile(
            movingavg[edges:-edges, highlocs], 10, axis = 0)
        refrange = high - low
        change = np.mean(refrange) + (2*np.std(refrange))
        try:
            change2 = np.percentile(refrange, 90)
        except:
            change2 = 1.
        change = np.minimum(change, change2)
        #print(i / 100, change, baseline - change, np.sum(highlocs > 0))
        counterfactuals.append(baseline - change)
    return counterfactuals

### FINAL GAIN FUNCTIONS ###

def min_filter1d(a, W):
    hW = (W-1)//2 # Half window size
    return minimum_filter1d(a, W)#[hW:-hW] ### MEDIAN

def check_for_gain_subtle(ma):
    gain_events = []
    threshes = [0.025, 0.05]
    for thresh in threshes:
        ma_below5 = np.argwhere(ma < thresh).flatten()
        for i in ma_below5:
            if i < (ma.shape[0] - 5) and (i >= 3):
            # check for two in a row at 0.025, and 3 in a row at 0.05
            # This indicates there was no tree before
                numb = 3 if thresh == 0.05 else 2
                if np.sum(ma[i:i + numb] <= thresh) == numb:
                    # Check for no loss in future
                    # Check for tree in future
                    #print(i, "CANDIDATE")
                    if np.sum(ma[i+2:i+22] < thresh) == 0:
                        previous_tree = (np.sum(ma[:i] > 0.10) >= 2)
                        future_tree = (np.sum(ma[i:] > 0.10) >= 10)
                        if previous_tree == False and future_tree == True:
                            gain_events.append(i)
    return gain_events


def check_for_gain_large(ma, deforested, 
    reference, counterfactual,
    cfs_trees, cfs_trees10, modifier = 0, verbose = True):
    
    minimum5win = min_filter1d(ma, 3)
    gain_events = []
    is_delta_after = 0.

    if deforested:
        deforested_date = np.maximum(np.argmin(ma), 3)
        upper_limit = np.max(ma[deforested_date:deforested_date + 6])
        upper_limit = np.maximum(upper_limit, 0.3)

    # If not deforested, we assume that the gain is non tree to tree
    # Rather than tree - no tree - tree, as the latter would
    # Indicate deforestation. 
    if not deforested:
        # We start by lookign at the first 3 (so 7) images
        # And, assuming that the pixel is not a tree
        # Calculate the 95th% confidence range for remaining non-tree
        baseline = round_down(np.mean(ma[:3]), 0.05)
        baseline = np.clip(baseline, 0.0, 0.40)
        target = counterfactual[int(baseline // 0.05)]
        #print(f"Target change from {baseline} is {target}")
        change = (target - baseline)
        reference = target

        # However, if it is a missed deforestation event, then we can "re-set"
        # the baseline
    else:
        # If it is deforested, then it just has to re-reach the 90th percentile
        # Of non-tree values.
        change = reference - 0.05
        change = np.clip(change, 0.15, 0.35)
    for i in range(ma.shape[0]):
        if i < (ma.shape[0] - 6) and (i >= 6):
            # Look for 6 dates in a row below reference = stable no-tree
            # And 5 dates in a row above reference = stable tree
            
            if deforested or (i < 6):
                n_lookback = 3 if (i - 3) > 0 else i
            else:
                n_lookback = i
                
            # If not deforested, look for the whole history to this point.
            # If deforested, just look at the last 3 images. 
            current_baseline = np.median(ma[i-n_lookback:i])

            baseline = current_baseline

            if (baseline <= 0.5) or deforested:
                baseline = round_down(baseline, 0.05)
                baseline = np.clip(baseline, 0.0, 0.60)
                reference = counterfactual[int(baseline // 0.05)]
                noise_factor = 0
                #if not deforested:
                #    bs = np.median(ma[i-n_lookback:i])
                #    positive_deviations = np.argwhere(ma[:i] > (bs))
                #    positive_deviations = positive_deviations[positive_deviations > (i - 10)]
                    #if len(positive_deviations) > 0:
                        #print()
                    #    noise_factor = np.mean(ma[positive_deviations] - bs)
                        #print(positive_deviations, noise_factor)
                        #noise_factor = np.std(ma[:i]) * 1.5
                    #    reference += 0.5*noise_factor

                change = (reference - baseline)
               #print(baseline, reference, change)
                # Do at least 3 image dates in the future reach the tree reference threshold? 
                endline = minimum5win[i+1:ma.shape[0]]
                lastdate_gain = False
                if np.argmax(endline - baseline) >= (endline.shape[0] - 2):
                    if baseline < 0.25:
                        endline = np.array(ma[-1])
                        reference -= ((noise_factor) / 2)
                        change = (reference - baseline)
                        lastdate_gain = True
                #if verbose:
                #    print(f"{i}, Base/end: {np.around(baseline, 3), np.max(endline)},"
                #          f" change/ref: {np.around(change, 3), reference}")
                # Identify whether change threshold is met, and reference is reached
                if (np.max(endline) - baseline) > (change + modifier) and (np.max(endline) > (reference + modifier)):
                    if lastdate_gain:
                        beforeidx = np.maximum(i, 10)
                        max_before = np.max(ma[:beforeidx])
                        max_after = np.max(ma[-6:])
                        no_cyclical_ndmi = max_after > ((max_before * 1.2) + modifier)
                        no_cyclical_ndmi = np.logical_or(no_cyclical_ndmi, deforested)
                        gain_date = ma.shape[0]
                        if no_cyclical_ndmi:
                            gain_events.append(i)
                    else:
                        #try:
                        gain_date = np.argwhere(np.logical_and(np.array(endline >= reference),
                                                           np.array((endline - baseline) > change)
                                                          )).flatten()[0] + i

                        imgs_after_gain = ma[gain_date:gain_date + 8]

                        if gain_date < (ma.shape[0] - 6):
                            try:
                                gain_date_after = np.argwhere(imgs_after_gain > np.percentile(
                                                imgs_after_gain, 75)).flatten()[0] + gain_date
                            except:
                                gain_date_after = gain_date
                        else:
                            gain_date_after = gain_date
                        gain_value = ma[gain_date]
                        gain_value_max = np.max(ma[gain_date:gain_date + 10])
                        max_value = np.max(ma[gain_date:])
                        #reference_min = calc_threshold_for_notree(gain_value, cfs_trees) + 0.10
                        reference_min_prior = calc_threshold_for_notree(max_value, cfs_trees) + 0.10
                        #reference_min_prior_gv = calc_threshold_for_notree(gain_value, cfs_trees10)
                        #reference_min_prior = np.minimum(reference_min_prior, reference_min_prior_gv)
                        #print(max_value, reference_min_prior, reference_min_prior_gv)
                        # TEST BLOCK
                        if gain_date >= 10:
                            notree_before = np.sum(ma[:gain_date] < (reference_min_prior)) > 0
                        else:
                            notree_before = np.sum(ma[:gain_date] < (reference_min_prior)) > 0
                        #notree_before = True
                        beforei = np.clip(gain_date-6, 3, ma.shape[0])
                        max_before = np.percentile(ma[:beforei], 90)
                        max_after = np.percentile(ma[gain_date:], 90)
                        no_cyclical_ndmi = (max_after - max_before) > (0.05 + modifier)
                        
                        #no_cyclical_ndmi = np.logical_or(no_cyclical_ndmi, np.argmin(ma) == 0)
                        no_cyclical_ndmi = np.logical_or(deforested, no_cyclical_ndmi)
                        #notree_before = np.logical_and(notree_before, no_cyclical_ndmi)
                        notree_before = np.logical_or(notree_before, deforested)

                        min_next_6 = (np.percentile(ma[gain_date:gain_date + 6], 25))
                        min_next_6 = np.maximum(min_next_6, (np.percentile(ma[gain_date_after:gain_date_after + 6], 25)))
                        reference_min = np.maximum(ma[gain_date] * 0.67, ma[i])
                        #reference_min = calc_threshold_for_notree(ma[gain_date], cfs_trees)
                        no_loss_after_gain = min_next_6 > (reference_min)
                        #if verbose and no_cyclical_ndmi:
                            #print(i, gain_date, gain_value, ma[i], max_before, max_after, max_after - max_before, reference_min, no_cyclical_ndmi)
                       # no_loss_after_gain_long = (np.sum(ma[gain_date_after:] < (reference_min)) == 0)

                        # What % decrease happens after the maximum is reached?
                        # And how does the min value compare with the gain value?
                        # And how does the total decrease compare to the total increase?
                        #argmax_after = np.argmax(ma[gain_date:]) + gain_date
                        #is_delta_after = (np.min(ma[argmax_after:]) / np.max(ma[gain_date:]))
                        #print(is_delta_after)
                        no_loss_val = np.maximum((reference + modifier) * 0.75, 0.15)
                        #no_loss_after_gain_long = (np.sum(ma[gain_date_after:] < (no_loss_val)) == 0)
                        #no_loss_after_gain_long = np.logical_or(is_delta_after > 0.75, )
                        #no_loss_after_gain = np.logical_and(no_loss_after_gain, no_loss_after_gain_long)
                        no_loss_after_gain = np.logical_or(deforested, no_loss_after_gain)
                        #no_loss_after_gain = True
                        if (gain_date - 5) > i:
                            no_decrease_until_gain = np.min(np.array(ma[i+5:gain_date])) >= (ma[i] - 0.10)
                        else:
                            no_decrease_until_gain = True
                        #if verbose and no_cyclical_ndmi:
                        #if no_loss_after_gain and no_decrease_until_gain:
                        #if notree_before:
                                #print(i)
                        if no_cyclical_ndmi and no_loss_after_gain:# and 
                            #print(i, gain_date, gain_value, ma[i], max_before, max_after,  max_after - max_before, no_cyclical_ndmi)
                            end = np.around(np.max(endline), 3)
                            _change = np.around(np.max(endline) - baseline, 3)
                            _thresh = np.around(change + modifier, 3)
                            #if np.logical_and(len(gain_events) > 3, len(gain_events) < 6):
                                #print(f'{i}, End: {end}, Change: {_change}, Thresh: {_thresh}')
                            gain_events.append(i)
                        #except:
                        #    print("exception")
                        #    continue
    if len(gain_events) > 0:
        if (len(gain_events) > 3) or ((np.max(gain_events) > (ma.shape[0] - 3))):
            return gain_events, np.around((reference + modifier), 2), gain_date
        elif (len(gain_events) >= 2) and deforested:
            return gain_events, np.around((reference + modifier), 2), gain_date
        else:
            return [], np.around((reference + modifier), 2), 0
    else:
        return [], np.around((reference + modifier), 2), 0


def calc_max_tc_decrease(mean_treecover):
    '''calculates a forward-looking, adjacent or non-adjacent
    maximum decrease in tree cover'''
    _max = 0
    _min = 100
    maxidx = 0
    minidx = 0
    maxdiff = 0
    for i in range(mean_treecover.shape[0]):
        if mean_treecover[i] <= _min:
            minidx = i
            _min = mean_treecover[i]
            maxdiff = _min - _max
        if mean_treecover[i] >= _max:
            if maxidx <= minidx:
                _max = mean_treecover[i]
                maxidx = i
    return maxdiff

### FILTER GAIN EVENTS BASED ON TIME SERIES ###
def filter_gain_px(gain2, loss2, percentiles, fs, cfs_flat, cfs_hill, cfs_steep, cfs_trees, 
    cfs_trees10, notree, dem, dates, n_imgs_per_year, modifier = 0):
    gain2 = remove_noise(gain2, 8)
    Zlabeled,Nlabels = ndimage.measurements.label(gain2)

    try:
        reference = np.percentile(percentiles[:, notree], 90)
    except:
        reference = 0.2
    reference = np.clip(reference, 0.20, 0.40)
    print(f"Reference: {reference}")

    struct = ndimage.generate_binary_structure(2, 1)
    loss_dilated = binary_dilation(np.copy(loss2), struct, 3)
    additional_gain = np.zeros_like(Zlabeled, dtype = np.int32)
    #deforested_gain = 0
    #nondeforested_gain = 0
    year = 0
    gainpx = []
    gaindates =[]
    #deforested_gain = np.zeros_like(Zlabeled)
    for idx in range(1, Nlabels):
        Npx = np.sum(Zlabeled == idx)
        if Npx > 0:
            means = np.mean(percentiles[:, Zlabeled == idx], axis = 1)

            deforested = np.mean(loss_dilated[Zlabeled == idx] > 0) > 0.1
            #print(dem.shape, Zlabeled.shape)
            #print((Zlabeled == idx).shape)
            mean_slope = np.mean(dem[Zlabeled == idx])
            counterfactual_events = cfs_flat if mean_slope < 10 else cfs_hill
            if mean_slope >= 20:
                counterfactual_events = cfs_steep
            verbose = True if Npx > 1000 else False

            if deforested and Npx > 125:
                # If deforestation is within the potential gain patch
                # Process the gain filtering separately for the deforested
                # And the non-deforested areas if the patch size is > 500 px
                # This removes really large degradation / drought events
                # But enables the small-scale rotations where loss might be missed
                # To be included
                deforest_area = np.logical_and(
                    Zlabeled == idx, loss2 > 0)
                nondeforest_area = np.logical_and(
                    Zlabeled == idx, loss2 == 0)
                mean_treecover = np.around(np.mean(fs[:, deforest_area], axis = 1), 1)
                deforest_area = np.mean(percentiles[:, deforest_area], axis = 1)
                nondeforest_area = np.mean(percentiles[:, nondeforest_area], axis = 1)
                deforest_area = moving_average(deforest_area, n = 5)
                nondeforest_area = moving_average(nondeforest_area, n = 5)
                if np.sum(np.isnan(deforest_area) == 0):
                    deforested_gain, gval, gdate = check_for_gain_large(ma = deforest_area,
                                                  deforested = True,
                                                  reference = reference, 
                                                  counterfactual = counterfactual_events,
                                                  cfs_trees = cfs_trees,
                                                  cfs_trees10 = cfs_trees10,
                                                  modifier = modifier,
                                                  verbose = verbose)
                else:
                    deforested_gain = []
                if np.sum(np.isnan(nondeforest_area) == 0):
                    # ADD Max decrease adj here
                    max_decrease = np.around(calc_max_tc_decrease(mean_treecover), 1)
                    #max_decrease = np.around(np.min(np.diff(mean_treecover)), 1)
                    if max_decrease < -30:
                        decrease_mod = 0.1
                    elif max_decrease < -20:
                        decrease_mod = 0.05
                    else:
                        decrease_mod = 0
                    nondeforested_gain, gval, gdate = check_for_gain_large(ma = nondeforest_area,
                                                  deforested = False,
                                                  reference = reference, 
                                                  counterfactual = counterfactual_events,
                                                  cfs_trees = cfs_trees,
                                                  cfs_trees10 = cfs_trees10,
                                                  modifier = modifier + decrease_mod,
                                                  verbose = verbose)
                    if len(nondeforested_gain) == 0:
                        yearlabeled = np.logical_and(Zlabeled == idx, gain2 == year, loss2 == 0)
                        yearlabeled = remove_noise(yearlabeled, 10)

                        yearlabeled, Nyear = ndimage.measurements.label(yearlabeled)
                        for i in range(1, Nyear + 1):
                            means = np.mean(percentiles[:, yearlabeled == i], axis = 1)
                            ma = moving_average(means, n = 7)

                            large_gainyear, gval, gdate = check_for_gain_large(ma = ma,
                                              deforested = False,
                                              reference = reference, 
                                              counterfactual = counterfactual_events,
                                              cfs_trees = cfs_trees,
                                              cfs_trees10 = cfs_trees10,
                                              modifier = modifier,
                                              verbose = False)
                            if np.sum(yearlabeled == i) > 10 and len(large_gainyear) > 0:
                                additional_gain[yearlabeled == i] = year
                else:
                    nondeforested_gain = []
                if len(nondeforested_gain) >= 1:
                    additional_gain[np.logical_and(
                    Zlabeled == idx, loss2 == 0)] = gain2[np.logical_and(
                    Zlabeled == idx, loss2 == 0)]
                if len(deforested_gain) >= 1:
                    additional_gain[np.logical_and(
                    Zlabeled == idx, loss2 > 0)] = gain2[np.logical_and(
                    Zlabeled == idx, loss2 > 0)]
            else:
                # If not deforested, process the patch as-is
                ma = moving_average(means, n = 5)
                n_before = 0
                kde_per_year = []
                for i in n_imgs_per_year:
                    kde_per_year.append(np.around(np.mean(ma[n_before:n_before+i]), 2))
                    n_before += i
                # If there is a dip in tree cover, but no deforestation
                # Make the modifier higher
                mean_treecover = np.around(np.mean(fs[:, Zlabeled == idx], axis = 1), 1)
                max_decrease = np.around(calc_max_tc_decrease(mean_treecover), 1)
                #max_decrease = np.around(np.min(np.diff(mean_treecover)), 1)
                if max_decrease < -30:
                    decrease_mod = abs(((max_decrease) + 30) / 100)
                    decrease_mod = decrease_mod + 0.1
                    decrease_mod = np.clip(decrease_mod, 0.1, 0.2)
                elif max_decrease < -20:
                    decrease_mod = abs(((max_decrease) + 20) / 200)
                    decrease_mod = decrease_mod + 0.05
                    decrease_mod = np.clip(decrease_mod, 0.05, 0.1)
                    #decrease_mod = 0.05
                else:
                    decrease_mod = 0
                large_gain, gval, gdate = check_for_gain_large(ma = ma,
                                                  deforested = deforested,
                                                  reference = reference, 
                                                  counterfactual = counterfactual_events,
                                                  cfs_trees = cfs_trees,
                                                  cfs_trees10 = cfs_trees10,
                                                  modifier = modifier + decrease_mod,
                                                  verbose = verbose)
                if mean_slope < 10:
                    gain_events = check_for_gain_subtle(ma)
                else:
                    gain_events = []

                # If no gain is detected for the entire gain patch
                # Process each year of possible gain separately
                if len(large_gain) == 0 and len(gain_events) == 0:
                    for year in np.unique(gain2[Zlabeled == idx]):
                        if np.sum( np.logical_and(Zlabeled == idx, gain2 == year)) > 50:
                            yearlabeled = np.logical_and(Zlabeled == idx, gain2 == year)
                            yearlabeled = remove_noise(yearlabeled, 8)

                            yearlabeled, Nyear = ndimage.measurements.label(yearlabeled)
                            for i in range(1, Nyear + 1):
                                means = np.mean(percentiles[:, yearlabeled == i], axis = 1)
                                patch = percentiles[:, yearlabeled == i]
                                deforested = np.mean(loss_dilated[yearlabeled == i] > 0) > 0.25
                                ma = moving_average(means, n = 5)

                                large_gainyear, gval, gdate = check_for_gain_large(ma = ma,
                                                  deforested = deforested,
                                                  reference = reference, 
                                                  counterfactual = counterfactual_events,
                                                  cfs_trees = cfs_trees,
                                                  cfs_trees10 = cfs_trees10,
                                                  modifier = modifier + decrease_mod,
                                                  verbose = False)
                                if np.sum(yearlabeled == i) > 10 and len(large_gainyear) > 0:
                                    if np.sum(yearlabeled == i) >100:
                                        mean_treecover = np.around(np.mean(fs[:, yearlabeled == i], axis = 1), 1)
                                        if year >= 2:
                                            max_previous_treecover = 0
                                            #max_previous_treecover = np.max(fs[:(year - 1), yearlabeled == i], axis = 1)
                                        else:
                                            max_previous_treecover = 0.
                                        print(f"{idx} Addtl gain: {len(large_gainyear)} events:"
                                            f" {deforested}, {year}, TC: {mean_treecover}, {max_previous_treecover}, {np.sum(yearlabeled == i)} px")
                                    additional_gain[yearlabeled == i] = year
                                elif len(large_gainyear) == 0 and np.sum(yearlabeled == i) > 100:
                                    print(f"No addtl gain: {i + 2017}, TC: {mean_treecover}, KDE: {kde_per_year}, {np.sum(yearlabeled == i)} px")
                                else:
                                    continue
                if len(large_gain) == 0 and  len(gain_events) == 0:
                    if np.sum(Zlabeled == idx) > 200:
                        mean_treecover = np.around(np.mean(fs[:, Zlabeled == idx], axis = 1), 1)
                        print(f"{idx} No gain: {deforested}, {mean_treecover}, {kde_per_year}, {np.sum(Zlabeled == idx)} px")
                        continue
                else:
                    n_gain_events = len(large_gain)
                    n_px = np.sum(Zlabeled == idx)
                    if n_px > 100:
                        if year > 2:
                            max_previous_treecover = np.max(fs[:(np.int32(year) - 1), Zlabeled == idx], axis = 0)
                            max_previous_treecover = np.around(np.mean(max_previous_treecover), 1)
                            #max_previous_treecover = 0.
                        else:
                            max_previous_treecover = 0.
                        mean_treecover = np.around(np.mean(fs[:, Zlabeled == idx], axis = 1), 1)
                        max_decrease = np.around(calc_max_tc_decrease(mean_treecover), 1)
                        #max_decrease = np.around(np.min(np.diff(mean_treecover)), 1)
                        gain_increase = np.around(mean_treecover[np.int32(year)] - max_previous_treecover, 1)
                        # If the year is 2022, if previously there was no deforestation but a big decrease
                        # And if the gain increase is < 40%, then remove the event
                        if (year == 5) and not deforested:
                            if (abs(max_decrease) > 25) and (gain_increase < 40):
                                print(f"Removing {idx}")
                                large_gain = []
                                n_gain_events = 0.
                        
                        print(f"{idx} Gain: {len(large_gain)} events, {gval}, {gdate},:"
                              f" {deforested}, {year}, {mean_treecover}, {kde_per_year}, {max_decrease}, {gain_increase}, {max_previous_treecover}, {np.sum(Zlabeled == idx)} px")
                    #if deforested and idx > 0:
                    #    deforested_gain += np.sum(Zlabeled == idx)
                    #else:
                    #    nondeforested_gain += np.sum(Zlabeled == idx)
                    if n_px < 25:
                        if n_gain_events > 5:
                            gainpx.append(idx)
                            gaindates.append(gdate)
                    elif n_gain_events > 0:
                        gainpx.append(idx)
                        gaindates.append(gdate)
    return gainpx, Zlabeled, additional_gain, gaindates


### WRITE TO DISK ###
def write_tif(arr: np.ndarray,
              point: list,
              x: int,
              y: int,
              out_folder: str,
              suffix="_FINAL") -> str:
    #! TODO: Documentation

    file = out_folder + f"{str(x)}X{str(y)}Y{suffix}.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]
    arr[np.isnan(arr)] = 255
    arr = arr.astype(np.int16)

    transform = rs.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])

    print("Writing", file)
    new_dataset = rs.open(file,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=1,
                                dtype="uint8",
                                compress='zstd',
                                predictor=2,
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    return file


def make_loss_plot(percentiles, loss2, gain2, dates, befores, afters, name):
    mean_loss18 = np.mean(percentiles[:, loss2 == 1], axis = (1))
    mean_loss19 = np.mean(percentiles[:, loss2 == 2], axis = (1))
    mean_loss20 = np.mean(percentiles[:, loss2 == 3], axis = (1))
    mean_loss21 = np.mean(percentiles[:, loss2 == 4], axis = (1))
    mean_loss22 = np.mean(percentiles[:, loss2 == 5], axis = (1))
    #mean_nochange_tree = np.mean(percentiles[:, np.logical_and(changemap <= 100, changemap > 40)], axis = (1))


    plt.figure(figsize=(15,15))
    fig, axs = plt.subplots(ncols=5, nrows = 2, figsize = (20, 15), sharey = True)
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss18, color = 'red', ax = axs[0, 0])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss18, color = 'red', label = 'Loss 2018', ax = axs[0, 0])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss19, color = 'red', ax = axs[0, 1])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss19, color = 'red', label = 'Loss 2019', ax = axs[0, 1])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss20, color = 'red', ax = axs[0, 2])
    sns.lineplot(x = (dates / 365) + 2017,y =  mean_loss20, color = 'red', label = 'Loss 2020', ax = axs[0, 2])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss21, color = 'red', ax = axs[0, 3])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss21, color = 'red', label = 'Loss 2021', ax = axs[0, 3])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss22, color = 'red', ax = axs[0, 4])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss22, color = 'red', label = 'Loss 2022', ax = axs[0, 4])

    mean_loss18 = np.mean(percentiles[:, gain2 == 1], axis = (1))
    mean_loss19 = np.mean(percentiles[:, gain2 == 2], axis = (1))
    mean_loss20 = np.mean(percentiles[:, gain2 == 3], axis = (1))
    mean_loss21 = np.mean(percentiles[:, gain2 == 4], axis = (1))
    mean_loss22 = np.mean(percentiles[:, gain2 == 5], axis = (1))
    for i in range(0, 5):
        axs[1, i].set_title(f"{np.around(befores[i] * 100, 2)}% -> {np.around(afters[i] * 100, 2)}%")
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss18, color = 'purple', ax = axs[1, 0])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss18, color = 'purple', label = 'Gain 2018', ax = axs[1, 0])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss19, color = 'purple', ax = axs[1, 1])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss19, color = 'purple', label = 'Gain 2019', ax = axs[1, 1])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss20, color = 'purple', ax = axs[1, 2])
    sns.lineplot(x = (dates / 365) + 2017,y =  mean_loss20, color = 'purple', label = 'Gain 2020', ax = axs[1, 2])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss21, color = 'purple', ax = axs[1, 3])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss21, color = 'purple', label = 'Gain 2021', ax = axs[1, 3])
    sns.scatterplot(x = (dates / 365) + 2017, y = mean_loss22, color = 'purple', ax = axs[1, 4])
    sns.lineplot(x = (dates / 365) + 2017, y = mean_loss22, color = 'purple', label = f'Gain 2022', ax = axs[1, 4])
    plt.savefig(name)
    plt.close()

### CURRENTLY NOT USED ###

"""
def check_for_gain_bootstrap(ma_upper, ma_lower, deforested, reference, counterfactual):
    
    #minimum5win = min_filter1d(ma_lower, 3)
    gain_events = []

    for i in range(ma_upper.shape[0]):
        if i < (ma_upper.shape[0] - 6) and (i >= 2):

            if deforested or (i < 6):
                n_lookback = 2 if (i - 2) > 0 else i
            else:
                n_lookback = i
            
            current_baseline = np.median(ma_upper[i-n_lookback:i])

            baseline = current_baseline
            if (baseline < 0.6) or deforested:
            
                baseline = round_down(baseline, 0.05)
                baseline = np.clip(baseline, 0.0, 0.60)
                reference = counterfactual[int(baseline // 0.05)]

                change = (reference - baseline)
                # Do at least 3 image dates in the future reach the tree reference threshold? 
                endline = ma_lower[i+1:ma_upper.shape[0]]
                lastdate_gain = False
                if np.argmax(endline - baseline) >= (endline.shape[0] - 2):
                    if baseline < 0.25:
                        endline = np.array(ma[-1])
                        #reference -= ((noise_factor) / 2)
                        change = (reference - baseline)
                        lastdate_gain = True
                #print(f"{i}, Base/end: {np.around(baseline, 3), np.max(endline)},"
                #      f" change/ref: {np.around(change, 3), reference}")
                # Identify whether change threshold is met, and reference is reached
                

                #print(f"{i}, Base/end: {np.around(baseline, 3), np.max(endline)},"
                #      f" change/ref: {np.around(change, 3), reference}")
                # Identify whether change threshold is met, and reference is reached
                if (np.max(endline) - baseline) > change and (np.max(endline) > reference):
                    #print("WTF")
                    if lastdate_gain:
                        gain_events.append(i)

                    try:
                        gain_date = np.argwhere(np.logical_and(np.array(endline >= reference),
                                                           np.array((endline - baseline) > change)
                                                          )).flatten()[0] + i

                        imgs_after_gain = ma[gain_date:gain_date + 8]

                        if gain_date < (ma.shape[0] - 4):
                            gain_date_after = np.argwhere(imgs_after_gain > np.percentile(
                                            imgs_after_gain, 75)).flatten()[0] + gain_date
                        else:
                            gain_date_after = gain_date
                        gain_value = ma[gain_date]
                        gain_value_max = np.max(ma[gain_date:gain_date + 10])
                        reference_min = calc_threshold_for_notree(gain_value, cfs_trees) + 0.10
                        min_next_6 = (np.min(ma[gain_date_after:gain_date + 8]))
                        no_loss_after_gain = min_next_6 > (reference_min)
                        reference_min = calc_threshold_for_notree(np.max(ma[gain_date:]), cfs_trees)
                        no_loss_after_gain_long = (np.sum(ma[gain_date_after:] < (reference_min + 0.05)) == 0)
                        no_loss_after_gain = np.logical_and(no_loss_after_gain, no_loss_after_gain_long)
                        no_loss_after_gain = np.logical_or(deforested, no_loss_after_gain)
                        if (gain_date - 5) > i:
                            no_decrease_until_gain = np.min(np.array(ma[i+5:gain_date])) >= ma[i]
                        else:
                            no_decrease_until_gain = True
                        #print(i, gain_value, ma[i], no_loss_after_gain, no_decrease_until_gain)
                        if no_loss_after_gain and no_decrease_until_gain:
                            gain_events.append(i)
                    except:
                        continue
                    gain_events.append(i)
    return gain_events
"""

"""
candidate2019 = (((ff[2] - np.min(ff[0:2], axis = 0)) > 50) * 
                       (ff[2] > 30)  * (ff[3] > 30)) # * (ff[1] < 30)
gain2[1] = remove_nonoverlapping_events(candidate2019, np.max(gain2[1:2], axis = 0), 3) * 2

candidate2020 = (((ff[3] - np.min(ff[1:3], axis = 0)) > 50) * 
                       (ff[3] > 30)* (ff[4] > 30)) #  * (ff[2] < 25) 

gain2[2] = remove_nonoverlapping_events(candidate2020, np.max(gain2[2:3], axis = 0), 3) * 3

candidate2021 = (((ff[4] - np.min(ff[2:4], axis = 0)) > 50) * 
                       (ff[4] > 30) * (ff[5] > 30)) # * (ff[3] < 25) 

gain2[3] = remove_nonoverlapping_events(candidate2021, np.max(gain2[3:4], axis = 0), 3) * 4
"""

import numpy as np
import sys
sys.path.append('../')
from src.downloading.utils import calculate_proximal_steps
from typing import List, Any, Tuple
from functools import reduce
from skimage.transform import resize
from tqdm import tnrange, tqdm_notebook
import math
from copy import deepcopy
import time

def hist_norm(source, template):
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    '''
    # convert the input to be 0 - 256
    '''
    if source.dtype != np.int:
        source = np.trunc(source * 256).astype(int)
        template = np.trunc(template * 256).astype(int)
    '''
        the np.unique funcitons should be fine as long as we only do it on the masked sections
        # the np.cumsum functions are fine as well
        # the np.interp is where i'm not sure! 
    '''
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

def adjust_interpolated_areas_new(array, interp):
    for time in range(array.shape[0]):
        for band in range(array.shape[-1]):
            interp_i = interp[time, :, :, band]
            array_i = array[time, :, :, band]
            if np.sum(interp_i) > 0:
                to_adjust = array_i[interp_i == 1]
                target = array_i[interp_i == 0]
                adjusted = hist_norm(to_adjust, array_i[interp_i == 0])
                adjusted = adjusted.astype(np.float32) / 256
                adjusted_idx = np.argwhere(interp_i.flatten() == 1).flatten()
                array_i = array_i.flatten()
                array_i[adjusted_idx] = adjusted
                array_i = np.reshape(array_i, (646, 646))
                array[time, :, :, band] = array_i
    return array

def adjust_interpolated_areas(array, interp):
    for time in range(array.shape[0]):
        for band in range(array.shape[-1]):
            interp_i = interp[time, :, :, band]
            array_i = array[time, :, :, band]
            if np.sum(interp_i) > 0:
                adj = (np.median(array_i[np.where(interp_i == 0)]) - 
                      (np.median(array_i[np.where(interp_i == 1)])))
                array_i[np.where(interp_i == 1)] += adj
                array[time, :, :, band] = array_i
    return array

def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray, 
                             shadows: np.ndarray,
                             image_dates: List[int], 
                             wsize: int = 35) -> np.ndarray:
    """ Interpolates clouds and shadows for each time step with 
        linear combination of proximal clean time steps for each
        region of specified window size
        
        Parameters:
         tiles (arr):
         probs (arr): 
         shadows (arr):
         image_dates (list):
         wsize (int): 
    
        Returns:
         tiles (arr): 
    """
    
    def _fspecial_gauss(size, sigma):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g

    c_arr = np.reshape(_fspecial_gauss(wsize, ((wsize/2) - 1 ) / 2), (1, wsize, wsize, 1))
    o_arr = 1 - c_arr

    c_probs = np.copy(probs) - np.min(probs, axis = 0)

    c_probs[np.where(c_probs >= 0.3)] = 1.
    c_probs[np.where(c_probs < 0.3)] = 0.
    
    shadows = shadows - np.min(shadows, axis = 0)
    c_probs += shadows
    c_probs[np.where(c_probs >= 1.)] = 1.
    n_interp = 0
    
    areas_interpolated = np.zeros_like(tiles)
    
    for x in range(0, tiles.shape[1] - (wsize - 1), 2):
        for y in range(0, tiles.shape[2] - (wsize - 1), 2):
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize)/20)
            if len(satisfactory) == 0:
                #print(f"There is a potential issue with the cloud removal at {x}, {y}")
                satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize)/2)
            for date in range(0, tiles.shape[0]):
                if np.sum(subs[date]) >= (wsize*wsize)/20:
                    n_interp += 1
                    before, after = calculate_proximal_steps(date, satisfactory)
                    before = date + before
                    after = date + after

                    before = (before - 1) if before > (tiles.shape[0] -1) else before
                    after = before if after > (tiles.shape[0] - 1) else after
                    before = after if before < 0 else before

                    if after > (tiles.shape[0] - 1):
                        print(f"There is an error, and after is {after} and before is {before}, \
                         for {tiles.shape[0]} and {date}, {satisfactory}")

                    before_array = deepcopy(tiles[before, x:x+wsize, y:y+wsize, : ])
                    after_array = deepcopy(tiles[after, x:x+wsize, y:y+wsize, : ])
                    #original_array = deepcopy(tiles[np.newaxis, date, x:x+wsize, y:y + wsize, :])
                    
                    n_days_before = abs(image_dates[date] - image_dates[before])
                    n_days_after = abs(image_dates[date] - image_dates[after])
                    before_weight = 1 - n_days_before / (n_days_before + n_days_after)
                    after_weight = 1 - before_weight
                    
                    candidate = before_weight*before_array + after_weight * after_array
                    #candidate = candidate * c_arr + tiles[date, x:x+wsize, y:y+wsize, :] * o_arr
                    tiles[date, x:x+wsize, y:y+wsize, : ] = candidate 
                    areas_interpolated[date, x:x+wsize, y:y+wsize, : ] = 1.

    tiles = adjust_interpolated_areas(tiles, areas_interpolated)

    #not_interpoted_mean = np.mean(tiles[np.where(areas_interpolated == 0)], axis = (1, 2, 3))
    #interpolated_mean = np.mean(tiles[np.where(areas_interpolated == 1)], axis = (1, 2, 3))

   # print(f"The non-interpolated mean is {not_interpolated_mean} and the interpolated means is {interpolated_mean}")
                    
    print("Interpolated {} px".format(n_interp))
    return tiles, areas_interpolated



def mcm_shadow_mask(arr: np.ndarray, c_probs: np.ndarray) -> np.ndarray:
    """ Calculates the multitemporal shadow mask for Sentinel-2 using
        the methods from Candra et al. 2020 on L1C images and matching
        outputs to the s2cloudless cloud probabilities

        Parameters:
         arr (arr): (Time, X, Y, Band) array of L1C data scaled from [0, 1]
         c_probs (arr): (Time, X, Y) array of S2cloudless cloud probabilities
    
        Returns:
         shadows_new (arr): cloud mask after Candra et al. 2020 and cloud matching 
         shadows_original (arr): cloud mask after Candra et al. 2020
    """
    import time
    def _rank_array(arr):
        order = arr.argsort()
        ranks = order.argsort()
        return ranks
    
    imsize = arr.shape[1]
    if imsize % 8 == 0:
        size = imsize
    else:
        size = imsize + (8 - (imsize % 8))
    
    arr = resize(arr, (arr.shape[0], size, size, arr.shape[-1]), order = 0)
    c_probs = resize(c_probs, (c_probs.shape[0], size, size), order = 0)
    
    mean_c_probs = np.mean(c_probs, axis = (1, 2))
    cloudy_steps = np.argwhere(mean_c_probs > 0.25)
    images_clean = np.delete(arr, cloudy_steps, 0)
    cloud_ranks = _rank_array(mean_c_probs)
    diffs = abs(np.sum(arr - np.mean(images_clean, axis = 0), axis = (1, 2, 3)))
    diff_ranks = _rank_array(diffs)
    overall_rank = diff_ranks + cloud_ranks
    reference_idx = np.argmin(overall_rank)
    ri = arr[reference_idx]
    shadows = np.zeros((arr.shape[0], size, size))  
    
    # Candra et al. 2020

    start = time.time()
    ri = np.tile(ri[np.newaxis], (arr.shape[0], 1, 1, 1))
    deltab2 = (arr[..., 0] - ri[..., 0]) < 0.10
    deltab8a = (arr[..., 1] - ri[..., 1]) < -0.04
    deltab11 = (arr[..., 2] - ri[..., 2]) < -0.04
    ti0 = arr[..., 0] < 0.095

    shadows = (deltab2 * deltab11 * deltab8a * ti0)
    shadows = shadows * 1
    end = time.time()
    print(f"Calculating shadows the new way took {end - start}")
    
    # Remove shadows if cannot coreference a cloud
    shadow_large = np.reshape(shadows, (shadows.shape[0], size // 8, 8, size // 8, 8))
    shadow_large = np.sum(shadow_large, axis = (2, 4))

    cloud_large = np.copy(c_probs)
    cloud_large[np.where(c_probs > 0.33)] = 1.
    cloud_large[np.where(c_probs < 0.33)] = 0.
    cloud_large = np.reshape(cloud_large, (shadows.shape[0], size // 8, 8, size // 8, 8))
    cloud_large = np.sum(cloud_large, axis = (2, 4))
    for time in tnrange(shadow_large.shape[0]):
        for x in range(shadow_large.shape[1]):
            x_low = np.max([x - 8, 0])
            x_high = np.min([x + 8, shadow_large.shape[1] - 2])
            for y in range(shadow_large.shape[2]):
                y_low = np.max([y - 8, 0])
                y_high = np.min([y + 8, shadow_large.shape[1] - 2])
                if shadow_large[time, x, y] < 8:
                    shadow_large[time, x, y] = 0.
                if shadow_large[time, x, y] >= 8:
                    shadow_large[time, x, y] = 1.
                c_prob_window = cloud_large[time, x_low:x_high, y_low:y_high]
                if np.max(c_prob_window) < 16:
                    shadow_large[time, x, y] = 0.
                    
    shadow_large = resize(shadow_large, (shadow_large.shape[0], size, size), order = 0)
    shadows = np.float32(shadows) * np.float32(shadow_large)
    
    # Go through and aggregate the shadow map to an 80m grid
    # and extend it one grid size around any positive ID
    shadows = np.reshape(shadows, (shadows.shape[0], size // 8, 8, size // 8, 8))
    shadows = np.sum(shadows, axis = (2, 4))
    shadows[np.where(shadows < 12)] = 0.
    shadows[np.where(shadows >= 12)] = 1.
    shadows = resize(shadows, (shadows.shape[0], size, size), order = 0)
    shadows = np.reshape(shadows, (shadows.shape[0], size//4, 4, size//4, 4))
    shadows = np.max(shadows, (2, 4))
    
    shadows_new = np.zeros_like(shadows)
    for time in range(shadows.shape[0]):
        for x in range(shadows.shape[1]):
            for y in range(shadows.shape[2]):
                if shadows[time, x, y] == 1:
                    min_x = np.max([x - 1, 0])
                    max_x = np.min([x + 2, size//4 - 1])
                    min_y = np.max([y - 1, 0])
                    max_y = np.min([y + 2, size//4 - 1])
                    for x_idx in range(min_x, max_x):
                        for y_idx in range(min_y, max_y):
                            shadows_new[time, x_idx, y_idx] = 1.
    shadows_new = resize(shadows_new, (shadows.shape[0], imsize, imsize), order = 0)
    return shadows_new


def remove_missed_clouds(img: np.ndarray) -> np.ndarray:
    """ Removes clouds that may have been missed by s2cloudless
        by looking at a temporal change outside of IQR
        
        Parameters:
         img (arr): 
    
        Returns:
         to_remove (arr): 
    """
    iqr = np.percentile(img[..., 0].flatten(), 75) - np.percentile(img[..., 0].flatten(), 25)
    thresh_t = np.percentile(img[..., 0].flatten(), 75) + iqr*1.75
    thresh_b = np.percentile(img[..., 0].flatten(), 25) - iqr*1.75
    outlier_percs = []
    for step in range(img.shape[0]):
        bottom = len(np.argwhere(img[step, ..., 0].flatten() > thresh_t))
        top = len(np.argwhere(img[step, ..., 0].flatten() < thresh_b))
        p = 100 * ((bottom + top) / (img.shape[1]*img.shape[2]))
        outlier_percs.append(p)
    to_remove = np.argwhere(np.array(outlier_percs) > 25)
    return to_remove


def calculate_cloud_steps(clouds: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """ Calculates the timesteps to remove based upon cloud cover and missing data
        
        Parameters:
         clouds (arr):
    
        Returns:
         to_remove (arr): 
    """
    
    def _check_month(month, thresh):
        month_idx = np.argwhere(np.logical_and(dates >= starting[month],
                                               dates < starting[month + 1]))
        cloud_month = cloud_percent[month_idx]
        month_good_idx = np.argwhere(cloud_month < thresh)
        if len(month_good_idx) > 0:
            month_good_dates = np.unique(dates[month_idx[month_good_idx]].flatten())
            min_distances = []
            for x in month_good_dates:
                clean_dates = dates[np.argwhere(cloud_percent < 0.15)].flatten()
                distances = x - clean_dates
                distances = np.delete(distances, month_idx.flatten())
                if np.min(distances) < 0:
                    lower_distance = abs(np.max(distances[np.argwhere(distances < 0)]))
                else: 
                    lower_distance = 0
                if np.max(distances) > 0:
                    upper_distance = np.min(distances[np.argwhere(distances > 0)])
                else:
                    upper_distance = 0
                min_distances.append(np.max([lower_distance, upper_distance]))
            min_distance = np.min(min_distances)
        else:
            month_good_dates = np.empty((0, 0))
            min_distance = 365
        return month_good_dates, min_distance
    
    good_steps = np.empty((0, ))
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 80]
    starting = np.cumsum(month_days)
    starting[0] = -30
    
    n_cloud_px = np.sum(clouds > 0.25, axis = (1, 2))
    cloud_percent = n_cloud_px / (clouds.shape[1]**2)
    thresh = [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.15]
    thresh_dist = [30, 30, 60, 60, 100, 100, 100, 100]
    for month in range(0, 12):
        finished = False
        for x in range(len(thresh)):
            if not finished:
                month_good_dates, min_distance = _check_month(month, thresh[x])
                # Put a cap of 4 images per month
                # Keep the first, last, middle, and the other one which has the lowest cloud cover
                # Max dates possible is 6, so you are either removing 1 or 2 of steps X in [1, X, X, X, X, 6]
                # keep = 0, (4, or 5)
                # if len(dates) == 5:
                #     potential_keep = 3
                if month == 0 and len(month_good_dates > 3):
                        month_good_dates = month_good_dates[-3:]
                if month == 11 and len(month_good_dates > 3):
                    month_good_dates = month_good_dates[:3]    
                if (min_distance < thresh_dist[x] or thresh[x] == 0.15):
                    finished = True
                    if len(month_good_dates) == 6:
                        month_good_dates = [val for i, val in enumerate(month_good_dates) if i in [0, 2, 3, 5]]
                        month_good_dates = np.array(month_good_dates)
                    print(f"{month}, Dates: {month_good_dates},"
                         f" Min dist: {min_distance}, thresh: {thresh[x]}")
                    good_steps = np.concatenate([good_steps, month_good_dates.flatten()])
                    
    good_steps_idx = [i for i, val in enumerate(dates) if val in good_steps]
    cloud_steps = np.array([x for x in range(dates.shape[0]) if x not in good_steps_idx])
    print(cloud_steps)
    if len(cloud_steps > 0):
        print(cloud_percent[cloud_steps])
    to_remove = cloud_steps
    data_filter = good_steps_idx
    return to_remove, good_steps_idx
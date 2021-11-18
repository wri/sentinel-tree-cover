import numpy as np
import sys
sys.path.append('../')
from src.downloading.utils import calculate_proximal_steps, calculate_proximal_steps_two
from typing import List, Any, Tuple
from functools import reduce
from skimage.transform import resize
from tqdm import tnrange, tqdm_notebook
import math
from copy import deepcopy
import time
from scipy.ndimage import label
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter


def hist_norm(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    '''
    Aligns the histograms of two input numpy arrays
    '''
    olddtype = source.dtype
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    if source.dtype != np.int:
        source = np.trunc(source * 256).astype(int)
        template = np.trunc(template * 256).astype(int)

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)


def adjust_interpolated_areas(array: np.ndarray, 
                              interp: np.ndarray) -> np.ndarray:
    interp_binary = interp > 0
    for time in range(array.shape[0]):
        if np.sum(interp[time] == 0) > 0:
            std_ref_all = np.nanstd(array[time][interp[time] == 0], axis = (0))
            mean_ref_all = np.nanmean(array[time][interp[time] == 0], axis = (0))
            
            interp_i = interp_binary[time]
            interp_map = interp[time]
            array_i = array[time]

            if np.sum(interp_i) > 0:# and np.sum(interp_i) < (216*216*0.90):
                labels, numL = label(interp_i)
                for section in range(1, numL + 1):
                    std_src = np.nanstd(array_i[labels == section], axis = (0))
                    std_ref = np.nanstd(array_i[interp_i == 0], axis = (0))

                    mean_src = np.nanmean(array_i[labels == section], axis = (0))
                    mean_ref = np.nanmean(array_i[interp_i == 0], axis = (0))
                    std_mult = (std_ref / std_src)
                    #std_mult = np.clip(std_mult, 0.5, 1.5)

                    addition = (mean_ref - (mean_src * (std_mult)))
                    #addition = np.clip(addition, -0.25, 0.25)

                    array_i[labels == section] = (
                            ((1 - interp_map[labels == section][..., np.newaxis]) * array_i[labels == section]) + \
                            (interp_map[labels == section][..., np.newaxis] * (array_i[labels == section] * std_mult + addition))
                    )
                    array[time] = array_i

    return array

def adjust_interpolated_groups(array: np.ndarray, 
                              interp: np.ndarray) -> np.ndarray:
    for time in range(array.shape[0]):
        #for group in range(interp.shape[-1]):
        if np.sum(interp[time] > 0) > 0 and np.sum(interp[time] == 0) > 0:

            interp_map = interp[time, ...]
            interp_all = interp_map#np.max(interp[time], axis = -1)
            array_i = array[time]
            aboves = [0.4, 0.6, 0.8, 1.]
            belows = [0.0, 0.4, 0.6, 0.8,]
            for above, below in zip(aboves, belows):
                interp_areas = array_i[np.logical_and(interp_map > below, interp_map <= above)]
                non_interp_areas = array_i[interp_all == 0]

                std_src = np.nanstd(interp_areas, axis = (0))
                std_ref = np.nanstd(non_interp_areas, axis = (0))

                mean_src = np.nanmean(interp_areas, axis = (0))
                mean_ref = np.nanmean(non_interp_areas, axis = (0))
                std_mult = (std_ref / std_src)
                #std_mult = np.clip(std_mult, 0.5, 1.5)

                addition = (mean_ref - (mean_src * (std_mult)))
                #addition = np.clip(addition, -0.25, 0.25)
                new_mult = interp_map[np.logical_and(interp_map > below, interp_map <= above)][..., np.newaxis] ** 0.1
                original_mult = (1 - new_mult)
                
                array_i[np.logical_and(interp_map > below, interp_map <= above)] = (
                        (original_mult * array_i[np.logical_and(interp_map > below, interp_map <= above)]) + \
                        ((array_i[np.logical_and(interp_map > below, interp_map <= above)] * std_mult + addition) * new_mult)
                    )
                array[time] = array_i

    return array


def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray, 
                             shadows: np.ndarray,
                             image_dates: List[int], 
                             wsize: int = 36, step = 8, thresh = 100) -> np.ndarray:
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

    def _fspecial_gauss_x(size, sigma):
        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**1 + y**2)/(2.0*sigma**2)))
        return g
    
    n_arr = _fspecial_gauss(wsize, 6.5)[..., np.newaxis]
    interp_gauss = _fspecial_gauss(wsize,6.5)[..., np.newaxis]
    left = _fspecial_gauss_x(wsize, 6.5)
    left[:, :wsize // 2] = 1.
    left = left * _fspecial_gauss_x(wsize, 6.5).T
    right = np.flip(left, 1)
    top = left.T
    down = right.T
    top = top[..., np.newaxis]
    down = down[..., np.newaxis]
    left = left[..., np.newaxis]
    right = right[..., np.newaxis]

    o_arr = 1 - n_arr

    c_probs = np.copy(probs)# - median_probs
    c_probs[np.where(c_probs >= 0.4)] = 1.
    c_probs[np.where(c_probs < 0.4)] = 0.
    
    if shadows.shape[1] != c_probs.shape[1]:
        shadows = shadows[:, 1:-1, 1:-1]
    c_probs += shadows
    c_probs[np.where(c_probs >= 1.)] = 1.

    areas_interpolated = np.zeros((tiles.shape[0], tiles.shape[1], tiles.shape[2]))
    dates_interpolated = np.zeros((tiles.shape[0], tiles.shape[1], tiles.shape[2], 10))

    x_range = [x for x in range(0, tiles.shape[1] - (wsize), step)] + [tiles.shape[1] - wsize]
    y_range = [x for x in range(0, tiles.shape[2] - (wsize), step)] + [tiles.shape[2] - wsize]
    dates_interp = []

    for x in x_range:
        for y in y_range:
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.05))

            if y == 0:
                array_to_use = left
            elif y == np.max(x_range):
                array_to_use = right
            elif x == 0:
                array_to_use = top
            elif x == np.max(y_range):
                array_to_use = down
            else:
                array_to_use = n_arr

            if len(satisfactory) == 0:
                print("Using median because there might only be cloudy images")
                areas_interpolated[date, x:x+wsize, y:y+wsize] = (
                    np.maximum(areas_interpolated[date, x:x+wsize, y:y+wsize], np.squeeze(n_arr)))

                median_retile = np.median(tiles[:, x:x+wsize, y:y+wsize, : ], axis = 0)
                median_retile = np.broadcast_to(median_retile, (tiles.shape[0], wsize, wsize, tiles.shape[-1]))
                tiles[date, x:x+wsize, y:y+wsize, : ] = (tiles[date, x:x+wsize, y:y+wsize, : ] * (1 - array_to_use)) + (array_to_use * median_retile)

            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(subs[date]) >= thresh:
                        before2, after2 = calculate_proximal_steps_two(date, satisfactory)

                        before = date + before2
                        after = date + after2
                        before = np.concatenate([before.flatten(), after.flatten()], axis = 0)
                        before[before < 0] = 0.
                        before[before > tiles.shape[0] - 1] = tiles.shape[0] - 1
                        before = np.unique(before)
                        if len(before) == 1:
                            if before == date:
                                before = np.arange(0, tiles.shape[0], 1)

                        is_center = False

                        candidate = np.median(tiles[before, x:x+wsize, y:y+wsize, :], axis = 0)
                        original = np.copy(tiles[date, x:x+wsize, y:y+wsize, : ])
                        if not is_center:
                            tiles[date, x:x+wsize, y:y+wsize, : ] = (original * (1 - array_to_use)) + (array_to_use * candidate) 
                            interp_window = areas_interpolated[date, x:x+wsize, y:y+wsize] 
                            areas_interpolated[date, x:x+wsize, y:y+wsize] = (
                                np.maximum(areas_interpolated[date, x:x+wsize, y:y+wsize], np.squeeze(array_to_use)))
                            dates_interpolated[date, x - 4:x+wsize + 4, y - 4:y+wsize + 4, 0] = 1.

    #tiles = adjust_interpolated_areas(tiles, areas_interpolated)
    print(f"Interpolated {np.sum(areas_interpolated > 0)} px"
          f" {np.sum(areas_interpolated) / (632 * 632 * tiles.shape[0])}%")

    x_range = [x for x in range(0, tiles.shape[1] - (wsize), step)] + [tiles.shape[1] - wsize]
    y_range = [x for x in range(0, tiles.shape[2] - (wsize), step)] + [tiles.shape[2] - wsize]
    for date in range(areas_interpolated.shape[0]):
        for x in x_range:
            for y in y_range:
                subs = areas_interpolated[date, x:x+wsize, y:y+wsize]
                if np.argmin(subs) == 105: #51 for 16x16
                    areas_interpolated[date, x:x + wsize, y:y+wsize] = 1.

    for date in range(dates_interpolated.shape[0]):
        for time in range(dates_interpolated.shape[-1]):
            dates_interpolated[date] = gaussian_filter(dates_interpolated[date], sigma=4)

    np.save("interp_dates.npy", dates_interpolated)
    return tiles, areas_interpolated


def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray, 
                             shadows: np.ndarray,
                             image_dates: List[int], 
                             wsize: int = 36, step = 8, thresh = 100) -> np.ndarray:
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
    c_probs = shadows

    areas_interpolated = np.zeros((tiles.shape[0], tiles.shape[1], tiles.shape[2]), dtype = np.float32)
    x_range = [x for x in range(0, tiles.shape[1] - (wsize), step)] + [tiles.shape[1] - wsize]
    y_range = [x for x in range(0, tiles.shape[2] - (wsize), step)] + [tiles.shape[2] - wsize]

    time1 = time.time()

    for x in x_range:
        for y in y_range:
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.2))

            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(subs[date]) >= thresh:
                        areas_interpolated[date, x:x+wsize, y:y+wsize] = 1.

            else:
                areas_interpolated[:, x:x+wsize, y:y+wsize] = 1.


    for date in range(areas_interpolated.shape[0]):
        blurred = gaussian_filter(areas_interpolated[date, ...], sigma=5, truncate = 2.)
        blurred[blurred < 0.2] = 0.
        areas_interpolated[date] = blurred
        
    areas_interpolated = areas_interpolated.astype(np.float32)

    for x in x_range:
        for y in y_range:
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.2))

            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(blurred[date] > 0.1) >= 0:
                        before2, after2 = calculate_proximal_steps_two(date, satisfactory)

                        before = date + before2
                        after = date + after2
                        before = np.concatenate([before.flatten(), after.flatten()], axis = 0)
                        before[before < 0] = 0.
                        before[before > tiles.shape[0] - 1] = tiles.shape[0] - 1
                        before = np.unique(before)
                        if len(before) <= 1:
                            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.5))
                            before2, after2 = calculate_proximal_steps_two(date, satisfactory)
                            before = date + before2
                            after = date + after2
                            before = np.concatenate([before.flatten(), after.flatten()], axis = 0)
                            before[before < 0] = 0.
                            before[before > tiles.shape[0] - 1] = tiles.shape[0] - 1
                            before = np.unique(before)
                           # print(before)
                        if len(before) == 0:
                            before = np.arange(0, tiles.shape[0], 1)

                        #array_to_use = areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis]
                        tiles[date, x:x+wsize, y:y+wsize, : ] = (
                            (tiles[date, x:x+wsize, y:y+wsize, : ] * (1 - areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis])) + \
                            (areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis] * np.median(tiles[before, x:x+wsize, y:y+wsize, :], axis = 0))
                        )
    time2 = time.time()
    return tiles, areas_interpolated


def mcm_shadow_mask(arr: np.ndarray, 
                    c_probs: np.ndarray) -> np.ndarray:
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
    imsize = arr.shape[1]

    # Create empty arrays for shadows, clouds
    shadows = np.empty_like(arr)[..., 0]
    clouds = np.empty_like(shadows)
    # Iterate through time steps, develop local reference images
    # and calculate cloud/shadow based on Candra et al. 2020
    for time in range(arr.shape[0]):
        lower = np.max([0, time - 3])
        upper = np.min([arr.shape[0], time + 4])
        ri = np.median(arr[lower:upper], axis = 0).astype(np.float32)

        deltab2 = (arr[time, ..., 0] - ri[..., 0]) > int(0.10 * 65535)
        deltab8a = (arr[time, ..., 3] - ri[..., 3]) < int(-0.04 * 65535)
        deltab11 = (arr[time, ..., 5] - ri[..., 5]) < int(-0.04 * 65535)
        deltab3 = (arr[time, ..., 1] - ri[..., 1]) > int(0.08 * 65535)
        deltab4 = (arr[time, ..., 2] - ri[..., 2]) > int(0.08 * 65535)
        ti0 = arr[time, ..., 0] < int(0.10 * 65535)
        ti10 = arr[time, ..., 4] > int(0.005 * 65535)
        clouds_i = (deltab2 * deltab3 * deltab4)
        clouds_i = clouds_i * 1
        clouds_i[clouds_i > 1] = 1.

        shadows_i = ((1 - clouds_i) * deltab11 * deltab8a * ti0)
        shadows_i = shadows_i * 1

        clouds[time] = clouds_i
        shadows[time] = shadows_i

    shadows = np.maximum(shadows, clouds)
    return shadows


def remove_missed_clouds(img: np.ndarray) -> np.ndarray:
    """ Removes clouds that may have been missed by s2cloudless
        by looking at a temporal change outside of IQR
        
        Parameters:
         img (arr): 
    
        Returns:
         to_remove (arr): 
    """


    # Implement mcm_shadow_mask based on available L2A bands
    shadows = np.empty_like(img)[..., 0]
    clouds = np.empty_like(shadows)

    for time in range(img.shape[0]):
        lower = np.max([0, time - 3])
        upper = np.min([img.shape[0], time + 4])
        ri = np.percentile(img[lower:upper], 25, axis = 0)

        deltab2 = (img[time, ..., 0] - ri[..., 0]) > 0.10#*(10/65)
        deltab8a = (img[time, ..., 7] - ri[..., 7]) < -0.04#*(10/65)
        deltab11 = (img[time, ..., 8] - ri[..., 8]) < -0.04#*(10/65)
        deltab3 = (img[time, ..., 1] - ri[..., 1]) > 0.08#*(10/65)
        deltab4 = (img[time, ..., 2] - ri[..., 2]) > 0.08#*(10/65)
        ti0 = (img[time, ..., 0] < 0.10)# * (10/65)
        clouds_i = (deltab2 * deltab3 * deltab4)
        clouds_i = clouds_i * 1

        shadows_i = ((1 - clouds_i) * deltab11 * deltab8a * ti0)
        shadows_i = shadows_i * 1

        clouds[time] = clouds_i
        shadows[time] = shadows_i

    clouds = np.maximum(clouds, shadows)
    for timestep in range(clouds.shape[0]):
        clouds[timestep] = binary_dilation(clouds[timestep], iterations = 5)
    return clouds


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
                clean_dates = dates[np.argwhere(cloud_percent <= 0.20)].flatten()
                clean_dates = clean_dates[np.argwhere(np.logical_or(clean_dates < starting[month],
                                               clean_dates >= starting[month + 1]))]
                distances = x - clean_dates
                distances = distances.flatten()
                if np.min(distances) < 0:
                    lower_distance = abs(np.max(distances[distances < 0]))
                else: 
                    lower_distance = 0
                if np.max(distances) > 0:
                    upper_distance = np.min(distances[distances > 0])
                else:
                    upper_distance = 0
                min_distances.append(np.max([lower_distance, upper_distance]))
            min_distance = np.min(min_distances)
        else:
            month_good_dates = starting[month] + 15
            clean_dates = dates[np.argwhere(cloud_percent <= 0.20)].flatten()
            clean_dates = clean_dates[np.argwhere(np.logical_or(clean_dates < starting[month],
                                           clean_dates >= starting[month + 1]))]
            distances = month_good_dates - clean_dates
            distances = distances.flatten()
            if np.min(distances) < 0:
                lower_distance = abs(np.max(distances[distances < 0]))
            else: 
                lower_distance = 0
            if np.max(distances) > 0:
                upper_distance = np.min(distances[distances > 0])
            else:
                upper_distance = 0
            min_distance = np.max([lower_distance, upper_distance])
            month_good_dates = np.empty((0, 0))
            #min_distance = 365
        return month_good_dates, min_distance
    
    good_steps = np.empty((0, ))
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 80]
    starting = np.cumsum(month_days)
    starting[0] = -30
    
    cloud_percent = clouds
    thresh = [0.20, 0.20, 0.25, 0.30]
    thresh_dist = [30, 60, 125, 150]
    for month in range(0, 12):
        finished = False
        for x in range(len(thresh)):
            if not finished:
                month_good_dates, min_distance = _check_month(month, thresh[x])
                # Put a cap of 3 images per month
                # Keep the first, last, middle, and the other one which has the lowest cloud cover
                # Max dates possible is 6, so you are either removing 1 or 2 of steps X in [1, X, X, X, X, 6]
                # keep = 0, (4, or 5)
                # if len(dates) == 5:
                #     potential_keep = 3
                if month == 0 and len(month_good_dates) > 3:
                        month_good_dates = month_good_dates[-3:]
                if month == 11 and len(month_good_dates) > 3:
                    month_good_dates = month_good_dates[:3]  
                if len(month_good_dates) > 3:
                    month_good_dates = np.array([month_good_dates[0],
                                        month_good_dates[1],
                                        month_good_dates[-1]]).flatten()
                if (min_distance < thresh_dist[x] or thresh[x] == 0.30):
                    finished = True
                    if len(month_good_dates) == 6:
                        month_good_dates = [val for i, val in enumerate(month_good_dates) if i in [0, 2, 3, 5]]
                        month_good_dates = np.array(month_good_dates)
                    print(f"{month + 1}, Dates: {month_good_dates},"
                         f" Dist: {min_distance}, Thresh: {thresh[x]}")
                    good_steps = np.concatenate([good_steps, month_good_dates.flatten()])
                    
    good_steps_idx = [i for i, val in enumerate(dates) if val in good_steps]
    cloud_steps = np.array([x for x in range(dates.shape[0]) if x not in good_steps_idx])

    to_remove = cloud_steps
    data_filter = good_steps_idx

    print(f"Utilizing {len(good_steps_idx)}/{dates.shape[0]} steps")
    return to_remove, good_steps_idx


def print_dates(dates, probs):
    month_days = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 80]
    starting = np.cumsum(month_days)
    starting[0] = -30
    
    for month in range(0, 12):
        month_idx = np.argwhere(np.logical_and(dates >= starting[month],
                                               dates < starting[month + 1]))
        month_dates = dates[month_idx]
        month_dates = [item for sublist in month_dates for item in sublist]
        month_probs = probs[month_idx]
        month_probs = [item for sublist in month_probs for item in sublist]
        month_probs = [round(x, 2) for x in month_probs]

        print(f"{month + 1}, Dates: {month_dates}, Probs: {month_probs}")


def subset_contiguous_sunny_dates(dates, probs):
    """
    For plots that have at least 24 image dates
    identifies 3-month windows where each month has at least three clean
    image dates, and removes one image per month within the window.
    Used to limit the number of images needed in low-cloud regions from
    a max of 36 to a max of 24.
    """
    begin = [-60, 0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    end = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 390]
    n_per_month = []
    months_to_adjust = []
    months_to_adjust_again = []
    indices_to_rm = []
    
    def _indices_month(dates, x, y, indices_to_rm):
        indices_month = np.argwhere(np.logical_and(
                    dates >= x, dates < y)).flatten()
        indices_month = [x for x in indices_month if x not in indices_to_rm]
        return indices_month

    if len(dates) >= 8:
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            n_per_month.append(len(indices_month))

        # Convert >3 image months to 2 by removing the cloudiest image
        # Enforces a maximum of 24 images
        for x in range(11):
            # This will only go from 3 images to 2 images
            three_m_sum = np.sum(n_per_month[x:x+3])
            # If at least 4/9 images and a minimum of 0:
            if three_m_sum >= 4 and np.min(n_per_month[x:x+3]) >= 0:
                months_to_adjust += [x, x+1, x+2]

        months_to_adjust = list(set(months_to_adjust))

        # This will sometimes take 3 month windows and cap them to 2 images/month
        for x in [0, 2, 4, 6, 8, 10]:
            three_m_sum = np.sum(n_per_month[x:x+3])
            if three_m_sum >= 5 and np.min(n_per_month[x:x+3]) >= 1: 
                if n_per_month[x + 1] == 3: # 3, 3, 3 or 2, 3, 2
                    months_to_adjust_again.append(x + 1)
                elif n_per_month[x] == 3: # 3, 2, 2 
                    months_to_adjust_again.append(x)
                elif n_per_month[x + 1] == 2: # 2, 2, 2 or 2, 2, 3
                    months_to_adjust_again.append(x + 1)
                elif n_per_month[x + 2] == 3: # 2, 2, 3
                    months_to_adjust_again.append(x + 2)

        if len(months_to_adjust) > 0:
            for month in months_to_adjust:
                indices_month = np.argwhere(np.logical_and(
                    dates >= begin[month], dates < end[month])).flatten()
                if len(indices_month) > 0:
                    if np.max(probs[indices_month]) >= 0.15:
                        cloudiest_idx = np.argmax(probs[indices_month].flatten())
                    else:
                        cloudiest_idx = 1
                    if len(indices_month) >= 3:
                        indices_to_rm.append(indices_month[cloudiest_idx])

        print(f"Removing {len(indices_to_rm)} sunny dates")
        n_remaining = len(dates) - len(indices_to_rm)

        if len(months_to_adjust_again) > 0 and n_remaining > 12:
            for month in months_to_adjust_again:
                indices_month = np.argwhere(np.logical_and(
                    dates >= begin[month], dates < end[month])).flatten()
                indices_month = [x for x in indices_month if x not in indices_to_rm]
                if np.max(probs[indices_month]) >= 0.15:
                    cloudiest_idx = np.argmax(probs[indices_month].flatten())
                else:
                    cloudiest_idx = 1
                if len(indices_month) >= 2:
                    indices_to_rm.append(indices_month[cloudiest_idx])

        if len(np.argwhere(probs > 0.4)) > 0:
            to_rm = [int(x) for x in np.argwhere(probs > 0.4)]
            print(f"Removing: {dates[to_rm]} missed cloudy dates")
            indices_to_rm.extend(to_rm)

        n_remaining = len(dates) - len(set(indices_to_rm))
        print(f"There are {n_remaining} left, max prob is {np.max(probs)}")

        if n_remaining > 12:
            probs[indices_to_rm] = 0.
            # logic flow here to remove all steps with > 10% cloud cover that leaves 14
            # or if no are >10% cloud cover, to remove the second time step for each month
            # with at least 1 image
            n_above_10_cloud = np.sum(probs >= 0.15)
            print(f"There are {n_above_10_cloud} steps above 10%")
            len_to_rm = (n_remaining - 12)
            print(f"Removing {len_to_rm} steps to leave a max of 12")
            if len_to_rm < n_above_10_cloud:
                max_cloud = np.argpartition(probs, -len_to_rm)[-len_to_rm:]
                print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                indices_to_rm.extend(max_cloud)
            else:
                # Remove the ones that are > 10% cloud cover
                if n_above_10_cloud > 0:
                    max_cloud = np.argpartition(probs, -n_above_10_cloud)[-n_above_10_cloud:]
                    print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                    indices_to_rm.extend(max_cloud)

                # And then remove the second time step for months with multiple images
                n_to_remove = len_to_rm - n_above_10_cloud
                print(f"Need to remove {n_to_remove} of {n_remaining}")
                n_removed = 0
                for x, y in zip(begin, end):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if len(indices_month) > 1:
                        if n_removed <= n_to_remove:
                            if indices_month[1] not in indices_to_rm:
                                indices_to_rm.append(indices_month[1])
                            else:
                                indices_to_rm.append(indices_month[0])
                            print(f"Removing second image in month: {x}, {indices_month[1]}")
                            print(len(set(indices_to_rm)), len(dates))
                        n_removed += 1

        elif np.max(probs) >= 0.20:
            max_cloud = np.argmax(probs)
            print(f"Removing cloudiest date (cloud_removal): {max_cloud}, {probs[max_cloud]}")
            indices_to_rm.append(max_cloud)

        n_remaining = len(dates) - len(set(indices_to_rm))
        print(f"There are {n_remaining} left, max prob is {np.max(probs)}")
        if n_remaining > 10:
            probs[indices_to_rm] = 0.

            # This first block will then remove months 3 and 9 if there are at least 10 months with images
            images_per_month = []
            for x, y, month in zip(begin, end, np.arange(0, 13, 1)):
                indices_month = _indices_month(dates, x, y, set(indices_to_rm))
                images_per_month.append(len(indices_month))
            print(images_per_month)

            months_with_images = np.sum(np.array(images_per_month) >= 1)
            if months_with_images >= 11:
                for x, y, month in zip(begin, end, np.arange(0, 13, 1)):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if months_with_images >= 12 and len(indices_month) > 0:
                        if (month == 3 or month == 9):
                                indices_to_rm.append(indices_month[0])
                    elif months_with_images == 11 and len(indices_month) > 0:
                        if month == 3:
                            if len(indices_month) > 0:
                                indices_to_rm.append(indices_month[0])

            # This second block will go back through and remove the second image from months with multiple images
            elif np.sum(np.array(images_per_month) >= 2) >= 1:
                n_to_remove = n_remaining - 10
                n_removed = 0
                for x, y in zip(begin, end):
                    indices_month = _indices_month(dates, x, y, indices_to_rm)
                    if len(indices_month) > 1:
                        if n_removed < n_to_remove:
                            if indices_month[1] not in indices_to_rm:
                                indices_to_rm.append(indices_month[1])
                            else:
                                indices_to_rm.append(indices_month[0])
                            print(f"Removing second image in month: {x}, {indices_month[1]}")
                        n_removed += 1

        print(f"Removing {len(indices_to_rm)} sunny/cloudy dates")
    return indices_to_rm
    

def subset_contiguous_sunny_dates(dates, probs):
    """
    Potential problems:

       1. One tile has an image at 0.28, the next has 0.34 - date mismatch
       2. A month has [0.28, 0.26, 0.02] and the [0.28, 0.26] get selected

    """
    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 401]
    n_per_month = []
    months_to_adjust = []
    months_to_adjust_again = []
    indices_to_rm = []
    indices = [x for x in range(len(dates))]
    
    def _indices_month(dates, x, y):
        indices_month = np.argwhere(np.logical_and(
                    dates >= x, dates < y)).flatten()
        return indices_month


    print_dates(dates, probs)
    # Select the best 2 images per month to start with
    best_two_per_month = []
    for x, y in zip(begin, end):
        indices_month = np.argwhere(np.logical_and(
            dates >= x, dates < y)).flatten()

        month_dates = dates[indices_month]
        month_clouds = probs[indices_month]
        month_good_dates = month_dates[month_clouds < 0.15]
        indices_month = indices_month[month_clouds < 0.15]

        if len(month_good_dates) >= 2:
            if x > 0:
                ideal_dates = [x, x + 15]
            else:
                ideal_dates = [0, 15]

            # We first pick the 2 images with <30% cloud cover that are the closest
            # to the 1st and 15th of the month
            # todo: if both these images are above 15%, and one below 15% is available, include it
            closest_to_first_img = np.argmin(abs(month_good_dates - ideal_dates[0]))
            closest_to_second_img = np.argmin(abs(month_good_dates - ideal_dates[1]))
            if closest_to_second_img == closest_to_first_img:
                distances = abs(month_good_dates - ideal_dates[1])
                closest_to_second_img = np.argsort(distances)[1]

            first_image = indices_month[closest_to_first_img]
            second_image = indices_month[closest_to_second_img]
            best_two_per_month.append(first_image)
            best_two_per_month.append(second_image)
                    
        elif len(month_good_dates) >= 1:
            if x > 0:
                ideal_dates = [x, x + 15]
            else:
                ideal_dates = [0, 15]

            closest_to_second_img = np.argmin(abs(month_good_dates - ideal_dates[1]))
            second_image = indices_month[closest_to_second_img]
            best_two_per_month.append(second_image)
                
    dates_round_2 = dates[best_two_per_month]
    probs_round_2 = probs[best_two_per_month]
    
    # We then select between those two images to keep a max of one per month
    # We select the least cloudy image if the most cloudy has >15% cloud cover
    # Otherwise we select the second image
    if len(dates_round_2) > 10:
        n_to_rm = len(dates_round_2) - 10
        monthly_dates = []
        monthly_probs = []
        monthly_dates_date = []
        removed = 0
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [val for i, val in enumerate(indices_month) if dates_month[i] in dates_round_2]
            if len(indices_month) > 1:
                month_dates = dates[indices_month]
                month_clouds = probs[indices_month]
                subset_month = True if x not in [-60, 334] else False
                if subset_month:
                    subset_month = True if removed <= n_to_rm else False
                if subset_month:
                    if np.max(month_clouds) >= 0.10:
                        month_best_date = [indices_month[np.argmin(month_clouds)]]
                    else:
                        month_best_date = [indices_month[1]]
                else:
                    month_best_date = indices_month
                monthly_dates.extend(month_best_date)
                monthly_probs.extend(probs[month_best_date])
                monthly_dates_date.extend(dates[month_best_date])
                removed += 1
            elif len(indices_month) == 1:
                monthly_dates.append(indices_month[0])
                monthly_probs.append(probs[indices_month[0]])
                monthly_dates_date.append(dates[indices_month[0]])
    else:
        monthly_dates = best_two_per_month
        
    indices_to_rm = [x for x in indices if x not in monthly_dates]


    dates_round_3 = dates[monthly_dates]
    probs_round_3 = probs[monthly_dates]
    
    if len(dates_round_3) > 10:
        delete_max = False
        if np.max(probs_round_3) >= 0.15:
            delete_max = True
            indices_to_rm.append(monthly_dates[np.argmax(probs_round_3)])
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [x for x in indices_month if x in monthly_dates]
            if len(indices_month) > 0:
                if len(monthly_dates) == 11 and delete_max:
                    continue
                elif len(monthly_dates) == 11 or (len(monthly_dates) == 12 and delete_max):
                    if x == 90:
                        indices_to_rm.append(indices_month[0])
                elif len(monthly_dates) >= 12:
                    if x == 90 or x == 273:
                        indices_to_rm.append(indices_month[0])
    return indices_to_rm

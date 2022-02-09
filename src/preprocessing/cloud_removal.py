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
import warnings

def adjust_interpolated_groups(array: np.ndarray, 
                              interp: np.ndarray) -> np.ndarray:
    for time in range(array.shape[0]):
        # For each time step where there is >0 and <100% interpolation
        if np.sum(interp[time] > 0) > 0 and np.sum(interp[time] == 0) > 0:
            interp_map = interp[time, ...]
            interp_all = interp_map#np.max(interp[time], axis = -1)
            array_i = array[time]
            # Conduct histogram normalization for each threshold of the 
            # Gaussian blur, since they will have varying amounts of brightness
            aboves = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
            belows = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
            for above, below in zip(aboves, belows):
                # Identify all of the areas that are, and aren't interpolated
                interp_areas = array_i[np.logical_and(interp_map > below, interp_map <= above)]
                interp_map[np.logical_and(interp_map > below, interp_map <= above)] = above
                non_interp_areas = array_i[interp_all == 0]

                # And calculate their means and standard deviation per band
                std_src = np.nanstd(interp_areas, axis = (0))
                std_ref = np.nanstd(non_interp_areas, axis = (0))
                mean_src = np.nanmean(interp_areas, axis = (0))
                mean_ref = np.nanmean(non_interp_areas, axis = (0))
                std_mult = (std_ref / std_src)

                addition = (mean_ref - (mean_src * (std_mult)))                
                array_i[np.logical_and(interp_map > below, interp_map <= above)] = (
                        array_i[np.logical_and(interp_map > below, interp_map <= above)] * std_mult + addition
                    )
                array[time] = array_i

    # If the full image is interpolated, then it is better to replace it
    # With the median of histogram matched images, rather than leave it
    # Because it could be interpolated with non-histogram matched images.
    for time in range(array.shape[0]):
        if np.mean(interp[time] > 0) == 1:
            candidate_lower = np.max([time - 1, 0])
            candidate_upper = np.min([time + 1, array.shape[0] - 1])
            candidate = np.mean(
                np.array([array[candidate_lower], array[candidate_upper]]), axis = 0
            )
            array[time] = candidate
    return array


def adjust_interpolated_areas_small(small_array: np.ndarray, 
                              std_ref: np.ndarray,
                              mean_ref) -> np.ndarray:
    if np.sum(std_ref) > 0:
        #(small_array.shape)
        std_src = np.std(small_array, axis = (0, 1))
        mean_src = np.mean(small_array, axis = (0, 1))
        std_mult = (std_ref / std_src)
        addition = (mean_ref - (mean_src * (std_mult)))   
        std_mult = np.reshape(std_mult, (1, 1, 10))
        addition = np.reshape(addition, (1, 1, 10)) 
        std_mult = np.clip(std_mult, 0.8, 1.2)
        addition = np.clip(addition, -0.1, 0.1) 
        #print(addition.shape)    
        #np.save("Small_array.npy", small_array)
        small_array = (small_array * std_mult) + addition
        #np.save("Small_array2.npy", small_array)
        #time.sleep(10)
    return small_array


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

    aboves = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
    belows = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for date in range(areas_interpolated.shape[0]):
        blurred = gaussian_filter(areas_interpolated[date, ...], sigma=6, truncate = 2.)
        blurred[blurred < 0.1] = 0.
        for above, below in zip(aboves, belows):
            blurred[np.logical_and(blurred > below, blurred <= above)] = above
        areas_interpolated[date] = blurred
        
    areas_interpolated = areas_interpolated.astype(np.float32)

    std_ref = np.zeros((tiles.shape[0], tiles.shape[-1]))
    mean_ref = np.zeros((tiles.shape[0], tiles.shape[-1]))
    for date in range(0, tiles.shape[0]):
        date_interp = areas_interpolated[date]
        date_tile = tiles[date]
        if np.sum(date_interp == 0) > 0:
            non_interp_areas = date_tile[date_interp == 0]
            std_ref[date] = np.nanstd(non_interp_areas, axis = (0))
            mean_ref[date] = np.nanmean(non_interp_areas, axis = (0))

    #print(mean_ref)
    tile_median = np.median(tiles, axis = 0)
    # masked_median = np.ma.masked_where(areas_interpolated > 0, tiles)
    # masked_median = np.ma.median
    # masked_median.data[masked_median.mask]
    #np.median()

    for x in x_range:
        for y in y_range:
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.1))

            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(areas_interpolated[date, x:x + wsize, y:y+wsize]) >= 0:
                        """
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
                        """
                        #before = np.arange(0, tiles.shape[0], 1)

                        #array_to_use = areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis]

                        candidate = tile_median[x:x+wsize, y:y+wsize, :]#, axis = 0)
                        #candidate = tile_median[x:x+wsize, y:y+wsize]

                        #candidate = adjust_interpolated_areas_small(small_array = candidate,
                        #    std_ref = std_ref[date],
                        #    mean_ref = mean_ref[date]
                        #)

                        candidate = (
                            (tiles[date, x:x+wsize, y:y+wsize, : ] * (1 - areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis])) + \
                            (areas_interpolated[date, x:x+wsize, y:y+wsize][..., np.newaxis] * candidate)
                        )

                        

                        #snp.save("after3.npy", candidate)
                        tiles[date, x:x+wsize, y:y+wsize, : ] = candidate
    #np.save("tiles_out.npy", tiles)
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
    warnings.warn("mcm_shadow_mask is deprecated; use remove_missed_clouds", category=DeprecationWarning)
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
        if img.shape[0] > 2:
            ri_lower = np.percentile(img[lower:upper], 25, axis = 0)
            ri_upper = ri_lower
        else:
            ri_lower = np.min(img[lower:upper], axis = 0)
            ri_upper = np.max(img[lower:upper], axis = 0)

        deltab2 = (img[time, ..., 0] - ri_lower[..., 0]) > 0.10
        deltab8a = (img[time, ..., 7] - ri_upper[..., 7]) < -0.04
        deltab11 = (img[time, ..., 8] - ri_upper[..., 8]) < -0.04
        deltab3 = (img[time, ..., 1] - ri_lower[..., 1]) > 0.08
        deltab4 = (img[time, ..., 2] - ri_lower[..., 2]) > 0.08
        ti0 = (img[time, ..., 0] < 0.10)
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

    duplicate_steps = []
    for month in range(0, 12):
        month_idx = np.argwhere(np.logical_and(dates % 365 >= starting[month],
                                               dates % 365 < starting[month + 1]))
        month_dates = dates[month_idx]
        month_dates = [item for sublist in month_dates for item in sublist]
        if len(month_dates) > 1:
            to_remove = month_idx[np.argsort(month_dates)[:-1]]
            if len(to_remove) > 1:
                duplicate_steps.extend(to_remove)
            else:
                duplicate_steps.append(to_remove)

        month_probs = probs[month_idx]
        month_probs = [item for sublist in month_probs for item in sublist]
        month_probs = [round(x, 2) for x in month_probs]

        print(f"{month + 1}, Dates: {month_dates}, Probs: {month_probs}")
    return duplicate_steps

def subset_contiguous_sunny_dates(dates, probs):
    """
    The general imagery subsetting strategy is as below:
        - Select all images with < 30% cloud cover
        - For each month, select up to 2 images that are <30% CC and are the closest to
          the beginning and the midde of the month
        - Select only one image per month for each month if the following criteria are met
              - Within Q1 and Q4, apply if at least 3 images in quarter
              - Otherwise, apply if at least 8 total images for year
              - Select the second image if max CC < 15%, otherwise select least-cloudy image
        - If more than 10 images remain, remove any images for April and September

    """
    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 410]
    n_per_month = []
    months_to_adjust = []
    months_to_adjust_again = []
    indices_to_rm = []
    indices = [x for x in range(len(dates))]
    
    def _indices_month(dates, x, y):
        indices_month = np.argwhere(np.logical_and(
                    dates >= x, dates < y)).flatten()
        return indices_month


    _ = print_dates(dates, probs)
    # Select the best 2 images per month to start with
    best_two_per_month = []
    for x, y in zip(begin, end):
        indices_month = np.argwhere(np.logical_and(
            dates >= x, dates < y)).flatten()

        month_dates = dates[indices_month]
        month_clouds = probs[indices_month]
        month_good_dates = month_dates[month_clouds < 0.20]
        indices_month = indices_month[month_clouds < 0.20]

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

    # If there are more than 8 images, subset so only 1 image per month,
    # To bring down to a min of 8 images
    if len(dates_round_2) >= 7:
        n_to_rm = len(dates_round_2) - 7
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

                subset_month = True
                if x == -60:
                    feb_mar = np.argwhere(np.logical_and(
                        dates >= 31, dates < 90)).flatten()
                    subset_month = False if len(feb_mar) < 1 else True
                if x == 334:
                    oct_nov = np.argwhere(np.logical_and(
                        dates >= 273, dates < 334)).flatten()
                    subset_month = False if len(oct_nov) < 1 else True

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

    if len(dates_round_3) >= 10:
        delete_max = False
        if np.max(probs_round_3) >= 0.15:
            delete_max = True
            indices_to_rm.append(monthly_dates[np.argmax(probs_round_3)])
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [x for x in indices_month if x in monthly_dates]

            n_removed = 0
            if len(indices_month) >= 1:
                if len(monthly_dates) == 11 and delete_max:
                    continue
                elif len(monthly_dates) >= 11:
                    if x in [90, 243]:
                        indices_to_rm.append(indices_month[0])

    return indices_to_rm

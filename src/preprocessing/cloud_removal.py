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

def hist_norm(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    '''
    Aligns the histograms of two input numpy arrays
    '''
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


def adjust_interpolated_areas_new(array: np.ndarray,
                                  interp: np.ndarray) -> np.ndarray:
    '''
    Aligns the histograms of the interpolated areas of an array with the
    histograms of the non-interpolated areas
    '''
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


def adjust_interpolated_areas(array: np.ndarray,
                              interp: np.ndarray) -> np.ndarray:
    for time in range(array.shape[0]):
        for band in range(array.shape[-1]):
            interp_i = interp[time]
            array_i = array[time, :, :, band]
            if np.sum(interp_i) > 0:
                adj = (np.median(array_i[interp_i == 0]) -  # 0.2
                      (np.median(array_i[interp_i == 1]))) # 0.
                if adj < 0:
                    adj = np.max([adj, -0.05])
                if adj > 0:
                    adj = np.min([adj, 0.05])

                array_i[interp_i == 1] += adj # - 0.1
                array[time, :, :, band] = array_i
    return array


def adjust_interpolated_areas(array: np.ndarray,
                              interp: np.ndarray) -> np.ndarray:
    for time in range(array.shape[0]):
        std_ref_all = np.nanstd(array[time][interp[time] == 0], axis = (0, 1))
        mean_ref_all = np.nanmean(array[time][interp[time] == 0], axis = (0, 1))
        for local_x in [x for x in range(0, array.shape[1] - 200, 200)] + [array.shape[1] - 200]:
            for local_y in [x for x in range(0, array.shape[2] - 200, 200)] + [array.shape[2] - 200]:
                interp_i = interp[time, local_x:local_x + 200, local_y:local_y + 200]
                array_i = array[time, local_x:local_x + 200, local_y:local_y + 200]

                if np.sum(interp_i) > 0 and np.sum(interp_i) < (200*200*0.5):
                    std_src = np.nanstd(array_i[interp_i == 1], axis = (0, 1))
                    std_ref = np.nanstd(array_i[interp_i == 0], axis = (0, 1))

                    mean_src = np.nanmean(array_i[interp_i == 1], axis = (0, 1))
                    mean_ref = np.nanmean(array_i[interp_i == 0], axis = (0, 1))

                    array_i[interp_i == 1] = \
                            array_i[interp_i == 1] * (std_ref / std_src) + (mean_ref - (mean_src * (std_ref / std_src)))
                    array[time, local_x:local_x + 200, local_y:local_y + 200] = array_i

                elif np.sum(interp_i) > 0 and np.sum(interp_i) > (200*200*0.5):
                    std_src = np.nanstd(array_i[interp_i == 1], axis = (0, 1))
                    std_ref = std_ref_all

                    mean_src = np.nanmean(array_i[interp_i == 1], axis = (0, 1))
                    mean_ref = mean_ref_all

                    array_i[interp_i == 1] = \
                            array_i[interp_i == 1] * (std_ref / std_src) + (mean_ref - (mean_src * (std_ref / std_src)))
                    array[time, local_x:local_x + 200, local_y:local_y + 200] = array_i
    return array


def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray,
                             shadows: np.ndarray,
                             image_dates: List[int],
                             wsize: int = 32) -> np.ndarray:
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

    c_arr = np.full((1, wsize,wsize, 1), 0.5)
    c_arr[:, 1:wsize-1, 1:wsize-1, :] += 0.25
    c_arr[:, 2:wsize-2, 2:wsize-2, :] += 0.25
    o_arr = np.ones((1, wsize, wsize, 1)) - c_arr

    # Subtract the median cloud binary mask, to remove pixels
    #  that are always clouds (false positive)
    #median_probs = np.percentile(probs, 66, axis = 0)
    #median_probs[median_probs < 0.10] = 0.
    c_probs = np.copy(probs)# - median_probs
    c_probs[np.where(c_probs >= 0.5)] = 1.
    c_probs[np.where(c_probs < 0.5)] = 0.

    initial_shadows = np.sum(shadows)
    after_shadows = np.sum(shadows)
    if shadows.shape[1] != c_probs.shape[1]:
        shadows = shadows[:, 1:-1, 1:-1]
    c_probs += shadows
    c_probs[np.where(c_probs >= 1.)] = 1.

    areas_interpolated = np.zeros((tiles.shape[0], tiles.shape[1], tiles.shape[2]))
    x_range = [x for x in range(0, tiles.shape[1] - (wsize), 8)] + [tiles.shape[1] - wsize]
    y_range = [x for x in range(0, tiles.shape[2] - (wsize), 8)] + [tiles.shape[2] - wsize]
    for x in x_range:
        for y in y_range:
            subs = c_probs[:, x:x + wsize, y:y+wsize]
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) == 0)
            if len(satisfactory) == 0:
                print("Using median because there might only be cloudy images")
                areas_interpolated[:, x:x+wsize, y:y+wsize] = 1.
                median_retile = np.median(tiles[:, x:x+wsize, y:y+wsize, : ], axis = 0)
                median_retile = np.broadcast_to(median_retile, (tiles.shape[0], wsize, wsize, tiles.shape[-1]))
                tiles[:, x:x+wsize, y:y+wsize, : ] = median_retile
            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(subs[date]) >= 40:
                        #before, after = calculate_proximal_steps(date, satisfactory)
                        before2, after2 = calculate_proximal_steps_two(date, satisfactory)

                        before = date + before2
                        after = date + after2
                        before = np.concatenate([before.flatten(), after.flatten()], axis = 0)
                        before[before < 0] = 0.
                        before[before > tiles.shape[0] - 1] = tiles.shape[0] - 1

                        candidate = np.median(tiles[before, x:x+wsize, y:y+wsize, :], axis = 0)
                        tiles[date, x:x+wsize, y:y+wsize, : ] = candidate

                        areas_interpolated[date, x:x+wsize, y:y+wsize] = 1.

    tiles = adjust_interpolated_areas(tiles, areas_interpolated)
    print(f"Interpolated {np.sum(areas_interpolated)} px"
          f" {np.sum(areas_interpolated) / (632 * 632 * tiles.shape[0])}%")
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

    if imsize % 8 != 0:
        pad_amt = 1 #int((imsize % 8) // 2)

        arr = np.pad(arr, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt), (0, 0)))
        c_probs = np.pad(c_probs, ((0, 0), (pad_amt, pad_amt), (pad_amt, pad_amt)))

    assert arr.dtype == np.uint16
    assert arr.shape[1] == c_probs.shape[1]
    size = arr.shape[1]

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
        deltab8a = (arr[time, ..., 3] - ri[..., 3]) < int(-0.06 * 65535)
        deltab11 = (arr[time, ..., 5] - ri[..., 5]) < int(-0.06 * 65535)
        deltab3 = (arr[time, ..., 1] - ri[..., 1]) > int(0.08 * 65535)
        deltab4 = (arr[time, ..., 2] - ri[..., 2]) > int(0.08 * 65535)
        ti0 = arr[time, ..., 0] < int(0.12 * 65535)
        ti10 = arr[time, ..., 4] > int(0.01 * 65535)
        clouds_i = (deltab2 * deltab3 * deltab4) + ti10
        clouds_i = clouds_i * 1
        clouds_i[clouds_i > 1] = 1.

        shadows_i = ((1 - clouds_i) * deltab11 * deltab8a * ti0)
        shadows_i = shadows_i * 1

        clouds[time] = clouds_i
        shadows[time] = shadows_i

    # Iterate through clouds, shadows, remove cloud/shadow where
    # The same px is positive in subsequent time steps (likely FP)
    clouds_new = np.copy(clouds)
    for time in range(1, clouds.shape[-1], 1):
        moving_sums = np.sum(clouds[time - 1:time + 2], axis = (0))
        moving_sums = moving_sums >= 3
        clouds_new[time - 1:time + 2, moving_sums] = 0.
    clouds = clouds_new

    # Remove shadows if multiple time steps are shadows
    """
    shadows_new = np.copy(shadows)
    for time in range(1, shadows.shape[-1], 1):
        moving_sums = np.sum(shadows[time - 1:time + 1], axis = 0)
        moving_sums = moving_sums >= 2
        if np.sum(moving_sums > 0):
        	print(f"Removing {np.sum(moving_sums)}, time {time}")
        shadows_new[time - 1:time + 1, moving_sums] = 0.
    shadows = shadows_new
    print(np.sum(shadows), np.sum(clouds))
    """

    # Combine cloud and shadow
    shadows = shadows + clouds
    shadows[shadows > 1] = 1.
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
        ri = np.median(img[lower:upper], axis = 0)

        deltab2 = (img[time, ..., 0] - ri[..., 0]) > 0.10
        deltab8a = (img[time, ..., 7] - ri[..., 7]) < -0.05
        deltab11 = (img[time, ..., 8] - ri[..., 8]) < -0.05
        deltab3 = (img[time, ..., 1] - ri[..., 1]) > 0.08
        deltab4 = (img[time, ..., 2] - ri[..., 2]) > 0.08
        ti0 = (img[time, ..., 0] < 0.095)
        clouds_i = (deltab2 * deltab3 * deltab4)
        clouds_i = clouds_i * 1

        shadows_i = ((1 - clouds_i) * deltab11 * deltab8a * ti0)
        shadows_i = shadows_i * 1

        if np.mean(clouds_i) > 0.1:
            print(f"Missed cloud {time}: {np.mean(clouds_i)}")
        if np.mean(shadows_i) > 0.1:
            print(f"Missed shadow {time}: {np.mean(shadows_i)}")

        clouds[time] = clouds_i
        shadows[time] = shadows_i

    clouds_new = np.copy(clouds)
    for time in range(1, clouds.shape[-1], 1):
        moving_sums = np.sum(clouds[time - 1:time + 2], axis = (0))
        moving_sums = moving_sums >= 3
        clouds_new[time - 1:time + 2, moving_sums] = 0.
    clouds = clouds_new

    shadows_new = np.copy(shadows)
    for time in range(1, shadows.shape[-1], 1):
        moving_sums = np.sum(shadows[time - 1:time + 1], axis = 0)
        moving_sums = moving_sums >= 2
        shadows_new[time - 1:time + 1, moving_sums] = 0.
    shadows = shadows_new

    clouds = clouds + shadows
    clouds[clouds > 1] = 1.
    clouds = np.mean(clouds, axis = (1, 2))
    clouds[clouds < 0.05] = 0.

    to_remove = np.argwhere(clouds > 0.25)

    delete_to_remove = []
    if len(to_remove) > 2:
        for i in range(1, len(to_remove) - 1):
            if to_remove[i - 1] == to_remove[i] - 1:
                if to_remove[i + 1] == to_remove[i] + 1:
                    delete_to_remove.append(i)
                    delete_to_remove.append(i - 1)
                    delete_to_remove.append(i + 1)

    if len(delete_to_remove) > 0:
        delete_to_remove = list(set(delete_to_remove))
        print(f"Removing: {delete_to_remove}")
        to_remove = np.delete(to_remove, delete_to_remove)

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

    n_cloud_px = np.sum(clouds > 0.50, axis = (1, 2))
    cloud_percent = n_cloud_px / (clouds.shape[1]**2)
    thresh = [0.01, 0.02, 0.05, 0.10, 0.15, 0.15, 0.20, 0.25, 0.30]
    thresh_dist = [30, 30, 60, 60, 100, 100, 100, 125, 150]
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

    if len(dates) >= 9:
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
                # Add the months to be adjusted
                months_to_adjust += [x, x+1, x+2]

        months_to_adjust = list(set(months_to_adjust))

        # Jan - Mar, Mar - May, May - Jul, Jul - Sep, Sep - Nov, Oct - Dec
        # This will sometimes take 3 images down to 1 image
        for x in [0, 2, 4, 6, 8, 10]:
            three_m_sum = np.sum(n_per_month[x:x+3])
            # For windows that are 2, 2, 2 to 3, 3, 3
            if three_m_sum >= 5 and np.min(n_per_month[x:x+3]) >= 1:
                # Prefer to adjust the middle month if possible
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
                    if np.max(probs[indices_month]) > 0.1:
                        cloudiest_idx = np.argmax(probs[indices_month].flatten())
                    else:
                        cloudiest_idx = 1
                    # Remove the cloudiest image of the 3
                    if len(indices_month) >= 3:
                        indices_to_rm.append(indices_month[cloudiest_idx])

        print(f"Removing {len(indices_to_rm)} sunny dates")
        n_remaining = len(dates) - len(indices_to_rm)

        if len(months_to_adjust_again) > 0 and n_remaining > 12:
            for month in months_to_adjust_again:
                indices_month = np.argwhere(np.logical_and(
                    dates >= begin[month], dates < end[month])).flatten()
                indices_month = [x for x in indices_month if x not in indices_to_rm]
                if np.max(probs[indices_month]) > 0.1:
                    cloudiest_idx = np.argmax(probs[indices_month].flatten())
                else:
                    cloudiest_idx = 1
                # Remove the cloudiest image of the 3
                if len(indices_month) >= 2:
                    indices_to_rm.append(indices_month[cloudiest_idx])

        if len(np.argwhere(probs > 0.4)) > 0:
            to_rm = [int(x) for x in np.argwhere(probs > 0.4)]
            print(f"Removing: {dates[to_rm]} missed cloudy dates")
            indices_to_rm.extend(to_rm)

        n_remaining = len(dates) - len(indices_to_rm)
        print(f"There are {n_remaining} left, max prob is {np.max(probs)}")

        if n_remaining > 13:
            probs[indices_to_rm] = 0.
            # logic flow here to remove all steps with > 10% cloud cover that leaves 14
            # or if no are >10% cloud cover, to remove the second time step for each month
            # with at least 1 image
            n_above_10_cloud = np.sum(probs > 0.10)
            len_to_rm = (n_remaining - 13)
            if len_to_rm < n_above_10_cloud:
                max_cloud = np.argpartition(probs, -len_to_rm)[-len_to_rm:]
                print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                indices_to_rm.extend(max_cloud)
            else:
                # Remove the ones that are > 10% cloud cover
                max_cloud = np.argpartition(probs, -n_above_10_cloud)[-n_above_10_cloud:]
                print(f"Removing cloudiest dates: {max_cloud}, {probs[max_cloud]}")
                indices_to_rm.extend(max_cloud)

                # And then remove the second time step for months with multiple images
                #dates_out = np.copy(dates)
                #dates_out = np.delete(dates_out, indices_to_rm)

                n_to_remove = len_to_rm - n_above_10_cloud
                n_removed = 0
                for x, y in zip(begin, end):
                    indices_month = np.argwhere(np.logical_and(
                        dates >= x, dates < y)).flatten()
                    indices_month = [x for x in indices_month if x not in indices_to_rm]
                    if len(indices_month) > 1:
                        if n_removed < n_to_remove:
                            indices_to_rm.append(indices_month[1])
                            print(f"Removing second image in month: {x}, {indices_month[1]}")
                        n_removed += 1

        elif np.max(probs) >= 0.20:
            max_cloud = np.argmax(probs)
            print(f"Removing cloudiest date: {max_cloud}, {probs[max_cloud]}")
            indices_to_rm.append(max_cloud)

        print(f"Removing {len(indices_to_rm)} sunny/cloudy dates")
    return indices_to_rm

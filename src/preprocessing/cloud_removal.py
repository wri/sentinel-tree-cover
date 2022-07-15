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
from scipy import signal
from scipy import ndimage
from scipy.ndimage import label, grey_closing
from scipy.ndimage.morphology import binary_dilation, generate_binary_structure
from scipy.ndimage.filters import gaussian_filter
import warnings
from scipy.ndimage import distance_transform_edt as distance
import bottleneck as bn


def align_interp_array(interp_array, array, interp):

    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    water_mask = _water_ndwi(np.median(array, axis = 0)) > 0.0
    print(np.mean(water_mask), "% water")

    for time in range(array.shape[0]):
        if np.sum(interp[time] > 0) > 0 and np.sum(interp[time] == 0) > 0:
            if np.mean(interp[time] > 0) < 1:
                interp_map = interp[time, ...]
                interp_all = interp_map
                array_i = array[time]
                interp_array_i = interp_array[time]

                # Identify all of the areas that are, and aren't interpolated
                interp_areas = interp_array_i[np.logical_and(interp[time] > 0, water_mask == 0)]
                non_interp_areas = array_i[np.logical_and(interp[time] == 0,  water_mask == 0)]

                # And calculate their means and standard deviation per band
                std_src = bn.nanstd(interp_areas, axis = (0))
                std_ref = bn.nanstd(non_interp_areas, axis = (0))
                mean_src = bn.nanmean(interp_areas, axis = (0))
                mean_ref = bn.nanmean(non_interp_areas, axis = (0))
                std_mult = (std_ref / std_src)

                addition = (mean_ref - (mean_src * (std_mult)))
                interp_array_i[np.logical_and(interp[time] > 0, water_mask == 0)] = (
                        interp_array_i[np.logical_and(interp[time] > 0, water_mask == 0)] * std_mult + addition
                    )

                interp_areas = interp_array_i[np.logical_and(interp[time] > 0, water_mask == 1)]
                non_interp_areas = array_i[np.logical_and(interp[time] == 0,  water_mask == 1)]

                if interp_areas.shape[0] > 200 and non_interp_areas.shape[0] > 618*618*.02:
                # And calculate their means and standard deviation per band
                    std_src = bn.nanstd(interp_areas, axis = (0))
                    std_ref = bn.nanstd(non_interp_areas, axis = (0))
                    mean_src = bn.nanmean(interp_areas, axis = (0))
                    mean_ref = bn.nanmean(non_interp_areas, axis = (0))
                    std_mult = (std_ref / std_src)

                    addition = (mean_ref - (mean_src * (std_mult)))
                    interp_array_i[np.logical_and(interp[time] > 0, water_mask == 1)] = (
                            interp_array_i[np.logical_and(interp[time] > 0, water_mask == 1)] * std_mult + addition
                        )
                
                interp_array[time] = interp_array_i

    for time in range(array.shape[0]):
        if np.mean(interp[time] > 0) == 1:
            candidate_lower = np.max([time - 1, 0])
            candidate_upper = np.min([time + 1, array.shape[0] - 1])
            candidate = np.mean(
                np.array([array[candidate_lower], array[candidate_upper]]), axis = 0
            )
            interp_array[time] = candidate

    return interp_array


def id_areas_to_interp(tiles: np.ndarray,
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
            satisfactory = np.argwhere(np.sum(subs, axis = (1, 2)) < (wsize*wsize*0.05))

            if len(satisfactory) > 0:
                for date in range(0, tiles.shape[0]):
                    if np.sum(subs[date]) >= thresh:
                        areas_interpolated[date, x:x+wsize, y:y+wsize] = 1.
            else:
                areas_interpolated[:, x:x+wsize, y:y+wsize] = 1.

    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
           # areas_interpolated[date] = binary_dilation(areas_interpolated[date], iterations = 4)
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 12] = 12
            blurred = blurred / 12
            blurred = 1 - blurred
            blurred[blurred < 0.1] = 0.
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)
    return areas_interpolated


def calc_cloudfree_medians(arr, interp, water_mask):
    source_refs = np.empty((arr.shape[-1]))
    for i in range(arr.shape[-1]):
        source_refs[i] = np.median(arr[..., i][np.logical_and(interp < 1, water_mask == 0)])
    return source_refs


def make_aligned_mosaic(arr, interp):
    def _ndwi(arr):
        return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])
    
    water_mask = np.median(_ndwi(arr), axis = 0)
    water_mask = water_mask > 0
    water_mask = binary_dilation(1 - water_mask, iterations = 2)
    water_mask = binary_dilation(1 - water_mask, iterations = 5)

    refs = calc_cloudfree_medians(arr, interp, water_mask)
    mosaic = np.zeros((arr.shape[1], arr.shape[2], arr.shape[3]), dtype = np.float32)
    for i in range(arr.shape[0]):
        
        target_refs = calc_cloudfree_medians(arr[i], interp[i], water_mask)
        adj = (refs - target_refs)
        adj = np.clip(adj, -0.05, 0.05)
        arr_i = np.copy(arr[i])
        arr_i[water_mask == 0] = arr_i[water_mask == 0] + adj
        increment = (1 - interp[i][..., np.newaxis]) * arr_i
        mosaic = mosaic + increment
    divisor = (np.sum(1 - interp, axis = 0))[..., np.newaxis]
    mosaic = mosaic / divisor
    mosaic[np.isnan(mosaic)] = np.percentile(arr, 25, axis = 0)[np.isnan(mosaic)]
    mosaic = np.clip(mosaic, 0, np.max(mosaic))
    return mosaic


def calculate_clouds_in_mosaic(mosaic, interp):
    # If there is only 1 availalble image, omission errors are
    # possible due to S2Cloudless and ESA SCL
    # We can assume that areas with > 1 image have no clouds,
    # as well as areas that we would consider false positives.
    # and use the red/blue band distributions of those areas
    # to threshold the non-saturated, non FCP areas with 1 image
    # to make a cloud mask.
    only_1_img = np.sum(1 - (interp > 0), axis = 0).squeeze() < 2
    fcp, _ = detect_pfcp(mosaic[np.newaxis])
    fcp = binary_dilation(fcp, iterations = 10)
    
    only_1_img = np.maximum(only_1_img, fcp.squeeze())
    
    reference_blue = np.percentile(mosaic[..., 0][~only_1_img], 99)
    reference_red = np.percentile(mosaic[..., 2][~only_1_img], 99)
    clouds_in_mosaic = ((mosaic[..., 0] > reference_blue) * \
                        (mosaic[..., 2] > reference_red) * \
                        only_1_img * \
                        (np.sum(mosaic[..., :3], axis = -1) < 1)
                       )

    
    clouds_in_mosaic[fcp.squeeze() > 0] = 0.
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations = 3)
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations = 8)
    return clouds_in_mosaic


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
    areas_interpolated = np.copy(c_probs)

    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 12] = 12
            blurred = (blurred / 12)
            blurred = 1 - blurred
            blurred[blurred < 0.1] = 0.
            blurred = grey_closing(blurred, size = 20)
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)
    interp_array = np.zeros_like(tiles, dtype = np.float32)

    interp_multiplier = (1 - areas_interpolated[..., np.newaxis])
    mosaic = make_aligned_mosaic(tiles, areas_interpolated)

    # There is a failure case here when the only image available is an ommission error cloud.
    # When there is only one image, we rely on 160m S2cloudless + ESA SCL...
    # which can miss tiny clouds, resulting in artifacts in the data. 
    # We attempt to remove these artifacts by thresholding red/blue bands based on cloud-free areas
    # This is a pretty conservative approach, but gets most of the problem areas.
    #np.save("mosaic.npy", mosaic)
    #np.save("interp.npy", areas_interpolated)
    #np.save("tiles.npy", tiles)
    clouds_in_mosaic = calculate_clouds_in_mosaic(mosaic, areas_interpolated.squeeze())
    
    for date in range(0, tiles.shape[0]):
        interp_array[date, areas_interpolated[date] > 0] = mosaic[areas_interpolated[date] > 0]

    interp_array = align_interp_array(interp_array, tiles, areas_interpolated)
    areas_interpolated = areas_interpolated[..., np.newaxis]
    tiles = (tiles * (1 - areas_interpolated) +  \
        (interp_array * areas_interpolated))

    areas_interpolated = areas_interpolated.squeeze()
    areas_interpolated += clouds_in_mosaic[np.newaxis]
    areas_interpolated[areas_interpolated > 1] = 1.
    interp_array = None
    #np.save("after.npy", tiles)
    return tiles, areas_interpolated


def detect_pfcp(arr):
    
    def _ndbi(arr):
        return ((arr[..., 8] - arr[..., 3]) / (arr[..., 8] + arr[..., 3]))
    
    def _ndvi(arr):
        return (arr[..., 3] - arr[..., 2]) / (arr[..., 3] + arr[..., 2])
    
    def _ndwi(arr):
        return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])
    
    pfps = np.logical_and(_ndbi(arr) > 0, _ndbi(arr) > _ndvi(arr))
    water_mask = _ndwi(arr)
    water_mask = np.median(water_mask, axis = 0)[np.newaxis]
    water_mask = water_mask > 0

    pfps = np.logical_and(pfps, water_mask == 0)
    
    cdis = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]), dtype = np.float32)
    
    for time in range(arr.shape[0]):
        
        b8down = np.copy(arr[time, ..., 3])
        if (b8down.shape[0] % 2 + b8down.shape[1] % 2) > 0:
            b8down = resize(b8down, (b8down.shape[0] + b8down.shape[0] % 2, 
                                     b8down.shape[1] + b8down.shape[1] % 2), 0)
        b8down = ndimage.gaussian_filter(b8down, sigma = 0.5, truncate = 3)
        b8down = np.reshape(b8down, (b8down.shape[0] // 2, 2, b8down.shape[1] // 2, 2))
        b8down = np.mean(b8down, axis = (1, 3))
        
        b8adown = np.copy(arr[time, ..., 7])
        if (b8adown.shape[0] % 2 + b8adown.shape[1] % 2) > 0:
            b8adown = resize(b8adown, (b8adown.shape[0] + b8adown.shape[0] % 2, 
                                       b8adown.shape[1] + b8adown.shape[1] % 2), 0)
        b8adown = np.reshape(b8adown, (b8adown.shape[0] // 2, 2, b8adown.shape[1] // 2, 2))
        b8adown = np.mean(b8adown, axis = (1, 3))
        
        b7down = np.copy(arr[time, ..., 6])
        if (b7down.shape[0] % 2 + b7down.shape[1] % 2) > 0:
            b7down = resize(b7down, (b7down.shape[0] + b7down.shape[0] % 2, 
                                           b7down.shape[1] + b7down.shape[1] % 2), 0)
        b7down = np.reshape(b7down, (b7down.shape[0] // 2, 2, b7down.shape[1] // 2, 2))
        b7down = np.mean(b7down, axis = (1, 3))
        
        r8a = b8down / b8adown
        r8a7 = b7down / b8adown

        mean_op = np.ones((7,7))/(7*7)
        mean_of_sq = signal.convolve2d( r8a**2, mean_op, mode='same', boundary='symm')
        sq_of_mean = signal.convolve2d( r8a   , mean_op, mode='same', boundary='symm') **2
        r8a = mean_of_sq - sq_of_mean

        mean_of_sq = signal.convolve2d( r8a7**2, mean_op, mode='same', boundary='symm')
        sq_of_mean = signal.convolve2d( r8a7   , mean_op, mode='same', boundary='symm') **2
        r8a7 = mean_of_sq - sq_of_mean

        cdi = (r8a7 -r8a)/(r8a7 + r8a)
        pfcps = cdi >= -0.7
        pfcps = pfcps.repeat(2, axis = 0).repeat(2, axis = 1)
        pfcps = resize(pfcps, (arr.shape[1], arr.shape[2]), 0)
        cdis[time] = pfcps
        
    cdis = binary_dilation(cdis, iterations = 3)
    pfps = binary_dilation(pfps, iterations = 3)
        
    fcps = (pfps * cdis)
    return fcps, pfps


def remove_missed_clouds(img: np.ndarray) -> np.ndarray:
    """ Removes clouds that may have been missed by s2cloudless
        by looking at a temporal change outside of IQR

        Parameters:
         img (arr):

        Returns:
         to_remove (arr):
    """
    def _water_ndwi(array):
        # For wate rmasking
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])
    
    def _hollstein_cld(arr):
        # Simple cloud detection algorithm
        # Generates "okay" cloud masks that are to be refined
        # This step could probably be improved
        # From Figure 6 in Hollstein et al. 2016
        step1 = arr[..., 7] > 0.156
        step2b = arr[..., 1] > 0.28
        step3 = arr[..., 5] / arr[..., 8] < 4.292
        cl = step1 * step2b * step3# * step4
        for i in range(cl.shape[0]):
            cl[i] = binary_dilation(1 - (binary_dilation(cl[i] == 0, iterations = 2)), iterations = 10)
        return cl

    def _winsum(in_arr, windowsize):
        # Sums pixels in a moving window
        in_arr = np.pad(in_arr, windowsize//2, mode='reflect')
        in_arr[windowsize:] -= in_arr[:-windowsize]
        in_arr[:, windowsize:] -= in_arr[:, :-windowsize]
        return in_arr.cumsum(0)[windowsize-1:].cumsum(1)[:, windowsize-1:]

    water_mask = bn.nanmedian(_water_ndwi(img), axis = 0)
    shadows = np.empty_like(img[..., 0], dtype = np.float32)
    clouds = np.empty_like(shadows, dtype = np.float32)
    
    # Generate a "okay" quality cloud mask
    # and use it to do multi-temporal cloud shadow masking
    # Where the delta B8A and delta B11 are < -0.04 and B2 is < 0.095 over land
    # The water shadow thresholds work -okay- and could be improved
    clm = _hollstein_cld(img)
    for time in range(img.shape[0]):
        lower = np.max([0, time - 2])
        upper = np.min([img.shape[0], time + 3])
        others = np.array([x for x in np.arange(lower, upper)])

        ri_shadow = np.copy(img[..., [0, 1, 7, 8]])
        ri_shadow = ri_shadow[others]
        ri_shadow[clm[others] > 0] = np.nan
        # Failure case if the median non-cloud image is a shadow
        # Non-cloud-shadow 5-window temporall median
        ri_shadow = bn.nanmedian(ri_shadow, axis = 0)
        ri_shadow[np.isnan(ri_shadow)] = np.min(img[..., [0, 1, 7, 8]],
                                                    axis = 0)[np.isnan(ri_shadow)]

        deltab8a = (img[time, ..., 7] - ri_shadow[...,2]) < -0.04
        deltab11 = (img[time, ..., 8] - ri_shadow[..., 3]) < -0.04
        ti0 = (img[time, ..., 0] < 0.095)

        shadows_i = (deltab11 * deltab8a * ti0)
        shadows_i[water_mask > 0] = 0.

        water_shadow = ((img[time, ..., 0] - ri_shadow[..., 0]) < -0.04) * \
                        ((img[time, ..., 1] - ri_shadow[..., 1]) < -0.04) * \
                        (img[time, ..., 7] < 0.03) * \
                        ((ri_shadow[..., 1] - img[time, ..., 1]) > 0.008) * \
                        (water_mask > 0)
        shadows[time] = shadows_i + water_shadow

    # Erode and dilate the shadow mask to remove small shadows
    # and increase boundary detection of shadows
    struct2 = generate_binary_structure(2, 2)
    for i in range(clouds.shape[0]):
        shadows_i = shadows[i]
        shadows_i = binary_dilation(1 - (binary_dilation(shadows_i == 0, iterations = 3)), iterations = 6)
        shadows_i = distance(1 - shadows_i)
        shadows_i[shadows_i <= 5] = 0.
        shadows_i[shadows_i > 5] = 1
        shadows_i = 1 - shadows_i
        shadows[i] = shadows_i
    
    # Use the cloud shadow masks to generate cloud-shadow-free
    # seasonal / local composites. Use these to generate
    # multi-temporal cloud candidate masks
    for time in range(img.shape[0]):
        lower = np.max([0, time - 2])
        upper = np.min([img.shape[0], time + 3])
        
        if (upper - lower) == 3:
            if upper == img.shape[0]:
                lower = np.maximum(lower - 1, 0)
            if lower == 0:
                upper = np.minimum(upper + 1, img.shape[0])
        others = np.array([x for x in np.arange(lower, upper)]) #if x != time])
        close = [np.max([0, time - 1]), np.min([img.shape[0] - 1, time + 1])]
        if close[1] - close[0] < 2:
            if close[0] == 0:
                close[0] += 1
                close[1] += 1
            else:
                close[1] -= 1
                close[0] -= 1
        
        ri_ref  = np.copy(img[..., [0, 1, 2]])
        # Go through and remove the cloud shadows and 
        # generate a darkest visible non-shadow composite
        if img.shape[0] > 2:
            ri_ref[shadows > 0] = np.nan
            ri_upper0 = bn.nanmin(ri_ref[others, ..., 0], axis = 0)
            ri_upper1 = bn.nanmin(ri_ref[others, ..., 1], axis = 0)
            ri_upper2 = bn.nanmin(ri_ref[others, ..., 2], axis = 0)
            nan_replace = np.isnan(ri_upper0)
            ri_upper0[nan_replace] = np.percentile(ri_ref[..., 0], 25, axis = 0)[nan_replace]
            ri_upper1[nan_replace] = np.percentile(ri_ref[..., 1], 25, axis = 0)[nan_replace]
            ri_upper2[nan_replace] = np.percentile(ri_ref[..., 2], 25, axis = 0)[nan_replace]
            ri_close = bn.nanmin(ri_ref[close], axis = 0).astype(np.float32)
            ri_close[np.isnan(ri_close)] = np.min(img[..., :3], axis = 0)[np.isnan(ri_close)]
        else:
            ri_close = np.min(ri_ref[others], axis = 0).astype(np.float32)
            ri_upper0 = ri_close[..., 0]
            ri_upper1 = ri_close[..., 1]
            ri_upper2 = ri_close[..., 2]

        # In tropical broadleaf forests (which tend to have low visible reflectances),
        # small clouds can have red bands < 0.1 reflectance, and < 0.05 delta
        # So the seasonal delta thresholds are scaled based on the reference image as: 
            # reference b2    0 - 0.03 = 0.03
            # reference b2 0.06 - 0.08 = 0.04
            # reference b2 0.08 - 0.10 = 0.05
            # reference b2 0.10 - 0.12 = 0.06
            # reference b2      > 0.12 = 0.07 (this is the 90th percentiel of b2 values)
        # Note that this is not done in Candra et al. 2020 but empirically is useful
        # These thresholds are slightly aggressive and result in some commission errors
        # But don't seem to significantly affect the overall % of data coverage.
        close_thresh = np.minimum(((ri_close[..., 0] / 0.02 / 100) + 0.01), 0.07)
        close_thresh = np.maximum(close_thresh, 0.03)

        deltab2 = (img[time, ..., 0] - ri_upper0) > 0.08
        deltab3 = (img[time, ..., 1] - ri_upper1) > 0.07
        deltab4 = (img[time, ..., 2] - ri_upper2) > 0.07
        
        closeb2 = (img[time, ..., 0] - ri_close[..., 0]) > close_thresh
        closeb3 = (img[time, ..., 1] - ri_close[..., 1]) > close_thresh
        closeb4  = (img[time, ..., 2] - ri_close[..., 2]) > close_thresh
    
        clouds_i = (deltab2 * deltab3 * deltab4)
        clouds_close = (closeb2 * closeb3 * closeb4)
        clouds[time] = np.maximum(clouds_i, clouds_close)

    # Remove urban false positives using b8a, b7, b8 paralax effect
    # and NDBI, NDVI, NDWI
    # This method is from Fmask 4.0
    fcps, pfcps = detect_pfcp(img)
    clouds[fcps > 0] = 0.
    
    # Remove bright surface false positives e.g. sand, rock
    # With a NIR to SWIR1 ratio threshold of < 0.75
    # This threshold is from Fmask 4.0
    nir_swir_ratio = (img[..., 3] / (img[..., 8] + 0.01))
    nir_swir_ratio = nir_swir_ratio < 0.75
    nir_swir_ratio = binary_dilation(nir_swir_ratio, iterations = 3)
    for i in range(img.shape[0]):
        nir_swir_ratio[i][water_mask < 0] = 0.
    clouds[nir_swir_ratio] = 0.

    # Remove false positive clouds over water based on NIR
    # A large dilation is necessary here because of shorelines
    # This threshold is from Fmask 4.0
    for i in range(img.shape[0]):
        clouds_i = clouds[i]
        fp = (water_mask > 0) * (img[i, ..., 8] < 0.11)
        fp = binary_dilation(fp, iterations = 10)
        clouds_i[fp] = 0.
        clouds[i] = clouds_i

    # Finally, exclude where only minority of 3x3 win are clouds
    # As done in Fmask 4.0
    for i in range(clouds.shape[0]):
        window_sum = _winsum(clouds[i],  3)
        clouds[i][window_sum < 5] = 0.
    
    # Dilate the non-urban clouds, erode the urban clouds
    struct2 = generate_binary_structure(2, 2)
    for i in range(clouds.shape[0]):
        pfcps[i] = binary_dilation(pfcps[i], iterations = 5)
        urban_clouds = clouds[i] * pfcps[i]
        urban_clouds = (1 - (binary_dilation(urban_clouds == 0, iterations = 3)))
        urban_clouds = binary_dilation(urban_clouds, iterations = 3)
        
        non_urban_clouds = clouds[i] * (1 - pfcps[i])
        window_sum = _winsum(non_urban_clouds,  3)
        is_large_cloud = np.copy(non_urban_clouds)
        is_large_cloud[window_sum < 6] = 0.
        is_small_cloud = np.copy(non_urban_clouds)
        is_small_cloud[window_sum >= 6] = 0.
        is_large_cloud = binary_dilation(is_large_cloud, iterations = 9)
        non_urban_clouds = np.maximum(is_large_cloud, is_small_cloud)

        #non_urban_clouds = binary_dilation(non_urban_clouds, iterations = 7, structure = struct2)
        non_urban_clouds = distance(1 - non_urban_clouds)
        non_urban_clouds[non_urban_clouds <= 3] = 0.
        non_urban_clouds[non_urban_clouds > 3] = 1
        non_urban_clouds = 1 - non_urban_clouds
        clouds[i] = (non_urban_clouds + urban_clouds)
        
    clouds = np.maximum(clouds, shadows)

    fcps = np.maximum(fcps, nir_swir_ratio)
    fcps = binary_dilation(fcps, iterations = 2)
    return clouds, fcps


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
        month_clouds = probs[month_idx]
        month_clouds = [item for sublist in month_clouds for item in sublist]
        #month_clouds = np.array(month_clouds)
        month_cl_arr = np.array(np.copy(month_clouds))
        if np.sum(month_cl_arr < 0.15) >= 1:
            maxcc = 0.15
        else:
            maxcc = 0.3
        #if np.max(probs)
        to_remove = month_idx[np.argwhere(np.logical_or(month_cl_arr >= maxcc,
                                          np.isnan(month_clouds))).flatten()]
        if len(to_remove) > 1:
            duplicate_steps.extend(to_remove)
        elif len(to_remove) > 0:
            duplicate_steps.append(to_remove)

        month_idx = month_idx[np.argwhere(month_cl_arr < maxcc)]

        month_dates = dates[month_idx]
        month_dates = np.array([item for sublist in month_dates for item in sublist])
        if len(month_dates) > 1:
            to_remove = month_idx[np.argsort(month_dates.flatten())[:-1]]
            if len(to_remove) > 1:
                duplicate_steps.extend(to_remove)
            else:
                duplicate_steps.append(to_remove)
        month_probs = probs[month_idx]
        month_probs = [item for sublist in month_probs for item in sublist]
        month_probs = [np.around(x, 2) for x in month_probs]

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

    if len(dates) > 6:
        # Select the best 2 images per month to start with
        best_two_per_month = []
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()

            month_dates = dates[indices_month]
            month_clouds = probs[indices_month]
            if np.sum(month_clouds < 0.1) >= 1:
                maxcc = 0.1
            elif np.sum(month_clouds < 0.2) >= 1:
                maxcc = 0.2
            else:
                maxcc = 0.3

            month_good_dates = month_dates[month_clouds < maxcc]
            indices_month = indices_month[month_clouds < maxcc]

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
    else:
        best_two_per_month = np.arange(0, len(dates))
    dates_round_2 = dates[best_two_per_month]

    probs_round_2 = probs[best_two_per_month]
    print(dates_round_2, probs_round_2)

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
    remove_next_month = False

    _ = print_dates(dates_round_3, probs_round_3)

    if len(dates_round_3) >= 10:
        delete_max = False
        n_removed = 0
        n_to_remove = len(dates_round_3) - 9
        print(f"There are {len(dates_round_3)} dates and need to remove {n_to_remove}")

        highest_n = np.argpartition(probs_round_3, -n_to_remove)[-n_to_remove:]
        highest_n = [x for x in highest_n if probs_round_3[x] > 0.15]
        date_of_highest_n = dates_round_3[highest_n]
        print(dates)
        print(date_of_highest_n)
        index_to_rm = np.argwhere(np.in1d(dates, date_of_highest_n)).flatten()
        print(index_to_rm)
        print(f"Removing cloudiest dates of {np.array(dates)[index_to_rm]}," 
            f"{date_of_highest_n}, {np.array(probs)[index_to_rm]}")

        indices_to_rm.extend(index_to_rm)
        n_removed += len(index_to_rm)

        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(
                dates >= x, dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [x for x in indices_month if x in monthly_dates]

            #print(f"Need to remove {n_to_remove} dates")
            if (len(indices_month) >= 1) and (len(monthly_dates) >= 10) and (n_removed < n_to_remove):
                print(x, indices_month, n_removed)
                to_remove = [90, 181, 243] #if len(monthly_dates) >= 10 else [90, 243]
                if x in to_remove or remove_next_month:
                    if len(indices_month) > 0:
                        if indices_month[0] not in indices_to_rm: #and probs_round_3[indices_month[0] < 0.15]:
                            indices_to_rm.append(indices_month[0])
                            remove_next_month = False
                            n_removed += 1
                            print(f"Removed {x}, {n_removed}")
                        else:
                            remove_next_month = True
                    else:
                        print("Removing the next month instead")
                        remove_next_month = True if not remove_next_month else False

    return indices_to_rm

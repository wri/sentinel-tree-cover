import numpy as np
import sys

sys.path.append('../')
from src.downloading.utils import calculate_proximal_steps, calculate_proximal_steps_two
from typing import List, Any, Tuple
import rasterio as rs
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
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import LinearRegression

np.seterr(invalid='ignore')



def identify_pifs(src, ref):
    # Canonical correlation analysis is used to identify psuedo-invariant features
    # We assume that phenological or atmospheric differences are linear
    # While land-use change or cloud cover is non-linear
    # We select the points between src and ref that are the most correlated
    # By doing CCA and selecting the pixels where the components are the closest
    cca = CCA(n_components=2, tol=1e-6, max_iter=800)
    xs, ys = cca.fit_transform(src[..., :4], ref[..., :4])
    diffs = xs - ys
    # Mean squared error
    diffs = np.sum((diffs / np.std(diffs, axis=0))**2, axis=1)
    diffs = np.argwhere(diffs < np.percentile(diffs, 10))
    return src[diffs].squeeze(), ref[diffs].squeeze()


def align_interp_array(interp_array,
                       array,
                       date,
                       interp,
                       mosaic,
                       water_mask,
                       linregress=False):
    # Normalizes interpolated areas to non-interpolated areas
    # By learning linear mappings based on pseudo-invariant features
    # And smoothly blending with a gaussian filter

    #def _identify_interp():
    #    non_interp_mosaic, non_interp_areas = identify_pifs(
    #                non_interp_mosaic, non_interp_areas)
    for time in range(1):

        if np.sum(interp[date] > 0) > 0 and np.sum(interp[date] == 0) > 0:
            if np.mean(np.logical_and(interp[date] < 1, water_mask == 0)) > 0.01:
                interp_map = interp[date, ...]
                interp_all = interp_map
                array_i = array[date]
                interp_array_i = interp_array[time]

                # Identify all of the areas that are, and aren't interpolated
                # interp_areas = interp_array_i[np.logical_and(interp[time] > 0, water_mask == 0)]
                non_interp_areas = []
                non_interp_mosaic = []

                min_time = np.maximum(date - 1, 0)
                max_time = np.minimum(date + 2, array.shape[0])
                #if np.mean(interp[date]) < 0.95:
                #    min_time = date
                #    max_time = date + 1
                n_current_time = np.sum(np.logical_and(
                        interp[date] < 1, water_mask == 0))
                for t in range(min_time, max_time):
                    non_interp_areasi = array[t][np.logical_and(
                        interp[t] < 1, water_mask == 0)][:n_current_time]
                    non_interp_mosaici = mosaic[np.logical_and(
                        interp[t] < 1, water_mask == 0)][:n_current_time]
                    #print(non_interp_areasi.shape, non_interp_mosaici.shape)
                    non_interp_areas.append(non_interp_areasi)
                    non_interp_mosaic.append(non_interp_mosaici)

                idx = 1 if non_interp_mosaic[1].shape[0] > 0 else 0
                non_interp_mid_mosaic = non_interp_mosaic[idx]
                non_interp_mid_areas = non_interp_areas[idx]
                non_interp_mid_mosaic, non_interp_mid_areas = identify_pifs(
                    non_interp_mid_mosaic, non_interp_mid_areas)

                non_interp_mosaic = np.concatenate(non_interp_mosaic, axis = 0)
                non_interp_areas = np.concatenate(non_interp_areas, axis = 0)

                non_interp_mosaic, non_interp_areas = identify_pifs(
                    non_interp_mosaic, non_interp_areas)


                if linregress:
                    # Learn a linear mapping with OLS. Empirically thihs does not seem to work better
                    # Than a simple mean / std deviation shift (below), which is the default
                    std_mult = np.ones((1, 1, 10))
                    addition = np.zeros((1, 1, 10))
                    for i in range(10):
                        model = LinearRegression().fit(
                            non_interp_mosaic[..., i][..., np.newaxis],
                            non_interp_areas[..., i][..., np.newaxis])
                        std_mult[..., i] = model.coef_
                        addition[
                            ...,
                            i] = model.intercept_  #(model.intercept_, model.coef_)
                else:
                    # And calculate their means and standard deviation per band
                    # First calculate the time difference between non-interpolated areas
                    # (non_interp_mosaic to non_interp_areas) -> (interp_mosaic -> interp_areas)
                    std_src = bn.nanstd(non_interp_mosaic, axis=(0))
                    std_ref = bn.nanstd(non_interp_areas, axis=(0))
                    mean_src = bn.nanmean(non_interp_mosaic, axis=(0))
                    mean_ref = bn.nanmean(non_interp_areas, axis=(0))
                    std_mult = (std_ref / std_src)

                    # Then calculate the mosaic diff between non interp and interp
                    addition = (mean_ref - (mean_src * (std_mult)))
                    addition = np.reshape(addition, (1, 1, 10))
                    std_mult = np.reshape(std_mult, (1, 1, 10))
                    #std_mult = np.clip(std_mult, 0.75, 1.25)
                    #addition = np.clip(addition, -0.05, 0.05)

                    # MIDDLE
                    std_src = bn.nanstd(non_interp_mid_mosaic, axis=(0))
                    std_ref = bn.nanstd(non_interp_mid_areas, axis=(0))
                    mean_src = bn.nanmean(non_interp_mid_mosaic, axis=(0))
                    mean_ref = bn.nanmean(non_interp_mid_areas, axis=(0))
                    std_mult_mid = (std_ref / std_src)
                    #! TODO we can assume that the mosaic will not have haze
                    # so if the mosaic is >2.5 the stdev of the source
                    # and the source is >20% cloudy, remove the source
                    # only let 1 image be removed.
                    #print(time, std_mult_mid)

                    # Then calculate the mosaic diff between non interp and interp
                    addition_mid = (mean_ref - (mean_src * (std_mult_mid)))
                    addition_mid = np.reshape(addition_mid, (1, 1, 10))
                    std_mult_mid = np.reshape(std_mult_mid, (1, 1, 10))

                    # As the reference image gets more clear (towards 50%), 
                    # weight the reference image more than the 3-image window
                    # 
                    multiplier = np.minimum((n_current_time) / (600*600*0.5), 1.)
                    print(f"The multiplier is {multiplier} and the clean is {n_current_time} for {date}")
                    addition = (addition * (1 - multiplier)) + (addition_mid * multiplier)
                    std_mult = (std_mult * (1 - multiplier)) + (std_mult_mid * multiplier)

                interp_array_i[np.logical_and(
                    interp[date] > 0,
                    water_mask == 0)] = ((interp_array_i[np.logical_and(
                        interp[date] > 0, water_mask == 0)] * std_mult) +
                                         addition)

                # Make sure that this normalization doesnt change the original range of the data.
                #for i in range(interp_array_i.shape[-1]):
                #    interp_array_i[..., i] = np.clip(interp_array_i[..., i],
                #                                     np.min(array_i[..., i]),
                #                                     np.max(array_i[..., i]))

                # Normalization for water areas is done separately since the spectral
                # Reflectances are so different than on land.
                interp_areas = interp_array_i[np.logical_and(
                    interp[date] > 0, water_mask == 1)]
                non_interp_areas = array_i[np.logical_and(
                    interp[date] == 0, water_mask == 1)]

                if interp_areas.shape[0] > 200 and non_interp_areas.shape[
                        0] > 618 * 618 * .02:
                    std_src = bn.nanstd(interp_areas, axis=(0))
                    std_ref = bn.nanstd(non_interp_areas, axis=(0))
                    mean_src = bn.nanmean(interp_areas, axis=(0))
                    mean_ref = bn.nanmean(non_interp_areas, axis=(0))
                    std_mult = (std_ref / std_src)

                    addition = (mean_ref - (mean_src * (std_mult)))
                    interp_array_i[np.logical_and(
                        interp[date] > 0,
                        water_mask == 1)] = (interp_array_i[np.logical_and(
                            interp[date] > 0, water_mask == 1)] * std_mult +
                                             addition)

                interp_array[time] = interp_array_i
    return interp_array


def make_aligned_mosaic(arr, interp):

    def _ndwi(arr):
        return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])

    water_mask = np.median(_ndwi(arr), axis=0)
    water_mask = water_mask > 0
    water_mask = binary_dilation(1 - water_mask, iterations=2)
    water_mask = binary_dilation(1 - water_mask, iterations=5)

    non_interp_areas = arr[np.logical_and(interp == 0, water_mask == 0)]

    mosaic = np.zeros((arr.shape[1], arr.shape[2], arr.shape[3]),
                      dtype=np.float32)
    divisor = (np.sum(1 - interp, axis=0))[..., np.newaxis]
    brightness_mask = bn.nanmax(np.sum(arr[..., :3], axis = -1), axis = 0)
    darkness_mask = bn.nanmin(np.sum(arr[..., :3], axis = -1), axis = 0)
    #brightness_mask[interp > 0] == np.nan
    #median_brightness = np.nanmedian(brightness_mask)
    for i in range(arr.shape[0]):

        # All areas within one image that are not interpolated
        non_interp_mosaic = arr[i][np.logical_and(interp[i] < 1,
                                                    water_mask == 0)]
        non_interp_areas = np.full((arr.shape[1], arr.shape[2], arr.shape[3]), 0, dtype = np.float32)
        non_interp_count = np.full((arr.shape[1], arr.shape[2], arr.shape[3]), 0, dtype = np.float32)

        for b in range(arr.shape[0]):
            if b != i:
                mask = np.logical_and(np.logical_and(interp[i] < 1, interp[b] < 1),
                                            water_mask == 0)
                arr_b = arr[b]
                non_interp_areas[mask] += arr_b[mask]
                non_interp_count[mask] += 1
        non_interp_areas = non_interp_areas / non_interp_count
        non_interp_areas = np.reshape(non_interp_areas, (non_interp_areas.shape[0] * non_interp_areas.shape[1], 10))
        non_interp_areas = non_interp_areas[~np.isnan(non_interp_areas).any(axis=1)]

        if non_interp_mosaic.shape[0] > 1000 and non_interp_areas.shape[0] > 1000:
            non_interp_mosaic = non_interp_mosaic[:non_interp_areas.shape[0]]
            non_interp_areas = non_interp_areas[:non_interp_mosaic.shape[0]]
            non_interp_mosaic, non_interp_areas_i = identify_pifs(
                        non_interp_mosaic, non_interp_areas)

            mean_ref = bn.nanmedian(non_interp_areas, axis=0)
            std_ref = bn.nanstd(non_interp_areas, axis=0)

            mean_src = bn.nanmedian(non_interp_mosaic, axis=0)
            std_src = bn.nanstd(non_interp_mosaic, axis=0)
            std_mult = (std_ref / std_src)
            addition = (mean_ref - (mean_src * (std_mult)))
            arr_i = np.copy(arr[i])
            arr_i[water_mask == 0] = arr_i[water_mask == 0] * std_mult + addition
            increment = (1 - interp[i][..., np.newaxis]) * arr_i
            # We want to not include pixels that are the brightest or the least bright
            # if there are >3 images
            #areas_to_subset = divisor >= 3
            #is_max_brightness = np.sum(arr[i, ..., :3], axis = -1) == brightness_mask
            #is_min_brightness = np.sum(arr[i, ..., :3], axis = -1) == darkness_mask
            #areas_to_subset = areas_to_subset.squeeze() * (is_min_brightness + is_max_brightness)
            #print(i, np.mean(areas_to_subset))
            #increment[areas_to_subset] = 0.
            #divisor[areas_to_subset] -= 1.
            mosaic = mosaic + increment
            
        else:
            interp[i] = 1.
    divisor[divisor < 0] = 0.
    mosaic = mosaic / divisor
    mosaic[np.isnan(mosaic)] = np.percentile(arr, 10, axis=0)[np.isnan(mosaic)]
    mosaic = np.clip(mosaic, 0, np.max(mosaic))
    np.save("mosaic.npy", mosaic)
    return mosaic



def calculate_clouds_in_mosaic(mosaic, interp, pfcps):
    # If there is only 1 availalble image, omission errors are
    # possible due to S2Cloudless and ESA SCL
    # We can assume that areas with > 1 image have no clouds,
    # as well as areas that we would consider false positives.
    # and use the red/blue band distributions of those areas
    # to threshold the non-saturated, non FCP areas with 1 image
    # to make a cloud mask.
    only_1_img = np.sum(1 - (interp > 0), axis=0).squeeze() < 2

    if len(pfcps.shape) == 3 and pfcps.shape[0] > 1:
        pfcps = pfcps[0]

    pfcps = binary_dilation(pfcps, iterations=10)

    only_1_img = np.maximum(only_1_img, pfcps.squeeze())

    reference_blue = np.percentile(mosaic[..., 0][~only_1_img], 99)
    reference_red = np.percentile(mosaic[..., 2][~only_1_img], 99)
    clouds_in_mosaic = ((mosaic[..., 0] > reference_blue) * \
                        (mosaic[..., 2] > reference_red) * \
                        only_1_img * \
                        (np.sum(mosaic[..., :3], axis = -1) < 1)
                       )

    clouds_in_mosaic[pfcps.squeeze() > 0] = 0.
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations=3)
    clouds_in_mosaic = binary_dilation(1 - clouds_in_mosaic, iterations=8)
    return clouds_in_mosaic


def mask_nonurban_areas(file, bbx, pfcps):
    # Use the ESA WorldCover (160m resample) to identify
    # Where urban areas are, to enable removal of urban
    # false-positive clouds

    with rs.open(file) as data:
        rst = data.read(1,
                        window=rs.windows.from_bounds(bbx[0], bbx[1], bbx[2],
                                                      bbx[3], data.transform))
        # 160m mask of majority urban-px in ESA Worldcover
        rst = binary_dilation(rst, iterations=1)
        rst_original = np.copy(rst)
        rst_original = resize(rst, pfcps.shape, 0)
        pfcps[rst_original == 1] = 1.

        # Remove false positives >1km from an urban px
        rst = binary_dilation(rst, iterations=5)
        rst = resize(rst, pfcps.shape, 0)
        pfcps[rst == 0] = 0.
        print(f"Tile is {np.mean(rst == 1) * 100}% urban")
        return pfcps


def adjust_cloudmask_in_forests(file, bbx, pfcps):
    # Use the ESA WorldCover (160m resample) to identify
    # Where urban areas are, to enable removal of urban
    # false-positive clouds

    with rs.open(file) as data:
        rst = data.read(1,
                        window=rs.windows.from_bounds(bbx[0], bbx[1], bbx[2],
                                                      bbx[3], data.transform))
        # 160m mask of majority urban-px in ESA Worldcover
        rst = binary_dilation(rst, iterations=2)
        rst_original = np.copy(rst)
        rst_original = resize(rst, pfcps.shape, 0)
        return rst_original


def id_areas_to_interp(tiles: np.ndarray, probs: np.ndarray,
                       shadows: np.ndarray, image_dates: List[int],
                       pfcps) -> np.ndarray:
    """ Identifies the areas that will be interpolated by 
    remove_cloud_and_shadows. This is used to remove images
    that are >95% interpolated before actually executinig
    the interpolation
    """
    areas_interpolated = np.copy(probs)
    areas_interpolated = areas_interpolated.astype(np.float32)
    
    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 14] = 14
            blurred = (blurred / 14)
            blurred = 1 - blurred
            blurred[blurred < 0.1] = 0.
            blurred = grey_closing(blurred, size=20)
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)

    return areas_interpolated


def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray,
                             shadows: np.ndarray,
                             image_dates: List[int],
                             pfcps,
                             wsize: int = 36,
                             step=8,
                             thresh=100) -> np.ndarray:
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

    areas_interpolated = np.copy(probs)
    areas_interpolated = areas_interpolated.astype(np.float32)
    
    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 16] = 16
            blurred = (blurred / 16)
            blurred = 1 - blurred
            blurred[blurred < 0.1] = 0.
            blurred = grey_closing(blurred, size=20)
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)
    
    mosaic = make_aligned_mosaic(tiles, areas_interpolated)

    def _water_ndwi(array):
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    water_mask = _water_ndwi(np.median(tiles, axis=0)) > 0.0
    to_remove = []
    for date in range(0, tiles.shape[0]):
        interp_array = np.zeros_like(tiles[date])
        interp_multiplier = (1 - areas_interpolated[date, ..., np.newaxis])
        interp_array[areas_interpolated[date] > 0] = mosaic[
            areas_interpolated[date] > 0]
        interp_array = align_interp_array(interp_array[np.newaxis],
                                          tiles,#tiles[date][np.newaxis],
                                          date,
                                          areas_interpolated,#[date][np.newaxis],
                                          mosaic, water_mask)
        tiles[date] = (tiles[date] * (1 - areas_interpolated[date][..., np.newaxis]) +  \
                      (interp_array * areas_interpolated[date][..., np.newaxis]))
        if np.mean(areas_interpolated[date] == 1) == 1:
            to_remove.append(date)

    areas_interpolated = areas_interpolated[..., np.newaxis]
    interp_array = None
    areas_interpolated = areas_interpolated.squeeze()
    clouds_in_mosaic = calculate_clouds_in_mosaic(mosaic,
                                                  areas_interpolated.squeeze(),
                                                  pfcps)
    areas_interpolated += clouds_in_mosaic[np.newaxis]
    areas_interpolated[areas_interpolated > 1] = 1.

    #p.save("right_interp.npy", areas_interpolated)
    return tiles, areas_interpolated, to_remove


def make_cloudfree_composite(arr, interp, time):

    #! TODO: Incorporate water mask

    arr_i = arr[time]
    mosaic = np.copy(arr[time])
    interp_mosaic = np.copy(interp)

    # Iterate through each band and timestep and make a cloud-free
    # Composite and normalize it to the reference time
    # first, generate the ref, src for each time.
    std_ref = np.zeros((arr.shape[0], arr.shape[-1]))
    std_src = np.zeros((arr.shape[0], arr.shape[-1]))
    mean_ref = np.zeros((arr.shape[0], arr.shape[-1]))
    mean_src = np.zeros((arr.shape[0], arr.shape[-1]))

    for i in range(arr.shape[0]):
        arr_band_i = arr[time]
        increment = np.copy(arr[i])
        ref = arr_band_i[np.logical_and(interp[time] < 1, interp[i] < 1)]
        src = increment[np.logical_and(interp[time] < 1, interp[i] < 1)]
        if src.shape[0] > 2000 and ref.shape[0] > 2000:
            ref = arr[time][np.logical_and(interp[time] < 1, interp[i] < 1)]
            src = arr[i][np.logical_and(interp[time] < 1, interp[i] < 1)]
            ref, src = identify_pifs(ref, src)
            std_src[i, :] = bn.nanstd(src, axis=(0))
            std_ref[i, :] = bn.nanstd(ref, axis=(0))
            mean_src[i, :] = bn.nanmean(src, axis=(0))
            mean_ref[i, :] = bn.nanmean(ref, axis=(0))

    for band in range(arr.shape[-1]):
        arr_band = arr[..., band]

        normalized = np.full_like(arr_band, np.nan)
        for i in range(arr.shape[0]):
            increment = np.copy(arr_band[i])
            arr_band_i = arr_band[time]

            ref = arr_band_i[np.logical_and(interp[time] < 1, interp[i] < 1)]
            src = increment[np.logical_and(interp[time] < 1, interp[i] < 1)]
            if src.shape[0] > 2000 and ref.shape[0] > 2000:
                # For the non-interpolated areas in BOTH the src and ref
                # Calculate the difference in mean/std
                # And apply that to make the src more similar to thhe ref
                #print(ref.shape, src.shape)
                #ref = arr[time][np.logical_and(interp[time] < 1, interp[i] < 1)]
                #src = arr[i][np.logical_and(interp[time] < 1, interp[i] < 1)]
                #ref, src = identify_pifs(ref, src)
                std_src_i = std_src[i, band]#bn.nanstd(src, axis=(0))
                std_ref_i = std_ref[i, band]#bn.nanstd(ref, axis=(0))
                mean_src_i = mean_src[i, band]# bn.nanmean(src, axis=(0))
                mean_ref_i = mean_ref[i, band]# bn.nanmean(ref, axis=(0))

                std_mult = (std_ref_i / std_src_i)
                addition = (mean_ref_i - (mean_src_i * (std_mult)))
                increment[interp[i] < 1] = increment[interp[i] < 1] * std_mult + addition
                normalized[i][interp[i] < 1] = increment[interp[i] < 1]
            else:
                # If we can't standardize the image using direct overlap
                # Then we just have to use the SEPARATE non-interpolated areas
                # Otherwise we'd be introducing NAN values to the mosaic
                ref = arr_band_i[interp[i] < 1]
                src = increment[interp[i] < 1]
                #ref, src = identify_pifs(ref, src)

                std_src_i = bn.nanstd(src, axis=(0))
                std_ref_i = bn.nanstd(ref, axis=(0))
                mean_src_i = bn.nanmean(src, axis=(0))
                mean_ref_i = bn.nanmean(ref, axis=(0))

                std_mult = (std_ref_i / std_src_i)
                addition = (mean_ref_i - (mean_src_i * (std_mult)))
                increment[interp[i] <
                          1] = increment[interp[i] < 1] * std_mult + addition
                normalized[i][interp[i] < 1] = increment[interp[i] < 1]

        # NANs should only persist where there were no non-interp areas to begin w/
        # And these will be picked up by the `interp` mask
        normalized_med = bn.nanmedian(normalized, axis=0)
        normalized_med[np.isnan(normalized_med)] = np.percentile(
            arr[..., band], 25, axis=0)[np.isnan(normalized_med)]
        mosaic[..., band] = normalized_med
    return mosaic

"""
def remove_cloud_and_shadows(tiles: np.ndarray,
                             probs: np.ndarray,
                             shadows: np.ndarray,
                             image_dates: List[int],
                             pfcps,
                             wsize: int = 36, step = 8, thresh = 100) -> np.ndarray:
    
    areas_interpolated = np.copy(probs)
    areas_interpolated = areas_interpolated.astype(np.float32)

    for date in range(areas_interpolated.shape[0]):
        if np.sum(areas_interpolated[date]) > 0:
            blurred = distance(1 - areas_interpolated[date])
            blurred[blurred > 15] = 15
            blurred = (blurred / 15)
            blurred = 1 - blurred
            blurred[blurred < 0.1] = 0.
            blurred = grey_closing(blurred, size = 20)
            areas_interpolated[date] = blurred

    areas_interpolated = areas_interpolated.astype(np.float32)
    for date in range(tiles.shape[0]):
        mosaic = make_cloudfree_composite(tiles, areas_interpolated, date)
        #areas_interpolated[nans[np.newaxis]] = 1.
        interp_array = np.zeros_like(tiles[date], dtype = np.float32)
        interp_multiplier = (1 - areas_interpolated[date, ..., np.newaxis])
        interp_array[areas_interpolated[date] > 0] = mosaic[areas_interpolated[date] > 0]
        #tiles[date] = mosaic
        tiles[date] = (tiles[date] * (interp_multiplier) +  \
            (interp_array * (1 - interp_multiplier)))
        #areas_interpolated[np.isnan()]

    #np.save("after.npy", tiles)


    #interp_array = None
    #areas_interpolated = areas_interpolated.squeeze()
    #clouds_in_mosaic = calculate_clouds_in_mosaic(mosaic, areas_interpolated.squeeze(), pfcps)
    #areas_interpolated += clouds_in_mosaic[np.newaxis]
    #areas_interpolated[areas_interpolated > 1] = 1.
    
    #np.save("after.npy", tiles)
    #areas_interpolated = areas_interpolated[..., np.newaxis]
    return tiles, areas_interpolated
"""

def detect_pfcp(arr, dem, bbx):
    # Detects potential false cloud pixels
    # By using the paralax effect of the NIR bands as
    # done in Fmask 4.0

    def _ndbi(arr):
        return ((arr[..., 8] - arr[..., 3]) / (arr[..., 8] + arr[..., 3]))

    def _ndvi(arr):
        return (arr[..., 3] - arr[..., 2]) / (arr[..., 3] + arr[..., 2])

    def _ndwi(arr):
        return (arr[..., 1] - arr[..., 3]) / (arr[..., 1] + arr[..., 3])

    ndvi = _ndvi(arr)
    ndwi = _ndwi(arr)
    ndbi = _ndbi(arr)
    ndwi = np.median(_ndwi(arr), axis=0)

    pfps = np.logical_and(ndbi > 0, ndbi > ndvi)
    pfps = np.median(pfps, axis=0)
    pfps = pfps * (ndwi < 0)
    try:
        pfps = mask_nonurban_areas("urbanmask.tif", bbx, pfps)
    except:
        print("Problem making PFPCPS, likely outside of tropics. continuing.")
        pfps = np.zeros_like(dem)
    pfps[(dem / 90) > 0.10] = 0.
    pfps = pfps[np.newaxis]

    water_mask = ndwi[np.newaxis]
    water_mask = water_mask > 0
    pfps = np.tile(pfps, (arr.shape[0], 1, 1))

    # Paralax
    cdis = np.zeros((arr.shape[0], arr.shape[1], arr.shape[2]),
                    dtype=np.float32)
    for time in range(arr.shape[0]):

        b8down = np.copy(arr[time, ..., 3])
        if (b8down.shape[0] % 2 + b8down.shape[1] % 2) > 0:
            b8down = resize(b8down, (b8down.shape[0] + b8down.shape[0] % 2,
                                     b8down.shape[1] + b8down.shape[1] % 2), 0)
        b8down = ndimage.gaussian_filter(b8down, sigma=0.5, truncate=3)
        b8down = np.reshape(b8down,
                            (b8down.shape[0] // 2, 2, b8down.shape[1] // 2, 2))
        b8down = np.mean(b8down, axis=(1, 3))

        b8adown = np.copy(arr[time, ..., 7])
        if (b8adown.shape[0] % 2 + b8adown.shape[1] % 2) > 0:
            b8adown = resize(b8adown, (b8adown.shape[0] + b8adown.shape[0] % 2,
                                       b8adown.shape[1] + b8adown.shape[1] % 2),
                             0)
        b8adown = np.reshape(
            b8adown, (b8adown.shape[0] // 2, 2, b8adown.shape[1] // 2, 2))
        b8adown = np.mean(b8adown, axis=(1, 3))

        b7down = np.copy(arr[time, ..., 6])
        if (b7down.shape[0] % 2 + b7down.shape[1] % 2) > 0:
            b7down = resize(b7down, (b7down.shape[0] + b7down.shape[0] % 2,
                                     b7down.shape[1] + b7down.shape[1] % 2), 0)
        b7down = np.reshape(b7down,
                            (b7down.shape[0] // 2, 2, b7down.shape[1] // 2, 2))
        b7down = np.mean(b7down, axis=(1, 3))

        r8a = b8down / b8adown
        r8a7 = b7down / b8adown

        mean_op = np.ones((7, 7)) / (7 * 7)
        mean_of_sq = signal.convolve2d(r8a**2,
                                       mean_op,
                                       mode='same',
                                       boundary='symm')
        sq_of_mean = signal.convolve2d(r8a,
                                       mean_op,
                                       mode='same',
                                       boundary='symm')**2
        r8a = mean_of_sq - sq_of_mean

        mean_of_sq = signal.convolve2d(r8a7**2,
                                       mean_op,
                                       mode='same',
                                       boundary='symm')
        sq_of_mean = signal.convolve2d(r8a7,
                                       mean_op,
                                       mode='same',
                                       boundary='symm')**2
        r8a7 = mean_of_sq - sq_of_mean

        cdi = (r8a7 - r8a) / (r8a7 + r8a)
        pfcps = (cdi >= -0.5)
        pfcps = pfcps.repeat(2, axis=0).repeat(2, axis=1)
        pfcps = resize(pfcps, (arr.shape[1], arr.shape[2]), 0)
        pfcps = pfcps * (_ndvi(arr[time]) < 0.4)
        cdis[time] = pfcps

    struct2 = generate_binary_structure(2, 2)
    for i in range(cdis.shape[0]):
        cdis[i] = binary_dilation(cdis[i], iterations=6, structure=struct2)
        pfps[i] = binary_dilation(pfps[i], iterations=6, structure=struct2)

    fcps = (pfps * cdis)
    print("False positive clouds", np.mean(fcps, axis=(1, 2)))
    return fcps, pfps


def identify_clouds_shadows(img: np.ndarray, dem, bbx) -> np.ndarray:
    """ Removes clouds that may have been missed by s2cloudless
        by looking at a temporal change outside of IQR

        Parameters:
         img (arr):

        Returns:
         to_remove (arr):
    """
    def _water_ndwi(array):
        # For water masking
        return (array[..., 1] - array[..., 3]) / (array[..., 1] + array[..., 3])

    def _hollstein_cld(arr):
        # Simple cloud detection algorithm
        # Generates "okay" cloud masks that are to be refined
        # From Figure 6 in Hollstein et al. 2016
        step1 = arr[..., 7] > 0.166
        step2b = arr[..., 1] > 0.28
        step3 = arr[..., 5] / arr[..., 8] < 4.292
        cl = step1 * step2b * step3
        for i in range(cl.shape[0]):
            cl[i] = binary_dilation(1 -
                                    (binary_dilation(cl[i] == 0, iterations=2)),
                                    iterations=10)
        return cl

    def _winsum(in_arr, windowsize):
        # Sums pixels in a moving window
        in_arr = np.pad(in_arr, windowsize // 2, mode='reflect')
        in_arr[windowsize:] -= in_arr[:-windowsize]
        in_arr[:, windowsize:] -= in_arr[:, :-windowsize]
        return in_arr.cumsum(0)[windowsize - 1:].cumsum(1)[:, windowsize - 1:]

    water_mask = bn.nanmedian(_water_ndwi(img), axis=0)
    shadows = np.zeros_like(img[..., 0], dtype=np.float32)
    clouds = np.zeros_like(shadows, dtype=np.float32)
    try:
        forest_mask = adjust_cloudmask_in_forests("forestmask.tif", bbx, dem)
    except:
        forest_mask = np.zeros_like(dem)
    print(f"{np.mean(forest_mask) * 100}% of the tile is forest in ESA WorldCover")

    # Generate a "okay" quality cloud mask
    # and use it to do multi-temporal cloud shadow masking
    # Where the delta B8A and delta B11 are < -0.04 and B2 is < 0.095 over land
    # The water shadow thresholds work -okay- and could be improved
    clm = _hollstein_cld(img)
    print("Slope shadow is turned off")
    for time in range(img.shape[0]):
        lower = np.max([0, time - 3])
        upper = np.min([img.shape[0], time + 3])
        if (upper - lower) == 3:
            if upper == img.shape[0]:
                lower = np.maximum(lower - 1, 0)
            if lower == 0:
                upper = np.minimum(upper + 1, img.shape[0])
        others = np.array([x for x in np.arange(lower, upper)])
        ri_shadow = np.copy(img[..., [0, 1, 7, 8]])
        ri_shadow = ri_shadow[others]
        ri_shadow[clm[others] > 0] = np.nan
        # Failure case if the median non-cloud image is a shadow
        # Non-cloud-shadow 5-window temporall median
        ri_shadow = bn.nanmedian(ri_shadow, axis=0)
        ri_shadow[np.isnan(ri_shadow)] = np.min(img[..., [0, 1, 7, 8]],
                                                axis=0)[np.isnan(ri_shadow)]

        deltab8a = (img[time, ..., 7] - ri_shadow[..., 2]) < -0.042
        deltab11 = (img[time, ..., 8] - ri_shadow[..., 3]) < -0.042
        ti0 = (img[time, ..., 0] < 0.09)
        deltablue = (img[time, ..., 0] - ri_shadow[..., 0]) < -0.02

        shadows_i = (deltab11 * deltab8a * ti0 * deltablue * (img[time, ..., 7] < 0.162))
        shadows_i[water_mask > 0] = 0.

        # Where slope > 0.25
        ri_shadow = np.copy(img[..., [0, 1, 7, 8]])
        ri_shadow[clm > 0] = np.nan
        ri_shadow = bn.nanmedian(ri_shadow, axis=0)
        ri_shadow[np.isnan(ri_shadow)] = np.median(img[..., [0, 1, 7, 8]],
                                                axis=0)[np.isnan(ri_shadow)]

        deltab8a = (img[time, ..., 7] - ri_shadow[..., 2]) < -0.02
        deltab11 = (img[time, ..., 8] - ri_shadow[..., 3]) < -0.02
        ti0 = (img[time, ..., 0] < 0.09)
        deltablue = (img[time, ..., 0] - ri_shadow[..., 0]) < 0

        shadows_slope = (deltab11 * deltab8a * ti0 * deltablue * (img[time, ..., 7] < 0.18))
        shadows_slope[water_mask > 0] = 0.
        shadows_slope = shadows_slope * (dem >= 0.25)
        
        #shadows_i = np.maximum(shadows_i, shadows_slope)

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
        shadows_i = binary_dilation(
            1 - (binary_dilation(shadows_i == 0, iterations=2)), iterations=3)
        shadows_i = distance(1 - shadows_i)
        shadows_i[shadows_i <= 4] = 0.
        shadows_i[shadows_i > 4] = 1
        shadows_i = 1 - shadows_i
        shadows[i] = shadows_i
    np.save("shadows.npy", shadows)

    # Use the cloud shadow masks to generate cloud-shadow-free
    # seasonal / local composites. Use these to generate
    # multi-temporal cloud candidate masks
    for time in range(img.shape[0]):
        lower = np.max([0, time - 2])
        upper = np.min([img.shape[0], time + 3])

        if (upper - lower) == 3:
            if upper == img.shape[0]:
                lower = np.maximum(lower - 2, 0)
            if lower == 0:
                upper = np.minimum(upper + 2, img.shape[0])
        others = np.array([x for x in np.arange(lower, upper)])
        close = [np.max([0, time - 1]), np.min([img.shape[0] - 1, time + 1])]
        if close[1] - close[0] < 2:
            if close[0] == 0:
                close[0] += 1
                close[1] += 1
            else:
                close[1] -= 1
                close[0] -= 1
        if len(close) == 2:
            #print(time, close[-1], img.shape[0] - 1)
            if close[-1] >= (img.shape[0] - 2) and img.shape[0] > 3:
                close = np.concatenate([np.array([close[0] - 1]), close])

        ri_ref = np.copy(img[..., [0, 1, 2]])
        # Go through and remove the cloud shadows and
        # generate a darkest visible non-shadow composite
        if img.shape[0] > 2:
            ri_ref[shadows > 0] = np.nan
            ri_upper0 = bn.nanmin(ri_ref[others, ..., 0], axis=0)
            ri_upper1 = bn.nanmin(ri_ref[others, ..., 1], axis=0)
            ri_upper2 = bn.nanmin(ri_ref[others, ..., 2], axis=0)
            nan_replace = np.isnan(ri_upper0)
            ri_upper0[nan_replace] = np.percentile(img[..., 0], 25,
                                                   axis=0)[nan_replace]
            ri_upper1[nan_replace] = np.percentile(img[..., 1], 25,
                                                   axis=0)[nan_replace]
            ri_upper2[nan_replace] = np.percentile(img[..., 2], 25,
                                                   axis=0)[nan_replace]
            ri_close = bn.nanmin(ri_ref[close], axis=0).astype(np.float32)

            min_i = close[0]
            max_i = close[-1]
            
            for iteration in range(10):
                if np.sum(np.isnan(ri_close) > 0):
                    min_i = np.maximum(min_i - 1, 0)
                    max_i = np.minimum(max_i + 1, img.shape[0])
                    close = np.array([x for x in np.arange(min_i, max_i) if x != time])
                    ri_close_new = bn.nanmin(ri_ref[close], axis=0).astype(np.float32)
                    ri_close[np.isnan(ri_close)] = ri_close_new[np.isnan(ri_close)]

            if np.sum(np.isnan(ri_close) > 0):
                ri_close[np.isnan(ri_close)] = np.min(img[..., :3], axis = 0)[np.isnan(ri_close)]
        else:
            ri_close = np.min(ri_ref, axis=0).astype(np.float32)
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
        close_thresh = np.minimum(((ri_close[..., 0] / 0.02 / 100) + 0.0025),
                                  0.10)
        close_thresh = np.maximum(close_thresh, 0.04)
        close_thresh[forest_mask == 1] -= 0.02
        close_thresh = np.maximum(close_thresh, 0.03)

        clouds_mean = 0.
        clouds_close_mean = 1.
        close_modifier = 0.
        # We only want the clouds close to increase cloudiness by a max of 15%
        # The idea here is that the "clouds close" identifies small clouds that are missed
        # And shouldn't be identifying absolutely massive clousd
        while clouds_close_mean - clouds_mean > 0.15:
            deltab2 = (img[time, ..., 0] - ri_upper0) > 0.08
            deltab3 = (img[time, ..., 1] - ri_upper1) > 0.08
            deltab4 = (img[time, ..., 2] - ri_upper2) > 0.07

            closeb2 = (img[time, ..., 0] - ri_close[..., 0]) > (close_thresh + close_modifier + 0.005)
            closeb3 = (img[time, ..., 1] - ri_close[..., 1]) > (close_thresh + close_modifier + 0.005)
            closeb4 = (img[time, ..., 2] - ri_close[..., 2]) > (close_thresh + close_modifier)

            clouds_i = (deltab2 * deltab3 * deltab4)
            clouds_close = (closeb2 * closeb3 * closeb4)
            clouds_mean = np.mean(clouds_i)
            clouds_close_mean = np.mean(clouds_close)
            
            print(time, close, others, clouds_mean, clouds_close_mean, np.mean(close_thresh) + close_modifier)
            close_modifier += 0.0025

        # The clouds_close should only be detecting relatively "dark" clouds
        brightness = np.sum(img[time, ..., :3], axis = -1) < 0.8
        clouds_close = clouds_close * brightness
        clouds_close_nonforest = 1 - (binary_dilation(clouds_close == 0, iterations = 2))
        clouds_close[forest_mask == 0] = clouds_close_nonforest[forest_mask == 0]
        clouds[time] = np.maximum(clouds_i, clouds_close)

    # Multitemporal brightness thresholds, TESTING in FEBRUARY 2023.
    # We calculate the median brightness (sum of R, G, B bands) for the non-interpolated pixels
    # For each image, we calculate the ratio of the per-pixel brightness to the median brightness
    # And then calculate the within-image z-score. Any pixel with > 4 Z-score is considered to be a cloud.
    # HOWEVER, we remove clouds identified this way that occur more than once, since agriculture etc. could
    # be picked up this way. We also do not apply this for pixels with >0 NDWI.

    # Pixels with < -3 Z-score are considered to be a shadow.
    brightness_mask = np.sum(img[..., :3], axis = -1)
    brightness_mask[np.logical_or(clouds > 0, shadows > 0)] = np.nan
    median_brightness = np.nanmedian(brightness_mask)

    brightness_clouds = np.zeros_like(clouds, dtype = np.float32)
    for i in range(img.shape[0]):
        brightness_i = np.sum(img[i, ..., :3], axis = -1)
        brightness_ratio = brightness_i / median_brightness
        brightness_ratio[water_mask > 0] = 1.
        if np.sum(clouds[i] < 0.90):
            brightness_zscore = (brightness_ratio - np.nanmean(brightness_ratio[clouds[i] == 0])) / np.nanstd(brightness_ratio[clouds[i] == 0])
            brightness_clouds[i][brightness_zscore > 5] = 1.
        else:
            brightness_zscore = (brightness_ratio - np.nanmean(brightness_ratio)) / np.nanstd(brightness_ratio)
            brightness_clouds[i][brightness_zscore > 3] = 1.

        brightness_ratio = brightness_i / median_brightness
        brightness_ratio[water_mask > 0] = 1.
        brightness_ratio = 1 / brightness_ratio
        brightness_zscore = (brightness_ratio - np.nanmean(brightness_ratio)) / np.nanstd(brightness_ratio)
        brightness_clouds[i][brightness_zscore > 4] = 1.
    sum_brightness_clouds = np.sum((brightness_clouds - clouds) > 0, axis = (0))

    for i in range(img.shape[0]):
        brightness_clouds[i][sum_brightness_clouds > 2] = 0.
    print("brightness clouds", np.mean(brightness_clouds, axis = (1, 2)))
    clouds = np.maximum(clouds, brightness_clouds)
    #np.save("clouds_b.npy", clouds)
    # Remove urban false positives using b8a, b7, b8 paralax effect
    # and NDBI, NDVI, NDWI
    # This method is from Fmask 4.0
    fcps, pfcps = detect_pfcp(img, dem, bbx)
    clouds[fcps > 0] = 0.
    shadows[fcps > 0] = 0.

    # Remove bright surface false positives e.g. sand, rock
    # With a NIR to SWIR1 ratio threshold of < 0.75
    # This threshold is from Fmask 4.0
    nir_swir_ratio = (img[..., 3] / (img[..., 8] + 0.01))
    nir_swir_ratio = nir_swir_ratio < 0.75
    nir_swir_ratio = binary_dilation(nir_swir_ratio, iterations=3)
    for i in range(img.shape[0]):
        nir_swir_ratio[i][water_mask < 0] = 0.
    clouds[nir_swir_ratio] = 0.
    #shadows[nir_swir_ratio] = 0.

    # Remove false positive clouds over water based on NIR
    # A large dilation is necessary here because of shorelines
    # This threshold is from Fmask 4.0
    for i in range(img.shape[0]):
        clouds_i = clouds[i]
        fp = (water_mask > 0) * (img[i, ..., 8] < 0.11)
        fp = binary_dilation(fp, iterations=10)
        clouds_i[fp] = 0.
        clouds[i] = clouds_i

    # Finally, exclude where only minority of 3x3 win are clouds
    # As done in Fmask 4.0
    for i in range(clouds.shape[0]):
        window_sum = _winsum(clouds[i], 3)
        clouds[i][window_sum < 5] = 0.

    for i in range(clouds.shape[0]):
        brightness_threshold = np.sum(img[i, ..., :3], axis = -1) < 0.21
        brightness_threshold = binary_dilation(brightness_threshold, iterations = 3)
        brightness_threshold = brightness_threshold * (1 - forest_mask)
        brightness_threshold = brightness_threshold.astype(np.uint8)
        clouds[i][brightness_threshold] = 0.
    #np.save("clouds_before.npy", clouds)

    # Dilate the non-urban clouds, erode the urban clouds
    struct2 = generate_binary_structure(2, 2)
    for i in range(clouds.shape[0]):
        clouds[i] = (1 - (binary_dilation(clouds[i] == 0, iterations = 1)))
        pfcps[i] = binary_dilation(pfcps[i], iterations=5)
        urban_clouds = clouds[i] * pfcps[i]
        urban_clouds = (1 - (binary_dilation(urban_clouds == 0, iterations=3)))
        #urban_clouds = binary_dilation(urban_clouds, iterations = 2)

        non_urban_clouds = clouds[i] * (1 - pfcps[i])
        window_sum = _winsum(non_urban_clouds, 3)
        is_large_cloud = np.copy(non_urban_clouds)
        is_large_cloud[window_sum < 6] = 0.
        is_small_cloud = np.copy(non_urban_clouds)
        is_small_cloud[window_sum >= 6] = 0.
        is_small_cloud = binary_dilation(is_small_cloud, iterations=1)
        is_large_cloud = binary_dilation(is_large_cloud, iterations=6)
        non_urban_clouds = np.maximum(is_large_cloud, is_small_cloud)

        #non_urban_clouds = binary_dilation(non_urban_clouds, iterations = 7, structure = struct2)
        non_urban_clouds = distance(1 - non_urban_clouds)
        non_urban_clouds[non_urban_clouds <= 4] = 0.
        non_urban_clouds[non_urban_clouds > 4] = 1
        non_urban_clouds = 1 - non_urban_clouds
        clouds[i] = (non_urban_clouds + urban_clouds)

    clouds = np.maximum(clouds, shadows)
    fcps = np.maximum(fcps, nir_swir_ratio)
    fcps = binary_dilation(fcps, iterations=2)
    print("Shadow percents", np.mean(shadows, axis = (1, 2)))
    print("Cloud/shadow percents", np.mean(clouds, axis=(1, 2)))

    #np.save("clouds.npy", clouds)
    return clouds, fcps


def calculate_cloud_steps(clouds: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """ Calculates the timesteps to remove based upon cloud cover and missing data
        This is basically the image selection algorithm, and will be renamed as such.
        In general, we want to balance selecting the best available images for each tile,
        with seelcting consistent images between nearby tiles. 

        This is DEPRECATED as of August 2022.

        Parameters:
         clouds (arr):

        Returns:
         to_remove (arr):
    """

    def _check_month(month, thresh):
        month_idx = np.argwhere(
            np.logical_and(dates >= starting[month],
                           dates < starting[month + 1]))
        cloud_month = cloud_percent[month_idx]
        month_good_idx = np.argwhere(cloud_month < thresh)
        if len(month_good_idx) > 0:
            month_good_dates = np.unique(
                dates[month_idx[month_good_idx]].flatten())
            min_distances = []
            for x in month_good_dates:
                clean_dates = dates[np.argwhere(
                    cloud_percent <= 0.20)].flatten()
                clean_dates = clean_dates[np.argwhere(
                    np.logical_or(clean_dates < starting[month],
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
            clean_dates = clean_dates[np.argwhere(
                np.logical_or(clean_dates < starting[month],
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

    good_steps = np.empty((0,))
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
                    month_good_dates = np.array([
                        month_good_dates[0], month_good_dates[1],
                        month_good_dates[-1]
                    ]).flatten()
                if (min_distance < thresh_dist[x] or thresh[x] == 0.30):
                    finished = True
                    if len(month_good_dates) == 6:
                        month_good_dates = [
                            val for i, val in enumerate(month_good_dates)
                            if i in [0, 2, 3, 5]
                        ]
                        month_good_dates = np.array(month_good_dates)
                    print(f"{month + 1}, Dates: {month_good_dates},"
                          f" Dist: {min_distance}, Thresh: {thresh[x]}")
                    good_steps = np.concatenate(
                        [good_steps, month_good_dates.flatten()])

    good_steps_idx = [i for i, val in enumerate(dates) if val in good_steps]
    cloud_steps = np.array(
        [x for x in range(dates.shape[0]) if x not in good_steps_idx])

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
        month_idx = np.argwhere(
            np.logical_and(dates % 365 >= starting[month],
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
            maxcc = 0.4
        #if np.max(probs)
        to_remove = month_idx[np.argwhere(
            np.logical_or(month_cl_arr >= maxcc,
                          np.isnan(month_clouds))).flatten()]
        if len(to_remove) > 1:
            duplicate_steps.extend(to_remove)
        elif len(to_remove) > 0:
            duplicate_steps.append(to_remove)

        month_idx = month_idx[np.argwhere(month_cl_arr < maxcc)]

        month_dates = dates[month_idx]
        month_dates = np.array(
            [item for sublist in month_dates for item in sublist])
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
    This is the currently used image selection algorithm.
    The general imagery subsetting strategy is as below:
        - Select all images with < 20% cloud cover.
        - If there are no images for a given month, expand search to < 30% cloud cover
        - For each month, select up to 2 images that are <20 or 30% CC and are the closest to
          the beginning and the midde of the month. this will NOT always select the least-cloudy
          image, since that would not be spatially stable. Temporal consistency here enforces
          spatial consistency of selected images between tiles in a region.
        - Select only one image per month for each month if n images > 6. Select the image closest to the
          middle of the month, unless that image is >20% CC, and the other image is at least 10% clearer.
              - The trade-off here is between date consistency (less artifacts) and selecting 
                clearer images (also less artifacts), so a line has to be drawn between
                the two cloud cover thresholds used here. A > 20% and (A - B) > 10% seem to work.
                But there is probably still some room to optimize.
        - If more than 10 images remain, remove any images for March, July, and October. This is
          purely done to reduce IO needs and save processing / storage / acquisition costs.

    Note that it --may-- be better to have a static cloud map (per 200x200km image), and 
    select the least-cloudy image each months. This would mitigate artifacts caused by the variable
    inclusion / exclusion of images close to the 30% threshold as a tile moves closer or further 
    away from a cloudy region. But would likely also cause increased artifacts near S2 image borders.

    """

    begin = [-60, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341]
    end = [31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 341, 410]
    n_per_month = []
    months_to_adjust = []
    months_to_adjust_again = []
    indices_to_rm = []
    indices = [x for x in range(len(dates))]

    def _indices_month(dates, x, y):
        indices_month = np.argwhere(np.logical_and(dates >= x,
                                                   dates < y)).flatten()
        return indices_month

    #_ = print_dates(dates, probs)

    if len(dates) > 6:
        # Select the best 2 images per month to start with
        best_two_per_month = []
        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(dates >= x,
                                                       dates < y)).flatten()

            month_dates = dates[indices_month]
            month_clouds = probs[indices_month]

            # If there is a < 0.2 image, go ahead and use it.
            # Otherwise, expand teh search to 0.2-0.4
            # If Make sure that the least CC image is at least 10% better
            # This is a very important step! We want to make sure that
            # Neighboring tiles have consistent images... so we expand the
            # Search up to 0.4 CC when necessary. However,
            # We don't want to avoid cloud-free images if they exist!
            if len(month_clouds) > 1:
                leastcc = np.min(month_clouds)
                secondleastcc = np.partition(month_clouds, 1)[1]
                maxcc = np.max(month_clouds)

                if np.logical_and(
                        np.logical_and(leastcc < 0.2, maxcc > 0.2),
                        # we don't want to choose e.g. 0.19 vs 0.22
                        # but we do want to choose e.g. 0.19 vs 0.30
                    ((maxcc - leastcc) > 0.10)):
                    maxcc = 0.2
                else:
                    maxcc = 0.4
                #print(leastcc, maxcc, month_clouds, maxcc)
            else:
                maxcc = 0.4

            month_good_dates = month_dates[month_clouds < maxcc]

            indices_month = indices_month[month_clouds < maxcc]

            if len(month_good_dates) >= 2:
                if x > 0:
                    ideal_dates = [x, x + 15]
                else:
                    ideal_dates = [0, 15]

                # We first pick the 2 images with <30% cloud cover that are the closest
                # to the 1st and 15th of the month
                closest_to_first_img = np.argmin(
                    abs(month_good_dates - ideal_dates[0]))
                closest_to_second_img = np.argmin(
                    abs(month_good_dates - ideal_dates[1]))
                # This is deprecated for now, as it leads to
                # Inconsistent image selection, and in some cases, artifacts.
                # We don't necessarily want these to be 2 different images, if the
                # Same image is the closest to both 0 and 15
                """
                if closest_to_second_img == closest_to_first_img:
                    distances = abs(month_good_dates - ideal_dates[1])
                    closest_to_second_img = np.argsort(distances)[1]
                """
                first_image = indices_month[closest_to_first_img]
                second_image = indices_month[closest_to_second_img]
                best_two_per_month.append(first_image)
                best_two_per_month.append(second_image)

            elif len(month_good_dates) >= 1:

                if x > 0:
                    ideal_dates = [x, x + 15]
                else:
                    ideal_dates = [0, 15]

                closest_to_second_img = np.argmin(
                    abs(month_good_dates - ideal_dates[1]))
                second_image = indices_month[closest_to_second_img]
                best_two_per_month.append(second_image)
    else:
        best_two_per_month = np.arange(0, len(dates))

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
            indices_month = np.argwhere(np.logical_and(dates >= x,
                                                       dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [
                val for i, val in enumerate(indices_month)
                if dates_month[i] in dates_round_2
            ]
            if len(indices_month) > 1:
                month_dates = dates[indices_month]
                month_clouds = probs[indices_month]
                
                subset_month = True
                if x == -60:
                    feb_mar = np.argwhere(
                        np.logical_and(dates >= 31, dates < 90)).flatten()
                    subset_month = False if len(feb_mar) < 1 else True
                if x == 334:
                    oct_nov = np.argwhere(
                        np.logical_and(dates >= 273, dates < 334)).flatten()
                    subset_month = False if len(oct_nov) < 1 else True

                if subset_month:
                    subset_month = True if removed <= n_to_rm else False
                if subset_month:
                    if month_clouds[1] >= 0.2 and (
                        (np.min(month_clouds) + 0.10) < (month_clouds[1])):
                        # This appears to be worthwhile, though a harsh cut-off
                        # Can cause inconsistent image selection, so be warned.
                        month_best_date = [
                            indices_month[np.argmin(month_clouds)]
                        ]
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
        print(
            f"There are {len(dates_round_3)} dates and need to remove {n_to_remove}"
        )

        highest_n = np.argpartition(probs_round_3, -n_to_remove)[-n_to_remove:]
        # In general, we do not want to remove the cloudiest image..
        # Since this leads to inconsistent selection between neighboring tiles.
        # However, in some cases, we do want to if it improves data availiability.
        # For now, this is turned off (set to 0.3), since it isn't clear that the trade-off
        # between data coverage and consiistenty is worth it

        highest_n = [x for x in highest_n if probs_round_3[x] > 0.4]
        date_of_highest_n = dates_round_3[highest_n]
        index_to_rm = np.argwhere(np.in1d(dates, date_of_highest_n)).flatten()
        print(f"Removing cloudiest dates of {np.array(dates)[index_to_rm]},"
              f"{date_of_highest_n}, {np.array(probs)[index_to_rm]}")

        indices_to_rm.extend(index_to_rm)
        n_removed += len(index_to_rm)

        for x, y in zip(begin, end):
            indices_month = np.argwhere(np.logical_and(dates >= x,
                                                       dates < y)).flatten()
            dates_month = dates[indices_month]
            indices_month = [x for x in indices_month if x in monthly_dates]

            #print(f"Need to remove {n_to_remove} dates")
            if (len(indices_month) >= 1) and (len(monthly_dates) >=
                                              10) and (n_removed < n_to_remove):
                to_remove = [59,
                             243]  #if len(monthly_dates) >= 10 else [90, 243]
                if x in to_remove or remove_next_month:
                    if len(indices_month) > 0:
                        if indices_month[
                                0] not in indices_to_rm:  #and probs_round_3[indices_month[0] < 0.15]:
                            indices_to_rm.append(indices_month[0])
                            remove_next_month = False
                            n_removed += 1
                            print(f"Removed {x}, {n_removed}")
                        else:
                            remove_next_month = True
                    else:
                        print("Removing the next month instead")
                        remove_next_month = True if not remove_next_month else False

    dates_round_4 = [
        val for x, val in enumerate(dates) if x not in indices_to_rm
    ]
    print(dates_round_4)

    return indices_to_rm

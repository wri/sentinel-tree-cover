import numpy as np
from preprocessing.indices import evi, bi, msavi2, grndvi
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage import label, grey_closing

def expand_interp(probs):
    '''deprecated'''
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
    return areas_interpolated


def rolling_mean(arr):
    """Deprecated"""
    if arr.shape[0] > 4:
        mean_arr = np.zeros_like(arr)
        start = np.arange(0, arr.shape[0] - 3, 1)
        start = np.concatenate([
            np.array([0]),
            start,
            np.full((2,), arr.shape[0] - 3)
        ])
        end = start + 3
        i = 0

        for s, e in zip(start, end):
            array_to_mean = arr[s:e]
            mean_arr[i] = np.median(array_to_mean, axis = 0)
            i += 1
        return mean_arr

    elif arr.shape[0] == 3 or arr.shape[0] == 4:
        mean_arr = np.median(arr, axis = 0)
        arr[0] = mean_arr
        arr[-1] = mean_arr
        return arr
    else:
        return arr


def normalize_first_last_quarter(arr, dates):
    """Deprecated"""
    dates_first_quarter = np.argwhere(np.logical_and(dates < 90, dates > -30))
    if len(dates_first_quarter) > 0:
        dates_first_quarter = np.argwhere(np.logical_and(dates < 90, dates > -30))
        arr[0]  = np.mean(arr[dates_first_quarter], axis = 0)

    if len(np.argwhere(dates > 270)) > 0:
        dates_last_quarter = np.argwhere(dates > 270)
        arr[-1] = np.mean(arr[dates_last_quarter], axis = 0)
    return arr, dates


def normalize_first_last_date(arr, dates):
    """Deprecated"""
    if len(dates) >= 4:
        arr[0] = np.median(arr[:3], axis = 0)
        arr[-1] = np.median(arr[-3:], axis = 0)
    return arr, dates


def make_indices(arr):
    indices = np.zeros(
        (arr.shape[0], arr.shape[1], arr.shape[2], 4), dtype = np.float32
    )
    indices[:, ..., 0] = evi(arr)
    indices[:, ...,  1] = bi(arr)
    indices[:, ...,  2] = msavi2(arr)
    indices[:, ...,  3] = grndvi(arr)
    return indices


def greenest_mosaic(arr):
    """Deprecated"""
    _evi = evi(arr)
    for i in range(1, arr.shape[0] - 1):
        three_window = _evi[i-1:i+1]
        three_window_median = bn.nanmedian(three_window, axis = 0)
        three_window_min = bn.nanmin(three_window, axis = 0)
        replace = _evi[i] == three_window_min
        arr[i][replace] = (arr[i - 1][replace] + arr[i + 1][replace]) / 2 
    return arr
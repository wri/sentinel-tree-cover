import numpy as np
import bottleneck as bn


def id_missing_px(sentinel2: np.ndarray, thresh: int = 11) -> np.ndarray:
    """
    Identifies missing (NA) values in a sentinel 2 array
    Parameters:
         sentinel2 (np.ndarray): multitemporal sentinel 2 array
         thresh (int): denominator for threshold (missing < 1 / thresh)

        Returns:
         missing_images (np.ndarray): (N,) array of time steps to remove
                                      due to missing imagery

    """
    missing_images_0 = np.sum(sentinel2[..., :10] == 0.0, axis=(-1))
    missing_images_p = np.sum(sentinel2[..., :10] >= 1., axis=(-1))
    missing_images = missing_images_0 + missing_images_p
    missing_images = np.sum(missing_images > 1., axis=(1, 2))
    missing_images = np.argwhere(
        missing_images >= (sentinel2.shape[1]**2) / thresh).flatten()
    return missing_images


def interpolate_missing_vals(s2: np.ndarray) -> np.ndarray:
    '''Interpolates NA values with closest time steps, to deal with
       the small potential for NA values in calculating indices
    '''
    print(f"There are {np.sum(s2 >= 1)} NAN values in the datacube")
    if np.sum(np.logical_and(s2 >= 1, s2 == 0)) > 0:
        nanmedian = np.median(s2, axis=0)
        for time in range(s2.shape[0]):
            s2_image = s2[time]
            s2_image[s2_image >= 1] = nanmedian[s2_image >= 1]
            s2_image[s2_image == 0] = nanmedian[s2_image == 0]
        print(f"After NAN removal: there are {np.sum(s2 >= 1)} NAN values in the datacube")

    return s2


def interpolate_na_vals(s2: np.ndarray) -> np.ndarray:
    '''Interpolates NA values with closest time steps, to deal with
       the small potential for NA values in calculating indices
    '''
    if np.sum(np.isnan(s2)) > 0:
        nanmedian = bn.median(s2, axis=0).astype(np.float32)
        nanmedian[np.isnan(nanmedian)] = 0.
        for time in range(s2.shape[0]):
            nanvals = np.isnan(s2[time])  # (X, Y, bands)
            s2[time, nanvals] = nanmedian[nanvals]
            if np.sum(nanvals) > 100:
                print(
                    f"There were {np.sum(nanvals)} missing values in {time} step"
                )
    return s2

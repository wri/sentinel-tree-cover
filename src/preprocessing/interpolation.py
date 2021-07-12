import numpy as np

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
    missing_images_0 = np.sum(sentinel2[..., :10] == 0.0, axis = (1, 2, 3))
    missing_images_p = np.sum(sentinel2[..., :10] >= 1., axis = (1, 2, 3))
    missing_images = missing_images_0 + missing_images_p
    
    missing_images = np.argwhere(missing_images >= (sentinel2.shape[1]**2) / thresh).flatten()
    return missing_images


def interpolate_na_vals(s2: np.ndarray) -> np.ndarray:
    '''Interpolates NA values with closest time steps, to deal with
       the small potential for NA values in calculating indices
    '''
    nanmedian = np.broadcast_to(np.nanmedian(s2, axis = 0), (s2.shape))
    nanvals = np.isnan(s2)
    print(f"There are {np.sum(nanvals)} NA values")
    
    s2[nanvals] = nanmedian[nanvals]
    numb_na = np.sum(np.isnan(s2), axis = (1, 2, 3))
    if np.sum(numb_na) > 0:
        print(f"There are {numb_na} NA values")
    return s2
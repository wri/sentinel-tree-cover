import numpy as np
import hickle as hkl
from skimage.transform import resize

def process_tile(x: int, y: int, data: pd.DataFrame, local_path) -> np.ndarray:
    """
    Processes raw data structure (in temp/raw/*) to processed data structure
        - align shapes of different data sources (clouds / shadows / s1 / s2 / dem)
        - superresolve 20m to 10m with bilinear upsampling for DSen2 input
        - remove (interpolate) clouds and shadows

    Parameters:
         x (int): x position of tile to be downloaded
         y (int): y position of tile to be downloaded
         data (pd.DataFrame): tile grid dataframe

        Returns:
         x (np.ndarray)
         image_dates (np.ndarray)
         interp (np.ndarray)
         s1 (np.ndarray)
    """

    x = str(int(x))
    y = str(int(y))
    x = x[:-2] if ".0" in x else x
    y = y[:-2] if ".0" in y else y

    folder = f"{local_path}{str(x)}/{str(y)}/"
    tile_idx = f'{str(x)}X{str(y)}Y'

    clouds_file = f'{folder}raw/clouds/clouds_{tile_idx}.hkl'
    shadows_file = f'{folder}raw/clouds/shadows_{tile_idx}.hkl'
    s1_file = f'{folder}raw/s1/{tile_idx}.hkl'
    s1_dates_file = f'{folder}raw/misc/s1_dates_{tile_idx}.hkl'
    s2_10_file = f'{folder}raw/s2_10/{tile_idx}.hkl'
    s2_20_file = f'{folder}raw/s2_20/{tile_idx}.hkl'
    s2_dates_file = f'{folder}raw/misc/s2_dates_{tile_idx}.hkl'
    s2_file = f'{folder}raw/s2/{tile_idx}.hkl'
    clean_steps_file = f'{folder}raw/clouds/clean_steps_{tile_idx}.hkl'
    dem_file = f'{folder}raw/misc/dem_{tile_idx}.hkl'

    clouds = hkl.load(clouds_file)
    shadows = hkl.load(shadows_file)
    s1 = hkl.load(s1_file)

    # The S1 data here needs to be bilinearly upsampled as it is in training time!
    s1 = s1.reshape((s1.shape[0], s1.shape[1] // 2, 2, s1.shape[2] // 2, 2, 2))
    s1 = np.mean(s1, (2, 4))
    s1 = resize(s1, (s1.shape[0], s1.shape[1] * 2, s1.shape[2] * 2, 2), order = 1)
    s1 = s1 / 65535
    s1[..., -1] = convert_to_db(s1[..., -1], 22)
    s1[..., -2] = convert_to_db(s1[..., -2], 22)

    s2_10 = to_float32(hkl.load(s2_10_file))
    s2_20 = to_float32(hkl.load(s2_20_file))


    dem = hkl.load(dem_file)
    dem = median_filter(dem, size = 5)
    image_dates = hkl.load(s2_dates_file)

    # The below code is somewhat ugly, but it is geared to ensure that the
    # Different data sources are all the same shape, as they are downloaded
    # with varying resolutions (10m, 20m, 60m, 160m)
    width = s2_10.shape[1]
    height = s2_20.shape[2] * 2

    if clouds.shape[1] < width:
        pad_amt =  (width - clouds.shape[1]) // 2
        clouds = np.pad(clouds, ((0, 0), (pad_amt, pad_amt), (0,0)), 'edge')

    if shadows.shape[1] < width:
        pad_amt =  (width - shadows.shape[1]) // 2
        shadows = np.pad(shadows, ((0, 0), (pad_amt, pad_amt), (0,0)), 'edge')

    if dem.shape[0] < width:
        pad_amt =  (width - dem.shape[0]) // 2
        dem = np.pad(dem, ((pad_amt, pad_amt), (0, 0)), 'edge')

    if s2_10.shape[2] < height:
        pad_amt =  (height - s2_10.shape[2]) / 2
        if pad_amt % 2 == 0:
            pad_amt = int(pad_amt)
            s2_10 = np.pad(s2_10, ((0, 0), (0, 0), (pad_amt, pad_amt), (0,0)), 'edge')
        else:
            s2_10 = np.pad(s2_10, ((0, 0), (0, 0), (0, int(pad_amt * 2)), (0,0)), 'edge')

    if s2_10.shape[2] > height:
        pad_amt =  abs(height - s2_10.shape[2])
        s2_10 = s2_10[:, :, :-pad_amt, :]
        print(s2_10.shape)

    if dem.shape[1] < height:
        pad_amt =  (height - dem.shape[1]) / 2
        if pad_amt % 2 == 0:
            pad_amt = int(pad_amt)
            dem = np.pad(dem, ((0, 0), (pad_amt, pad_amt)), 'edge')
        else:
            dem = np.pad(dem, ( (0, 0), (0, int(pad_amt * 2))), 'edge')

    if dem.shape[1] > height:
        pad_amt =  abs(height - dem.shape[1])
        dem = dem[:, :-pad_amt]

    print(f'Clouds: {clouds.shape}, \nShadows: {shadows.shape} \n'
          f'S1: {s1.shape} \nS2: {s2_10.shape}, {s2_20.shape} \nDEM: {dem.shape}')

    # The 20m bands must be bilinearly upsampled to 10m as input to superresolve_tile
    #! TODO: Parallelize this function such that
         # sentinel2 = np.reshape(sentinel2, sentinel2.shape[0]*sentinel2.shape[-1], width, height)
         # parallel_apply_along_axis(resize, sentinel2, 0)
         # sentinel2 = np.reshape(sentinel2, ...)
    sentinel2 = np.empty((s2_10.shape[0], width, height, 10))
    sentinel2[..., :4] = s2_10
    for band in range(6):
        for time in range(sentinel2.shape[0]):
            sentinel2[time, ..., band + 4] = resize(s2_20[time,..., band], (width, height), 1)


    lower_thresh, upper_thresh = id_iqr_outliers(sentinel2)
    if lower_thresh is not None and upper_thresh is not None:
        above = np.sum(sentinel2 > upper_thresh, axis = (1, 2))
        below = np.sum(sentinel2 < lower_thresh, axis = (1, 2))
        probs = above + below
        n_bands_outlier = (np.sum(probs > (0.5 * sentinel2.shape[1] * sentinel2.shape[2]), axis = (1)))
        print(n_bands_outlier)
        to_remove = np.argwhere(n_bands_outlier >= 1)
        if len(to_remove) > 0:
            print(f"Removing {to_remove} dates due to IQR threshold")
            clouds = np.delete(clouds, to_remove, axis = 0)
            shadows = np.delete(shadows, to_remove, axis = 0)
            image_dates = np.delete(image_dates, to_remove)
            sentinel2 = np.delete(sentinel2, to_remove, axis = 0)


    # Identifies missing imagery (either in sentinel acquisition, or induced in preprocessing)
    # If more than 50% of data for a time step is missing, then remove them....
    missing_px = interpolation.id_missing_px(sentinel2, 2)

    if len(missing_px) > 0:
        print(f"Removing {missing_px} dates due to missing data")
        clouds = np.delete(clouds, missing_px, axis = 0)
        shadows = np.delete(shadows, missing_px, axis = 0)
        image_dates = np.delete(image_dates, missing_px)
        sentinel2 = np.delete(sentinel2, missing_px, axis = 0)

    # Otherwise... set the missing values to the median value.
    sentinel2 = interpolation.interpolate_missing_vals(sentinel2)

    # interpolate cloud and cloud shadows linearly
    sentinel2, interp = cloud_removal.remove_cloud_and_shadows(sentinel2, clouds, shadows, image_dates)
    to_remove_interp = np.argwhere(np.sum(interp, axis = (1, 2)) > (sentinel2.shape[1] * sentinel2.shape[2] * 0.5) ).flatten()

    if len(to_remove_interp > 0):
        print(f"Removing: {to_remove_interp}")
        sentinel2 = np.delete(sentinel2, to_remove_interp, 0)
        image_dates = np.delete(image_dates, to_remove_interp)
        interp = np.delete(interp, to_remove_interp, 0)

    dem = dem / 90
    sentinel2 = np.clip(sentinel2, 0, 1)
    return sentinel2, image_dates, interp, s1, dem

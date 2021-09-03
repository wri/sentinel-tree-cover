import sys
sys.path.append('../')
from src.preprocessing.cloud_removal import mcm_shadow_mask
from src.preprocessing.slope import calcSlope
from src.downloading.utils import calculate_and_save_best_images
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants, DataSource, CustomUrlParam
from typing import Tuple, List
import numpy as np
import datetime
from skimage.transform import resize
from scipy.ndimage import median_filter
import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc
from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f'{f.__name__}, {np.around(te-ts, 2)}')
        return result
    return wrap


def extract_dates(date_dict: dict, year: int) -> List:
    """ Transforms a SentinelHub date dictionary to a
         list of integer calendar dates
    """
    dates = []
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    starting_days = np.cumsum(days_per_month)
    for date in date_dict:
        if date.year == year - 1:
            dates.append(-365 + starting_days[(date.month-1)] + date.day)
        if date.year == year:
            dates.append(starting_days[(date.month-1)] + date.day)
        if date.year == year + 1:
            dates.append(365 + starting_days[(date.month-1)]+date.day)
    return dates


def to_int16(array: np.array) -> np.array:
    '''Converts a float32 array to uint16, reducing storage costs by three-fold'''
    assert np.min(array) >= 0, np.min(array)
    assert np.max(array) <= 1, np.max(array)
    
    array = np.clip(array, 0, 1)
    array = np.trunc(array * 65535)
    assert np.min(array >= 0)
    assert np.max(array <= 65535)
    
    return array.astype(np.uint16)


def to_float32(array: np.array) -> np.array:
    """Converts an int_x array to float32"""
    print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.float32(array) / 65535.
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    return array


def process_sentinel_1_tile(sentinel1: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Converts a (?, X, Y, 2) Sentinel 1 array to (12, X, Y, 2)

        Parameters:
         sentinel1 (np.array):
         dates (np.array):

        Returns:
         s1 (np.array)
    """
    s1, _ = calculate_and_save_best_images(sentinel1, dates)
    monthly = np.empty((12, sentinel1.shape[1], sentinel1.shape[2], 2))
    index = 0
    for start, end in zip(range(0, 72 + 6, 72 // 12), #0, 72, 6
                          range(72 // 12, 72 + 6, 72 // 12)): # 6, 72, 6
        monthly[index] = np.median(s1[start:end], axis = 0)
        index += 1
    return monthly


def identify_clouds(cloud_bbx, shadow_bbx: List[Tuple[float, float]], dates: dict,
                api_key: str,
                year: int) -> (np.ndarray, np.ndarray, np.ndarray):
    """ Downloads and calculates cloud cover and shadow
        This downloads cartesian WGS 84 coords that are snapped to the 
        ESA LULC pixels -- so it will not return a square! 
        
        Parameters:
         bbox (list): output of calc_bbox
         epsg (float): EPSG associated with bbox 
         dates (tuple): YY-MM-DD - YY-MM-DD bounds for downloading 
    
        Returns:
         cloud_img (np.array):
         shadows (np.array): 
         clean_steps (np.array):
    """
    # Download 160 x 160 meter cloud masks, 0 - 255
    box = BBox(cloud_bbx, crs = CRS.WGS84)
    cloud_request = WcsRequest(
        layer='CLOUD_NEW',
        bbox=box, time=dates,
        resx='160m',resy='160m',
        image_format = MimeType.TIFF_d8,
        maxcc=0.75, instance_id=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
        time_difference=datetime.timedelta(hours=72),
    )

    # Download 160 x 160 meter bands for shadow masking, 0 - 65535
    box = BBox(shadow_bbx, crs = CRS.WGS84)
    shadow_request = WcsRequest(
        layer='SHADOW',
        bbox=box, time=dates,
        resx='160m', resy='160m',
        image_format = MimeType.TIFF_d16,
        maxcc=0.75, instance_id=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
        time_difference=datetime.timedelta(hours=72))
    
    cloud_img = np.array(cloud_request.get_data())
    cloud_img = cloud_img.repeat(16,axis=1).repeat(16,axis=2).astype(np.uint8)
    print(f"Clouds: {cloud_img.shape}")
    
    # Identify steps with at least 20% cloud cover
    n_cloud_px = np.sum(cloud_img > int(0.5 * 255), axis = (1, 2))
    cloud_steps = np.argwhere(n_cloud_px > (cloud_img.shape[1]*cloud_img.shape[2] * 0.30))
    clean_steps = [x for x in range(cloud_img.shape[0]) if x not in cloud_steps]
    cloud_img = np.delete(cloud_img, cloud_steps, 0)
    
    # Align cloud and shadow imagery dates
    cloud_dates_dict = [x for x in cloud_request.get_dates()]
    cloud_dates = extract_dates(cloud_dates_dict, year)
    cloud_dates = [val for idx, val in enumerate(cloud_dates) if idx in clean_steps]

    shadow_dates_dict = [x for x in shadow_request.get_dates()]
    shadow_dates = extract_dates(shadow_dates_dict, year)
    shadow_steps = [idx for idx, val in enumerate(shadow_dates) if val in cloud_dates] 

    to_remove_cloud = [idx for idx, val in enumerate(cloud_dates) if val not in shadow_dates]  

    if len(to_remove_cloud) > 0:
        cloud_img = np.delete(cloud_img, to_remove_cloud, 0)
        cloud_dates = list(np.delete(np.array(cloud_dates), to_remove_cloud))

    shadow_img = np.array(shadow_request.get_data(data_filter = shadow_steps))
    shadow_pus = (shadow_img.shape[1]*shadow_img.shape[2])/(512*512) * shadow_img.shape[0] * (6 / 3)
    shadow_img = shadow_img.repeat(16,axis=1).repeat(16,axis=2)

    n_remove_x = (cloud_img.shape[1] - shadow_img.shape[1]) // 2
    n_remove_y = (cloud_img.shape[2] - shadow_img.shape[2]) // 2
    if n_remove_x > 0 and n_remove_y > 0:
        cloud_img = cloud_img[:, n_remove_x:-n_remove_x, n_remove_y : -n_remove_y]
    
    print(shadow_img.shape, cloud_img.shape)

    # Make sure that the cloud_img and the shadow_img are the same shape
    # using the cloud_img as reference
    cloud_img = resize(cloud_img, (cloud_img.shape[0], shadow_img.shape[1], shadow_img.shape[2]), order = 0,
                       anti_aliasing = False,
                       preserve_range = True).astype(np.uint8)
    
    # Type assertions, size assertions
    if not isinstance(cloud_img.flat[0], np.floating):
        assert np.max(cloud_img) > 1
        cloud_img = np.float32(cloud_img) / 255.
    assert np.max(cloud_img) <= 1
    assert cloud_img.dtype == np.float32
    assert shadow_img.dtype == np.uint16
    assert shadow_img.shape[0] == cloud_img.shape[0], (shadow_img.shape, cloud_img.shape)
    
    # Calculate shadow+cloud masks with multitemporal images (Candra et al. 2020)
    print(f"Shadows ({shadow_img.shape}) used {round(shadow_pus, 1)} processing units")
    shadows = mcm_shadow_mask(shadow_img, cloud_img)
    
    return cloud_img, shadows, clean_steps, np.array(cloud_dates), shadow_img


def download_dem(bbox: List[Tuple[float, float]],
                 api_key: str) -> np.ndarray:
    """ Downloads the DEM layer from Sentinel hub
        
        Parameters:
         bbox (list): output of calc_bbox
         epsg (float): EPSG associated with bbox 
    
        Returns:
         dem_image (arr):
    """
    # Download imagery
    box = BBox(bbox, crs = CRS.WGS84)

    dem_request = WcsRequest(
        data_source = DataSource.DEM,
        layer='DEM',
        bbox=box, 
        resx='10m', resy='10m',
        image_format = MimeType.TIFF_d16,
        maxcc=0.75, instance_id=api_key,
        custom_url_params = {CustomUrlParam.SHOWLOGO: False,
                            constants.CustomUrlParam.UPSAMPLING: 'NEAREST'})

    dem_image = dem_request.get_data()[0]
    dem_image = dem_image - 12000
    dem_image = dem_image.astype(np.float32)
    width = dem_image.shape[0]
    height = dem_image.shape[1]
    
    # Calculate median filter, slopde
    dem_image = median_filter(dem_image, size = 5)
    dem_image = calcSlope(dem_image.reshape((1, width, height)),
                          np.full((width, height), 10), 
                          np.full((width, height), 10), zScale = 1, minSlope = 0.02)
    dem_image = dem_image.reshape((width,height, 1))

    dem_image = dem_image[1:width-1, 1:height-1, :]
    dem_image = dem_image.squeeze()
    print(f"DEM used {round(((width*height)/(512*512))*1/3, 1)} processing units")
    return dem_image


def identify_dates_to_download(dates: list) -> list:
    """ Identify the S1 dates to download"""
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    days_per_month = np.array(days_per_month)
    days_per_month = np.reshape(days_per_month, (4, 3))
    days_per_month = np.sum(days_per_month, axis = 1)

    starting_days = np.cumsum(days_per_month)

    dates = np.array(dates)
    dates_to_download = []
    for i in starting_days:
        s1_month = dates[dates > i]
        s1_month = s1_month[s1_month < (i + 30)]
        if len(s1_month) > 0:
            dates_to_download.append(s1_month[0])
    return dates_to_download


def make_overlapping_windows(tiles: np.ndarray, diff = 7) -> np.ndarray:
    """ Takes the A x B window IDs (n, 4)for an
     X by Y rectangle and enures that the windows are the right
     size (e.g. square, 150 x 150) for running predictions on """
    tiles2 = np.copy(tiles)
    n_x = np.sum(tiles2[:, 0] == 0)
    n_y = np.sum(tiles2[:, 1] == 0)

    tiles2[:n_x, 2] += diff
    tiles2[-n_x:, 2] += diff
    to_adjust = np.full((tiles.shape[0]), diff * 2).astype(np.uint16)
    
    for i in range(len(to_adjust)):
        if (i % n_y == 0) or ((i + 1) % n_y == 0):
            to_adjust[i] -= diff
    tiles2 = tiles2.astype(np.int64)
    tiles2[:, 3] += to_adjust
    tiles2[n_x:-n_x, 2] += (diff * 2)
    tiles2[n_x:, 0] -= diff
    tiles2[:, 1] -= diff
    
    tiles2[tiles2 < 0] = 0.
    return tiles2


def redownload_sentinel_1(dates: np.ndarray, layer: str, bbox: list, 
                          source: str, api_key: str, data_filter: np.ndarray) -> np.ndarray:
    """Worker function to redownload individual time steps if the
    preceding time step had null values, likely the case in areas with
    both ascending and descending orbits"""

    image_request = WcsRequest(
            layer=layer, bbox=bbox,
            time=dates,
            image_format = MimeType.TIFF_d16,
            data_source=source, maxcc=1.0,
            resx='20m', resy='20m',
            instance_id=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=72),
        )
    return np.array(image_request.get_data(data_filter = data_filter))


def download_sentinel_1(bbox: List[Tuple[float, float]],
                        api_key,
                        year: int,
                        dates: dict,
                        layer: str = "SENT"
                        ) -> (np.ndarray, np.ndarray):
    """ Downloads the GRD Sentinel 1 VV-VH layer from Sentinel Hub
        
        Parameters:
         bbox (list): output of calc_bbox
         epsg (float): EPSG associated with bbox 
         imsize (int):
         dates (tuple): YY-MM-DD - YY-MM-DD bounds for downloading 
         layer (str):
         year (int): 
    
        Returns:
         s1 (arr):
         image_dates (arr): 
    """
    days_per_month = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30]
    starting_days = np.cumsum(days_per_month)

    # Identify the S1 orbit, imagery dates
    source = DataSource.SENTINEL1_IW_DES if layer == "SENT_DESC" else DataSource.SENTINEL1_IW_ASC
    box = BBox(bbox, crs = CRS.WGS84)

    image_request = WcsRequest(
            layer=layer, bbox=box,
            time=dates,
            image_format = MimeType.TIFF_d16,
            data_source=source, maxcc=1.0,
            resx='20m', resy='20m',
            instance_id=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=72),
        )
    
    s1_dates_dict = [x for x in image_request.get_dates()]
    s1_dates = extract_dates(s1_dates_dict, year)
    dates_to_download = identify_dates_to_download(s1_dates)

    steps_to_download = [i for i, val in enumerate(s1_dates) if val in dates_to_download]
    print(f"The following dates will be downloaded: {dates_to_download}")
    
    # If the correct orbit is selected, download imagery
    if len(image_request.download_list) >= 4 and len(steps_to_download) >= 4:
        try:
            s1 = np.array(image_request.get_data(data_filter = steps_to_download))
            if not isinstance(s1.flat[0], np.floating):
                assert np.max(s1) > 1
                s1 = np.float32(s1) / 65535.
            assert np.max(s1) <= 1

            height = s1.shape[1]
            width = s1.shape[2]

            s1_usage = (4/3) * s1.shape[0] * ((s1.shape[1]*s1.shape[2]) / (512*512))
            print(f"Sentinel 1 used {round(s1_usage, 1)} PU for "
                  f" {s1.shape[0]} out of {len(image_request.download_list)} images")

            image_dates_dict = [x for x in image_request.get_dates()]
            image_dates = extract_dates(image_dates_dict, year)
            image_dates = [val for idx, val in enumerate(image_dates) if idx in steps_to_download]
            image_dates = np.array(image_dates)

            n_pix_oob = np.sum(s1 >= 1, axis = (1, 2, 3))
            print(f"N_oob: {n_pix_oob}")
            to_remove = np.argwhere(n_pix_oob > (height*width)/5)
            
            if len(to_remove) > 0:
                for index in to_remove:

                    index = index[0]
                    step = steps_to_download[index]
                    new_step = step + 1

                    print(f'Redownloading {new_step} because {step} had '
                          f'{n_pix_oob[index]} missing or null values')
                    new_s1 = redownload_sentinel_1(dates, layer, box, source, api_key, [new_step])
                    new_s1 = np.float32(new_s1) / 65535.
                    print(f"The new step has {np.sum(new_s1 >= 1)} and is {new_s1.shape}")
                    s1[index] = new_s1
                    image_dates[index] = image_dates[index] + 5

            n_pix_oob = np.sum(s1 >= 1, axis = (1, 2, 3))
            print(f"N_oob: {n_pix_oob}")
            to_remove = np.argwhere(n_pix_oob > (height*width)/10)
            print(to_remove)
            print(s1.shape)
            if len(to_remove) > 0:
                s1 = np.delete(s1, to_remove, 0)
                image_dates = np.delete(image_dates, to_remove)
            print(s1.shape)
            s1 = np.clip(s1, 0, 1)
            s1 = s1.repeat(3, axis = 0)
            image_dates = np.array(image_dates).repeat(2, axis = 0)
            s1 = s1.repeat(2,axis=1).repeat(2,axis=2)
            return s1, image_dates

        except:
            return np.empty((0,)), np.empty((0,))
    else: 
        return np.empty((0,)), np.empty((0,))


def identify_s1_layer(coords: Tuple[float, float]) -> str:
    """ Identifies whether to download ascending or descending 
        sentinel 1 orbit based upon predetermined geographic coverage
        
        Reference: https://sentinel.esa.int/web/sentinel/missions/
                   sentinel-1/satellite-description/geographical-coverage
        
        Parameters:
         coords (tuple): 
    
        Returns:
         layer (str): either of SENT, SENT_DESC 
    """
    results = rg.search(coords)
    country = results[-1]['cc']
    try:
        continent_name = pc.country_alpha2_to_continent_code(country)
        print(continent_name, country)
    except:
        continent_name = 'AF'
    layer = None
    if continent_name in ['AF', 'OC']:
        layer = "SENT"
    if continent_name in ['SA']:
        if coords[0] > -7.11:
            layer = "SENT"
        else:
            layer = "SENT_DESC"
    if continent_name in ['AS']:
        if coords[0] > 23.3:
            layer = "SENT"
        else:
            layer = "SENT_DESC"
    if continent_name in ['NA']:
        layer = "SENT_DESC"
    if not layer:
        layer = "SENT"
    print(f"Country: {country}, continent: {continent_name}, orbit: {layer}")
    return layer


def download_sentinel_2(bbox: List[Tuple[float, float]],
                   clean_steps: np.ndarray, api_key,
                   dates: dict, year: int) -> (np.ndarray, np.ndarray):
    """ Downloads the L2A sentinel layer with 10 and 20 meter bands
        
        Parameters:
         bbox (list): output of calc_bbox
         clean_steps (list): list of steps to filter download request
         epsg (float): EPSG associated with bbox 
         time (tuple): YY-MM-DD - YY-MM-DD bounds for downloading 
    
        Returns:
         img (arr):
         img_request (obj): 
    """
    
    # Download 20 meter bands
    box = BBox(bbox, crs = CRS.WGS84)

    image_request = WcsRequest(
            layer='L2A20',
            bbox=box, time=dates,
            image_format = MimeType.TIFF_d16,
            maxcc=0.75, resx='20m', resy='20m',
            instance_id=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=72),
        )
    image_dates_dict = [x for x in image_request.get_dates()]
    image_dates = extract_dates(image_dates_dict, year)
    steps_to_download = [i for i, val in enumerate(image_dates) if val in clean_steps]
    dates_to_download = [val for i, val in enumerate(image_dates) if val in clean_steps]

    quality_request = WcsRequest(
            layer='DATA_QUALITY',
            bbox=box, time=dates,
            image_format = MimeType.TIFF_d8,
            maxcc=0.75, resx='160m', resy='160m',
            instance_id=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=72),
    )
    quality_img = np.array(quality_request.get_data(data_filter = steps_to_download))
    quality_per_img = np.mean(quality_img, axis = (1, 2)) / 255
    print("QUALITY:", quality_per_img)
    steps_to_rm = np.argwhere(quality_per_img > 0.2).flatten()
    if len(steps_to_rm) > 0:
        steps_to_download = np.array(steps_to_download)
        steps_to_download = list(np.delete(steps_to_download, steps_to_rm))
        dates_to_download = np.array(dates_to_download)
        dates_to_download = list(np.delete(dates_to_download, steps_to_rm))
    
    img_20 = np.array(image_request.get_data(data_filter = steps_to_download))
    s2_20_usage = (img_20.shape[1]*img_20.shape[2])/(512*512) * (6/3) * img_20.shape[0]
    
    # Convert 20m bands to np.float32, ensure correct dimensions
    if not isinstance(img_20.flat[0], np.floating):
        assert np.max(img_20) > 1
        img_20 = np.float32(img_20) / 65535.
        assert np.max(img_20) <= 1
        assert img_20.dtype == np.float32
    
    print(f"Original 20 meter bands size: {img_20.shape}, using {round(s2_20_usage, 1)} PU")

    # Download 10 meter bands
    image_request = WcsRequest(
            layer='L2A10',
            bbox=box, time=dates,
            image_format = MimeType.TIFF_d16,
            maxcc=0.75, resx='10m', resy='10m',
            instance_id=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'BICUBIC',
                                constants.CustomUrlParam.UPSAMPLING: 'BICUBIC'},
            time_difference=datetime.timedelta(hours=72),
    )
    img_10 = np.array(image_request.get_data(data_filter = steps_to_download))
    s2_10_usage = (img_10.shape[1]*img_10.shape[2])/(512*512) * (4/3) * img_10.shape[0]
    
    # Convert 10 meter bands to np.float32, ensure correct dimensions
    if not isinstance(img_10.flat[0], np.floating):
        print(f"Converting S2, 10m to float32, with {np.max(img_10)} max and"
                  f" {s2_10_usage} PU")
        assert np.max(img_10) > 1
        img_10 = np.float32(img_10) / 65535.
        assert np.max(img_10) <= 1
        assert img_10.dtype == np.float32

    # Ensure output is within correct range
    img_10 = np.clip(img_10, 0, 1)
    img_20 = np.clip(img_20, 0, 1)

    s2_10_usage = (img_10.shape[1]*img_10.shape[2])/(512*512) * (4/3) * img_10.shape[0]

    return img_10, img_20, np.array(dates_to_download)

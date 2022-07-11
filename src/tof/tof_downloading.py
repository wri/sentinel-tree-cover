import sys
sys.path.append('../')
from src.preprocessing.cloud_removal import mcm_shadow_mask
from src.preprocessing.slope import calcSlope
from src.downloading.utils import calculate_and_save_best_images
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants, DataSource, CustomUrlParam, SentinelHubRequest
from sentinelhub.geo_utils import bbox_to_dimensions
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
import warnings
from scipy.ndimage.morphology import binary_dilation


def timing(f):
    """ Decorator used to time function execution times
    """
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
        #if date.year == year:
        dates.append(((date.year - year)*365) + starting_days[(date.month-1)] + date.day)
       # if date.year == year + 1:
        #    dates.append(365 + starting_days[(date.month-1)]+date.day)
    return dates


def to_int16(array: np.array) -> np.array:
    '''Converts a float32 array to uint16'''
    assert np.min(array) >= 0, np.min(array)
    assert np.max(array) <= 1, np.max(array)

    array = np.clip(array, 0, 1)
    array = np.trunc(array * 65535)
    assert np.min(array >= 0)
    assert np.max(array <= 65535)

    return array.astype(np.uint16)


def to_float32(array: np.array) -> np.array:
    """Converts an int array to float32"""
    print(f'The original max value is {np.max(array)}')
    if not isinstance(array.flat[0], np.floating):
        assert np.max(array) > 1
        array = np.float32(array) / 65535.
    assert np.max(array) <= 1
    assert array.dtype == np.float32
    return array


def process_sentinel_1_tile(sentinel1: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Converts a (?, X, Y, 2) Sentinel 1 array to a regular monthly grid

        Parameters:
         sentinel1 (np.array):
         dates (np.array):

        Returns:
         s1 (np.array)
    """
    s1, _ = calculate_and_save_best_images(sentinel1, dates)
    monthly = np.empty((12, sentinel1.shape[1], sentinel1.shape[2], 2))
    index = 0
    for start, end in zip(range(0, 24 + 2, 24 // 12), #0, 72, 6
                          range(24 // 12, 24 + 2, 24 // 12)): # 6, 72, 6
        monthly[index] = np.mean(s1[start:end], axis = 0)
        index += 1
    return monthly


def identify_clouds(cloud_bbx, shadow_bbx: List[Tuple[float, float]], dates: dict,
                api_key: str,
                year: int,
                maxcc = 0.9) -> (np.ndarray, np.ndarray, np.ndarray):
    """ DEPRECATED: This version of the cloud identification is currently deprecated
        because it can cause tile artifacts between neighboring tiles, if the
        imagery selection is done entirely independently!

        Downloads and calculates cloud cover and shadow
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
    warnings.warn("identify_clouds is deprecated; use identify_clouds_big_bbx", warnings.DeprecationWarning)
    # Download 160 x 160 meter cloud masks, 0 - 255
    box = BBox(cloud_bbx, crs = CRS.WGS84)
    cloud_request = WcsRequest(
        layer='CLOUD_NEWEST',
        bbox=box, time=dates,
        resx='160m',resy='160m',
        image_format = MimeType.TIFF,
        maxcc=maxcc, config=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
        time_difference=datetime.timedelta(hours=48),
    )

    # Download 160 x 160 meter bands for shadow masking, 0 - 65535
    box = BBox(shadow_bbx, crs = CRS.WGS84)
    shadow_request = WcsRequest(
        layer='SHADOW',
        bbox=box, time=dates,
        resx='160m', resy='160m',
        image_format = MimeType.TIFF,
        maxcc=maxcc, config=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
        time_difference=datetime.timedelta(hours=48))

    cloud_img = np.array(cloud_request.get_data())
    cloud_img = cloud_img.repeat(16,axis=1).repeat(16,axis=2).astype(np.uint8)

    cloud_dates_dict = [x for x in cloud_request.get_dates()]
    cloud_dates = extract_dates(cloud_dates_dict, year)

    # Remove steps with >30% cloud cover
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

    # Make sure that the cloud_img and the shadow_img are the same shape
    # using the cloud_img as reference
    cloud_img = resize(
        cloud_img, (cloud_img.shape[0],
                    shadow_img.shape[1],
                    shadow_img.shape[2]),
                    order = 0,
        anti_aliasing = False,
        preserve_range = True
    ).astype(np.uint8)

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


def identify_clouds_big_bbx(cloud_bbx, shadow_bbx: List[Tuple[float, float]], dates: dict,
                api_key: str,
                year: int,
                maxclouds = 0.3) -> (np.ndarray, np.ndarray, np.ndarray):
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
    box = BBox(shadow_bbx, crs = CRS.WGS84)
    cloud_filter_request = WcsRequest(
        layer='CLOUD_SCL_PREVIEW',
        bbox=box, time=dates,
        resx='640m', resy='640m',
        image_format = MimeType.TIFF,
        maxcc=0.4, config=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST',
                             constants.CustomUrlParam.PREVIEW: 'TILE_PREVIEW'},
        time_difference=datetime.timedelta(hours=48),
    )

    image_dates_dict = [x for x in cloud_filter_request.get_dates()]
    cloud_filter_dates = np.array(extract_dates(image_dates_dict, year))
    print(cloud_filter_dates)

    box = BBox(cloud_bbx, crs = CRS.WGS84)
    cloud_request = WcsRequest(
        layer='CLOUD_SCL_PREVIEW',
        bbox=box, time=dates,
        resx='640m', resy='640m',
        image_format = MimeType.TIFF,
        maxcc=0.4, config=api_key,
        custom_url_params = {constants.CustomUrlParam.UPSAMPLING: 'NEAREST',
                             constants.CustomUrlParam.PREVIEW: 'TILE_PREVIEW'},
        time_difference=datetime.timedelta(hours=48),
    )
    image_dates_dict = [x for x in cloud_request.get_dates()]
    cloud_dates = extract_dates(image_dates_dict, year)
    #closest_small_date = [np.min(abs(cloud_filter_dates - x)) for x in cloud_dates]
    #keep_steps = list(np.argwhere(np.array(closest_small_date) <= 2).flatten())
    cloud_dates = np.array(cloud_dates)#[keep_steps]

    cloud_img = np.array(cloud_request.get_data()).astype(np.float32)

    print(f"Clouds: {cloud_img.shape}, {np.max(cloud_img)}")
    #cloud_dates_dict = [x for x in cloud_request.get_dates(data_filter = keep_steps)]
    #cloud_dates = extract_dates(cloud_dates_dict, year)
    

    # Make sure that there is imagery for the tile
    mid_idx = cloud_img.shape[1] // 2
    mid_idx_y = cloud_img.shape[2] // 2
    is_valid = np.sum(cloud_img[:, mid_idx-5:mid_idx+5, mid_idx_y-5:mid_idx_y+5] == 255, axis = (1, 2))
    is_invalid = np.argwhere(is_valid > 0).flatten()

    #for i in range(len(cloud_dates)):
    #    print(cloud_dates[i], np.mean(cloud_img[i] == 100), is_valid[i])

    if len(is_invalid) > 0:
        print(f"deleting {np.array(cloud_dates)[is_invalid]}")
        cloud_dates = np.delete(cloud_dates, is_invalid)
        cloud_img = np.delete(cloud_img, is_invalid, 0)

    # And make sure to remove areas within the tile that are only cloudy
    is_invalid = np.mean(cloud_img[:, mid_idx-5:mid_idx+5, mid_idx_y-5:mid_idx_y+5] == 100, axis = (1, 2))
    is_invalid = np.argwhere(is_invalid > 0.75).flatten()
    print(f"deleting {np.array(cloud_dates)[is_invalid]}")
    cloud_dates = np.delete(cloud_dates, is_invalid)
    cloud_img = np.delete(cloud_img, is_invalid, 0)

    cloud_img = np.float32(cloud_img)
    cloud_img[cloud_img == 255] = np.nan
    cloud_percent = np.nanmean(cloud_img, axis = (1, 2)) / 100
    cloud_img[np.isnan(cloud_img)] = 255
    cloud_img = cloud_img / 255

    cloud_steps = np.argwhere(cloud_percent > maxclouds)

    # Remove steps with at least 15% cloud cover
    """
    n_cloud_px = np.sum(cloud_img > int(0.5 * 255), axis = (1, 2))
    cloud_percent = n_cloud_px / (cloud_img.shape[1]*cloud_img.shape[2])
    cloud_steps = np.argwhere(n_cloud_px >= (cloud_img.shape[1]*cloud_img.shape[2] * maxclouds))
    """
    #clean_steps = [x for x in range(cloud_img.shape[0]) if x not in cloud_steps]
    cloud_img = np.delete(cloud_img, cloud_steps, 0)
    cloud_percent = np.delete(cloud_percent, cloud_steps, 0)
    cloud_dates = np.delete(cloud_dates, cloud_steps, 0)
    #cloud_dates_dict = [x for x in cloud_request.get_dates()]
    #cloud_dates = extract_dates(cloud_dates_dict, year)
    #cloud_dates = [val for idx, val in enumerate(cloud_dates) if idx in clean_steps]
    # Type assertions, size assertions
    if not isinstance(cloud_img.flat[0], np.floating):
        #assert np.max(cloud_img) > 1
        cloud_img = np.float32(cloud_img) / 255.
    assert np.max(cloud_img) <= 1, np.max(cloud_img)
    assert cloud_img.dtype == np.float32

    return cloud_img, cloud_percent, np.array(cloud_dates)


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
        image_format = MimeType.TIFF,
        maxcc=0.75, config=api_key,
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


def download_sentinel_1_composite(bbox: List[Tuple[float, float]],
                        api_key,
                        year: int,
                        dates: dict,
                        size,
                        layer: str = "SENT",
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

    # Identify the S1 orbit, imagery dates
    if layer == "SENT_DESC":
        source = DataSource.SENTINEL1_IW_DES
    if layer == "SENT":
        source = DataSource.SENTINEL1_IW_ASC
    if layer == "SENT_ALL":
        source = DataSource.SENTINEL1_IW
    box = BBox(bbox, crs = CRS.WGS84)
    #size = bbox_to_dimensions(box, resolution=20)

    # If the correct orbit is selected, download imagery
    s1_all = []
    image_dates = []
    image_date = 60

    dates_q1 = (f'{str(year)}-01-15' , f'{str(year)}-04-15')
    dates_q2 = (f'{str(year)}-05-15' , f'{str(year)}-08-15')
    dates_q3 = (f'{str(year)}-09-15' , f'{str(year)}-12-15')
    #dates_q4 = (f'{str(year)}-10-15' , f'{str(year)}-12-15')

    try:
        for date in [dates_q1, dates_q2, dates_q3]:
            evalscript = """
            //VERSION3
            function mean(values) {
                var total = 0, i;
                for (i = 0; i < values; i += 1) {
                    total += values[i];
                }
                return total / values.length;
            }



            function median(values) {
                if (values.length === 0) return 0;
                if (values.length === 1) return values[0];

                values.sort(function(a, b) {
                    return a - b;
                });

                var half = Math.floor(values.length / 2);
                if (values.length % 2 === 0) {
                    return (values[half - 1] + values[half]) / 2;
                }

                return values[half];
            }


            function evaluatePixel(samples) {
                // Initialise arrays
                var VV_samples = [];
                var VH_samples = [];

                // Loop through orbits and add data
                for (let i=0; i<samples.length; i++){
                  // Ignore noData
                  if (samples[i].dataMask != 0){
                    VV_samples.push(samples[i].VV);
                    VH_samples.push(samples[i].VH);
                   }
                }

                const factor = 65535;

                if (VV_samples.length == 0){
                  var VV_median = [factor];
                } else {
                  var VV_median = median(VV_samples) * factor;
                }
                if (VH_samples.length == 0){
                  var VH_median = [factor];
                } else{
                  var VH_median = median(VH_samples) * factor;
                }

                return [VV_median, VH_median];
            }

            function setup() {
              return {
                input: [{
                  bands: [
                    "VV",
                    "VH",
                    "dataMask",
                         ]
                }],
                output: {
                  bands: 2,
                  sampleType:"UINT16"
                },
                mosaicking: "ORBIT"
              }
            }
            """

            request = SentinelHubRequest(
                evalscript=evalscript,
                input_data=[
                    SentinelHubRequest.input_data(
                        data_collection=source,
                        time_interval=date,
                        other_args={
                            "processing": {
                                "backCoeff": "GAMMA0_TERRAIN",
                                "speckleFilter": {
                                    "type": "NONE"
                                },
                                "orthorectify": "true",
                                "demInstance": "MAPZEN",
                                "type": "S1GRD",
                                "resolution": "HIGH",
                                "polarization": "DV",
                            }
                        },
                    ),
                ],
                responses=[
                SentinelHubRequest.output_response('default', MimeType.TIFF)
                ],
                bbox=box,
                size = [size[1] // 2, size[0] // 2],
                config=api_key
            )

            s1 = np.array(request.get_data())
            #print(f"Shape of returned images for {date} = {s1[0].shape[0:2]}")

            if not isinstance(s1.flat[0], np.floating):
                assert np.max(s1) > 1
                s1 = np.float32(s1) / 65535.
            assert np.max(s1) <= 1

            height = s1.shape[1]
            width = s1.shape[2]

            s1_usage = (4/3) * s1.shape[0] * ((s1.shape[1]*s1.shape[2]) / (512*512))
            print(f"Sentinel 1 used {round(s1_usage, 1)} PU for {image_date}")
            print(np.sum(s1 == 1) / (height * width))
            if np.sum(s1 == 1) < (height * width / 4):
                s1_all.append(s1)
                image_dates.append(image_date)
            elif date == dates_q1 and np.sum(s1 == 1) >= (height * width):
                return np.empty((0,)), np.empty((0,))

            image_date += 120

        s1 = np.concatenate(s1_all, axis = 0)
        s1 = np.clip(s1, 0, 1)
        image_dates = np.array(image_dates).repeat(12 // s1.shape[0], axis = 0)
        #print(image_dates)

        s1 = s1.repeat(12 // s1.shape[0], axis = 0)
        s1 = s1.repeat(4,axis=1).repeat(4,axis=2)
        return s1, image_dates

    except:
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
                   dates: dict, year: int, maxclouds: float) -> (np.ndarray, np.ndarray):
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
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='20m', resy='20m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
        )
    image_dates_dict = [x for x in image_request.get_dates()]
    image_dates = extract_dates(image_dates_dict, year)

    steps_to_download = []
    dates_to_download = []
    for min_thresh in [1, 2, 3]:
        if len(steps_to_download) < 3:
            for i, val in enumerate(clean_steps):
                closest_image = np.argmin(abs(val - image_dates))
                if np.min(abs(val - image_dates)) < 1:
                    steps_to_download.append(closest_image)
                    dates_to_download.append(image_dates[closest_image])
                else:
                    print(f"There is a date/orbit mismatch, and the closest image is"
                          f" {np.min(abs(val - image_dates))} days away")

    quality_request = WcsRequest(
            layer='DATA_QUALITY',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='160m', resy='160m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
    )


    quality_img = np.array(quality_request.get_data(data_filter = steps_to_download))
    quality_per_img = np.mean(quality_img, axis = (1, 2)) / 255
    print("Image quality:", quality_per_img)
    steps_to_rm = np.argwhere(quality_per_img > 0.1).flatten()
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
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='10m', resy='10m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'BICUBIC',
                                constants.CustomUrlParam.UPSAMPLING: 'BICUBIC'},
            time_difference=datetime.timedelta(hours=48),
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


def remove_noise_clouds(arr):
        for t in range(arr.shape[0]):
            for x in range(1, arr.shape[1] - 1, 1):
                for y in range(1, arr.shape[2] - 1, 1):
                    window = arr[t, x-1:x+2, y-1:y+2]
                    if window[1, 1] > 0:
                        # if one pixel and at least (n - 2) are cloudy
                        if np.sum(window > 0) <= 1 and np.sum(arr[:, x, y]) > arr.shape[0] - 1:
                            window = 0.
                            arr[t, x-1:x+2, y-1:y+2] = window
        return arr


def download_sentinel_2_new(bbox: List[Tuple[float, float]],
                   clean_steps: np.ndarray, api_key,
                   dates: dict, year: int, maxclouds:float = 1.) -> (np.ndarray, np.ndarray):
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
            layer='L2A20_ORBIT',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='20m', resy='20m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
        )
    image_dates_dict = [x for x in image_request.get_dates()]
    image_dates = extract_dates(image_dates_dict, year)

    steps_to_download = []
    dates_to_download = []

    for i, val in enumerate(clean_steps):
        closest_image = np.argmin(abs(val - image_dates))
        if np.min(abs(val - image_dates)) < 3:
            steps_to_download.append(closest_image)
            dates_to_download.append(image_dates[closest_image])
        else:
            print(f"There is a date/orbit mismatch, and the closest image is"
                  f" {np.min(abs(val - image_dates))} days away")
    print(steps_to_download)

    quality_request = WcsRequest(
            layer='DATA_QUALITY',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='160m', resy='160m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
    )

    cirrus = WcsRequest(
            layer='CIRRUS_CLOUDS',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='160m', resy='160m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
    )

    quality_img = np.array(quality_request.get_data(data_filter = steps_to_download))
    quality_per_img = np.mean(quality_img, axis = (1, 2)) / 255
    quality_per_img = quality_per_img# + cirrus_img
    print("Image quality:", quality_per_img)
    steps_to_rm = np.argwhere(quality_per_img > 0.2).flatten()
    if len(steps_to_rm) > 0:
        steps_to_download = np.array(steps_to_download)
        steps_to_download = list(np.delete(steps_to_download, steps_to_rm))
        dates_to_download = np.array(dates_to_download)
        dates_to_download = list(np.delete(dates_to_download, steps_to_rm))

    img_20 = np.array(image_request.get_data(data_filter = steps_to_download))
    s2_20_usage = (img_20.shape[1]*img_20.shape[2])/(512*512) * (4/3) * img_20.shape[0]

    # Convert 20m bands to np.float32, ensure correct dimensions
    if not isinstance(img_20.flat[0], np.floating):
        assert np.max(img_20) > 1
        img_20 = np.float32(img_20) / 65535.
        assert np.max(img_20) <= 1
        assert img_20.dtype == np.float32

    image_request = WcsRequest(
            layer='L2A40_ORBIT',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            maxcc=maxclouds, resx='40m', resy='40m',
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
            time_difference=datetime.timedelta(hours=48),
        )
    img_40 = np.array(image_request.get_data(data_filter = steps_to_download))

    if not isinstance(img_40.flat[0], np.floating):
        assert np.max(img_40) > 1
        img_40 = np.float32(img_40) / 65535.
        assert np.max(img_40) <= 1
        assert img_40.dtype == np.float32

    s2_40_usage = (img_40.shape[1]*img_40.shape[2])/(512*512) * (2/3) * img_40.shape[0]
    img_40 = img_40.repeat(2, axis = 1).repeat(2, axis = 2)

    if (img_20.shape[1] > img_40.shape[1]) or (img_20.shape[2] > img_40.shape[2]):
        img_40 = resize(img_40, (img_20.shape[0], img_20.shape[1], img_20.shape[2], 2), order = 0)

    if img_40.shape[1] > img_20.shape[1]:
        to_remove = (img_40.shape[1] - img_20.shape[1])
        if to_remove == 2:
            img_40 = img_40[:, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis = 1).repeat(2, axis = 2)
            img_40 = img_40[:, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, 2))
            img_40 = np.mean(img_40, axis = (2, 4))


    if img_40.shape[2] > img_20.shape[2]:
        to_remove = (img_40.shape[2] - img_20.shape[2])
        if to_remove == 2:
            img_40 = img_40[:, :, 1:-1, ...]
        if to_remove == 1:
            img_40 = img_40.repeat(2, axis = 1).repeat(2, axis = 2)
            img_40 = img_40[:, :, 1:-1, ...]
            img_40 = np.reshape(img_40, (img_40.shape[0], img_40.shape[1] // 2, 2, img_40.shape[2] // 2, 2, 2))
            img_40 = np.mean(img_40, axis = (2, 4))

    img_20 = np.concatenate([img_20, img_40], axis = -1)
    print(f"Original 20 meter bands size: {img_20.shape}, using {round(s2_20_usage + s2_40_usage, 1)} PU")
    # Download 10 meter bands
    image_request = WcsRequest(
            layer='L2A10_ORBIT',
            bbox=box, time=dates,
            image_format = MimeType.TIFF,
            resx='10m', resy='10m',
            maxcc=maxclouds,
            config=api_key,
            custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'BICUBIC',
                                constants.CustomUrlParam.UPSAMPLING: 'BICUBIC'},
            time_difference=datetime.timedelta(hours=48),
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

    cirrus_img = np.array(cirrus.get_data(data_filter = steps_to_download))
    print(np.sum(cirrus_img == 2, axis = (1, 2)))
    print(np.sum(cirrus_img == 1, axis = (1, 2)))
    print(np.sum(cirrus_img == 3, axis = (1, 2)))
    print(np.sum(cirrus_img == 9, axis = (1, 2)))
    print(np.sum(cirrus_img == 10, axis = (1, 2)))

    cirrus_img = remove_noise_clouds(cirrus_img)

    print(np.sum(cirrus_img == 2, axis = (1, 2)))
    print(np.sum(cirrus_img == 1, axis = (1, 2)))
    print(np.sum(cirrus_img == 3, axis = (1, 2)))
    print(np.sum(cirrus_img == 9, axis = (1, 2)))
    print(np.sum(cirrus_img == 10, axis = (1, 2)))

    cirrus_img = cirrus_img > 0
    cirrus_img = resize(cirrus_img, (cirrus_img.shape[0], img_20.shape[1], img_20.shape[2]), order = 0, preserve_range = True)
    # Ensure output is within correct range
    img_10 = np.clip(img_10, 0, 1)
    img_20 = np.clip(img_20, 0, 1)

    s2_10_usage = (img_10.shape[1]*img_10.shape[2])/(512*512) * (4/3) * img_10.shape[0]
    return img_10, img_20, np.array(dates_to_download), cirrus_img

#!/usr/bin/env python3

import functools
from time import time, strftime
import os
import os.path
import boto3
import confuse
import rasterio as rs
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.enums import Resampling

import numpy as np
import numpy.ma as ma
import geopandas as gpd
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import pandas as pd
import pandas.api.types as ptypes
import fiona
from contextlib import contextmanager
import math
import requests
import urllib.request
from urllib.error import HTTPError
import osgeo
from osgeo import gdal
from osgeo import gdalconst
import glob
from copy import copy
from datetime import datetime
import psutil

parser = argparse.ArgumentParser(description='Provide a capitalized country name, extent and analysis type.')
parser.add_argument('country', type=str)
parser.add_argument('extent', type=str)
parser.add_argument('incl_hansen', type=bool)
args = parser.parse_args()

def timer(func):
    '''
    Prints the runtime of the decorated function.
    '''

    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start = datetime.now()
        value = func(*args, **kwargs)
        end = datetime.now()
        run_time = end - start
        print(f'Completed {func.__name__!r} in {run_time}.')
        return value
    return wrapper_timer

def download_inputs(country):

    if not os.path.exists(f'{country}/'):
        os.makedirs(f'{country}/')

    config = confuse.Configuration('sentinel-tree-cover')
    config.set_file('jessica-config.yaml')
    aws_access_key = config['aws']['aws_access_key_id']
    aws_secret_key = config['aws']['aws_secret_access_key']
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())

    # download 10m res country tif
    s3.download_file('tof-output',
                     f'2020/mosaics/{country}.tif',
                     f'{country}/{country}.tif')

    # download admin 1 boundaries
    s3.download_file('tof-output',
                     f'2020/analysis/2020-full/admin_boundaries/{country}_adminboundaries.geojson',
                     f'{country}/{country}_adminboundaries.geojson')
    print(f'{country} files downloaded.')

    return None


def create_hansen_tif(country):
    '''
    Identifies the lat/lon coordinates for a single country
    to download Hansen 2010 tree cover and 2020 tree cover loss tif files.
    Returns combined tifs as one file in the country's folder.

    Attributes
    ----------
    country : str
        a string indicating the country files to import

    '''
    gdal.UseExceptions()
    shapefile = gpd.read_file(f'{country}/{country}_adminboundaries.geojson')

    if not os.path.exists(f'hansen_treecover2010'):
        os.makedirs(f'hansen_treecover2010')

    if not os.path.exists(f'hansen_lossyear2020'):
        os.makedirs(f'hansen_lossyear2020')

    # identify min/max bounds for the country
    bounds = shapefile.geometry.bounds
    min_x = bounds.minx.min()
    min_y = bounds.miny.min()
    max_x = bounds.maxx.max()
    max_y = bounds.maxy.max()

    # identify the lowest and highest 10 lat/lon increments for the country
    lower_x = math.floor(min_x / 10) * 10
    lower_y = math.ceil(min_y / 10) * 10
    upper_x = math.ceil(max_x / 10) * 10
    upper_y = math.ceil(max_y / 10) * 10

    print('Downloading files from GLAD...')

    for x_grid in range(lower_x, upper_x, 10):
        for y_grid in range(lower_y, upper_y + 10, 10):

            lon = 'N' if y_grid >= 0 else 'S'
            lat = 'E' if x_grid >= 0 else 'W'

            # establish urls
            lon_lat = f'{str(np.absolute(y_grid)).zfill(2)}{lon}_{str(np.absolute(x_grid)).zfill(3)}{lat}.tif'
            cover_url = f'https://storage.googleapis.com/earthenginepartners-hansen/GFC2015/Hansen_GFC2015_treecover2000_{lon_lat}'
            cover_dest = f'hansen_treecover2010/{lon_lat}'
            loss_url = f'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/Hansen_GFC-2020-v1.8_lossyear_{lon_lat}'
            loss_dest = f'hansen_lossyear2020/{lon_lat}'

            # download tree cover and loss files from UMD website
            try:
                urllib.request.urlretrieve(cover_url, cover_dest)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    print(f'HTTP Error 404 for tree cover data: {cover_url}')
                    pass

            try:
                urllib.request.urlretrieve(loss_url, loss_dest)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    print(f'HTTP Error 404 for tree cover loss data: {loss_url}')
                    pass

    # if the tree cover file doesn't exist, remove loss file
    for tif in os.listdir('hansen_lossyear2020/'):
        if tif not in os.listdir('hansen_treecover2010/'):
            os.remove(f'hansen_lossyear2020/{tif}')

    # create list of tifs and ensure no duplicates
    tree_tifs = glob.glob('hansen_treecover2010/*.tif')
    loss_tifs = glob.glob('hansen_lossyear2020/*.tif')

    # convert tree cover and loss tifs into a virtual raster tile
    gdal.BuildVRT(f'{country}/{country}_hansen_treecover2010.vrt', tree_tifs)
    gdal.BuildVRT(f'{country}/{country}_hansen_loss2020.vrt', loss_tifs)

    # open vrts and convert to a single .tif -- adding tfw=yes increases file size significantly
    translateoptions = gdal.TranslateOptions(format='Gtiff',
                                              outputSRS='EPSG:4326',
                                              outputType=gdal.GDT_Byte,
                                              noData=255,
                                              creationOptions=['COMPRESS=LZW'],
                                              resampleAlg='nearest')

    source = gdal.Open(f'{country}/{country}_hansen_treecover2010.vrt', )
    ds = gdal.Translate(f'{country}/{country}_hansen_treecover2010.tif', source, options=translateoptions)
    os.remove(f'{country}/{country}_hansen_treecover2010.vrt')
    source = None
    ds = None

    source = gdal.Open(f'{country}/{country}_hansen_loss2020.vrt')
    ds = gdal.Translate(f'{country}/{country}_hansen_loss2020.tif', source, options=translateoptions)
    os.remove(f'{country}/{country}_hansen_loss2020.vrt')
    source = None
    ds = None

    assert os.path.exists(f'{country}/{country}_hansen_treecover2010.tif')
    assert os.path.exists(f'{country}/{country}_hansen_loss2020.tif')

    # if new files are properly create, delete what is not needed
    for file in tree_tifs:
        os.remove(file)

    for file in loss_tifs:
        os.remove(file)

    return None


def remove_loss(country):
    '''
    Imports hansen tree cover loss tifs for a single country. Updates tree cover
    to 0 if loss was detected between 2011-2020. Returns updated tif in the country's
    folder.

    Attributes
    ----------
    country : str
        a string indicating the country files to import
    '''
    gdal.UseExceptions()
    hansen_cover = rs.open(f'{country}/{country}_hansen_treecover2010.tif').read(1)
    hansen_loss = rs.open(f'{country}/{country}_hansen_loss2020.tif').read(1)

     # assert raster shape, datatype and max/min values
    assert hansen_cover.dtype == 'uint8'
    assert hansen_cover.shape != (0, ) and len(hansen_cover.shape) <= 2
    assert hansen_cover.max() <= 100 and hansen_cover.min() >= 0
    assert hansen_loss.dtype == 'uint8'
    assert hansen_loss.shape != (0, ) and len(hansen_loss.shape) <= 2
    assert hansen_loss.max() <= 20 and hansen_cover.min() >= 0

    # If there was loss between 2011-2020, make then 0 in tree cover
    sum_before_loss = np.sum(hansen_cover > 0)
    hansen_cover[(hansen_loss >= 11)] = 0.

    # check bin counts after loss removed
    print(f'{sum_before_loss - (np.sum(hansen_cover > 0))} tree cover pixels converted to loss.')

    # write as a new file
    out_meta = rs.open(f'{country}/{country}_hansen_treecover2010.tif').meta
    out_meta.update({'driver': 'GTiff',
                     'dtype': 'uint8',
                     'height': hansen_cover.shape[0],
                     'width': hansen_cover.shape[1],
                     'count': 1,
                     'compress':'lzw'})
    outpath = f'{country}/{country}_hansen_treecover2010_wloss.tif'
    with rs.open(outpath, 'w', **out_meta) as dest:
            dest.write(hansen_cover, 1)

    # remove original hansen tree cover and loss files
    os.remove(f'{country}/{country}_hansen_treecover2010.tif')
    os.remove(f'{country}/{country}_hansen_loss2020.tif')
    hansen_cover = None
    hansen_loss = None

    print('Hansen raster built.')
    return None

def pad_tml_raster(country):

    '''
    Increase the TML raster extent to match the bounds of a country's shapefile
    and fill with no data value to facilitate clipping.

    Attributes
    ----------
    country : str
        a string indicating the country files to import
    '''

    shapefile = gpd.read_file(f'{country}/{country}_adminboundaries.geojson')

    # identify min/max bounds for the country
    bounds = shapefile.geometry.bounds
    min_x = bounds.minx.min()
    min_y = bounds.miny.min()
    max_x = bounds.maxx.max()
    max_y = bounds.maxy.max()

    # create new bounds by rounding to the nearest .1 lat/lon
    lower_x = math.floor(min_x * 10) / 10
    lower_y = math.floor(min_y * 10) / 10
    upper_x = math.ceil(max_x * 10) / 10
    upper_y = math.ceil(max_y * 10) / 10

    # create tif with new bounds
    warp_options = gdal.WarpOptions(format='GTiff',
                                    dstSRS='EPSG:4326',
                                    dstNodata=255,
                                    outputBounds=[lower_x, lower_y, upper_x, upper_y],
                                    resampleAlg='near',
                                    outputType=osgeo.gdalconst.GDT_Byte,
                                    creationOptions=['TFW=YES', 'COMPRESS=LZW', 'BIGTIFF=YES'])

    ds = gdal.Warp(f'{country}/{country}_tof_padded.tif',
                   f'{country}/{country}.tif',
                   options=warp_options)

    ds = None

    return None


def create_clippings(country, multi_analysis):
    '''
    Takes in a country name to import tof/hansen rasters and masks out administrative
    boundaries based on the shapefile. Saves exploded shapefile as a geojson with polygons
    split/numbered for each admin boundary. Returns clipped rasters as individual
    files in the country's "clipped_rasters" folder. Deletes the original Hansen file.

    Attributes
    ----------
    country : str
        a string indicating the country files to import
    '''

    if multi_analysis:
        if not os.path.exists(f'{country}/clipped_rasters/hansen'):
            os.makedirs(f'{country}/clipped_rasters/hansen')

    if not os.path.exists(f'{country}/clipped_rasters/tof'):
        os.makedirs(f'{country}/clipped_rasters/tof')

    if not os.path.exists(f'{country}/clipped_rasters/esa'):
        os.makedirs(f'{country}/clipped_rasters/esa')

    orig_shapefile = gpd.read_file(f'{country}/{country}_adminboundaries.geojson')

    # preprocess shapefile from multipolygon to single
    counter = 0
    for idx, row in orig_shapefile.iterrows():
        counter += 1 if type(row.geometry) == MultiPolygon else 0

    if counter > 0:
        shapefile = orig_shapefile.explode()

        # add integer to admin name if multi polys
        shapefile.NAME_1 = np.where(shapefile.NAME_1.duplicated(keep=False),
                                     shapefile.NAME_1 + shapefile.groupby('NAME_1').cumcount().add(1).astype(str),
                                     shapefile.NAME_1)

        shapefile = shapefile.reset_index()
        shapefile.drop(columns=['level_0', 'level_1'], inplace=True)

    # if no multi polys save original shapefile under new name
    else:
        shapefile = orig_shapefile

    shapefile.to_file(f'{country}/{country}_adminboundaries_exp.geojson', driver='GeoJSON')

    def mask_raster(polygon, admin, raster, folder):
        out_img, out_transform = mask(dataset=raster, shapes=[polygon], crop=True, nodata=255, filled=True)
        out_meta = raster.meta
        out_meta.update({'driver': 'GTiff',
                         'dtype': 'uint8',
                         'height': out_img.shape[1],
                         'width': out_img.shape[2],
                         'transform': out_transform,
                         'compress':'lzw'})
        outpath = f'{country}/clipped_rasters/{folder}/{admin}.tif'
        with rs.open(outpath, 'w', **out_meta) as dest:
            dest.write(out_img)
        out_img = None
        out_transform = None
        return None

    tof_raster_path = f'{country}/{country}_tof_padded.tif'
    esa_raster_path = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif'

    files_to_process = [tof_raster_path, esa_raster_path]
    types_to_process = ['tof', 'esa']

    if multi_analysis:
        files_to_process.append(f'{country}/{country}_hansen_treecover2010_wloss.tif')
        types_to_process.append('hansen')

    for file, file_type in zip(files_to_process, types_to_process):
        with rs.open(file) as raster:
            for polygon, admin in zip(shapefile.geometry, shapefile.NAME_1):
                mask_raster(polygon, admin, raster, file_type)

    # delete Tof and Hansen files once clippings created
    os.remove(f'{country}/{country}_tof_padded.tif')
    os.remove(f'{country}/{country}_tof_padded.tfw')
    if multi_analysis:
        os.remove(f'{country}/{country}_hansen_treecover2010_wloss.tif')

    print(f"{country}'s rasters clipped and saved.")

    return None


def match_extent_and_res(source, reference, out_filename, tof=False, esa=False):

    '''
    GDAL’s nearest neighbor interpolation is used match the
    projection, bounding box and dimensions of the source dataset
    to the reference dataset.
    '''

    # set up the source file
    src = gdal.Open(source, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()
    src_geotrans = src.GetGeoTransform()

    # set up the reference file (esa)
    ref_ds = gdal.Open(reference, gdalconst.GA_ReadOnly)
    ref_proj = ref_ds.GetProjection()
    ref_geotrans = ref_ds.GetGeoTransform()

    # create height/width for the interpolation (ref dataset except for tof)
    width = ref_ds.RasterXSize if not tof else src.RasterXSize
    height = ref_ds.RasterYSize if not tof else src.RasterYSize

    out = gdal.GetDriverByName('GTiff').Create(out_filename, width, height, 1, gdalconst.GDT_Byte, options=['COMPRESS=LZW'])
    rb = out.GetRasterBand(1)
    rb.SetNoDataValue(255)

    # do not adjust the bounds for esa, use source (esa)
    if esa:
        ref_proj = src_proj

    # set geotrans, proj and no data val for the out file
    out.SetGeoTransform(ref_geotrans)
    out.SetProjection(ref_proj)

    interpolation = gdalconst.GRA_NearestNeighbour
    gdal.ReprojectImage(src, out, src_proj, ref_proj, interpolation)

    ref_ds = None
    src = None

    return None


@timer
def apply_extent_res(country, multi_analysis):

    '''
    Applies match_raster_extent_and_res() to all admin files
    for a country. The ESA and Hansen data are upsampled to match
    TOF at 10m resolution. TOF and Hansen et al. data are resized to
    match the dimensions and bounding box of the ESA data.

    Attributes
    ----------
    country : str
        a string indicating the country files to import
    '''

    if multi_analysis:
        if not os.path.exists(f'{country}/resampled_rasters/hansen'):
            os.makedirs(f'{country}/resampled_rasters/hansen')

    if not os.path.exists(f'{country}/resampled_rasters/tof'):
        os.makedirs(f'{country}/resampled_rasters/tof')

    if not os.path.exists(f'{country}/resampled_rasters/esa'):
        os.makedirs(f'{country}/resampled_rasters/esa')


    # import new shapefile containing only polygons
    shapefile = gpd.read_file(f'{country}/{country}_adminboundaries_exp.geojson')
    admin_boundaries = list(shapefile.NAME_1)

    for admin in admin_boundaries:

        # apply to esa
        match_extent_and_res(f'{country}/clipped_rasters/esa/{admin}.tif', # source
                             f'{country}/clipped_rasters/tof/{admin}.tif', # reference
                             f'{country}/resampled_rasters/esa/{admin}.tif', # outpath
                             tof = False,
                             esa = True)

        # apply to tof
        match_extent_and_res(f'{country}/clipped_rasters/tof/{admin}.tif',
                             f'{country}/resampled_rasters/esa/{admin}.tif',
                             f'{country}/resampled_rasters/tof/{admin}.tif',
                             tof = True,
                             esa = False)

        # apply to hansen
        if multi_analysis:
            match_extent_and_res(f'{country}/clipped_rasters/hansen/{admin}.tif',
                                 f'{country}/resampled_rasters/esa/{admin}.tif',
                                 f'{country}/resampled_rasters/hansen/{admin}.tif',
                                 tof = False,
                                 esa = False)

        # assert no data value added correctly in tof rasters
        tof = rs.open(f'{country}/resampled_rasters/tof/{admin}.tif').read(1)
        assert tof.max() <= 255
        tof = None

    return None


def merge_polygons(country, multi_analysis):
    '''
    Takes in a country's resampled rasters and identifies
    which admin boundaries are composed of multipolygons. Combines individual files
    into one for the admin district, then deletes the individual files.

    Attributes
    ----------
    country : str
        a string indicating the country files to import
    '''

    shapefile = gpd.read_file(f'{country}/{country}_adminboundaries_exp.geojson')
    admin_boundaries_all = list(shapefile.NAME_1)

    # creates a list of admins that need to be merged (digits in filename)
    no_ints = []
    for admin in admin_boundaries_all:

        # if any characters are digits, remove them and ad admin to list
        if any(char.isdigit() for char in admin):
            clean_admin = ''.join([char for char in admin if not char.isdigit()])
            no_ints.append(clean_admin)

    no_ints = list(set(no_ints))
    print(f'{len(no_ints)} admins will be merged: {no_ints}')

    datasets = ['tof', 'esa']
    if multi_analysis:
        datasets.append('hansen')

    for data in datasets:
        for admin_2 in no_ints:

            # gather list of files for that admin (ex: Puntarenas1.tif, Puntarenas2.tif, Puntarenas3.tif)
            files_to_merge = [] # items need to be in dataset reader mode
            files_to_delete = [] # items are just string of the file name

            for path in glob.glob(f'{country}/resampled_rasters/{data}/{admin_2}?.tif'):
                filename = os.path.basename(path)
                files_to_delete.append(filename)
                src = rs.open(f'{country}/resampled_rasters/{data}/{filename}')
                files_to_merge.append(src)

            # capture double digits
            for path in glob.glob(f'{country}/resampled_rasters/{data}/{admin_2}??.tif'):
                filename = os.path.basename(path)
                files_to_delete.append(filename)
                src = rs.open(f'{country}/resampled_rasters/{data}/{filename}')
                files_to_merge.append(src)

            # capture triple digits
            for path in glob.glob(f'{country}/resampled_rasters/{data}/{admin_2}???.tif'):
                filename = os.path.basename(path)
                files_to_delete.append(filename)
                src = rs.open(f'{country}/resampled_rasters/{data}/{filename}')
                files_to_merge.append(src)

            if len(files_to_merge) < 1:
                print(f'No files to merge in {data}.')

            mosaic, out_transform = merge(files_to_merge)

            outpath = f'{country}/resampled_rasters/{data}/{admin_2}.tif'
            out_meta = src.meta.copy()
            out_meta.update({'driver': "GTiff",
                             'dtype': 'uint8',
                             'height': mosaic.shape[1],
                             'width': mosaic.shape[2],
                             'transform': out_transform,
                             'compress':'lzw'})

            with rs.open(outpath, "w", **out_meta) as dest:
                dest.write(mosaic)

            # delete the old separated tifs
            for file in files_to_delete:
                os.remove(f'{country}/resampled_rasters/{data}/{file}')

    return None

def processing_check(country):
    '''
    Calculate the area of an admin district in hectares. Convert hectares to bytes to determine
    if the admin can be processed on r5a.2xlarge instance. If it exceeds the processing threshold
    flag the country and save to a csv file.
    '''
    # print size of TML tif


    # Return the current process id
    process = psutil.Process(os.getpid())

    # get rss and calculate return the memory usage in MB
    print(f'Current memory usage: {process.memory_info()[0] / float(2 ** 20)} MB')

    # return the memory usage in percentage like top
    mem = process.memory_percent()
    print(f'Perc memory usage: %{round(mem, 2)}')

    # import and create a copy
    shapefile = gpd.read_file(f'{country}/{country}_adminboundaries.geojson')
    shapefile = shapefile.copy()

    # convert the crs to an equal-area projection to get polygon area in m2
    # then convert to hectares (divide the area value by 10000)
    shapefile['area'] = shapefile['geometry'].to_crs({'init': 'epsg:3395'}).map(lambda x: x.area / 10**4)

    # calculate the size of the largest area, ha --> bytes
    max_area = shapefile['area'].max()
    max_bytes = max_area * 3200
    admin = shapefile.loc[shapefile['area'] == max_area]['NAME_1'].item()

    # create a dataframe to store details
    too_large = pd.DataFrame(columns=['country','admin','file_size','date'], dtype=object)

    # check if it can fit into RAM, otherwise save to csv
    # should be checking the max area in ha?
    if max_bytes >= 6.4e10:
        print(f'The largest admin in {country} is {admin}. Area: {round(max_area, 2)} ha')
        print(f'Warning: That largest admin {admin} is too large to process. As np.float32, array is ({round(max_bytes/10e9, 2)} GB)')
        too_large.append({'country': country,
                         'largest_admin': admin,
                         'file_size': max_bytes,
                         'area': round(max_area, 2),
                         'date': datetime.now().strftime("%m/%d/%Y, %H:%M:%S")}, ignore_index=True)

        too_large.to_csv('bigtiff_full_2020.csv', mode='a', header=False)
    else:
        print('Passed processing check.')
    return None

def reshape_to_4d(raster):

    '''
    Takes in a GTiff, identifies the dimensions and them down to the nearest 10th.
    Then uses those dimensions and reshapes to a 4 dimensional, 10x10 grid.

    Attributes
    ----------
    raster : str
        GTiff that will be reshaped
    '''

    def round_down(num, divisor):
         return num - (num%divisor)

    # round down rows and cols to nearest 10th
    rows, cols = round_down(raster.shape[0], 10), round_down(raster.shape[1], 10)

    # clip according to rounded numbers and reshape
    rounded = raster[:rows, :cols]
    reshaped = np.reshape(rounded, (rounded.shape[0] // 10, 10, rounded.shape[1] // 10, 10))

    return reshaped

@timer
def calculate_stats_tml(country, extent):

    '''
    Takes in a country and extent (full or partial) to import appropriate rasters. Returns a csv
    with statistics per administrative district, per land cover class and per tree cover
    threshold. Only produces statistics for TML.

    Attributes
    ----------
    country : str
        a string indicating the country files to import

    '''

    if not os.path.exists(f'{country}/stats'):
        os.makedirs(f'{country}/stats')

    df = pd.DataFrame({'country': pd.Series(dtype='str'),
                       'admin': pd.Series(dtype='str'),
                       'esa_id': pd.Series(dtype='str'),
                       'esa_class': pd.Series(dtype='str'),
                       'esa_sampled_ha': pd.Series(dtype='float64'),
                       'esa_total_ha': pd.Series(dtype='float64'),
                       'tree_cover_class': pd.Series(dtype='str'),
                       'tof_ha': pd.Series(dtype='int64'),
                       'tof_mean': pd.Series(dtype='float64')})
    counter = 0

    folder_contents = [f for f in os.listdir(f'{country}/resampled_rasters/tof') if f != '.ipynb_checkpoints']

    # iterate through the admins
    for file in folder_contents:
        counter += 1

        tof = rs.open(f'{country}/resampled_rasters/tof/{file}').read(1)
        esa = rs.open(f'{country}/resampled_rasters/esa/{file}').read(1)

        lower_rng = [x for x in range(0, 100, 10)]
        upper_rng = [x for x in range(10, 110, 10)]
        esa_classes = np.unique(esa)

        for cover in esa_classes:

            # replace all values not equal to the current lcc with no data values
            tof_class = tof.copy()
            tof_class[esa != cover] = 255

            # reshape to a 4d array and apply mask
            tof_reshaped = reshape_to_4d(tof_class)
            tof_reshaped = np.ma.masked_equal(tof_reshaped, 255)

            # count the number of non-masked entries per hectare
            tof_class_count_per_ha = np.sum(~tof_reshaped.mask, axis=(1,3), dtype=np.uint8)

            # get sum of values themselves that are not masked
            tof_class_sum_per_ha = np.sum(tof_reshaped, axis=(1,3), dtype=np.uint16)

            # divide the sum by the count (to avoid using np.mean which will use np.float)
            tof_class_mean_per_ha = np.divide(tof_class_sum_per_ha, tof_class_count_per_ha, dtype=np.float32)

            # Return all the non-masked data as a 1-D array (prevent mask from propagating)
            tof_class_mean_per_ha = tof_class_mean_per_ha.compressed()

            tof_class_mean = np.round(np.mean(tof_class_mean_per_ha), 2)

            # calculate the area sampled
            lc_total = np.sum(esa == cover)/100
            lc_sampled = np.sum(~tof_reshaped.mask)/100

            # iterate through the thresholds (0-10, 10-20, 20-30)
            for lower, upper in zip(lower_rng, upper_rng):

                # calculate total ha for that threshold
                tof_bin = np.sum((tof_class_mean_per_ha >= lower) & (tof_class_mean_per_ha < upper))
                bin_name = (f'{str(lower)}-{str(upper - 1)}')

                # confirm masked array doesn't propogate
                vars_to_check = [lc_sampled, lc_total, tof_bin, tof_class_mean]

                for index, var in enumerate(vars_to_check):
                    if var == '--':
                        var = 0

                # check for erroneous values
                assert lc_sampled <= lc_total, f'Sampled area is greater than total area for land cover {cover} in {file}.'

                df = df.append({'country': country,
                               'admin': file[:-4],
                               'esa_id': cover,
                               'esa_sampled_ha': lc_sampled,
                               'esa_total_ha': lc_total,
                               'tree_cover_class': bin_name,
                               'tof_ha': tof_bin,
                               'tof_mean': tof_class_mean},
                                ignore_index=True)

                # reinforce datatypes
                convert_dict = {'esa_sampled_ha':'float64',
                                'esa_total_ha':'float64',
                                'tof_ha':'int64',
                                'tof_mean': 'float64'}
                df = df.astype(convert_dict)

        # map ESA id numbers to lcc labels
        esa_legend = {0: 'ESA No Data',
                10: 'Cropland, rainfed',
                11: 'Cropland, rainfed',
                12: 'Cropland, rainfed',
                20: 'Cropland, irrigated or post-flooding',
                30: 'Mosaic cropland / natural vegetation',
                40: 'Mosaic natural vegetation / cropland',
                50: 'Tree cover, broadleaved, evergreen',
                60: 'Tree cover, broadleaved, deciduous',
                61: 'Tree cover, broadleaved, deciduous',
                62: 'Tree cover, broadleaved, deciduous',
                70: 'Tree cover, needleleaved, evergreen',
                71: 'Tree cover, needleleaved, evergreen',
                72: 'Tree cover, needleleaved, evergreen',
                80: 'Tree cover, needleleaved, deciduous',
                81: 'Tree cover, needleleaved, deciduous',
                82: 'Tree cover, needleleaved, deciduous',
                90: 'Tree cover, mixed leaf type',
                100: 'Mosaic tree and shrub / herbaceous cover',
                110: 'Mosaic herbaceous cover / tree and shrub',
                120: 'Shrubland',
                121: 'Shrubland',
                122: 'Shrubland',
                130: 'Grassland',
                140: 'Lichens and mosses',
                150: 'Sparse vegetation',
                151: 'Sparse vegetation',
                152: 'Sparse vegetation',
                153: 'Sparse vegetation',
                160: 'Tree cover, flooded, fresh or brakish water',
                170: 'Tree cover, flooded, saline water',
                180: 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water',
                190: 'Urban areas',
                200: 'Bare areas',
                201: 'Bare areas',
                202: 'Bare areas',
                210: 'Water bodies',
                220: 'Permanent snow and ice',
                255: 'No Data (flag)'}

        df['esa_class'] = df['esa_id'].map(esa_legend)

        tof = None
        esa = None

        if counter % 3 == 0:
            print(f'{counter}/{len(folder_contents)} admins processed...')

    cols_to_check = ['esa_sampled_ha', 'esa_total_ha', 'tof_ha', 'tof_mean']
    assert all(ptypes.is_numeric_dtype(df[col]) for col in cols_to_check)

    df.to_csv(f'{country}/stats/{country}_statistics_{extent}_tmlonly.csv', index=False)
    print('Analysis complete.')

    return None

@timer
def calculate_stats(country, extent):

    '''
    Takes in a country and extent (partial or full) to import appropriate tml/hansen/esa rasters.
    Returns a csv with statistics per administrative district, per land cover class and per tree cover
    threshold. Includes Hansen and TML statistics.

    Attributes
    ----------
    country : str
        a string indicating the country files to import

    '''

    if not os.path.exists(f'{country}/stats'):
        os.makedirs(f'{country}/stats')

    df = pd.DataFrame({'country': pd.Series(dtype='str'),
                       'admin': pd.Series(dtype='str'),
                       'esa_id': pd.Series(dtype='str'),
                       'esa_class': pd.Series(dtype='str'),
                       'esa_sampled_ha': pd.Series(dtype='float64'),
                       'esa_total_ha': pd.Series(dtype='float64'),
                       'tree_cover_class': pd.Series(dtype='str'),
                       'tof_ha': pd.Series(dtype='int64'),
                       'hans_ha': pd.Series(dtype='int64'),
                       'tof_mean': pd.Series(dtype='float64'),
                       'hans_mean': pd.Series(dtype='float64')})

    counter = 0

    folder_contents = [f for f in os.listdir(f'{country}/resampled_rasters/tof') if f != '.ipynb_checkpoints']

    # iterate through the admins
    for file in folder_contents:
        counter += 1
        tof = rs.open(f'{country}/resampled_rasters/tof/{file}').read(1)
        hans = rs.open(f'{country}/resampled_rasters/hansen/{file}').read(1)
        esa = rs.open(f'{country}/resampled_rasters/esa/{file}').read(1)

        lower_rng = [x for x in range(0, 100, 10)]
        upper_rng = [x for x in range(10, 110, 10)]

        # iterate through the land cover classes
        esa_classes = np.unique(esa)

        for cover in esa_classes:

            # replace all values not equal to the current lcc with no data values
            tof_class = tof.copy()
            tof_class[esa != cover] = 255

            # reshape to a 4d array and apply mask
            tof_reshaped = reshape_to_4d(tof_class)
            tof_reshaped = np.ma.masked_equal(tof_reshaped, 255)

            # count the number of non-masked entries per hectare
            tof_class_count_per_ha = np.sum(~tof_reshaped.mask, axis=(1,3), dtype=np.uint8)

            # get sum of values themselves that are not masked
            tof_class_sum_per_ha = np.sum(tof_reshaped, axis=(1,3), dtype=np.uint16)

            # divide the sum by the count (to avoid using np.mean which will use np.float)
            tof_class_mean_per_ha = np.divide(tof_class_sum_per_ha, tof_class_count_per_ha, dtype=np.float32)

            # check the conversion to hectares - should be 10x smaller than tof_class
            #print(f'Conversion check. Original: {tof_class.shape} New: {tof_class_mean_per_ha.shape}')

            # Return all the non-masked data as a 1-D array (prevent mask from propagating)
            tof_class_mean_per_ha = tof_class_mean_per_ha.compressed()

            tof_class_mean = np.round(np.mean(tof_class_mean_per_ha), 2)

            # apply same steps to hansen
            hans_class = hans.copy()
            hans_class[esa != cover] = 255
            hans_reshaped = reshape_to_4d(hans_class)
            hans_reshaped = np.ma.masked_equal(hans_reshaped, 255)
            hans_class_count_per_ha = np.sum(~hans_reshaped.mask, axis=(1,3), dtype=np.uint8)
            hans_class_sum_per_ha = np.sum(hans_reshaped, axis=(1,3), dtype=np.uint16)
            hans_class_mean_per_ha = np.divide(hans_class_sum_per_ha, hans_class_count_per_ha, dtype=np.float32)
            hans_class_mean_per_ha = hans_class_mean_per_ha.compressed()
            hans_class_mean = np.round(np.mean(hans_class_mean_per_ha), 2)

            # calculate the area sampled
            lc_total = np.sum(esa == cover)/100
            lc_sampled = np.sum(~tof_reshaped.mask)/100

            # iterate through the thresholds (0-10, 10-20, 20-30)
            for lower, upper in zip(lower_rng, upper_rng):

                # calculate total ha for that threshold
                tof_bin = np.sum((tof_class_mean_per_ha >= lower) & (tof_class_mean_per_ha < upper))
                bin_name = (f'{str(lower)}-{str(upper - 1)}')
                hans_bin = np.sum((hans_class_mean_per_ha >= lower) & (hans_class_mean_per_ha < upper))

                # confirm masked array doesn't propogate
                vars_to_check = [lc_sampled, lc_total, tof_bin, hans_bin, tof_class_mean, hans_class_mean]

                for index, var in enumerate(vars_to_check):
                    if var == '--':
                        var = 0

                # check for erroneous values
                assert lc_sampled <= lc_total, f'Sampled area is greater than total area for land cover {cover} in {file}.'

                df = df.append({'country': country,
                               'admin': file[:-4],
                               'esa_id': cover,
                               'esa_sampled_ha': lc_sampled,
                               'esa_total_ha': lc_total,
                               'tree_cover_class': bin_name,
                               'tof_ha': tof_bin,
                               'hans_ha': hans_bin,
                               'tof_mean': tof_class_mean,
                               'hans_mean': hans_class_mean},
                                ignore_index=True)

                # reinforce datatypes
                convert_dict = {'esa_sampled_ha':'float64',
                                'esa_total_ha':'float64',
                                'tof_ha':'int64',
                                'hans_ha':'int64',
                                'tof_mean': 'float64',
                                'hans_mean': 'float64'}
                df = df.astype(convert_dict)

        # map ESA id numbers to lcc labels
        esa_legend = {0: 'ESA No Data',
                10: 'Cropland, rainfed',
                11: 'Cropland, rainfed',
                12: 'Cropland, rainfed',
                20: 'Cropland, irrigated or post-flooding',
                30: 'Mosaic cropland / natural vegetation',
                40: 'Mosaic natural vegetation / cropland',
                50: 'Tree cover, broadleaved, evergreen',
                60: 'Tree cover, broadleaved, deciduous',
                61: 'Tree cover, broadleaved, deciduous',
                62: 'Tree cover, broadleaved, deciduous',
                70: 'Tree cover, needleleaved, evergreen',
                71: 'Tree cover, needleleaved, evergreen',
                72: 'Tree cover, needleleaved, evergreen',
                80: 'Tree cover, needleleaved, deciduous',
                81: 'Tree cover, needleleaved, deciduous',
                82: 'Tree cover, needleleaved, deciduous',
                90: 'Tree cover, mixed leaf type',
                100: 'Mosaic tree and shrub / herbaceous cover',
                110: 'Mosaic herbaceous cover / tree and shrub',
                120: 'Shrubland',
                121: 'Shrubland',
                122: 'Shrubland',
                130: 'Grassland',
                140: 'Lichens and mosses',
                150: 'Sparse vegetation',
                151: 'Sparse vegetation',
                152: 'Sparse vegetation',
                153: 'Sparse vegetation',
                160: 'Tree cover, flooded, fresh or brakish water',
                170: 'Tree cover, flooded, saline water',
                180: 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water',
                190: 'Urban areas',
                200: 'Bare areas',
                201: 'Bare areas',
                202: 'Bare areas',
                210: 'Water bodies',
                220: 'Permanent snow and ice',
                255: 'No Data (flag)'}

        df['esa_class'] = df['esa_id'].map(esa_legend)

        tof = None
        esa = None
        hans = None

        if counter % 3 == 0:
            print(f'{counter}/{len(folder_contents)} admins processed...')

    cols_to_check = ['esa_sampled_ha', 'esa_total_ha', 'tof_ha', 'hans_ha', 'tof_mean', 'hans_mean']
    assert all(ptypes.is_numeric_dtype(df[col]) for col in cols_to_check)

    df.to_csv(f'{country}/stats/{country}_statistics_{extent}.csv', index=False)
    print('Analysis complete.')

    return None

@timer
def upload_dir(source_dir, bucket, object_name):
    """
    Upload a directory to an S3 bucket.

    file_name: File to upload
    bucket: Bucket to upload to
    object_name: S3 object name. If not specified then file_name is used

    """
    config = confuse.Configuration('sentinel-tree-cover')
    config.set_file('jessica-config.yaml')
    aws_access_key = config['aws']['aws_access_key_id']
    aws_secret_key = config['aws']['aws_secret_access_key']
    session = boto3.Session(aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())
    s3 = session.resource('s3')
    bucket = s3.Bucket(bucket)

    # use directory tree generator to get list of file paths and upload each
    for subdir, dirs, files in os.walk(source_dir):

        for file in files:
            dest_path = os.path.join(subdir, file)

            with open(dest_path, 'rb') as data:
                bucket.put_object(Key=object_name+dest_path, Body=data)

    print('Upload complete.')

    return None

@timer
def execute_pipe(country, extent, incl_hansen=True):
    print(f'Started at: {datetime.now().strftime("%H:%M:%S")}')
    print('Downloading input data...')
    download_inputs(country)
    if incl_hansen:
        print('Building Hansen tree cover raster...')
        create_hansen_tif(country)
        print('Removing tree cover loss...')
        remove_loss(country)
    print('Padding tml raster...')
    pad_tml_raster(country)
    print('Clipping rasters by admin boundary...')
    create_clippings(country, multi_analysis=incl_hansen)
    print('Resampling to match raster extents and resolutions...')
    apply_extent_res(country, multi_analysis=incl_hansen)
    print('Merging admins containing multiple polygons...')
    merge_polygons(country, multi_analysis=incl_hansen)
    print('Checking size...')
    processing_check(country)
    print('Calculating statistics...')
    if incl_hansen:
        calculate_stats(country, extent)
    else:
        calculate_stats_tml(country, extent)
    print('Uploading files to s3...')
    upload_dir(country, 'tof-output', '2020/analysis/2020-full/')
    print(f'Finished {extent} processing at: {datetime.now().strftime("%H:%M:%S")}')
    return None

def main():
    country = args.country
    extent = args.extent
    incl_hansen = args.incl_hansen
    execute_pipe(country, extent, incl_hansen)

if __name__ ==  "__main__":
    main()

#!/usr/bin/env python3

import os
import rasterio as rs
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.enums import Resampling
from rasterio import Affine, MemoryFile

import numpy as np 
import numpy.ma as ma 
import pyproj
import geopandas as gpd 
import shapely
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon
import pandas as pd
import fiona
from contextlib import contextmanager  
from skimage.transform import resize
import math
import urllib.request
import osgeo
from osgeo import gdal
from osgeo import gdalconst
import glob
from copy import copy
import argparse

parser = argparse.ArgumentParser(description='Provide a capitalized country name.')
parser.add_argument('country', type=str)
args = parser.parse_args()

def shp_to_gjson(country):
    '''
    Imports a country shapefile, translates and saves it as 
    a geojson, confirming the correct CRS and absence of 
    duplicates. Prints the number of admin 1 districts.
    
    Attributes
    ----------
    country : str
        a string indicating the country files to import
    
    '''
    if country == 'Costa Rica':
        return 'Using existing geojson file for Costa Rica.'
    else: 
        shapefile = glob.glob(f'{country}/shapefile/*.shp')
        new_shp = gpd.read_file(shapefile[0])
        new_shp.to_file(f'{country}/{country}_adminboundaries.geojson', driver='GeoJSON')
        print(f'There are {len(new_shp)} admins in {country}.')
        assert new_shp.crs == 'epsg:4326'
        assert new_shp.NAME_1.duplicated().sum() == 0
    return None


def create_hansen_tif(country):
    '''
    Identifies the latitude and longitude coordinates for a country 
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

    # create a list of tif file names for the country
    tree_cover_files = []
    loss_files = []
    
    print('Downloading files from GLAD...')
    
    for x_grid in range(lower_x, upper_x, 10):
        for y_grid in range(lower_y, upper_y + 10, 10):
            
            lon = 'N' if y_grid >= 0 else 'S'
            lat = 'E' if x_grid >= 0 else 'W'

            # download tree cover and loss files from UMD
            cover_url =  f'https://glad.umd.edu/Potapov/TCC_2010/treecover2010_' \
                         f'{str(y_grid).zfill(2)}{lon}_{str(np.absolute(x_grid)).zfill(3)}{lat}.tif'
            cover_dest = f'hansen_treecover2010/{str(y_grid).zfill(2)}{lon}_{str(np.absolute(x_grid)).zfill(3)}{lat}.tif'

            try:
                urllib.request.urlretrieve(cover_url, cover_dest)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    print(f'HTTP Error 404: {cover_url}')
                    
            loss_url =  f'https://storage.googleapis.com/earthenginepartners-hansen/GFC-2020-v1.8/Hansen_GFC-2020-v1.8_lossyear_' \
                         f'{str(y_grid).zfill(2)}{lon}_{str(np.absolute(x_grid)).zfill(3)}{lat}.tif'
            loss_dest = f'hansen_lossyear2020/{str(y_grid).zfill(2)}{lon}_{str(np.absolute(x_grid)).zfill(3)}{lat}.tif'

            try:
                urllib.request.urlretrieve(loss_url, loss_dest)
            except urllib.error.HTTPError as err:
                if err.code == 404:
                    print(f'HTTP Error 404: {loss_url}')
            
            if not os.path.exists(cover_dest) or not os.path.exists(loss_dest):
                print(f'Files did not download.')
                
            tree_cover_files.append(cover_dest)
            loss_files.append(loss_dest)
    
    # remove duplicate file names
    tree_tifs = [x for x in tree_cover_files if os.path.exists(x)] 
    loss_tifs = [x for x in loss_files if os.path.exists(x)]
    
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
    for file in tree_cover_files:
        os.remove(file)
    
    for file in loss_files:
        os.remove(file)
    
    print('Hansen raster built.')
    return None


def remove_loss(country):
    '''
    Takes in a country name to import hansen tree cover loss tifs. Updates tree cover 
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
    hansen_cover_new = np.where((hansen_loss >= 11) & (hansen_loss <= 20), 0, hansen_cover)
    
    # check bin counts after loss removed
    print(f'{(np.sum(hansen_cover > 0)) - (np.sum(hansen_cover_new > 0))} pixels converted to loss.')
    
    # write as a new file
    out_meta = rs.open(f'{country}/{country}_hansen_treecover2010.tif').meta
    out_meta.update({'driver': 'GTiff',    
                     'dtype': 'uint8',
                     'height': hansen_cover_new.shape[0],
                     'width': hansen_cover_new.shape[1],
                     'count': 1,
                     'compress':'lzw'})
    outpath = f'{country}/{country}_hansen_treecover2010_wloss.tif'
    with rs.open(outpath, 'w', **out_meta) as dest:
            dest.write(hansen_cover_new, 1) 
    
    # remove original hansen tree cover and loss files
    os.remove(f'{country}/{country}_hansen_treecover2010.tif')
    os.remove(f'{country}/{country}_hansen_loss2020.tif')
    hansen_cover = None
    hansen_loss = None 
    
    return None

def pad_tcl_raster(country):
    
    '''
    Increase the raster extent to match the bounds of a country's shapefile
    and fill with no data value.
    
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


def create_clippings(country):
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
        out_img, out_transform = mask(dataset=raster, shapes=[polygon], crop=True, nodata=255, filled = False)
        out_meta = raster.meta
        out_meta.update({'driver': 'GTiff',    
                         'dtype': 'uint8',
                         'height': out_img.shape[1],
                         'width': out_img.shape[2],
                         'transform': out_transform})
        outpath = f'{country}/clipped_rasters/{folder}/{admin}.tif'
        with rs.open(outpath, 'w', **out_meta) as dest:
            dest.write(out_img)
        out_img = None
        out_transform = None
        return None
    
    tof_raster_path = f'{country}/{country}_tof_padded.tif'
    hansen_raster_path = f'{country}/{country}_hansen_treecover2010_wloss.tif'
    esa_raster_path = 'ESACCI-LC-L4-LCCS-Map-300m-P1Y-2015-v2.0.7.tif'
    
    files_to_process = [tof_raster_path, hansen_raster_path, esa_raster_path]
    types_to_process = ['tof', 'hansen', 'esa']
    
    for file, file_type in zip(files_to_process, types_to_process):
        with rs.open(file) as raster:
            for polygon, admin in zip(shapefile.geometry, shapefile.NAME_1):
                #print(f"Clipping {admin}: {file_type}")
                mask_raster(polygon, admin, raster, file_type)

    
    # delete Tof and Hansen files once clippings created
    os.remove(f'{country}/{country}_hansen_treecover2010_wloss.tif')
    os.remove(f'{country}/{country}_tof_padded.tif')
    os.remove(f'{country}/{country}_tof_padded.tfw')
    
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

    return None


def apply_extent_res(country):
    
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
        match_extent_and_res(f'{country}/clipped_rasters/hansen/{admin}.tif', 
                             f'{country}/resampled_rasters/esa/{admin}.tif', 
                             f'{country}/resampled_rasters/hansen/{admin}.tif', 
                             tof = False, 
                             esa = False) 
        
        # assert no data value added correctly in tof rasters
        tof = rs.open(f'{country}/resampled_rasters/tof/{admin}.tif').read(1)
        assert tof.max() <= 255
        
    return None


def merge_polygons(country):
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

    datasets = ['tof', 'hansen', 'esa']
    
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
                             'transform': out_transform})

            with rs.open(outpath, "w", **out_meta) as dest:
                dest.write(mosaic)

            # delete the old separated tifs
            for file in files_to_delete:
                os.remove(f'{country}/resampled_rasters/{data}/{file}')

    return None

def reshape_to_4d(raster):
    
    '''
    Takes in a GTiff, identifies the dimensions and them down to the nearest 10th.
    Returns a reshaped 10x10 grid array. 
    
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


def calculate_stats(country):
    
    '''
    Takes in a country to import appropriate tof/hansen/esa rasters. Returns a csv 
    with statistics per administrative district, per land cover class and per tree cover
    threshold.
    
    Attributes
    ----------
    country : str
        a string indicating the country files to import

    '''
    
    if not os.path.exists(f'{country}/stats'):
        os.makedirs(f'{country}/stats')
        
    # set up the dataframe
    df = pd.DataFrame(columns=['country','admin','esa_id','esa_class',
                               'esa_sampled_ha','esa_total_ha','tree_cover_class',
                               'tof_ha','hans_ha', 'tof_mean', 'hans_mean'], dtype=object) 
    counter = 0
    
    folder_contents = [f for f in os.listdir(f'{country}/resampled_rasters/tof') if f != '.ipynb_checkpoints']
    
    # iterate through the admins
    for file in folder_contents:
        
        counter += 1
        print(f'Importing {file} rasters.')
        tof = rs.open(f'{country}/resampled_rasters/tof/{file}').read(1).astype(np.float32)
        hans = rs.open(f'{country}/resampled_rasters/hansen/{file}').read(1).astype(np.float32)
        esa = rs.open(f'{country}/resampled_rasters/esa/{file}').read(1).astype(np.float32)
        
        lower_rng = [x for x in range(0, 100, 10)]
        upper_rng = [x for x in range(10, 110, 10)]

        # convert values to their median for binning
        for lower, upper in zip(lower_rng, upper_rng):
            
            tof[(tof >= lower) & (tof < upper)] = lower + 4.5
            hans[(hans >= lower) & (hans < upper)] = lower + 4.5
    
        # iterate through the land cover classes
        esa_classes = np.unique(esa)
        
        if 0 and 255 in esa_classes:
            print('ESA contains lc labels 0 and 255.')
        
        for cover in esa_classes:
            print(f'Calculating means for {cover}.')
            # change all values that are not equal to the lcc to NaN including no data vals
            tof_class = tof.copy()
            tof_class[esa != cover] = np.nan 
            tof_class[tof_class == 255] = np.nan

            # reshape and calculate stats
            # if the entire array in NaNs then tof mean = 0
            tof_reshaped = reshape_to_4d(tof_class) 
            tof_class_mean = np.nanmean(tof_reshaped)
            tof_class_mean_per_ha = np.nanmean(tof_reshaped, axis=(1,3))

            # same for Hansen
            hans_class = hans.copy()
            hans_class[esa != cover] = np.nan
            hans_class[hans_class == 255] = np.nan

            hans_reshaped = reshape_to_4d(hans_class)
            hans_class_mean = np.nanmean(hans_reshaped)
            hans_class_mean_per_ha = np.nanmean(hans_reshaped, axis=(1,3)) 

            # iterate through the thresholds (0-10, 10-20, 20-30)
            for lower, upper in zip(lower_rng, upper_rng):
                print('Thresholding values.')
                # calculate total ha for that threshold 
                tof_bin = np.sum((tof_class_mean_per_ha >= lower) & (tof_class_mean_per_ha < upper))
                hans_bin = np.sum((hans_class_mean_per_ha >= lower) & (hans_class_mean_per_ha < upper))
                bin_name = (f'{str(lower)}-{str(upper - 1)}')
    
                # area of lc sampled (tof is NOT null) and total area (esa raster equals cover)
                # /100 converts 10m data to hectares
                lc_sampled = np.sum(~np.isnan(tof_class)) / 100   
                
                # need to ensure this counts the no data class correctly (no data label is 0.0)
                lc_total = np.count_nonzero(esa == cover)/100 if cover == 0.0 else np.sum(esa == cover)/100
                
                # check for erroneous calculations
                if lc_sampled > lc_total:
                    raise ValueError(f'Sampled area is greater than total area for land cover {cover} in {file}.')
                    
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
            print('Appended to dataframe.')
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
                70: 'Tree cover, needleleaved, evergreen',
                80: 'Tree cover, needleleaved, deciduous',
                90: 'Tree cover, mixed leaf type',
                100: 'Mosaic tree and shrub / herbaceous cover',
                110: 'Mosaic herbaceous cover / tree and shrub',
                120: 'Shrubland',
                130: 'Grassland',
                140: 'Lichens and mosses',
                150: 'Sparse vegetation',
                160: 'Tree cover, flooded, fresh or brakish water',
                170: 'Tree cover, flooded, saline water',
                180: 'Shrub or herbaceous cover, flooded, fresh/saline/brakish water',
                190: 'Urban areas',
                200: 'Bare areas',
                210: 'Water bodies',
                220: 'Permanent snow and ice',
                255: 'No Data (flag)'}
     
        df['esa_class'] = df['esa_id'].map(esa_legend)
        
        if counter % 3 == 0:
            print(f'{counter}/{len(folder_contents)} admins processed...')
    
    df.to_csv(f'{country}/stats/{country}_statistics.csv', index=False)
    
    return None

def execute_pipe(country):

    print('Converting shapefile to geojson...')
    shp_to_gjson(country)
    print('Building Hansen tree cover raster...')
    create_hansen_tif(country)
    print('Removing tree cover loss...')
    remove_loss(country)
    print('Padding tof raster...')
    pad_tcl_raster(country)
    print('Clipping rasters by admin boundary...')
    create_clippings(country)
    print('Resampling to match raster extents and resolutions...')
    apply_extent_res(country)
    print('Merging admins containing multiple polygons...')
    merge_polygons(country)
    print('Data preparation complete.')
    print('Calculating statistics...')
    calculate_stats(country)
    print('Analysis complete.')

    return None


def main():
    country = args.country
    execute_pipe(country)

if __name__ ==  "__main__":
    main()
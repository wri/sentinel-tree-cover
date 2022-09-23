import datetime
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
import scipy.sparse as sparse
import yaml

from collections import Counter
from random import shuffle
from scipy.sparse.linalg import splu
from sentinelhub import WmsRequest, WcsRequest, MimeType
from sentinelhub import CRS, BBox, constants, DataSource, CustomUrlParam
from skimage.transform import resize
from pyproj import Proj, transform
from typing import List, Any, Tuple
from shapely.geometry import Point, Polygon

def calculate_bbx_pyproj(coord: Tuple[float, float],
                         step_x: int, step_y: int,
                         expansion: int, multiplier: int = 1.) -> (Tuple[float, float], 'CRS'):
    ''' Calculates the four corners of a bounding box
        [bottom left, top right] as well as the UTM EPSG using Pyproj

        Note: The input for this function is (x, y), not (lat, long)

        Parameters:
         coord (tuple): Initial (long, lat) coord
         step_x (int): X tile number of a 6300x6300 meter tile
         step_y (int): Y tile number of a 6300x6300 meter tile
         expansion (int): Typically 10 meters - the size of the border for the predictions
         multiplier (int): Currently deprecated

        Returns:
         coords (tuple):
         CRS (int):
    '''

    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))



    coord_utm =  transform(inproj, outproj, coord[1], coord[0])
    coord_utm_bottom_left = (coord_utm[0] + step_x*6300 - expansion,
                             coord_utm[1] + step_y*6300 - expansion)

    coord_utm_top_right = (coord_utm[0] + (step_x+multiplier) * 6300 + expansion,
                           coord_utm[1] + (step_y+multiplier) * 6300 + expansion)

    zone = str(outproj_code)[3:]
    direction = 'N' if coord[1] >= 0 else 'S'
    utm_epsg = "UTM_" + zone + direction
    return (coord_utm_bottom_left, coord_utm_top_right), CRS[utm_epsg]


def calculate_epsg(points: Tuple[float, float]) -> int:
    """ Calculates the UTM EPSG of an input WGS 84 lon, lat

        Parameters:
         points (tuple): input longitiude, latitude tuple

        Returns:
         epsg_code (int): integer form of associated UTM EPSG
    """
    lon, lat = points[0], points[1]
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(epsg_code)


def PolygonArea(corners: Tuple[float, float]) -> float:
    """ Calculates the area in meters squared of an input bounding box
    """
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area)
    return area


def offset_x(coord: Tuple[float, float], offset: int) -> tuple:
    ''' Converts a WGS 84 to UTM, adds meters, and converts back'''

    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))

    coord_utm = transform(inproj, outproj, coord[1], coord[0])
    coord_utm = list(coord_utm)
    coord_utm[0] += offset
    return coord_utm


def offset_y(coord: Tuple[float, float], offset: int) -> tuple:
    ''' Converts a WGS 84 to UTM, adds meters, and converts back'''
    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))

    coord_utm = transform(inproj, outproj, coord[1], coord[0])
    coord_utm = list(coord_utm)
    coord_utm[1] += offset
    return coord_utm


def bounding_box(point: Tuple[float, float],
                 x_offset_max: int = 140,
                 y_offset_max: int = 140,
                 expansion: int = 10) -> Tuple[float, float]:

    tl = point

    epsg = calculate_epsg(tl)
    tl = convertCoords(tl, 4326, epsg)

    br = (tl[0], tl[1])
    tl = ((tl[0] + (x_offset_max)), (tl[1] + (y_offset_max )))
    distance1 = tl[0] - br[0]
    distance2 = tl[1] - br[1]

    br = [a - expansion for a in br]
    tl = [a + expansion for a in tl]

    after = [b - a for a,b in zip(br, tl)]
    br = convertCoords(br, epsg, 4326)
    tl = convertCoords(tl, epsg, 4326)

    min_x = tl[0] # original X offset - 10 meters
    max_x = br[0] # original X offset + 10*GRID_SIZE meters

    min_y = tl[1] # original Y offset - 10 meters
    max_y = br[1] # original Y offset + 10 meters + 140 meters
    # (min_x, min_y), (max_x, max_y)
    # (bl, tr)
    return [(min_x, min_y), (max_x, max_y)]


def calculate_and_save_best_images(img_bands: np.ndarray,
                                   image_dates: np.ndarray) -> (np.ndarray, int):
    """ Interpolate input data of (Time, X, Y, Band) to a constant
        (72, X, Y, Band) shape with one time step every five days

        Parameters:
         img_bands (arr):
         image_dates (list):

        Returns:
         keep_steps (arr):
         max_distance (int)
    """
    image_dates = np.array(image_dates)
    image_dates[image_dates < -100] = image_dates[image_dates < -100] % 365
    biweekly_dates = [day for day in range(0, 360, 15)] # ideal imagery dates are every 15 days

    # Identify the dates where there is < 20% cloud cover
    satisfactory_ids = [x for x in range(0, img_bands.shape[0])]
    satisfactory_dates = [value for idx, value in enumerate(image_dates) if idx in satisfactory_ids]

    selected_images = {}
    for i in biweekly_dates:
        distances = np.array([(date - i) for date in satisfactory_dates])
        # Number of days prior and after the selected date of the nearest clean imagery
        closest = np.min(abs(distances))
        closest_id = np.argmin(abs(distances))
        # If there is imagery within 10 days, select it
        #if closest <= 30:
        #    image_idx = np.argwhere(abs(distances) <= 30).flatten()
        #    date = np.array(satisfactory_dates)[image_idx]
        #    image_ratio = [1 / len(image_idx)] * len(image_idx)
            #image_idx = int(np.argwhere(np.array(image_dates) == date)[0])
        #    selected_images[i] = {'image_date': [date], 'image_ratio': image_ratio, 'image_idx': [image_idx]}
        #else:
            # Fill gaps with the median of all images that are within 2 months of the closest prior
            # and after image

        # 
        prior = distances[np.where(distances < 5)][-2:]
        if prior.shape[0] > 0:
            prior = np.array(prior[prior > (-100 + np.max(prior))]).flatten()

        after = distances[np.where(distances >= -5)][:2]
        if after.shape[0] > 0:
            after = np.array(after[after < (100 + np.min(after))])
        after_flag = 0
        prior_flag = 0
        if len(prior) == 0:
            if np.min(satisfactory_dates) >= 90:
                prior = distances[-1:]
                prior_flag = 365
            else:
                prior = after
        if len(after) == 0:
            if np.max(satisfactory_dates) <= 270:
                after = distances[:1]
                after_flag = 365
            else:
                after = prior
        
        # Wrap when no data at end or beginning of year
        prior_calc = prior - prior_flag
        after_calc = after + after_flag 

        prior_calc = abs(prior_calc)
        after_calc = abs(after_calc)
        prior_calc = np.maximum(prior_calc, 1.)
        after_calc = np.maximum(after_calc, 1.)

        total_distances = np.sum(np.concatenate([abs(prior_calc), abs(after_calc)]))
        if total_distances == 0:
            prior_calc += 1
            after_calc += 1
            total_distances = np.sum(np.concatenate([abs(prior_calc), abs(after_calc)]))
        closest_distances = np.maximum(abs(prior_calc[-1]) + abs(after_calc[0]), 2)
        
        # Each 15 day grid is a 4-window weighted average based on temporal distance
        # This will always work for the closest distances
        prior_mults = abs(1 - (abs(prior_calc) / closest_distances))
        after_mults = abs(1 - (abs(after_calc) / closest_distances))
        # But for the further away distances, if distance > sum(closest)... it fails
        # So we overwrite the value and manually calculate it here, but keep the
        # above vectorization since the code is cleaner
        if len(prior_mults) == 2:
            prior_mults[0] = abs((prior_calc[1] / prior_calc[0]) * prior_mults[1])
        if len(after_mults) == 2:
            after_mults[1] = abs((after_calc[0] / after_calc[1]) * after_mults[0])

        # Rescale to ensure that the multiplications add up to 1.
        divisor = np.sum(np.concatenate([abs(prior_mults), abs(after_mults)]))
        prior_ratio = prior_mults / divisor
        after_ratio = after_mults / divisor

        # Extract the image date and imagery index for the prior and after values
        prior_dates = i + prior
        prior_images_idx = [i for i, val in enumerate(image_dates) if val in prior_dates]
        prior_images_idx = np.array(prior_images_idx).reshape(-1)
        after_dates = i + after
        after_images_idx = [i for i, val in enumerate(image_dates) if val in after_dates]
        after_images_idx = np.array(after_images_idx).reshape(-1)
        after_images_idx = sorted(list(set(after_images_idx)))
        prior_images_idx = sorted(list(set(prior_images_idx)))
        if len(after_images_idx) > 2:
            after_images_idx = after_images_idx[-2:]
        if len(prior_images_idx) > 2:
            prior_images_idx = prior_images_idx[:2]
        #print(np.concatenate([prior_images_idx, after_images_idx]))
        selected_images[i] = {'image_date': np.array([prior_dates, after_dates]).flatten(),
                              'image_ratio': [prior_ratio, after_ratio],
                              'image_idx': [prior_images_idx, after_images_idx]}

    max_distance = 0
    for i in sorted(selected_images.keys()):
        #print(i, selected_images[i])
        if len(selected_images[i]['image_date']) == 2:
            dist = (np.min(selected_images[i]['image_date'][1]) -
                    np.max(selected_images[i]['image_date'][0]))
            if dist > max_distance:
                max_distance = dist

    keep_steps = []
    for i in sorted(selected_images.keys()):
        info = selected_images[i]
        if len(info['image_idx']) == 1:
            step = np.median(img_bands[info['image_idx']], axis = 0)
        if len(info['image_idx']) >= 2:
            step1 = img_bands[info['image_idx'][0]]
            if len(step1.shape) == 3:
                step1 = step1[np.newaxis]

            step1mult = np.array(info['image_ratio'][0], dtype = np.float32)[..., np.newaxis, np.newaxis, np.newaxis]
            step1 = np.sum(np.copy(step1) * step1mult, axis = 0)
            step2 = img_bands[info['image_idx'][1]]
            step2mult = np.array(info['image_ratio'][1], dtype = np.float32)[..., np.newaxis, np.newaxis, np.newaxis]
            if len(step2.shape) == 3:
                step2 = step2[np.newaxis]
            step2 = np.sum(np.copy(step2) * step2mult, axis = 0)
            step = step1 + step2
            """
            if info['image_ratio'][0] > 0.5:
                step = img_bands[info['image_idx'][0]]
                step = np.median(step, axis = 0)#* info['image_ratio'][0]
            else:
                step = img_bands[info['image_idx'][1]]
                step = np.median(step, axis = 0)#$ * info['image_ratio'][1]
                #step = step1 + step2
            """
        keep_steps.append(step)

    keep_steps = np.stack(keep_steps)
    return keep_steps, max_distance


def calculate_proximal_steps(date: int, satisfactory: list) -> (int, int):
    """Returns proximal steps that are cloud and shadow free

         Parameters:
          date (int): current time step
          satisfactory (list): time steps with no clouds or shadows

         Returns:
          arg_before (str): index of the prior clean image
          arg_after (int): index of the next clean image
    """
    arg_before, arg_after = None, None
    if date > 0:
        idx_before = satisfactory - date
        arg_before = idx_before[np.where(idx_before < 0, idx_before, -np.inf).argmax()]
    if date < np.max(satisfactory):
        idx_after = satisfactory - date
        arg_after = idx_after[np.where(idx_after > 0, idx_after, np.inf).argmin()]
    if not arg_after and not arg_before:
        arg_after = date
        arg_before = date
    if not arg_after:
        arg_after = arg_before
    if not arg_before:
        arg_before = arg_after
    return arg_before, arg_after

def calculate_proximal_steps_two(date: int, satisfactory: list) -> (int, int):
    """Returns proximal steps that are cloud and shadow free

         Parameters:
          date (int): current time step
          satisfactory (list): time steps with no clouds or shadows

         Returns:
          arg_before (str): index of the prior clean image
          arg_after (int): index of the next clean image
    """
    arg_before, arg_after = [], []
    if date > 0:
        idx_before = satisfactory - date

        arg_before = np.array(np.where(idx_before < 0, idx_before, -np.inf).flatten())
        to_print = np.copy(arg_before)
        n_before = 2#if date < np.max(satisfactory) else 3

        if np.sum(arg_before > -np.inf) == 0:
            arg_before = np.empty((0))
        elif np.sum(arg_before > -np.inf) > n_before:
            arg_before = np.argpartition(arg_before, -n_before)[-n_before:]
        elif np.sum(arg_before > -np.inf) == n_before:
            arg_before = np.argwhere(arg_before > -np.inf).flatten()
        else:
            arg_before = np.array(arg_before.argmax())
        if arg_before != np.empty((0)):
            arg_before = list(idx_before[arg_before])
    if date < np.max(satisfactory):
        idx_after = satisfactory - date
        arg_after = np.array(np.where(idx_after > 0, idx_after, np.inf).flatten())
        n_after = 2# if date > 0 else 3

        if np.sum(arg_after < np.inf) == 0:
            arg_after = np.empty((0))
        if np.sum(arg_after < np.inf) > n_after:
            arg_after = np.argpartition(arg_after, n_after)[:n_after]
        elif np.sum(arg_after < np.inf) == n_after:
            arg_after = np.argwhere(arg_after < np.inf).flatten()
        else:
            arg_after = np.array(arg_after.argmin())
        if arg_after != np.empty((0)):
            arg_after = list(idx_after[arg_after])

    if arg_after == np.empty((0)) and arg_before == np.empty((0)):
        arg_after = date
        arg_before = date
    elif arg_after == np.empty((0)):
        arg_after = arg_before
    elif arg_before == np.empty((0)):
        arg_before = arg_after

    return np.array(arg_before).astype(int), np.array(arg_after).astype(int)


def tile_window(h: int, w: int, tile_width: int=None,
                tile_height: int=None,
                window_size: int=100) -> List:
    """Calculates overlapping tiles of tile_width x tile_height
    for an input h x w array
    """

    np.seterr(divide='ignore', invalid='ignore')

    if not tile_width:
        tile_width = window_size

    if not tile_height:
        tile_height = window_size

    wTile = tile_width
    hTile = tile_height

    if tile_width > w or tile_height > h:
        raise ValueError("tile dimensions cannot be larger than origin dimensions")

    # Number of tiles
    nTilesX = np.uint8(np.ceil(w / wTile))
    nTilesY = np.uint8(np.ceil(h / hTile))

    # Total remainders
    remainderX = nTilesX * wTile - w
    remainderY = nTilesY * hTile - h

    # Set up remainders per tile
    remaindersX = np.ones((nTilesX-1, 1)) * np.uint16(np.floor(remainderX / (nTilesX-1)))
    remaindersY = np.ones((nTilesY-1, 1)) * np.uint16(np.floor(remainderY / (nTilesY-1)))
    remaindersX[0:np.remainder(remainderX, np.uint16(nTilesX-1))] += 1
    remaindersY[0:np.remainder(remainderY, np.uint16(nTilesY-1))] += 1

    # Initialize array of tile boxes
    tiles = np.zeros((nTilesX * nTilesY, 4), np.uint16)

    k = 0
    x = 0
    for i in range(nTilesX):
        y = 0
        for j in range(nTilesY):
            tiles[k, :] = (x, y, hTile, wTile)
            k += 1
            if j < (nTilesY-1):
                y = y + hTile - remaindersY[j]
        if i < (nTilesX-1):
            x = x + wTile - remaindersX[i]

    return tiles


def check_contains(coord: tuple, step_x: int, step_y:
                   int, folder: str) -> bool:
    """Given an input .geojson, identifies whether a given tile intersections
       the geojson

        Parameters:
         coord (tuple):
         step_x (int):
         step_y (int):
         folder (path):

        Returns:
         contains (bool)
    """
    contains = False
    bbx, epsg = calculate_bbx_pyproj(coord, step_x, step_y, expansion = 80)
    inproj = Proj('epsg:' + str(str(epsg)[5:]))
    outproj = Proj('epsg:4326')
    bottomleft = transform(inproj, outproj, bbx[0][0], bbx[0][1])
    topright = transform(inproj, outproj, bbx[1][0], bbx[1][1])

    if os.path.exists(folder):
            if any([x.endswith(".geojson") for x in os.listdir(folder)]):
                geojson_path = folder + [x for x in os.listdir(folder) if x.endswith(".geojson")][0]

                bool_contains = pts_in_geojson(lats = [bottomleft[1], topright[1]],
                                                       longs = [bottomleft[0], topright[0]],
                                                       geojson = geojson_path)
                contains = bool_contains
    return contains


def hist_match(source: np.ndarray, template: np.ndarray) -> np.ndarray:
    """
    Adjust the pixel values of a grayscale image such that its histogram
    matches that of a target image

    Arguments:
    -----------
        source: np.ndarray
            Image to transform; the histogram is computed over the flattened
            array
        template: np.ndarray
            Template image; can have different dimensions to source
    Returns:
    -----------
        matched: np.ndarray
            The transformed output image
    """

    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    # get the set of unique pixel values and their corresponding indices and
    # counts
    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True,
                                            return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    # take the cumsum of the counts and normalize by the number of pixels to
    # get the empirical cumulative distribution functions for the source and
    # template images (maps pixel value --> quantile)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    # interpolate linearly to find the pixel values in the template image
    # that correspond most closely to the quantiles in the source image
    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)

    return interp_t_values[bin_idx].reshape(oldshape)

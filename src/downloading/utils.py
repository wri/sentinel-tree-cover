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
from osgeo import ogr, osr
from random import shuffle
from scipy.sparse.linalg import splu
from sentinelhub import WmsRequest, WcsRequest, MimeType
from sentinelhub import CRS, BBox, constants, DataSource, CustomUrlParam
from skimage.transform import resize
from pyproj import Proj, transform
from typing import List, Any, Tuple


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


def calculate_area(bbx: list) -> int:
    '''
    Calculates the area in ha of a [(min_x, min_y), (max_x, max_y)] bbx
    '''

    #epsg = calculate_epsg(bbx[0])
    
    #mins = convertCoords(bbx[0], 4326, epsg)
    #maxs = convertCoords(bbx[1], 4326, epsg)
    mins = bbx[0]
    maxs = bbx[1]
    area = PolygonArea([(mins[0], mins[1]), # BL
                        (mins[0], maxs[1]), # BR
                        (maxs[0], mins[1]), # TL
                        (maxs[0], mins[1]) # TR
                        ])
    hectares = math.floor(area / 1e4)
    print(hectares)


def convertCoords(xy, src='', targ=''):

    srcproj = osr.SpatialReference()
    srcproj.ImportFromEPSG(src)
    targproj = osr.SpatialReference()
    if isinstance(targ, str):
        targproj.ImportFromProj4(targ)
    else:
        targproj.ImportFromEPSG(targ)
    transform = osr.CoordinateTransformation(srcproj, targproj)

    pt = ogr.Geometry(ogr.wkbPoint)
    pt.AddPoint(xy[0], xy[1])
    pt.Transform(transform)
    return([pt.GetX(), pt.GetY()])


def convert_to_float(array):
    return (array.astype(float) / 65535)


def bounding_box(point, x_offset_max = 140, y_offset_max = 140, expansion = 10):

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
    biweekly_dates = [day for day in range(0, 360, 5)] # ideal imagery dates are every 15 days
    
    # Identify the dates where there is < 20% cloud cover
    satisfactory_ids = [x for x in range(0, img_bands.shape[0])]
    satisfactory_dates = [value for idx, value in enumerate(image_dates) if idx in satisfactory_ids]
    
    selected_images = {}
    for i in biweekly_dates:
        distances = [abs(date - i) for date in satisfactory_dates]
        closest = np.min(distances)
        closest_id = np.argmin(distances)
        # If there is imagery within 8 days, select it
        if closest < 8:
            date = satisfactory_dates[closest_id]
            image_idx = int(np.argwhere(np.array(image_dates) == date)[0])
            selected_images[i] = {'image_date': [date], 'image_ratio': [1], 'image_idx': [image_idx]}
        # If there is not imagery within 8 days, look for the closest above and below imagery
        else:
            distances = np.array([(date - i) for date in satisfactory_dates])
            # Number of days above and below the selected date of the nearest clean imagery
            above = distances[np.where(distances < 0, distances, -np.inf).argmax()]
            below = distances[np.where(distances > 0, distances, np.inf).argmin()]
            if abs(above) > 240: # If date is the last date, occassionally argmax would set above to - number
                above = below
            if abs(below) > 240:
                below = above
            if above != below:
                below_ratio = above / (above - below)
                above_ratio = 1 - below_ratio
            else:
                above_ratio = below_ratio = 0.5
                
            # Extract the image date and imagery index for the above and below values
            above_date = i + above
            above_image_idx = int(np.argwhere(np.array(image_dates) == above_date)[0])
            
            below_date = i + below
            below_image_idx = int(np.argwhere(np.array(image_dates) == below_date)[0])
            
            selected_images[i] = {'image_date': [above_date, below_date], 
                                  'image_ratio': [above_ratio, below_ratio],
                                  'image_idx': [above_image_idx, below_image_idx]}
                            
    max_distance = 0
    
    for i in sorted(selected_images.keys()):
        if len(selected_images[i]['image_date']) == 2:
            dist = selected_images[i]['image_date'][1] - selected_images[i]['image_date'][0]
            if dist > max_distance:
                max_distance = dist
    
    print("Maximum time distance: {}".format(max_distance))
        
    keep_steps = []
    for i in sorted(selected_images.keys()):
        info = selected_images[i]
        if len(info['image_idx']) == 1:
            step = img_bands[info['image_idx'][0]]
        if len(info['image_idx']) == 2:
            step1 = img_bands[info['image_idx'][0]] * 0.5 #info['image_ratio'][0]
            step2 = img_bands[info['image_idx'][1]] * 0.5 #info['image_ratio'][1]
            step = step1 + step2
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


def tile_window(h, w, tile_width=None, tile_height=None, window_size=100):
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
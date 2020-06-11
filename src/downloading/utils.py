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


def calculate_epsg(points):
    """ Calculates the UTM EPSG of an input WGS 84 lon, lat

        Parameters:
         points (tuple): input longitiude, latitude tuple
    
        Returns:
         epsg_code (int): integer form of associated UTM EPSG
    """
    lon, lat = points[0], points[1]
    print(lon, lat)
    utm_band = str((math.floor((lon + 180) / 6 ) % 60) + 1)
    if len(utm_band) == 1:
        utm_band = '0'+utm_band
    if lat >= 0:
        epsg_code = '326' + utm_band
    else:
        epsg_code = '327' + utm_band
    return int(epsg_code)
    

def PolygonArea(corners):
    n = len(corners) # of corners
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += corners[i][0] * corners[j][1]
        area -= corners[j][0] * corners[i][1]
    area = abs(area)
    return area
    

def offset_x(coord, offset):
    ''' Converts a WGS 84 to UTM, adds meters, and converts back'''

    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))
    
    coord_utm = transform(inproj, outproj, coord[1], coord[0])
    coord_utm = list(coord_utm)
    coord_utm[0] += offset 
    return coord_utm
    

def offset_y(coord, offset):
    ''' Converts a WGS 84 to UTM, adds meters, and converts back'''
    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))
    
    coord_utm = transform(inproj, outproj, coord[1], coord[0])
    coord_utm = list(coord_utm)
    coord_utm[1] += offset 
    return coord_utm


def calculate_area(bbx):
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


def calculate_and_save_best_images(img_bands, image_dates):
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
    
    for i in selected_images.keys():
        if len(selected_images[i]['image_date']) == 2:
            dist = selected_images[i]['image_date'][1] - selected_images[i]['image_date'][0]
            if dist > max_distance:
                max_distance = dist
    
    print("Maximum time distance: {}".format(max_distance))
        
    keep_steps = []
    for i in selected_images.keys():
        info = selected_images[i]
        if len(info['image_idx']) == 1:
            step = img_bands[info['image_idx'][0]]
        if len(info['image_idx']) == 2:
            step1 = img_bands[info['image_idx'][0]] * 0.5#info['image_ratio'][0]
            step2 = img_bands[info['image_idx'][1]] * 0.5 #info['image_ratio'][1]
            step = step1 + step2
        keep_steps.append(step)
        
    keep_steps = np.stack(keep_steps)
    return keep_steps, max_distance


def calculate_proximal_steps(date, satisfactory):
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
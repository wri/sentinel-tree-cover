import numpy as np
import yaml
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
from pyproj import Proj, transform
import pytest
from src.downloading.utils import calculate_epsg
import os
import unittest
import datetime

def calculate_bbx_pyproj(coord, step_x, step_y, expansion, multiplier = 1.):
    ''' Calculates the four corners of a bounding box as above
        but uses pyproj instead of OGR. It seems sentinelhub uses
        pyproj, so this may be more pixel accurate (?)
        x, y format
    '''

    inproj = Proj('epsg:4326')
    outproj_code = calculate_epsg(coord)
    outproj = Proj('epsg:' + str(outproj_code))



    coord_utm =  transform(inproj, outproj, coord[1], coord[0])
    coord_utm_bottom_left = (coord_utm[0] + step_x*6300 - expansion,
                             coord_utm[1] + step_y*6300 - expansion)
    coord_utm_top_right = (coord_utm[0] + (step_x+multiplier) * 6300 + expansion,
                           coord_utm[1] + (step_y+multiplier) * 6300 + expansion)

    coord_bottom_left = transform(outproj, inproj,
                                  coord_utm_bottom_left[0],
                                  coord_utm_bottom_left[1])

    coord_top_right = transform(outproj, inproj,
                                  coord_utm_top_right[0],
                                  coord_utm_top_right[1])

    zone = str(outproj_code)[3:]
    direction = 'N' if coord[1] >= 0 else 'S'
    utm_epsg = "UTM_" + zone + direction
    print(utm_epsg)
    return (coord_utm_bottom_left, coord_utm_top_right), CRS[utm_epsg]


def load_api_key(path = "./config.yaml"):
    with open(path, 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']
    print("API KEY LOADED")
    return API_KEY

class TestDownload(unittest.TestCase):

    def setUp(self):
        self.api_key = load_api_key()
        self.coords = (-90.8, 15.3)

    def test_download(self):
        bbx, epsg = calculate_bbx_pyproj(self.coords, 0, 0, 10)
        bbx = BBox(bbx, crs = epsg)

        image_request = WcsRequest(
                    layer='L2A20',
                    bbox=bbx,
                    time=('2019-12-15', '2020-01-15'),
                    image_format = MimeType.TIFF_d16,
                    maxcc=0.7,
                    resx='10m', resy='10m',
                    instance_id=self.api_key,
                    custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                        constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
                    time_difference=datetime.timedelta(hours=72),
        )

        data = image_request.get_data()
        self.assertEqual(np.array(data).shape[1], 632)
        self.assertEqual(np.array(data).shape[2], 632)

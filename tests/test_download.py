import numpy as np
import yaml
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
from pyproj import Proj, transform

def calculate_epsg(points):
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

def test_download(key, coord):
  bbx, epsg = calculate_bbx_pyproj(coords, 0, 0, 10)
  bbx = BBox(bbx, crs = CRS[epsg])

  image_request = WcsRequest(
              layer='L2A20',
              bbox=bbx,
              time=('2019-12-15', '2020-01-15'),
              image_format = MimeType.TIFF_d16,
              maxcc=0.7,
              resx='10m', resy='10m',
              instance_id=API_KEY,
              custom_url_params = {constants.CustomUrlParam.DOWNSAMPLING: 'NEAREST',
                                  constants.CustomUrlParam.UPSAMPLING: 'NEAREST'},
              time_difference=datetime.timedelta(hours=72),
  )

  data = image_request.get_data()
  assert np.array(data).shape[1] == 632
  assert np.array(data).shape[2] == 632

if __name__ == "__main__":
	with open("../config.yaml", 'r') as stream:
	    key = (yaml.safe_load(stream))
	    API_KEY = key['key']
	    AWSKEY = key['awskey']
	    AWSSECRET = key['awssecret']

	coord = (-90.8, 15.3)
  test_download(API_key, coord)	
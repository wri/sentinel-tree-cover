import geopandas as gpd
import shutil
import pycountry
import os
import argparse
import fiona
import osgeo
from osgeo import gdal
from osgeo import gdalconst

parser = argparse.ArgumentParser(description='Provide a list of capitalized country names and boolean value if you want directories deleted.')
parser.add_argument('country_list', nargs='+', type=str)
parser.add_argument('--delete', action='store_true')
parser.add_argument('--keep', dest='delete', action='store_false')
parser.set_defaults(delete=False)
args = parser.parse_args()

def create_geojsons(country_list, delete=False):

    '''
    Uses gadm admin 1 shapefiles to create a geojson for each country.
    Option to delete shapefile folders upon completion.

    Attributes
    ----------
    country_list : list
    '''

    for country in country_list:
        if os.path.exists(f'admin_boundaries/{country}_adminboundaries.geojson'):
            pass
        else:
            try:
                iso = pycountry.countries.get(name = country).alpha_3
            except AttributeError as e:
                print(f'Potential country name:{pycountry.countries.search_fuzzy(country)}')
                return e

            shapefile = f'admin_boundaries/gadm40_{iso}_shp/gadm40_{iso}_1.shp'
            new_shp = gpd.read_file(shapefile)
            new_shp.to_file(f'admin_boundaries/{country}_adminboundaries.geojson', driver='GeoJSON')

            assert new_shp.crs == 'epsg:4326'
            assert new_shp.NAME_1.duplicated().sum() == 0

            if delete:
                shutil.rmtree(f'admin_boundaries/gadm40_{iso}_shp')

    return None

def main():
    country_list = args.country_list
    delete = args.delete
    create_geojsons(country_list, delete=False)

if __name__ ==  "__main__":
    main()

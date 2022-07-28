import geopandas as gpd
import shutil
import pycountry
import os
import argparse
from botocore.errorfactory import ClientError
import confuse
import boto3
import os
import os.path


parser = argparse.ArgumentParser(description='Provide a list of capitalized country names and boolean value if you want directories deleted.')
parser.add_argument('country_list', nargs='+', type=str, help='list of capitalized country names')
parser.add_argument('--delete', action='store_true', help='remove local shapefile directories')
parser.add_argument('--keep', dest='delete', action='store_false', help='keep local shapefile directories')
parser.set_defaults(delete=False)
args = parser.parse_args()


def create_geojsons(country_list, delete=False):

    '''
    checks s3 buckets for existing admin boundaries geojson. If not present, uses country name
    to get ISO code and uses gadm shapefile to create a geojson

    Attributes
    ----------
    country_list : list
    '''
    config = confuse.Configuration('sentinel-tree-cover')
    config.set_file('/Users/jessica.ertel/sentinel-tree-cover/jessica-config.yaml')
    aws_access_key = config['aws']['aws_access_key_id']
    aws_secret_key = config['aws']['aws_secret_access_key']
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())

    on_s3 = []
    no_geoj = []

    for country in country_list:

        # Checks s3 buckets for admin boundaries geojson for a country, if file is not present appends country name to a list
        try:
            s3.head_object(Bucket='tof-output', Key=f'2020/analysis/2020-full-v1/{country}/{country}_adminboundaries.geojson')
            on_s3.append(country)
        except ClientError:
            no_geoj.append(country)

    for country in no_geoj:
        # get the iso code to identify shapefile folder locally
        try:
            iso = pycountry.countries.get(name = country).alpha_3

        except AttributeError as e:
            print(f'Potential country name:{pycountry.countries.search_fuzzy(country)}')
            return e

        # create geojson
        shapefile = f'admin_boundaries/gadm40_{iso}_shp/gadm40_{iso}_1.shp'
        new_shp = gpd.read_file(shapefile)
        new_shp.to_file(f'admin_boundaries/{country}_adminboundaries.geojson', driver='GeoJSON')
        assert new_shp.crs == 'epsg:4326'
        assert new_shp.NAME_1.duplicated().sum() == 0

        # remove shapefile folder
        if delete:
            shutil.rmtree(f'admin_boundaries/gadm40_{iso}_shp')

    return None


def main():
    country_list = args.country_list
    delete = args.delete
    create_geojsons(country_list, delete=False)

if __name__ ==  "__main__":
    main()

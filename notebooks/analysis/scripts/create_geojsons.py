import geopandas as gpd
import shutil
import pycountry
import os
import argparse
from botocore.errorfactory import ClientError
from botocore.exceptions import ClientError
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
    Checks s3 buckets for existing admin boundaries geojson. If not present, uses country name
    to get ISO code and searches locally for a gadm shapefile to create a geojson.

    Attributes
    ----------
    country_list : list
    '''
    config = confuse.Configuration('sentinel-tree-cover')
    config.set_file('/Users/jessica.ertel/sentinel-tree-cover/jessica-config.yaml')
    aws_access_key = config['aws']['aws_access_key_id']
    aws_secret_key = config['aws']['aws_secret_access_key']

    on_s3 = []
    no_geoj = []

    # if country was not previously processed, appends country to list
    for country in country_list:
        try:
            s3 = boto3.resource('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())
            copy_src = {'Bucket': 'tof-output',
                        'Key': f'2020/analysis/2020-full-v1/{country}/{country}_adminboundaries.geojson'}
            bucket = s3.Bucket('tof-output')
            bucket.copy(copy_src, f'2020/analysis/2020-full/admin_boundaries/{country}_adminboundaries.geojson')
            print(f'{country} geojson is on s3 and was copied.')
            # s3 = boto3.client('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())
            # s3.head_object(Bucket='tof-output', Key=f'2020/analysis/2020-full-v1/{country}/{country}_adminboundaries.geojson')
            # on_s3.append(country)

        except ClientError:
            no_geoj.append(country)

    # if it's already in s3, copy to the new admin boundaries folder
    # for country in on_s3:
    #     s3 = boto3.resource('s3', aws_access_key_id=aws_access_key.as_str(), aws_secret_access_key=aws_secret_key.as_str())
    #     copy_src = {'Bucket': 'tof-output',
    #                 'Key': f'2020/analysis/2020-full-v1/{country}/{country}_adminboundaries.geojson'}
    #     bucket = s3.Bucket('tof-output')
    #     bucket.copy(copy_src, f'2020/analysis/2020-full/admin_boundaries/{country}_adminboundaries.geojson')
    #     print(f'{country} geojson is on s3 and was copied.')


    # if it's not in s3, create the geojson locally using gadm shapefile
    for country in no_geoj:

        try:
            iso = pycountry.countries.get(name = country).alpha_3

        except AttributeError as e:
            print(f'Potential country name:{pycountry.countries.search_fuzzy(country)}')
            return e

        # create geojson
        shapefile = f'/Users/jessica.ertel/sentinel-tree-cover/notebooks/analysis/admin_boundaries/gadm41_{iso}_shp/gadm41_{iso}_1.shp'
        new_shp = gpd.read_file(shapefile)
        new_shp.to_file(f'admin_boundaries/{country}_adminboundaries.geojson', driver='GeoJSON')
        assert new_shp.crs == 'epsg:4326'
        assert new_shp.NAME_1.duplicated().sum() == 0

        # remove shapefile folder
        if delete:
            shutil.rmtree(f'admin_boundaries/gadm40_{iso}_shp')

        # and upload geojson to s3
        s3_client = boto3.client('s3')
        try:
            s3_client.upload_file(f'admin_boundaries/{country}_adminboundaries.geojson',
                                  'tof-output',
                                  f'2020/analysis/2020-full/admin_boundaries/{country}_adminboundaries.geojson')
            print(f'Created and uploaded shapefile for {country}.')
        except ClientError as e:
            logging.error(e)

    return None



def geojson_admin2(country, gadm_filepath):

    filepath = f'admin_boundaries/{gadm_filepath}.json'
    shapefile = gpd.read_file(filepath)

    # if there are duplicate admin 2 names
    if shapefile.NAME_2.duplicated().sum() > 0:

        # create a df of the duplicates
        dups = shapefile[shapefile.NAME_2.duplicated()]

        # iterate by index and update the name to combine admin 1 and 2 names
        for row, column in dups.iterrows():
            shapefile.loc[row,['NAME_2']] = shapefile.loc[row,['NAME_1']][0] + '_' + shapefile.loc[row,['NAME_2']][0]

    # run assertions
    assert shapefile.NAME_2.duplicated().sum() == 0
    assert shapefile.crs == 'epsg:4326'

    # save file
    shapefile.to_file(f'admin_boundaries/{country}_adminboundaries2.geojson', driver='GeoJSON')

    return None


def main():
    country_list = args.country_list
    delete = args.delete
    create_geojsons(country_list, delete)

if __name__ ==  "__main__":
    main()

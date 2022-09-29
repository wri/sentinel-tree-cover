import boto3
import numpy
import yaml
import os
from boto3.s3.transfer import TransferConfig
from botocore.errorfactory import ClientError
import botocore
import boto3.s3.transfer as s3transfer
import argparse
import re
import reverse_geocoder as rg
import pycountry
import pycountry_convert as pc
import rasterio
from rasterio.transform import from_origin
import numpy as np
from glob import glob


class FileUploader:

    def __init__(self, awskey, awssecret, stream=False, overwrite=False):
        self.total = 0
        self.uploaded = 0
        self.percent = 0
        self.awskey = awskey
        self.awssecret = awssecret
        self.config = botocore.config.Config(max_pool_connections=20)
        self.s3client = boto3.client(
            's3',
            config=self.config,
            aws_access_key_id=self.awskey,
            aws_secret_access_key=self.awssecret,
        )
        self.stream = stream
        self.overwrite = overwrite

    def upload_callback(self, size):
        if self.total == 0:
            return
        self.uploaded += size
        percent = int(self.uploaded / self.total * 100)
        if percent > self.percent and percent % 5 == 0:
            print("{} %".format(int(self.uploaded / self.total * 100)))
            self.percent = percent

    def upload(self, bucket, key, file):
        self.total = os.stat(file).st_size

        # check if the file exists
        if self.overwrite:
            #print(f'uploading {file} to {bucket} as {key}')
            self.s3client.upload_file(
                file,
                bucket,
                key,
                Config=TransferConfig(5 * (1024**3),
                                      use_threads=True,
                                      max_concurrency=20),
                Callback=self.upload_callback,
                ExtraArgs={'ACL': 'bucket-owner-full-control'})
        else:
            try:
                boto3.client(
                    's3',
                    aws_access_key_id=self.awskey,
                    aws_secret_access_key=self.awssecret,
                ).head_object(Bucket=bucket, Key=key)
                #print(f'removing {file}')
                #os.remove(file)

            # if the file doesn't exist, upload it
            except ClientError:
                print(f'uploading {file} to {bucket} as {key}')
                if self.stream:
                    with open(file, 'rb') as data:
                        self.s3client.upload_fileobj(
                            file,
                            bucket,
                            key,
                            Config=TransferConfig(5 * (1024**3),
                                                  use_threads=True,
                                                  max_concurrency=20),
                            Callback=self.upload_callback,
                            ExtraArgs={'ACL': 'bucket-owner-full-control'})

                else:
                    self.s3client.upload_file(
                        file,
                        bucket,
                        key,
                        Config=TransferConfig(5 * (1024**3),
                                              use_threads=True,
                                              max_concurrency=20),
                        Callback=self.upload_callback,
                        ExtraArgs={'ACL': 'bucket-owner-full-control'})
                #print(f'removing {file}')
                #os.remove(file)


def get_folder_prefix(coordinates, params):
    geolocation = rg.search((coordinates[0], coordinates[1]))
    country = geolocation[-1]['cc']
    regex = re.compile('[^a-zA-Z-]')

    country = pc.country_alpha2_to_country_name(country)
    country = country.replace(" ", "-").lower()
    country = regex.sub('', country)
    admin1 = geolocation[-1]['admin1'].replace(" ", "-").lower()
    admin1 = regex.sub('', admin1)
    name = geolocation[-1]['name'].replace(" ", "-").lower()
    name = regex.sub('', name)
    path = f'{params["bucket-prefix"]}/{country}/{admin1}/{name}/'
    return path


def save_file(obj, path, params, save_bucket=True):

    hkl.dump(obj, params['prefix'] + path, mode='w', compression='gzip')
    uploader = FileUploader(awskey=AWSKEY, awssecret=AWSSECRET)
    key = params['bucket_prefix'] + path
    if save_bucket:
        uploader.upload(bucket=params['bucket'],
                        key=key,
                        file=params['prefix'] + path)


def make_output_and_temp_folders(output_folder: str) -> None:
    """Makes necessary folder structures for input/output of raw/processed data

        Parameters:
         idx (str)
         output_folder (path)

        Returns:
         None
    """

    def _find_and_make_dirs(dirs: list) -> None:
        if not os.path.exists(os.path.realpath(dirs)):
            os.makedirs(os.path.realpath(dirs))

    folders = [
        'raw/', 'raw/clouds/', 'raw/misc/', 'raw/s1/', 'raw/s2_10/',
        'raw/s2_20/'
    ]

    for folder in folders:
        _find_and_make_dirs(output_folder + folder)


def upload_raw_processed_s3(path_to_tile, x, y, uploader):
    '''
    Uploads temp/raw/*, temp/processed/* to s3 bucket
    and then deletes temporary files

    Parameters:
         path_to_tile (os.path)
         x (int)
         y (int)

        Returns:
         None
    '''
    for folder in glob(path_to_tile + "raw/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            internal_folder = folder[len(path_to_tile):]
            key = f'2020/raw/{x}/{y}/' + internal_folder + file
            uploader.upload(bucket='tof-output', key=key, file=_file)
            os.remove(_file)

    for folder in glob(path_to_tile + "processed/*/"):
        for file in os.listdir(folder):
            _file = folder + file
            internal_folder = folder[len(path_to_tile):]
            key = f'2020/processed/{x}/{y}/' + internal_folder + file
            uploader.upload(bucket='tof-output', key=key, file=_file)
            os.remove(_file)


def file_in_local_or_s3(file, key, apikey, apisecret, bucket):
    """
    Checks to see if a file/key pair exists locally or on s3 or neither
    """

    exists = False
    s3 = boto3.resource('s3',
                        aws_access_key_id=apikey,
                        aws_secret_access_key=apisecret)
    bucket = s3.Bucket(bucket)
    objs = list(bucket.objects.filter(Prefix=key))

    if len(objs) > 0:
        print(f"The s3 resource s3://{bucket}/{key} exists")
        exists = True

    if not exists:
        print(f"The s3 resource s3://{bucket}/{key} does not exist")
        if os.path.isdir(file):
            files = [x for x in os.listdir(file) if '.tif' in x]
            exists = True if len(files) > 0 else False
            print(f"The local path {file} contains {files}")
    return exists


def write_tif(arr: np.ndarray,
              point: list,
              x: int,
              y: int,
              out_folder: str,
              suffix="_FINAL") -> str:
    #! TODO: Documentation

    file = out_folder + f"{str(x)}X{str(y)}Y{suffix}.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]
    arr = arr.T.astype(np.uint8)

    transform = rasterio.transform.from_bounds(west=west,
                                               south=south,
                                               east=east,
                                               north=north,
                                               width=arr.shape[1],
                                               height=arr.shape[0])

    print("Writing", file)
    new_dataset = rasterio.open(file,
                                'w',
                                driver='GTiff',
                                height=arr.shape[0],
                                width=arr.shape[1],
                                count=1,
                                dtype="uint8",
                                compress='lzw',
                                crs='+proj=longlat +datum=WGS84 +no_defs',
                                transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    return file


def download_folder(s3_folder, local_dir, apikey, apisecret, bucket):
    """
    Checks to see if a file/key pair exists locally or on s3 or neither, and downloads the folder
    """

    s3 = boto3.resource('s3',
                        aws_access_key_id=apikey,
                        aws_secret_access_key=apisecret)
    bucket = s3.Bucket(bucket)

    for obj in bucket.objects.filter(Prefix=s3_folder):
        target = obj.key if local_dir is None \
            else os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target)


def delete_folder(s3_folder, apikey, apisecret, bucket):
    """
    Checks to see if a file/key pair exists locally or on s3 or neither, and downloads the folder
    """

    s3 = boto3.resource('s3',
                        aws_access_key_id=apikey,
                        aws_secret_access_key=apisecret)
    bucket = s3.Bucket(bucket)
    bucket.objects.filter(Prefix=s3_folder).delete()


def download_file(s3_file, local_file, apikey, apisecret, bucket):
    """
    Checks to see if a file/key pair exists locally or on s3 or neither,
    if exists -- download the file
    """

    s3 = boto3.resource('s3',
                        aws_access_key_id=apikey,
                        aws_secret_access_key=apisecret)
    bucket = s3.Bucket(bucket)

    print(f"Starting download of {s3_file} to {local_file}")

    for obj in bucket.objects.filter(Prefix=s3_file):
        target = obj.key if local_file is None \
            else os.path.join(local_file, os.path.relpath(obj.key, s3_file))
        if not os.path.exists(os.path.dirname(target)):
            os.makedirs(os.path.dirname(target))

        file_name = s3_file.split("/")[-1]
        print(f"Downloaded {s3_file} to {local_file + file_name}")
        bucket.download_file(obj.key, target[:-1] + file_name)
    return file_name


def make_subtiles(folder: str, tiles) -> None:

    y_tiles = np.unique(tiles[:, 1])
    x_tiles = np.unique(tiles[:, 0])

    def _find_and_make_dirs(dirs):
        if not os.path.exists(os.path.realpath(dirs)):
            os.makedirs(os.path.realpath(dirs))

    for y_tile in y_tiles:
        _find_and_make_dirs(folder + str(y_tile) + '/')

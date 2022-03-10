#!/usr/bin/env python
import boto3
import numpy
import yaml
import os
from boto3.s3.transfer import TransferConfig
from botocore.errorfactory import ClientError
import botocore
import boto3.s3.transfer as s3transfer
import argparse

class FileDownloader:
    def __init__(self, stream = False):
        self.total = 0
        self.uploaded = 0
        self.percent = 0
        self.config = botocore.config.Config(max_pool_connections=20)
        self.s3resource = boto3.resource('s3', config=self.config,
            aws_access_key_id=AWSKEY,
            aws_secret_access_key=AWSSECRET,
        )
        self.s3client = boto3.client('s3', config=self.config,
            aws_access_key_id=AWSKEY,
            aws_secret_access_key=AWSSECRET,
        )
        self.stream = stream
        self.files = None

    def list_files(self, bucket):
        bucket = self.s3resource.Bucket(args.bucket)
        self.files =  [x.key for x in bucket.objects.all()]


    def download(self, bucket):
        prefix = "../data/train-csv/"
        for file in self.files:
            if 'train-csv' in file:
                if ".csv" in file:
                    #id_npy = file[-13:]
                    filename = prefix + file.split("/")[-1]
                    print(file, filename)
                    #print(file, id_npy)
                    self.s3client.download_file(Bucket = bucket,
                        Key = file,
                        Filename = prefix + file.split("/")[-1])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket", help = "Bucket to upload to", required=True)
    #parser.add_argument("--files", help = "File pattern", required=True)
    parser.add_argument("--subset", help = "Subset pattern", required=False)
    parser.add_argument("--prefix", required = False)
    parser.add_argument("--filetype", required = False)
    args = parser.parse_args()

    with open("../config.yaml", 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        AWSKEY = key['awskeywri']
        AWSSECRET = key['awssecretwri']


    downloader = FileDownloader(stream = False)
    downloader.list_files('restoration-monitoring')
    downloader.download('restoration-monitoring')
    #for file in files:
    #   downloader.download(bucket = args.bucket, # rm-guatemala
    #                   key = args.prefix + file,
    #                   file = base_path + file)

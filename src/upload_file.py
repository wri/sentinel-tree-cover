#!/usr/bin/env python
import boto3
import numpy
import yaml
import os
from boto3.s3.transfer import TransferConfig
from botocore.errorfactory import ClientError

class FileUploader:
    def __init__(self, stream = False):
        self.total = 0
        self.uploaded = 0
        self.percent = 0
        self.session = boto3.Session(
            aws_access_key_id=AWSKEY,
            aws_secret_access_key=AWSSECRET,
        )
        self.s3 = boto3.client('s3')
        self.stream = stream

    def upload_callback(self, size):
        if self.total == 0:
            return
        self.uploaded += size
        percent = int(self.uploaded / self.total * 100)
        if percent > self.percent:
            print("{} %".format(int(self.uploaded / self.total * 100)))
            self.percent = percent

    def upload(self, bucket, key, file):
        self.total = os.stat(file).st_size

        # check if the file exists
        try:
             boto3.client('s3',
                        aws_access_key_id=AWSKEY,
                        aws_secret_access_key=AWSSECRET,
                    ).head_object(Bucket=bucket, Key=key)
             print(f'removing {file}')
             os.remove(file)

        # if the file doesn't exist, upload it
        except ClientError:
            print(f'uploading {file} to {bucket} as {key}')
            if self.stream:
                with open(file, 'rb') as data:
                    boto3.client('s3',
                        aws_access_key_id=AWSKEY,
                        aws_secret_access_key=AWSSECRET,
                    ).upload_fileobj(
                        data, bucket, key, 
                        Config=TransferConfig(5*(1024**3)), Callback=self.upload_callback
                    )
                
            else:
                 boto3.client('s3',
                        aws_access_key_id=AWSKEY,
                        aws_secret_access_key=AWSSECRET,
                    ).upload_file(
                        file, bucket, key, 
                        Config=TransferConfig(5*(1024**3)), Callback=self.upload_callback
                    )
            print(f'removing {file}')
            os.remove(file)

if __name__ == "__main__":

    with open("../config.yaml", 'r') as stream:
        key = (yaml.safe_load(stream))
        API_KEY = key['key']
        AWSKEY = key['awskey']
        AWSSECRET = key['awssecret']


    uploader = FileUploader(stream = False)
    base_path = "../tile_data/guatemala-coban/"

    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(base_path + "raw/s2/")) for f in fn]
    files = [x for x in files if ".hkl" in x]
    files = [x.replace(base_path, "") for x in files]

    for file in files:
    	uploader.upload(bucket = 'rm-guatemala',
    	                key = file,
    	                file = base_path + file)
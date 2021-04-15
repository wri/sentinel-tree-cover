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


class FileUploader:
	def __init__(self, awskey, awssecret, stream = False, ):
		self.total = 0
		self.uploaded = 0
		self.percent = 0
		self.awskey = awskey
		self.awssecret = awssecret
		self.config = botocore.config.Config(max_pool_connections=20)
		self.s3client = boto3.client('s3', config=self.config,
			aws_access_key_id= self.awskey,
			aws_secret_access_key= self.awssecret,
		)
		self.stream = stream

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
		try:
			 boto3.client('s3',
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
						file, bucket, key, 
						Config=TransferConfig( 5*(1024**3), use_threads=True, max_concurrency=20), 
						Callback=self.upload_callback
					)
				
			else:
				 self.s3client.upload_file(
						file, bucket, key, 
						Config=TransferConfig( 5*(1024**3), use_threads=True, max_concurrency=20),
						Callback=self.upload_callback
					)
			#print(f'removing {file}')
			#os.remove(file)

def get_folder_prefix(coordinates, params):
    geolocation = rg.search((coordinates[0], coordinates[1]))
    country = geolocation[-1]['cc']
    regex = re.compile('[^a-zA-Z-]')
   
    country =  pc.country_alpha2_to_country_name(country)
    country = country.replace(" ", "-").lower()
    country = regex.sub('', country)
    admin1 = geolocation[-1]['admin1'].replace(" ", "-").lower()
    admin1 = regex.sub('', admin1)
    name = geolocation[-1]['name'].replace(" ", "-").lower()
    name = regex.sub('', name)
    path = f'{params["bucket-prefix"]}/{country}/{admin1}/{name}/'
    return path


def save_file(obj, 
              path,
              params, 
              save_bucket = True):
    
    hkl.dump(obj, params['prefix'] + path, mode = 'w', compression = 'gzip')
    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
    key = params['bucket_prefix'] + path
    if save_bucket:
        uploader.upload(bucket = params['bucket'], key = key,
                        file = params['prefix'] + path
                       )



'''
if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--bucket", help = "Bucket to upload to", required=True)
	parser.add_argument("--files", help = "File pattern", required=True)
	parser.add_argument("--subset", help = "Subset pattern", required=False)
	parser.add_argument("--prefix", required = False)
	parser.add_argument("--filetype", required = False)
	args = parser.parse_args()

	with open("../config.yaml", 'r') as stream:
		key = (yaml.safe_load(stream))
		API_KEY = key['key']
		AWSKEY = key['awskeywri']
		AWSSECRET = key['awssecretwri']


	uploader = FileUploader(stream = False)
	base_path = args.files #"../tile_data/guatemala-coban/"

	files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(base_path)) for f in fn]
	files = [x.replace(base_path, "") for x in files]
	if args.filetype:
		files = [x for x in files if args.filetype in x]
	if args.subset:
		files = [x for x in files if args.subset in x]

	print(files)

	for file in files:
		uploader.percent = 0
		uploader.uploaded = 0
		uploader.upload(bucket = args.bucket, # rm-guatemala
						key = args.prefix + file,
						file = base_path + file)
'''
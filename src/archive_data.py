import numpy as np
import os
import yaml
from glob import glob
import boto3
from downloading.io import FileUploader,  get_folder_prefix, make_output_and_temp_folders, upload_raw_processed_s3
from downloading.io import file_in_local_or_s3, write_tif, make_subtiles, download_folder, delete_folder
import shutil
import pandas as pd

def download_raw_tile(tile_idx, local_path, subfolder = "raw"):
    x = tile_idx[0]
    y = tile_idx[1]

    path_to_tile = f'{local_path}{str(x)}/{str(y)}/'
    s3_path_to_tile = f'2020/{subfolder}/{str(x)}/{str(y)}/'
    if subfolder == "tiles":
        folder_to_check = len(glob(path_to_tile + "*.tif")) > 0
    if subfolder == "processed":
        folder_to_check = os.path.exists(path_to_tile + subfolder + "/0/")
    if subfolder == "raw":
        folder_to_check = os.path.exists(path_to_tile + subfolder + "/clouds/")
    if not folder_to_check:
        print(f"Downloading {s3_path_to_tile}")
        download_folder(bucket = "tof-output",
                       apikey = AWSKEY,
                       apisecret = AWSSECRET,
                       local_dir = path_to_tile,
                       s3_folder = s3_path_to_tile)
    return None


def cleanup(path_to_tile, path_to_s3, path_to_local_archive, delete = True, upload = True):

    for folder in glob(path_to_tile + "*"):
        print(folder)
        for file in os.listdir(folder):
            print(file)
            if delete:
                os.remove(folder + "/" + file)
                print(f"Deleting {folder}/{file}")
    if delete:
        delete_folder(bucket = args.s3_bucket,
                      apikey = AWSKEY,
                      apisecret = AWSSECRET,
                      s3_folder = path_to_s3)
        os.remove(path_to_local_archive)

    return None


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tiles/')
    parser.add_argument("--db_path", dest = "db_path", default = "processing_area_june_28.csv")
    parser.add_argument("--s3_bucket", dest = "s3_bucket", default = "tof-output")
    parser.add_argument("--yaml_path", dest = "yaml_path", default = "../config.yaml")
    args = parser.parse_args()


    if os.path.exists(args.yaml_path):
        with open(args.yaml_path, 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']
        print(f"Successfully loaded key from {args.yaml_path}")
        uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET, overwrite = True)
    else:
        raise Exception(f"The API keys do not exist in {args.yaml_path}")

    if os.path.exists(args.db_path):
        data = pd.read_csv(args.db_path)
        data = data[data['country'] == args.country]
        data = data.reset_index(drop = True)
        print(f"There are {len(data)} tiles for {args.country}")
    else:
        raise Exception(f"The database does not exist at {args.db_path}")

    country = str(args.country).replace(" ", "_").lower()
    for index, row in data.iterrows():
        try:
            x = str(int(row['X_tile']))
            y = str(int(row['Y_tile']))
            x = x[:-2] if ".0" in x else x
            y = y[:-2] if ".0" in y else y
            
            path_to_tile = f'{args.local_path}{str(x)}/{str(y)}/raw/'
            path_to_tile_s3 = f'2020/raw/{str(x)}/{str(y)}/'
            path_to_tile_archive = f'2020/raw-archive/{country}/{str(x)}/{str(y)}/'
            path_to_local_archive = f"{str(x)}X{str(y)}Y"


            download_raw_tile((x, y), args.local_path, "raw")
            shutil.make_archive(path_to_local_archive, 'zip', path_to_tile)

            uploader.upload(bucket = 'tof-output', 
                            key = path_to_tile_archive + path_to_local_archive + ".zip",
                            file = path_to_local_archive + ".zip")

            cleanup(path_to_tile, path_to_tile_s3, path_to_local_archive + ".zip")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Ran into {str(e)}")


from tqdm import tnrange
import multiprocessing
import pandas as pd
import numpy as np
from random import shuffle
from osgeo import ogr, osr
from sentinelhub import WmsRequest, WcsRequest, MimeType, CRS, BBox, constants
from s2cloudless import S2PixelCloudDetector, CloudMaskRequest
import logging
from collections import Counter
import datetime
import os
import yaml
from sentinelhub import DataSource
import scipy.sparse as sparse
from scipy.sparse.linalg import splu
from sentinelhub import CustomUrlParam

if __name__ == '__main__':

    df = pd.read_csv("interim/{}.csv".format(str(month)))
    urls = df['to_scrape'].unique()

    if not os.path.exists("text/{}".format(str(month))):
        os.makedirs("text/{}".format(str(month)))

    existing = os.listdir("text/{}/".format(str(month)))
    existing = [int(x[:5]) for x in existing if ".DS" not in x]
    if args.warm_start:
        potential = range(max(existing), len(urls))
    else:
        potential = range(0, len(urls))
    
    potential = [x for x in potential if x not in existing]
    print(len(potential))

    threads = 16
    print("Multiprocessing enabled with {}".format(threads))
    pool = multiprocessing.Pool(threads)
    zip(*pool.map(download_url, potential))
    pool.close()
    pool.join()


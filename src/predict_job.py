from preprocessing.indices import evi, bi, msavi2, grndvi
from downloading.upload import FileUploader

import tensorflow as tf
sess = tf.Session()
from keras import backend as K
K.set_session(sess)
import numpy as np 
import os
import rasterio
from rasterio.transform import from_origin
from tqdm import tnrange, tqdm_notebook
from scipy.ndimage import median_filter
from skimage.transform import resize
import hickle as hkl
from time import sleep
import pandas as pd
import copy
import yaml

from downloading import utils
from models import utils

path = '../models/master-2021-13000/'
new_saver = tf.train.import_meta_graph(path + 'model.meta')
new_saver.restore(sess, tf.train.latest_checkpoint(path))

for i in range(50):
    try:
        logits = tf.get_default_graph().get_tensor_by_name("conv2d_{}/Sigmoid:0".format(i))
    except Exception:
        pass
    
inp = tf.get_default_graph().get_tensor_by_name("Placeholder:0")
length = tf.get_default_graph().get_tensor_by_name("PlaceholderWithDefault:0")



def make_smooth_predicts():

    def _fspecial_gauss(size, sigma):

        """Function to mimic the 'fspecial' gaussian MATLAB function
        """

        x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
        return g

    arr = _fspecial_gauss(14, 4.5)[:7, :7]

    SIZE = 10
    SIZE_N = SIZE*SIZE
    SIZE_UR = (SIZE - 1) * (SIZE - 1)

    arr = np.concatenate([arr, np.flip(arr, 0)], 0)
    base_filter = np.concatenate([arr, np.flip(arr, 1)], 1)
    normal = np.tile(base_filter, (SIZE, SIZE))
    normal[:, 0:7] = 1.
    normal[:, -7:] = 1.
    normal[0:7, :] = 1.
    normal[-7:, :] = 1.
    upright = np.tile(base_filter, (SIZE - 1, SIZE - 1))
    upright = np.pad(upright, (7, 7), 'constant', constant_values = 0)

    sums = (upright + normal)
    upright /= sums
    normal /= sums

    return upright, normal


def make_bbox(initial_bbx, expansion = 10):
    
    multiplier = 1/360
    bbx = copy.deepcopy(initial_bbx)
    bbx[0] -= expansion * multiplier
    bbx[1] -= expansion * multiplier
    bbx[2] += expansion * multiplier
    bbx[3] += expansion * multiplier
    return bbx


def convert_to_db(x, min_db):
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = x + min_db
    x = x / min_db
    x = np.clip(x, 0, 1)
    return x
 

def load_and_predict_folder(pred_files, histogram_match = False):
    """Insert documentation here
    """
      

    upright, normal = make_smooth_predicts()
    
    pred_x = []
    x = hkl.load(pred_files)
    
    if np.sum(x) > 0:
        if not isinstance(x.flat[0], np.floating):
            assert np.max(x) > 1
            x = x / 65535.

        x[..., -1] = convert_to_db(x[..., -1], 50)
        x[..., -2] = convert_to_db(x[..., -2], 50)

        indices = np.empty((12, x.shape[1], x.shape[2], 4))
        indices[..., 0] = evi(x)
        indices[..., 1] = bi(x)
        indices[..., 2] = msavi2(x)
        indices[..., 3] = grndvi(x)

        x = np.concatenate([x, indices], axis = -1)

        med = np.median(x, axis = 0)
        med = med[np.newaxis, :, :, :]
        x = np.concatenate([x, med], axis = 0)

        filtered = median_filter(x[0, :, :, 10], size = 5)
        x[:, :, :, 10] = np.stack([filtered] * x.shape[0])
        x = tile_images(x)
        
        pred_x = np.stack(x)   
        for band in range(0, pred_x.shape[-1]):
            mins = min_all[band]
            maxs = max_all[band]
            pred_x[..., band] = np.clip(pred_x[..., band], mins, maxs)
            midrange = (maxs + mins) / 2
            rng = maxs - mins
            standardized = (pred_x[..., band] - midrange) / (rng / 2)
            pred_x[..., band] = standardized

        preds = []
        batches = [x for x in range(0, 180, 40)] + [181]
        for i in range(len(batches)-1):
            batch_x = pred_x[batches[i]:batches[i+1]]
            lengths = np.full((batch_x.shape[0],), 12)
            batch_pred = sess.run(logits,
                                  feed_dict={inp:batch_x, 
                                             length:lengths}).reshape(batch_x.shape[0], 14, 14)
            for sample in range(batch_pred.shape[0]):
                preds.append(batch_pred[sample, :, :])


        preds_stacked = []
        for i in range(0, SIZE_N, SIZE):
            preds_stacked.append(np.concatenate(preds[i:i + SIZE], axis = 1))
        stacked = np.concatenate(preds_stacked, axis = 0) * normal

        preds_overlap = []
        for scene in range(SIZE_N, SIZE_N+SIZE_UR, SIZE - 1):
            to_concat = np.concatenate(preds[scene:scene+ (SIZE - 1)], axis = 1)
            preds_overlap.append(to_concat)    
        overlapped = np.concatenate(preds_overlap, axis = 0)
        overlapped = np.pad(overlapped, (7, 7), 'constant', constant_values = 0)
        overlapped = overlapped * upright


        stacked = stacked + overlapped
    else:
        stacked = np.full((140, 140), 255)
    
    return stacked

def tile_images(arr: np.ndarray) -> list:
    """ Converts a 142x142 array to a 289, 24, 24 array
        
        Parameters:
         arr (np.ndaray): (142, 142) float array
    
        Returns:
         images (list): 
    """

    # Normal
    images = []
    for x_offset, cval in enumerate([x for x in range(0, 140, 14)]):
        for y_offset, rval in enumerate([x for x in range(0, 140, 14)]):
            min_x = np.max([cval - 0, 0])
            max_x = np.min([cval + 24, 150])
            min_y = np.max([rval - 0, 0])
            max_y = np.min([rval + 24, 150])
            subs = arr[:, min_x:max_x, min_y:max_y]
            images.append(subs)
            
    # Upright  
    for x_offset, cval in enumerate([x for x in range(7,  140-7, 14)]):
        for y_offset, rval in enumerate([x for x in range(7, 140-7, 14)]):
            base_id = 9*9
            min_x = np.max([cval - 0, 0])
            max_x = np.min([cval + 24, 150])
            min_y = np.max([rval - 0, 0])
            max_y = np.min([rval + 24, 150])
            subs = arr[:, min_x:max_x, min_y:max_y]
            images.append(subs)

    return images



def write_tif(arr, point, x, y):
    
    file = out_folder[:-7] + f"{str(x)}X{str(y)}Y_POST.tif"

    west, east = point[0], point[2]
    north, south = point[3], point[1]
    arr = arr.T.astype(np.uint8)

    transform = rasterio.transform.from_bounds(west = west, south = south,
                                               east = east, north = north,
                                               width = arr.shape[1], 
                                               height = arr.shape[0])

    print("Writing", file)
    new_dataset = rasterio.open(file, 'w', driver = 'GTiff',
                               height = arr.shape[0], width = arr.shape[1], count = 1,
                               dtype = "uint8",
                               crs = '+proj=longlat +datum=WGS84 +no_defs',
                               transform=transform)
    new_dataset.write(arr, 1)
    new_dataset.close()
    return file

if __name__ == '__main__':
    import argparse
    if os.path.exists("../config.yaml"):

        with open("../config.yaml", 'r') as stream:
            key = (yaml.safe_load(stream))
            API_KEY = key['key']
            AWSKEY = key['awskey']
            AWSSECRET = key['awssecret']

    uploader = FileUploader(awskey = AWSKEY, awssecret = AWSSECRET)
    data = pd.read_csv("../notebooks/processing_area.csv")

    min_all = [0.012558175020981156, 0.025696192874036773, 0.015518425268940261, 0.04415960936903945,
               0.040497444113832305, 0.04643320363164721, 0.04924086366063935, 0.04289311055161364, 
               0.027450980392156862, 0.019760433356221865, 0.0, 0.5432562150454495, 0.2969113383797463,
                -0.03326967745787883, -0.4014989586557378, -0.023132966289995487, -0.4960341058778109]

    max_all = [0.21116960402838178, 0.30730144197756926, 0.4478065156023499, 0.5342488746471351,
               0.4942702372777905, 0.5072556649118791, 0.5294422827496758, 0.5418631265735866, 0.6813458457312886,
               0.6285648889906157, 0.4208438239108873, 0.9480767549203932, 0.8130214090572532, 0.7444347421954634,
               0.3268904303046983, 0.6872429594867983, 0.7129084148772861]

    SIZE = 10
    SIZE_N = SIZE*SIZE
    SIZE_UR = (SIZE - 1) * (SIZE - 1)

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--country", dest = 'country')
    parser.add_argument("--local_path", dest = 'local_path', default = '../project-monitoring/tof/')
    args = parser.parse_args()
    data = data[data['country'] == args.country]
    data = data.reset_index(drop = True)
    print(f"There are {len(data)} tiles for {args.country}")

    tifs = []
    for index, row in data.iterrows():
        x = str(int(row['X_tile']))
        y = str(int(row['Y_tile']))
        dir_i = f"{args.local_path}{x}/{y}/"
        if os.path.exists(dir_i):
            files = [file for file in os.listdir(dir_i)  if os.path.splitext(file)[-1] == '.tif']
            if len(files) == 0 and os.path.exists(dir_i + "processed/"):
                initial_bbx = [row['X'], row['Y'], row['X'], row['Y']]
                point = make_bbox(initial_bbx, expansion = 300/30)

                inp_folder = f'{args.local_path}{str(x)}/{str(y)}/processed/'
                out_folder = f'{args.local_path}{str(x)}/{str(y)}/output/'
                x_tiles = [int(x) for x in os.listdir(inp_folder) if '.DS' not in x]
                max_x = np.max(x_tiles) + 140

                for x_tile in x_tiles:
                    y_tiles = [int(y[:-4]) for y in os.listdir(inp_folder + str(x_tile) + "/") if '.DS' not in y]
                    max_y = np.max(y_tiles) + 140
                    for y_tile in y_tiles:
                        output_file = f"{out_folder}{str(x_tile)}/{str(y_tile)}.npy"
                        input_file = f"{inp_folder}{str(x_tile)}/{str(y_tile)}.hkl"
                        if os.path.exists(input_file) and not os.path.exists(output_file):
                            print(output_file)
                            prediction = load_and_predict_folder(input_file, histogram_match = False)
                            if not os.path.exists(f"{out_folder}{str(x_tile)}/"):
                                os.makedirs(f"{out_folder}{str(x_tile)}/")
                            np.save(output_file, prediction)

                predictions = np.full((max_x, max_y), 0, dtype = np.uint8)
                x_tiles = [int(x) for x in os.listdir(out_folder) if '.DS' not in x]
                for x_tile in x_tiles:
                    y_tiles = [int(y[:-4]) for y in os.listdir(out_folder + str(x_tile) + "/") if '.DS' not in y]
                    for y_tile in y_tiles:
                        output_file = out_folder + str(x_tile) + "/" + str(y_tile) + ".npy"
                        if os.path.exists(output_file):
                            prediction = np.load(output_file)
                            prediction = prediction * 100
                            prediction = prediction.T.astype(np.uint8)
                            predictions_tile = predictions[ (x_tile ): (x_tile+140),
                                       y_tile:y_tile + 140]

                            if np.max(prediction) <= 100:
                                predictions_tile[np.logical_and(predictions_tile != 0, predictions_tile <= 100)] = (
                                    predictions_tile[np.logical_and(predictions_tile != 0, predictions_tile <= 100)] + 
                                    prediction[np.logical_and(predictions_tile != 0, predictions_tile <= 100)] ) / 2
                                predictions_tile[predictions_tile == 0] = prediction[predictions_tile == 0]
                            else:
                                predictions[ (x_tile ): (x_tile+140),
                                       y_tile:y_tile + 140] = prediction
                            

                for x_i in range(0, predictions.shape[0] - 3):
                    for y_i in range(0, predictions.shape[1] - 3):
                        window = predictions[x_i:x_i+3, y_i:y_i+3]
                        if np.max(window) < 35:
                            if np.sum(np.logical_and(window > 10, window < 35)) > 5:
                                predictions[x_i:x_i+3, y_i:y_i+3] = 0.

                predictions[predictions <= .20*100] = 0.        
                predictions = np.around(predictions / 20, 0) * 20
                predictions[predictions > 100] = 255.
                file = write_tif(predictions, point, x, y)
                key = f'2020/tiles/{x}/{y}/{str(x)}X{str(y)}Y_POST.tif'
                uploader.upload(bucket = 'tof-output', key = key, file = file)

                # Delete the local processed/ folder

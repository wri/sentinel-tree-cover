import numpy as np
import pandas as pd
import os
import rasterio as rs
import hickle as hkl


def load_individual_sample(fpath, ypath, f, df):
    if df is not None:
        #print(df['name'])[0]
        dfi = df[df['name'] == int(f)]
        if len(dfi) > 0:
            dfi = dfi.reset_index(drop = True)
            bestx = dfi['bestx'][0].astype(np.int32)
            besty = dfi['besty'][0].astype(np.int32)
        else:
            bestx = 0
            besty = 0
    else:
        bestx = 0
        besty = 0
    #print(bestx, besty)
    ishkl = os.path.exists(fpath + f + '.hkl')
    if ishkl:
        x = hkl.load(fpath + f + '.hkl') / 65535
    else:
        x = np.load(fpath + f + ".npy") / 65535
    
    nsize = (x.shape[1] - input_size) // 2
    x = x[:, nsize+bestx:nsize+bestx+input_size,
                                    nsize+besty:nsize+besty+input_size, :]
    if x.shape[-1] == 13:
        i = make_and_smooth_indices(x)
        out = np.zeros((x.shape[0], x.shape[1], x.shape[2], 17), dtype = np.float32)
        out[..., :13] = x 
        out[..., 13:] = i
    else:
        out = x
        out[..., -1] *= 2
        out[..., -1] -= 0.7193834232943873
        
        #out[-1] -= 0.7193834232943873
        out[..., -2] -= 0.09731556326714398
        out[..., -3] -= 0.4973397113668104,
        out[..., -4] -= 0.1409399364817101
        #out[]
    #median = np.median(out, axis = 0)
    #out = np.reshape(out, (4, 3, out.shape[1], out.shape[2], out.shape[3]))
    #out = np.median(out, axis = 1, overwrite_input = True)
    #out = np.concatenate([out, median[np.newaxis]], axis = 0)
    
    # Account for image size -> model size
    # And georeferencing from satellite -> label based on provided DF for each sample
    y_size = input_size - 14
    if os.path.exists(ypath + f + '.tif'):
        y = rs.open(ypath + f + '.tif').read(1)
    elif os.path.exists(ypath + f + '.npy'):
        y = np.load(ypath + f + '.npy')
    else:
        y = None
    if np.max(y) > 1:
        y = y / 255
    y[y < 0.3] = 0.
    if y.shape[0] > y_size:
        torm = (y.shape[0] - y_size) // 2
        print(torm)
        y = y[torm:-torm, torm:-torm]
        print(y.shape)
    return normalize_subtile(out), y


def normalize_subtile(subtile):
    for band in range(0, subtile.shape[-1]):
        mins = min_all[band]
        maxs = max_all[band]
        subtile[..., band] = np.clip(subtile[..., band], mins, maxs)
        midrange = (maxs + mins) / 2
        rng = maxs - mins
        standardized = (subtile[..., band] - midrange) / (rng / 2)
        subtile[..., band] = standardized
    return subtile


def convert_to_db(x, min_db):
    x = 10 * np.log10(x + 1/65535)
    x[x < -min_db] = -min_db
    x = x + min_db
    x = x / min_db
    x = np.clip(x, 0, 1)
    return x

def grndvi(array):
    nir = np.clip(array[..., 3], 0, 1)
    green = np.clip(array[..., 1], 0, 1)
    red = np.clip(array[..., 2], 0, 1)
    denominator = (nir+(green+red)) + 1e-5
    return (nir-(green+red)) / denominator

def evi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    '''
    Calculates the enhanced vegetation index
    2.5 x (08 - 04) / (08 + 6 * 04 - 7.5 * 02 + 1)
    '''

    BLUE = x[..., 0]
    GREEN = x[..., 1]
    RED = x[..., 2]
    NIR = x[..., 3]
    evis = 2.5 * ( (NIR-RED) / (NIR + (6*RED) - (7.5*BLUE) + 1))
    evis = np.clip(evis, -1.5, 1.5)
    return evis

def msavi2(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    '''
    Calculates the modified soil-adjusted vegetation index 2
    (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-RED)) / 2
    '''
    BLUE = x[..., 0]
    GREEN = x[..., 1]
    RED = np.clip(x[..., 2], 0, 1)
    NIR = np.clip(x[..., 3], 0, 1)

    msavis = (2 * NIR + 1 - np.sqrt( (2*NIR+1)**2 - 8*(NIR-RED) )) / 2
    return msavis

def bi(x: np.ndarray, verbose: bool = False) -> np.ndarray:
    B11 = np.clip(x[..., 8], 0, 1)
    B4 = np.clip(x[..., 2], 0, 1)
    B8 = np.clip(x[..., 3], 0, 1)
    B2 = np.clip(x[..., 0], 0, 1)
    bis = ((B11 + B4) - (B8 + B2)) / ((B11 + B4) + (B8 + B2))
    return bis


def augment_batch(batch_ids, batch_size, train_x, train_y, args):
    '''Performs random flips and rotations of the X and Y
       data for a total of 4 x augmentation
    
         Parameters:
          batch_ids (list):
          batch_size (int):
          
         Returns:
          x_batch (arr):
          y_batch (arr):
    '''
    
    def _unapply(x, idx):
        _max = args['maxs'][idx]
        _min =  args['mins'][idx]
        midrange = (_max + _min) / 2
        rng = _max - _min
        return x * (rng / 2) + midrange
    
    def _apply(x, idx):
        _max = args['maxs'][idx]
        _min =  args['mins'][idx]
        midrange = (_max + _min) / 2
        rng = _max - _min
        return (x - midrange) / (rng / 2)
    
    x = np.copy(train_x[batch_ids])
    samples_to_median = np.random.randint(0, 12, size=(batch_size, 12)) #[32, 6]
    samples_to_select = np.zeros((batch_size, 4))
    flast = np.array([0, 1, 2, 3, 8, 9, 10, 11])
    #samples_to_select[:, 0] = np.random.choice(flast, size=(batch_size))
    samples_to_select[:, 0] = np.random.randint(0, 4, size=(batch_size))
    samples_to_select[:, 1] = np.random.randint(3, 7, size=(batch_size))
    samples_to_select[:, 2] = np.random.randint(6, 10, size=(batch_size))
    samples_to_select[:, 3] = np.random.randint(9, 12, size=(batch_size))
    samples_to_select = samples_to_select.astype(np.int)
    n_samples = np.random.randint(2, 5, size=(batch_size)) 
    
    x_batch = np.zeros((x.shape[0], args['length'] + 1, 28, 28, args['n_bands']))
    for samp in range(batch_size):
        samps = samples_to_median[samp, :]#:np.random.randint(6, 12)]
        x_samp = train_x[samp]
        samps = np.unique(samps)
        med_samp = np.median(x_samp[samps], axis = 0)

        x_batch[samp, :-1, ...] = x[samp, samples_to_select[samp]]
        x_batch[samp, -1, ...] = med_samp
        
    y = train_y[batch_ids]
    
    y_batch = np.zeros_like(y)
    
    flips = np.random.choice(np.array([0, 1, 2, 3]), batch_size, replace = True)
    for i in range(x_batch.shape[0]):
        current_flip = flips[i]
        if current_flip == 0:
            x_batch[i] = x_batch[i]
            y_batch[i] = y[i]
        if current_flip == 1:
            x_batch[i] = np.flip(x_batch[i], 1)
            y_batch[i] = np.flip(y[i], 0)
        if current_flip == 2:
            x_batch[i] = np.flip(x_batch[i], [2, 1])
            y_batch[i] = np.flip(y[i], [1, 0])
        if current_flip == 3:
            x_batch[i] = np.flip(x_batch[i], 2)
            y_batch[i] = np.flip(y[i], 1)
    
    for b in range(10, 11):
        slope = _unapply(x_batch[..., b], b)
        mults = np.clip(np.random.normal(1, 0.06, size = (batch_size, 1, 1, 1,)), 0.5, 2)
        slope = slope * mults
        slope = _apply(slope, b)
        x_batch[..., b] = slope
 
    
    y_batch = y_batch.reshape((batch_size, 14, 14))
    return x_batch, y_batch


def equibatch(train_ids, train_y):
        '''Docstring
        
             Parameters:
              train_ids (list):
              p (list):

             Returns:
              equibatches (list):
        '''
        
        
        percents = [9.0, 17.0, 27.0, 40.0, 63.0, 105.0, 158.0]
      
        train_ids_cp = (train_ids) 
        np.random.shuffle(train_ids_cp) 
        ix = train_ids_cp
        percs = [np.sum(x) for x in train_y[ix]]
        ids0 = [x for x, z in zip(ix, percs) if z <= 2]
        ids30 = [x for x, z in zip(ix, percs) if 2 < z <= percents[0]]
        ids40 = [x for x, z in zip(ix, percs) if percents[0] < z <= percents[1]]
        ids50 = [x for x, z in zip(ix, percs) if percents[1] < z <= percents[2]]
        ids60 = [x for x, z in zip(ix, percs) if percents[2] < z <= percents[3]]
        ids70 = [x for x, z in zip(ix, percs) if percents[3] < z <= percents[4]]
        ids80 = [x for x, z in zip(ix, percs) if percents[4] < z <= percents[5]]
        ids90 = [x for x, z in zip(ix, percs) if percents[5] < z <= percents[6]]
        ids100 = [x for x, z in zip(ix, percs) if percents[6] < z]
        
        new_batches = []
        maxes = [len(ids0), len(ids30), len(ids40), len(ids50), len(ids60), len(ids70),
                 len(ids80), len(ids90), len(ids100)]

        cur_ids = [0] * len(maxes)
        iter_len = len(train_ids)//(len(maxes))
        for i in range(0, iter_len):
            for i, val in enumerate(cur_ids):
                if val > maxes[i] - 1:
                    cur_ids[i] = 0
            if cur_ids[0] >= (maxes[0] - 3):
                cur_ids[0] = 0
            to_append = [ids0[cur_ids[0]], ids0[cur_ids[0] + 1], ids0[cur_ids[0] + 2],
                        ids30[cur_ids[1]], ids40[cur_ids[2]],
                        ids50[cur_ids[3]], ids60[cur_ids[4]], 
                        ids70[cur_ids[5]], ids80[cur_ids[6]],
                        ids90[cur_ids[7]], ids100[cur_ids[8]]]#,
                        #ids100[cur_ids[8] + 1]]
            
            np.random.shuffle(to_append)
            new_batches.append(to_append)
            cur_ids = [x + 1 for x in cur_ids]
            cur_ids[0] += 2        
            
        new_batches = [item for sublist in new_batches for item in sublist]
        print(f"This batch uses {len(set(new_batches))} samples")
        
        return new_batches
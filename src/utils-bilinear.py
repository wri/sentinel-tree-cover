import numpy as np
import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
import itertools
from tensorflow.python.keras.layers import Conv2D, Lambda, Dense, Multiply, Add

def ndvi(x, verbose = False):
    # (B8 - B4)/(B8 + B4)
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    ndvis = (NIR-RED) / (NIR+RED)
    if verbose:
        mins = np.min(ndvis)
        maxs = np.max(ndvis)
        if mins < -1 or maxs > 1:
            print("ndvis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, ndvis[:, :, :, np.newaxis]], axis = -1)
    return x

def evi(x, verbose = False):
    # 2.5 x (08 - 04) / (08 + 6 * 04 - 7.5 * 02 + 1)
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    BLUE = x[:, :, :, 0]
    evis = 2.5 * ( (NIR-RED) / (NIR + (6*RED) - (7.5*BLUE) + 1))
    if verbose:
        amin = np.argwhere(np.array([np.min(evis[i]) for i in range(x.shape[0])]) < -3)
        len_amin = len(np.argwhere(evis.flatten() < -3))
        len_amax = len(np.argwhere(evis.flatten() > 3))
        print("There are: {} out of bounds EVI".format(len_amin + len_amax))
        amax = np.argwhere(np.array([np.max(evis[i]) for i in range(x.shape[0])]) > 3)
        amin = np.concatenate([amin, amax])
        amin = np.unique(amin)
        mins = np.min(evis)
        maxs = np.max(evis)
        if mins < -1 or maxs > 1:
            idx = np.argmin(evis)
            print(idx.shape)
            #print(np.argmin(evis, (0, 1)).shape)
            #min_nir = NIR[idx[0], idx[1], idx[2]]
            #min_red = RED[idx[0], idx[1], idx[2]]
            #min_blue = BLUE[idx[0], idx[1], idx[2]]
            #print("NIR at lowest: {}, RED at lowest: {}, BLUE at lowest: {}".format(
            #    min_nir, min_red, min_blue))
            print("evis error: {}, {}, {} steps, clipping to -1.5, 1.5".format(mins, maxs, len(amin)))
    evis = np.clip(evis, -1.5, 1.5)
    x = np.concatenate([x, evis[:, :, :, np.newaxis]], axis = -1)
    return x, amin
    
def savi(x, verbose = False):
    # (1.5) * ((08 - 04)/ (08 + 04 + 0.5))
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    savis = 1.5 * ( (NIR-RED) / (NIR+RED +0.5))
    if verbose:
        mins = np.min(savis)
        maxs = np.max(savis)
        if mins < -1.0 or maxs > 1.0:
        	print("SAVI: {} {}".format(mins, maxs))
    x = np.concatenate([x, savis[:, :, :, np.newaxis]], axis = -1)
    return x

def msavi2(x, verbose = False):
    # (2 * NIR + 1 - sqrt((2*NIR + 1)^2 - 8*(NIR-RED)) / 2
    NIR = x[:, :, :, 3]
    RED = x[:, :, :, 2]
    for i in range(x.shape[0]):
        NIR_i = x[i, :, :, 3]
        RED_i = x[i, :, :, 2]
        under_sqrt = (2*NIR_i+1)**2 - 8*(NIR_i-RED_i)
        under_sqrt = np.min(under_sqrt)
        if under_sqrt <= 0:
            print("MSAVI2 negative sqrt at: {}, {}".format(i, under_sqrt))
    msavis = (2 * NIR + 1 - np.sqrt( (2*NIR+1)**2 - 8*(NIR-RED) )) / 2
    if verbose:
        mins = np.min(msavis)
        maxs = np.max(msavis)
        if mins < -1 or maxs > 1:
            print("MSAVIS error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, msavis[:, :, :, np.newaxis]], axis = -1)
    return x

def bi(x, verbose = False):
    # (2 + 0 - 1) / (2 + 0 + 1)
    BLUE = x[:, :, :, 0]
    RED = x[:, :, :, 2]
    GREEN = x[:, :, :, 1]
    bis = (BLUE + RED - GREEN) / (BLUE + RED + GREEN)
    if verbose:
        mins = np.min(bis)
        maxs = np.max(bis)
        if mins < -1.5 or maxs > 1.5:
            print("bis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, bis[:, :, :, np.newaxis]], axis = -1)
    return x

def si(x, verbose = False):
    # (1 - B2) * (1 - B3) * (1 - B4) ** 1/3
    BLUE = x[:, :, :, 0]
    RED = x[:, :, :, 2]
    GREEN = x[:, :, :, 1]
    sis = np.power( (1-BLUE) * (1 - GREEN) * (1 - RED), 1/3)
    if verbose:
        mins = np.min(sis)
        maxs = np.max(sis)
        if mins < -1 or maxs > 1:
            print("sis error: {}, {}".format(mins, maxs))
    x = np.concatenate([x, sis[:, :, :, np.newaxis]], axis = -1)
    return x

def ndmi(x):
    ndmis = [(im[:, :, 5] - im[:, :, 9]) / (im[:, :, 5] + im[:, :, 9]) for im in x]
    ndmis = np.stack(ndmis)
    x = np.concatenate([x, ndmis[:, :, :, np.newaxis]], axis = -1)
    return x

#tf.reset_default_graph()

def Fully_connected(x, units, layer_name='fully_connected') :
    with tf.name_scope(layer_name) :
        return tf.layers.dense(inputs=x, use_bias=True, units=units)

def Relu(x):
    return tf.nn.relu(x)

def Sigmoid(x):
    return tf.nn.sigmoid(x)

def Global_Average_Pooling(x):
    return global_avg_pool(x, name='Global_avg_pooling')

def Squeeze_excitation_layer(input_x, out_dim, ratio, layer_name):
    with tf.name_scope(layer_name) :
        squeeze = global_avg_pool(input_x)

        excitation = Fully_connected(squeeze, units=out_dim / ratio, layer_name=layer_name+'_fully_connected1')
        excitation = Relu(excitation)
        excitation = Fully_connected(excitation, units=out_dim, layer_name=layer_name+'_fully_connected2')
        excitation = Sigmoid(excitation)

        excitation = tf.reshape(excitation, [-1,1,1,out_dim])

        scale = input_x * excitation

        return scale
    
def convGRU(x, cell_fw, cell_bw, ln):
        output, final = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, ln, dtype=tf.float32)
        #output = tf.concat(output, -1)
        #final = tf.concat(final, -1)
        return output, final

def remove_blank_steps(array):
    to_update = {}
    sets = []
    for k in range(2):
        for i in range(array.shape[0]):
            for k in range(array.shape[-1]):
                mean = (np.mean(array[i, :, :, k]))
                if mean == 0:
                    print("blank step")
                    sets.append(i)
                    if i < array.shape[0] - 1:
                        array[i, :, :, k] = array[i + 1, :, :, k]
                    else:
                        array[i, :, :, k] = array[i - 1, :, :, k]
                if mean == 1:
                    print("blank step")
                    sets.append(i)
                    if i < array.shape[0] - 1:
                        array[i, :, :, k] = array[i + 1, :, :, k]
                    else:
                        array[i, :, :, k] = array[i - 1, :, :, k]
    for i in range(array.shape[0]):
        for k in range(array.shape[-1]):
            mean = (np.mean(array[i, :, :, k]))
            if mean == 0:
                if i < array.shape[0] - 2:
                    array[i, :, :, k] = array[i + 2, :, :, k]
                else:
                    array[i, :, :, k] = array[i - 2, :, :, k]
            if mean == 1:
                if i < array.shape[0] - 2:
                    array[i, :, :, k] = array[i + 2, :, :, k]
                else:
                    array[i, :, :, k] = array[i - 2, :, :, k]
    return array

def thirty_meter(true, pred, thresh = 0.4):
    subs_pred = pred.reshape(196, 1)
    subs_pred[np.where(subs_pred > thresh)] = 1
    subs_pred[np.where(subs_pred <= thresh)] = 0
    subs_true = true.reshape(196, 1)
    pred = [np.sum(x) for x in subs_pred]
    true = [np.sum(x) for x in subs_true]
    true_positives = []
    false_positives = []
    false_negatives = []
    for p, t in zip(pred, true):
        if p == 1 and t == 1:
            tp = 1
            true_positives.append(tp)
        if p == 1 and t == 0:
            fp = 1
            false_positives.append(fp)
        if p == 0 and t == 1:
            fn = 1
            false_negatives.append(fn)
            
    if sum(true_positives) + sum(false_positives) > 0:
        prec = sum(true_positives) / (sum(true_positives) + sum(false_positives))
        prec = prec * sum(subs_true)
    else:
        prec = np.nan
    if sum(true_positives) + sum(false_negatives) > 0:
        rec = sum(true_positives) / (sum(true_positives) + sum(false_negatives))
        rec = rec * sum(subs_true)
    else:
        rec = np.nan
    return sum(true_positives), sum(false_positives), sum(false_negatives)#rec, prec, sum(subs_true)
    
def get_shifts(arr):
    true_m = arr[1:13, 1:13]
    true_l = arr[0:12, 1:13]
    true_r = arr[2:14, 1:13]
    true_u = arr[1:13, 0:12]
    true_d = arr[1:13, 2:14]
    true_dr = arr[2:14, 0:12]
    true_dl = arr[0:12, 0:12]
    true_ur = arr[2:14, 2:14]
    true_ul = arr[0:12, 2:14]
    true_shifts = [true_m, true_l, true_r, true_u, true_d, true_dr, true_dl, true_ur, true_ul]
    return true_shifts
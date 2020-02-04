def dsc_np(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.flatten().astype(np.float32)
    y_pred_f = y_pred.flatten().astype(np.float32)
    intersection = sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (sum(y_true_f) + sum(y_pred_f) + smooth)
    return score

def bce_shift(true, pred, power):
    losses = []
    for i in range(BATCH_SIZE):
        true_i = tf.reshape(true[i], (1, 14, 14, 1))
        pred_i = tf.reshape(pred[i], (1, 12, 12, 1))
        true_p = true_i
        #loss_o = binary_crossentropy(true_p, pred)
        # extract out the candidate shifts
        true_m = true_i[:, 1:13, 1:13]
        true_l = true_i[:, 0:12, 1:13]
        true_r = true_i[:, 2:14, 1:13]
        true_u = true_i[:, 1:13, 0:12]
        true_d = true_i[:, 1:13, 2:14]
        true_dr = true_i[:, 2:14, 0:12]
        true_dl = true_i[:, 0:12, 0:12]
        true_ur = true_i[:, 2:14, 2:14]
        true_ul = true_i[:, 0:12, 2:14]
        true_shifts = [true_m, true_l, true_r, true_u, true_d, true_dr, true_dl, true_ur, true_ul]
        bce_shifts = tf.stack([binary_crossentropy(x, pred_i) for x in true_shifts])
        jac_shifts = tf.stack([smooth_jaccard(x, pred_i) for x in true_shifts])

        # Calculate BCE
        
        
        bce_power = tf.math.pow(1/(tf.reduce_mean(bce_shifts, axis = [2,3])), power)
        jac_power = tf.math.pow(1/(jac_shifts+0.1), power)
        
        sums = tf.reduce_sum(bce_power)
        sum_jac = tf.reduce_sum(jac_power)
        weights = bce_power/sums
        weights_jac = jac_power/sum_jac
    
        weights = (2*weights + weights_jac)/3
        loss = tf.reshape(bce_shifts, (1, 9, 12, 12)) * tf.reshape(weights, (1, 9, 1, 1))
        loss = tf.reduce_sum(loss, axis = 1)
        loss_j = tf.reshape(jac_shifts, (1, 9)) * tf.reshape(weights, (1, 9))
        loss_j = tf.reduce_sum(loss_j, axis = 1)
        losses.append(loss + 0.5*loss_j)
    loss = tf.reshape(tf.stack(losses), (BATCH_SIZE, 12, 12, 1))
    return loss

def get_shifts_batched(arr):
    true_m = arr[:, 1:13, 1:13]
    true_l = arr[:, 0:12, 1:13]
    true_r = arr[:, 2:14, 1:13]
    true_u = arr[:, 1:13, 0:12]
    true_d = arr[:, 1:13, 2:14]
    true_dr = arr[:, 2:14, 0:12]
    true_dl = arr[:, 0:12, 0:12]
    true_ur = arr[:, 2:14, 2:14]
    true_ul = arr[:, 0:12, 2:14]
    true_shifts = [true_m, true_l, true_r, true_u, true_d, true_dr, true_dl, true_ur, true_ul]
    return true_shifts

def ce(targets, predictions, epsilon=1e-12):
    targets = targets.reshape(1, 144)
    predictions = predictions.reshape(1, 144)
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.mean(targets*np.log(predictions+1e-9))/N
    return ce

from sklearn.metrics import precision_score
def prec_shift(true, pred):
    true_m = true[1:13, 1:13]
    true_l = true[0:12, 1:13]
    true_r = true[2:14, 1:13]
    true_u = true[1:13, 0:12]
    true_d = true[1:13, 2:14]
    true_dr = true[2:14, 0:12]
    true_dl = true[0:12, 0:12]
    true_ur = true[2:14, 2:14]
    true_ul = true[0:12, 2:14]
    
    match = dsc_np(true_m, pred)
    match_l = dsc_np(true_l, pred)
    match_r = dsc_np(true_r, pred)
    match_u = dsc_np(true_u, pred)
    match_d = dsc_np(true_d, pred)
    match_dr = dsc_np(true_dr, pred)
    match_dl = dsc_np(true_dl, pred)
    match_ur = dsc_np(true_ur, pred)
    match_ul = dsc_np(true_ul, pred)
    return max([match, match_l, match_r, match_u, match_d, match_dr, match_dl, match_ur, match_ul])

import keras.backend as K
import tensorflow as tf 

epsilon = 1e-5
smooth = 1

def dsc(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return score

def confusion(y_true, y_pred):
    smooth=1
    y_pred_pos = K.clip(y_pred, 0, 1)
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.clip(y_true, 0, 1)
    y_neg = 1 - y_pos
    tp = K.sum(y_pos * y_pred_pos)
    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg) 
    prec = (tp + smooth)/(tp+fp+smooth)
    recall = (tp+smooth)/(tp+fn+smooth)
    return prec, recall

def tp(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pos = K.round(K.clip(y_true, 0, 1))
    tp = (K.sum(y_pos * y_pred_pos) + smooth)/ (K.sum(y_pos) + smooth) 
    return tp 

def tn(y_true, y_pred):
    smooth = 1
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos
    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos 
    tn = (K.sum(y_neg * y_pred_neg) + smooth) / (K.sum(y_neg) + smooth )
    return tn 

def tversky(y_true, y_pred):
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

def focal_tversky(y_true,y_pred):
    pt_1 = tversky(y_true, y_pred)
    gamma = 0.75
    return K.pow((1-pt_1), gamma)
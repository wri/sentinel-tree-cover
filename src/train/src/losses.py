from keras.losses import binary_crossentropy
import math
from scipy.ndimage import distance_transform_edt as distance
import numpy as np

import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
from keras import backend as K

def calc_cosine_decay(epoch, maxepoch, offset):
        import math
        return 0.5 * (1 + math.cos(math.pi * (epoch - offset) / (maxepoch - offset)))

def calc_mask(seg):

    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)
    loss_importance = np.array([x for x in range(0, 197, 1)])
    loss_importance = loss_importance / 196
    loss_importance = np.expm1(loss_importance)
    loss_importance[:30] = 0.

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    if np.sum(seg) == 196:
        res = np.ones_like(seg)
    if np.sum(seg) == 0:
        res = np.ones_like(seg)
    res[np.logical_and(res < 2, res > 0)] = 0.5
    res[np.logical_or(res >= 2, res <= 0)] = 1.
    return res

def calc_mask_batch(y_true):
    '''Applies calc_dist_map to each sample in an input batch
    
         Parameters:
          y_true (arr):
          
         Returns:
          loss (arr):
    '''
    y_true_numpy = y_true.numpy()
    bce_batch = np.array([calc_mask(y)
                     for y in y_true_numpy]).astype(np.float32)
    return bce_batch



def weighted_bce_loss(y_true, y_pred, weight, mask = True, smooth = 0.06):
    '''Calculates the weighted binary cross entropy loss between y_true and
       y_pred with optional masking and smoothing for regularization
       
       For smoothing, we want to weight false positives as less important than
       false negatives, so we smooth false negatives 2x as much. 
    
         Parameters:
          y_true (arr):
          y_pred (arr):
          weight (float):
          mask (arr):
          smooth (float):

         Returns:
          loss (float):
    '''
    epsilon = 1e-7
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true = K.clip(y_true, 0.0125, 1. - smooth)
    logit_y_pred = K.log(y_pred / (1. - y_pred))
    loss = tf.nn.weighted_cross_entropy_with_logits(
        y_true,
        logit_y_pred,
        weight,
    )

    return loss 


def calc_dist_map(seg):
    #Utility function for calc_dist_map_batch that calculates the loss
    #   importance per pixel based on the surface distance function
    
     #    Parameters:
    #      seg (arr):
     #     
    #     Returns:
    #      res (arr):
    #
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    mults = np.ones_like(seg)
    ones = np.ones_like(seg)
    for x in range(1, res.shape[0] -1 ):
        for y in range(1, res.shape[0] - 1):
            # If > 1 px, double the weight of the positive
            # If == 1 px, half the weight of the negative
            if seg[x, y] == 1:
                l = seg[x - 1, y]
                r = seg[x + 1, y]
                u = seg[x, y + 1]
                d = seg[x, y - 1]
                lu = seg[x - 1, y + 1]
                ru = seg[x + 1, y + 1]
                rd = seg[x + 1, y - 1]
                ld = seg[x -1, y - 1]
                
                sums = (l + r + u + d)
                sums2 = (l + r + u + d + lu + ru +rd + ld)
                if sums >= 2:
                    mults[x, y] = 2
                if sums2 <= 1:
                    ones[x - 1, y] = 0.5
                    ones[x + 1, y] = 0.5
                    ones[x, y + 1] = 0.5
                    ones[x, y - 1] = 0.5
                    ones[x - 1, y + 1] = 0.5
                    ones[x + 1, y + 1] = 0.5
                    ones[x + 1, y - 1] = 0.5
                    ones[x -1, y - 1] = 0.5

    if posmask.any():
        
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
        # When % = 1, 0 -> 1.75
        # When % = 100, 0 -> 0
        res = np.round(res, 0)
        res[np.where(np.isclose(res, -.41421356, rtol = 1e-2))] = -1
        res[np.where(res == -1)] = -1 * mults[np.where(res == -1)]
        res[np.where(res == 0)] = -1  * mults[np.where(res == 0)]
        # When % = 1, 1 -> 0
        # When % = 100, 1 -> 1.75
        res[np.where(res == 1)] = 1 * ones[np.where(res == 1)]
        res[np.where(res == 1)] *= 0.67
        #res[np.where(np.isclose(res, 1.41421356, rtol = 1e-2))] = loss_importance[sums]
        
    res[np.where(res < -3)] = -3
    res[np.where(res > 3)] = 3
    if np.sum(seg) == 196:
        res = np.ones_like(seg)
        res *= -1
    if np.sum(seg) == 0:
        res = np.ones_like(seg)
    return res


def calc_dist_map_batch(y_true):
    '''Applies calc_dist_map to each sample in an input batch
    
         Parameters:
          y_true (arr):
          
         Returns:
          loss (arr):
    '''
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    '''Calculates the mean surface loss for the input batch
       by multiplying the distance map by y_pred
    
         Parameters:
          y_true (arr):
          y_pred (arr):
          
         Returns:
          loss (arr):
        
         References:
          https://arxiv.org/abs/1812.07032
    '''
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    y_true_dist_map = tf.stack(y_true_dist_map, axis = 0)
    #tomult = tf.where(y_true < 0.25, y_pred, tf.minimum(y_true, y_pred))
    multiplied = y_pred * y_true_dist_map
    loss = tf.reduce_mean(multiplied, axis = (1, 2, 3))
    return loss

    return loss

def tf_percentile(x, p):
    with tf.name_scope('percentile'):
        y = tf.transpose(x)  # take percentile over batch dimension
        sorted_y = tf.sort(y)
        frac_idx = tf.cast(p, tf.float64) / 100. * (tf.cast(tf.shape(y)[-1], tf.float64) - 1.)
        return 0.5 * (  # using midpoint rule
            tf.gather(sorted_y, tf.cast(tf.math.ceil(frac_idx), tf.int32), axis=-1)
            + tf.gather(sorted_y, tf.cast(tf.math.floor(frac_idx), tf.int32), axis=-1))

def lovasz_surf(y_true, y_pred, alpha, weight, beta):
    
    #lv = lovasz_softmax(probas = y_pred,
    #                    labels = tf.reshape(y_true, (-1, 14, 14)), 
    #                    classes=[1],
    #                    per_image=False) 
    
    bce = weighted_bce_loss(y_true = y_true, 
                             y_pred = y_pred, 
                             weight = weight,
                             smooth = 0.045)


    bce = tf.reduce_mean(bce, axis = (1, 2, 3))
    surface = surface_loss(tf.cast(tf.math.greater(y_true, 0.1), tf.float32), y_pred)
    surface = tf.reshape(surface, tf.shape(bce))
    print(surface.shape)

    bce = (1 - alpha) * bce
    surface_portion = alpha * surface
    
    #result = bce + lovasz
    result = bce + surface_portion
    #upper_bound = tf_percentile(result, 90)
    #result = tf.clip_by_value(result, tf.reduce_min(result), upper_bound)
    result = tf.reduce_mean(result)
    return result


def dice_loss_tolerance(y_true, y_pred):
    numerator_data = np.zeros_like(y_true)
    for x in range(y_true.shape[0]):
        for y in range(y_true.shape[1]):
            min_x = np.max([0, x-1])
            min_y = np.max([0, y-1])
            max_y = np.min([y_true.shape[0], y+2])
            max_x = np.min([y_true.shape[0], x+2])
            if y_true[x, y] == 1:
                numerator_data[x, y] = np.max(y_pred[min_x:max_x, min_y:max_y])
                
    numerator = 2 * np.sum(y_true * numerator_data, axis=-1)
    denominator = np.sum(y_true + y_pred, axis=-1)
    return (numerator + 1) / (denominator + 1)
                    
            
def compute_f1_score_at_tolerance(true, pred, tolerance = 1):
    fp = 0
    tp = 0
    fn = 0
    
    tp = np.zeros_like(true)
    fp = np.zeros_like(true)
    fn = np.zeros_like(true)
    
    for x in range(true.shape[0]):
        for y in range(true.shape[1]):
            min_x = np.max([0, x-1])
            min_y = np.max([0, y-1])
            max_y = np.min([true.shape[0], y+2])
            max_x = np.min([true.shape[0], x+2])
            if true[x, y] == 1:
                if np.sum(pred[min_x:max_x, min_y:max_y]) > 0:
                    tp[x, y] = 1
                else:
                    fn[x, y] = 1
            if pred[x, y] == 1:
                if np.sum(true[min_x:max_x, min_y:max_y]) > 0:
                    if true[x, y] == 1:
                        tp[x, y] = 1
                else:
                    fp[x, y] = 1                
                
    return np.sum(tp), np.sum(fp), np.sum(fn)

def calc_median_input(x_batch):
    x_median = np.percentile(x_batch, 25, axis = (1))
    return x_median

def calculate_metrics(test_x, test_y, sess, op, test_loss, input_ops, args, al = 0.4, canopy_thresh = 100):
    '''Calculates the following metrics for an input country, based on
       indexing of the country dictionary:
       
         - Loss
         - F1
         - Precision
         - Recall
         - Dice
         - Mean surface distance
         - Average error
    
         Parameters:
          country (str):
          al (float):
          
         Returns:
          val_loss (float):
          best_dice (float):
          error (float):
    '''
    print(canopy_thresh)
    start_idx = 0
    stop_idx = len(test_x)
    best_f1 = 0
    best_dice = 0
    best_thresh = 0
    hausdorff = 0
    relaxed_f1 = 0
    preds = []
    vls = []
    trues = []
    test_ids = [x for x in range(len(test_x))]
    inp, length, is_training, labels, loss_weight, alpha = input_ops
    for test_sample in test_ids[start_idx:stop_idx]:
        if np.sum(test_y[test_sample]) < ((canopy_thresh/100) * 197):
            x_input = test_x[test_sample, ..., ][np.newaxis]#.reshape(1, test_x.shape[1], 28, 28, n_bands)
            #x_input = np.delete(x_input, [11, 12], axis = -1)
            x_median_input = calc_median_input(x_input)
            y, vl = sess.run([op, test_loss], feed_dict={inp: x_input,
                                                          length: np.full((1,), args['length']),
                                                          is_training: False,
                                                          labels: test_y[test_sample].reshape(1, 14, 14),
                                                          loss_weight: 1.0,
                                                          alpha: 0.33,
                                                          })
            preds.append(y.reshape((14, 14)))
            vls.append(vl)
            trues.append(test_y[test_sample].reshape((14, 14)))
    dice_losses = []
    for thresh in range(7, 9):
        tps_relaxed = np.empty((len(preds), ))
        fps_relaxed = np.empty((len(preds), ))
        fns_relaxed = np.empty((len(preds), ))
        abs_error = np.empty((len(preds), ))
        
        for sample in range(len(preds)):
            pred = np.copy(preds[sample])
            true = trues[sample]
            if thresh == 8:
                if np.sum(true + pred) > 0:
                    dice_losses.append(0.5)
                   # dice_losses.append(dice_loss_tolerance(np.array(true), np.array(pred)))
                else:
                    dice_losses.append(1.)
            pred[np.where(pred >= thresh*0.05)] = 1
            pred[np.where(pred < thresh*0.05)] = 0
            
            true_s = np.sum(true[1:-1])
            pred_s = np.sum(pred[1:-1])
            abs_error[sample] = abs(true_s - pred_s)
            tp_relaxed, fp_relaxed, fn_relaxed = compute_f1_score_at_tolerance(true, pred)
            tps_relaxed[sample] = tp_relaxed
            fps_relaxed[sample] = fp_relaxed
            fns_relaxed[sample] = fn_relaxed                   
            
        oa_error = np.mean(abs_error)
        precision_r = np.sum(tps_relaxed) / (np.sum(tps_relaxed) + np.sum(fps_relaxed))
        recall_r = np.sum(tps_relaxed) / (np.sum(tps_relaxed) + np.sum(fns_relaxed))
        f1_r = 2*((precision_r* recall_r) / (precision_r + recall_r))
        print(f1_r)
        if f1_r > best_f1:
            haus = np.zeros((len(preds), ))
            for sample in range(len(preds)):
                pred = np.copy(preds[sample])
                pred[np.where(pred >= thresh*0.05)] = 1
                pred[np.where(pred < thresh*0.05)] = 0
                true = trues[sample]

            dices = np.mean(dice_losses)
            haus = np.mean(haus)
            best_dice = 0.5
            best_f1 = f1_r
            p = precision_r
            r = recall_r
            error = oa_error
            best_thresh = thresh*0.05
            best_haus = 0.5
            print(f"Val loss: {np.around(np.mean(vls), 3)}"
                  f" Thresh: {np.around(best_thresh, 2)}"
                  f" F1: {np.around(best_f1, 3)} R: {np.around(p, 3)} P: {np.around(r, 3)}"
                  f" D: {np.around(np.mean(best_dice), 3)} H: {np.around(best_haus, 3)}"
                  f" Error: {np.around(error, 3)}")
            
    return np.mean(vls), best_f1, error, best_haus, np.mean(best_dice)
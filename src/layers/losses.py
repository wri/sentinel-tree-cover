import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    #tf.disable_v2_behavior()
    #tf.logging.set_verbosity(tf.logging.ERROR)
import tensorflow.compat.v1.keras.backend as K
import numpy as np
from keras.losses import binary_crossentropy
import math
from scipy.ndimage import distance_transform_edt as distance

def weighted_bce_loss(y_true, y_pred, weight, mask = True, smooth = 0.03):
    '''Calculates the weighted binary cross entropy loss between y_true and
       y_pred with optional masking and smoothing for regularization
       
       For smoothing, we want to weight false positives as less important than
       false negatives, so we smooth false negatives 2x as much. 
    
         Parameters:
          y_true (arr):
          y_pred (arr):
          weight (float):
          mask (arr): DEPRECATED
          smooth (float):

         Returns:
          loss (float):
    '''
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)
    y_true = tf.clip_by_value(y_true, 0.0125, 1. - smooth)
    logit_y_pred = tf.math.log(y_pred / (1. - y_pred))
    #loss = tf.nn.weighted_cross_entropy_with_logits(
    #    y_true,
    #    logit_y_pred,
    #    weight,
    #)
    loss = tf.nn.sigmoid_cross_entropy_with_logits(
      labels = y_true, logits = logit_y_pred)

    return loss


def calc_dist_map(seg):
    
    """
    This is a modified version of surface loss to reduce
    the loss importance of samples at the boundary. THe original
    paper specifies a boundary loss of 0, but the low-resolution of
    Sentinel data requires some amount of boundary importance, and
    this function works well to accomplish that task
    """

    res = np.zeros_like(seg)
    posmask = seg.astype(bool)

    mults = np.ones_like(seg)
    ones = np.ones_like(seg)
    for x in range(1, res.shape[0] -1 ):
        for y in range(1, res.shape[0] - 1):
            # If > 1 px distance, double the weight of the positive
            # If == 1 px, half the weight of the negative
            # This is important because the calc_mask fn
            # leaves borders with 0 weight otherwise
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
        
    # Empirically capping the loss at -3 to 3 is better
    res[np.where(res < -2)] = 2
    res[np.where(res > 2)] = 2
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
    #tomult = tf.where(y_true < 0.45, y_pred, tf.minimum(y_true, y_pred))
    multiplied = y_pred * y_true_dist_map
    #loss = tf.reduce_mean(multiplied, axis = (1, 2, 3))
    return multiplied


def bce_surface_loss(y_true, y_pred, alpha, weight, beta, mask):
    
    #lv = lovasz_softmax(probas = y_pred,
    #                    labels = tf.reshape(y_true, (-1, 14, 14)), 
    #                    classes=[1],
    #                    per_image=False) 
    
    bce = weighted_bce_loss(y_true = y_true, 
                             y_pred = y_pred, 
                             weight = weight,
                             smooth = 0.06)


    #bce = tf.reduce_mean(bce, axis = (1, 2, 3))
    sums = tf.reduce_sum(tf.squeeze(bce) * mask, axis = (1, 2))
    denom = tf.reduce_sum(mask, axis = (1, 2))
    bce = sums / denom


    surface = surface_loss(tf.cast(tf.math.greater(y_true, 0.1), tf.float32), y_pred)
    sums = tf.reduce_sum(tf.squeeze(surface) * mask, axis = (1, 2))
    #denom = tf.reduce_sum(mask, axis = (1, 2))
    surface = sums / denom
    #surface = tf.reduce_mean(surface, axis = (1, 2, 3))
    #surface = tf.reshape(surface, tf.shape(bce))

    bce = (1 - alpha) * bce
    surface_portion = alpha * surface
    
    result = bce + surface_portion
    result = tf.reduce_mean(result)
    return result


"""
def logcosh(y_true, y_pred):
  
    y_pred = ops.convert_to_tensor(y_pred)
    y_true = math_ops.cast(y_true, y_pred.dtype)
    
    def _logcosh(x):
        return x + tf.nn.softplus(-2. * x) - math_ops.log(2.)
    return tf.reduce_mean(tf.squared_difference(y_true, y_pred))#tf.losses.mean_squared_error(y_true, y_pred)
"""
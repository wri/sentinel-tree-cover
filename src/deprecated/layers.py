def a2_block(inp, in_channels, c_m, c_n, k = 1):
    # c_m is the output channel number of convs for V == 1/4 in_channels
    # c_n is the output channel number of convs for A, B == 1/4 in_channels
    # implementation based on https://arxiv.org/pdf/1810.11579.pdf
    b, d, h, w, c = inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3], inp.shape[4]
    inp = tf.reshape(inp, (-1, h, w, c*d))
    #inp = tf.transpose(inp, perm = [0, 3, 1, 2])
    b, h, w, c = inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]
    batch = inp.shape[0]

    A = Conv2D(filters = c_m, kernel_size = (1, 1), activation = 'linear',
                        kernel_initializer = lecun_normal(), data_format = 'channels_last')(inp)
    B = Conv2D(filters = c_n, kernel_size = (1, 1), activation = 'linear',
                        kernel_initializer = lecun_normal(), data_format = 'channels_last')(inp)
    V = Conv2D(filters = c_n, kernel_size = (1, 1), activation = 'linear',
                        kernel_initializer = lecun_normal(), data_format = 'channels_last')(inp)
    A = tf.transpose(A, perm = [0, 3, 1, 2])
    B = tf.transpose(B, perm = [0, 3, 1, 2])
    V = tf.transpose(V, perm = [0, 3, 1, 2])

    tmpA = tf.reshape(A, (-1, k, c_m, h*w))
    tmpA = tf.transpose(tmpA, perm=[0, 2, 1, 3])
    tmpA = tf.reshape(tmpA, (-1, c_m, k*h*w))

    tmpB = tf.reshape(B, (-1, k, c_n, h*w))
    tmpB = tf.transpose(tmpB, perm=[0,2,1,3])
    tmpB = tf.reshape(tmpB, (-1, c_n, k*h*w))

    tmpV = tf.reshape(V, (-1, k, c_n, h*w))
    tmpV = tf.transpose(tmpV, perm = [0, 1, 3, 2])
    tmpV = tf.reshape(tmpV, (inp.shape[0], c_m, h, w))

    softmaxB = tf.nn.softmax(tmpB)
    softmaxB = tf.reshape(softmaxB, (-1, c_n, k*h*w))
    softmaxB = tf.transpose(softmaxB, perm = [0, 2, 1])

    softmaxV = tf.nn.softmax(tmpV)
    softmaxV = tf.reshape(softmaxV, (-1, k*h*w, c_n))
    softmaxV = tf.transpose(softmaxV, perm = [0, 2, 1])

    tmpG = tf.linalg.matmul(tmpA, softmaxB)
    tmpZ = tf.linalg.matmul(tmpG, softmaxV)
    tmpZ = tf.reshape(tmpZ, (-1, c_m, k, h*w))
    tmpZ = tf.transpose(tmpV, perm = [0, 2, 1, 3])
    tmpZ = tf.reshape(tmpZ, (b, c_m, h, w))
    print(tmpZ.shape)

    return tmpZ


def temporal_attention(inp, units):
    # This rescales each output
    # Timesteps that are more important get weighted higher
    # Timesteps that are least important get weighted lower --> B, N, H, W, C
    conved = TimeDistributed(Conv2D(units, (1, 1), padding = 'same', kernel_initializer = 'glorot_uniform',
                            activation = 'tanh', strides = (1, 1)))(inp)


    #conved = tf.reshape(conved, (-1, units, 16, 16, STEPS))
    print("Attention weight shape: {}".format(conved.shape))
    conved = TimeDistributed(Conv2D(1, (1, 1), padding = 'same', kernel_initializer = 'glorot_uniform',
                            activation = 'sigmoid', use_bias = False, strides = (1, 1)))(conved)
    print("Conved sigmoid shape: {}".format(conved.shape))
    #conved = tf.reshape(conved, (-1, 24, 1, 1, 1))

    alphas = tf.reduce_sum(conved, axis = 1, keep_dims = True)
    print("Attention alphas: {}".format(alphas.shape))
    # We need to calculate the total sum for each pixel for each channel, so that we can combine them
    alphas = conved / alphas
    print("Attention weight shapes {}".format(alphas.shape))

    # This actually multiplies the Conv by the input
    multiplied = tf.reduce_sum(alphas * inp, axis = 1)
    return multiplied


def create_deconv_init(filter_size, num_channels):
    bilinear_kernel = np.zeros([filter_size, filter_size], dtype=np.float32)
    scale_factor = (filter_size + 1) // 2
    if filter_size % 2 == 1:
        center = scale_factor - 1
    else:
        center = scale_factor - 0.5
    for x in range(filter_size):
        for y in range(filter_size):
            bilinear_kernel[x,y] = (1 - abs(x - center) / scale_factor) * \
                                   (1 - abs(y - center) / scale_factor)
    weights = np.zeros((filter_size, filter_size, num_channels, num_channels))
    for i in range(num_channels):
        weights[:, :, i, i] = bilinear_kernel

    #assign numpy array to constant_initalizer and pass to get_variable
    bilinear_init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return bilinear_init

def get_deconv2d(inp, filter_count, num_channels, scope, is_training):
    bilinear_init = create_deconv_init(4, filter_count)
    x = tf.keras.layers.Conv2DTranspose(filters = filter_count, kernel_size = (4, 4),
                                        strides=(2, 2), padding='same',
                                        use_bias = False,
                                        kernel_initializer = bilinear_init)(inp)
    #x = ELU()(x)
    #x = tf.nn.relu(x)
    x = Batch_Normalization(x, training=is_training, scope = scope + "bn")
    return x

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


sum_pos = np.sum(train_y[batch], axis = (1, 2))
sum_pos = sum_pos[sum_pos != 196]
n_pos = len(train_y) - len(sum_pos)
sum_pos = np.sum(sum_pos)
sum_neg = np.sum(train_y[batch], axis = (1, 2))
sum_neg = sum_neg[sum_neg != 0]
n_neg = len(train_y) - len(sum_neg)
sum_neg = (len(train_y) - (n_neg + n_pos)) * 196
print(sum_pos, sum_neg)
beta = 0.9995
print("Beta: {}".format(beta))
samples_per_cls = np.array([sum_neg, sum_pos]) / 196
print(samples_per_cls)
effective_num = 1.0 - np.power(beta, samples_per_cls)
print(effective_num)
weights = (1.0 - beta) / np.array(effective_num)
weights = weights / np.sum(weights)
print("Neg and pos weights: {}".format(weights))
weight = weights[1] / weights[0]
print(weight)
weight = 1.45

print("Baseline: The positive is: {}".format(weights[0]))
print("Baseline: The negative is: {}".format(weights[1]))
print("\n")
print("Balanced: The positive is: {}".format(weight*weights[0]))
print("Balanced: The negative is: {}".format(weights[1]))


def make_stc(subtile, sess, s1_subtile, dem_subtile):

    b2 = subtile[..., 0]
    b3 = subtile[..., 1]
    b4 = subtile[..., 2]
    b8 = subtile[..., 3]
    b8a = subtile[..., 7]
    b11 = subtile[..., 8]
    b12 = subtile[..., 9]

    mndwi = (b3 - b11) / (b3 + b11)
    ndvi = (b8 - b4) / (b8 + b4)
    tcb = ((0.3029 * b2) + (0.2786 * b3) + (0.47 * b4) + (0.5599 * b8a) +\
    (0.508 * b11) + (0.1872 * b12))

    stc = np.nan((SIZE, SIZE, 10))

    first_criteria = np.logical_and(np.mean(mndwi, axis = 0) < -0.55, (ndvi[ndvi_argmax] - np.mean(ndvi, axis = 0)) < 0.05)
    second_criteria = np.logical_and(np.mean(ndvi, axis = 0) < -0.3, (np.mean(mndwi, axis = 0) - np.min(ndvi, axis = 0)) < 0.05)
    third_criteria = np.logical_and(np.mean(ndvi, axis = 0) > 0.6, np.mean(tcb, axis = 0) < 0.45)
    fourth_criteria = np.mean(ndvi, axis = 0) < - 0.2

    # pixels with the max NDVI
    # argmax of ndvi for each pixel
    # argmax

    stc[first_criteria] = maxndvi
    stc[np.logical_and(second_criteria, np.isnan(stc))] = maxndwi
    stc[np.logical_and(third_criteria, np.isnan(stc))] = maxndvi
    stc[np.logical_and(fourth_criteria, np.isnan(stc))] = maxndwi
    stc[np.isnan(stc)] = maxndvi
    stc = superresolve(stc, sess)
    stc_tile = np.empty((SIZE, SIZE, 13))
    stc_tile[..., :10] = stc
    stc_tile[..., 10] = dem_subtile.repeat(12, axis = 0)
    subtile[..., 11:] = s1_subtile
    return stc



def predict_stc(subtile, sess) -> np.ndarray:
    """ Runs non-temporal predictions on a (12, 174, 174, 13) array:
        - Calculates remote sensing indices
        - Normalizes data
        - Returns predictions for subtile

        Parameters:
         subtile (np.ndarray): monthly sentinel 2 + sentinel 1 mosaics
                               that will be median aggregated for model input
         sess (tf.Session): tensorflow session for prediction

        Returns:
         preds (np.ndarray): (160, 160) float32 [0, 1] predictions
    """

    if np.sum(subtile) > 0:
        if not isinstance(subtile.flat[0], np.floating):
            assert np.max(subtile) > 1
            subtile = subtile / 65535.
        indices = np.empty((subtile.shape[1], subtile.shape[2], 17))
        indices[..., :13] = subtile
        indices[..., 13] = evi(subtile)
        indices[...,  14] = bi(subtile)
        indices[...,  15] = msavi2(subtile)
        indices[...,  16] = grndvi(subtile)

        subtile = indices
        subtile = subtile.astype(np.float32)
        subtile = np.clip(subtile, min_all, max_all)
        subtile = (subtile - midrange) / (rng / 2)
        subtile = subtile[-1]


        batch_x = subtile[np.newaxis]
        lengths = np.full((batch_x.shape[0]), 12)
        preds = sess.run(gap_logits,
                              feed_dict={gap_inp:batch_x})

        preds = preds.squeeze()
        preds = preds[1:-1, 1:-1]
    else:
        preds = np.full((SIZE, SIZE), 255)
    return preds

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
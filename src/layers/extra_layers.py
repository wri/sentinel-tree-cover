import tensorflow as tf
from tflearn.layers.conv import global_avg_pool
import itertools
from tensorflow.python.keras.layers import Conv2D, Lambda, Dense, Multiply, Add

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

def fpa(inp, is_training, filter_count, clipping_params, 
        keep_rate, upsample = "upconv"):
    '''Feature pyramid attention layer block, that allows for cross-scale combination
       of different size features without making blurry feature maps.

         Parameters:
          inp (tf.Variable): input tensorflow layer
          is_training (str): flag to differentiate between train/test ops
          filter_count (int): number of filters for convolution
          clipping_params (dict): specifies clipping of 
                                  rmax, dmax, rmin for renormalization

         Returns:
          concat_1 (tf.Variable): output of FPA
          
         References:
          https://arxiv.org/abs/1805.10180
    '''
    one = conv_bn_relu(inp = inp, is_training = is_training, 
                       kernel_size = 1, scope =  'forward1',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = False,
                       use_bias = False, batch_norm = True,
                       dropblock = False,
                       csse = False)
    inp_pad = ReflectionPadding2D(padding = (2, 2))(inp)
    seven = conv_bn_relu(inp = inp_pad, is_training = is_training, 
                       kernel_size = 5, scope =  'down1', stride = (2, 2),
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True, csse = True, dropblock = False)
    seven_pad = ReflectionPadding2D(padding = (2, 2))(seven)
    seven_f = conv_bn_relu(inp = seven_pad, is_training = is_training, 
                       kernel_size = 5, scope =  'down1_f',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = False,
                       use_bias = False, batch_norm = True, csse = False, dropblock = False)
    
    print("Seven: {}".format(seven.shape))
    print("Seven f: {}".format(seven_f.shape))
    
    five_pad = ReflectionPadding2D(padding = (1, 1))(seven)
    five = conv_bn_relu(inp = five_pad, is_training = is_training,  stride = (2, 2),
                       kernel_size = 3, scope =  'down2',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True, csse = True, dropblock = False)
    
    five_pad2 = ReflectionPadding2D(padding = (1, 1))(five)
    five_f = conv_bn_relu(inp = five_pad2, is_training = is_training, 
                       kernel_size = 3, scope =  'down2_f',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = False,
                       use_bias = False, batch_norm = True, csse = False, dropblock = False)
    print("Five: {}".format(five.shape))
    print("Five_F: {}".format(five_f.shape))
    '''
    three_pad = ReflectionPadding2D(padding = (1, 1))(five)
    three = conv_bn_relu(inp = three_pad, is_training = is_training,  stride = (2, 2),
                       kernel_size = 3, scope =  'down3',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True, csse = True, dropblock = False)
    
    three_pad2 = ReflectionPadding2D(padding = (1, 1))(three)
    three_f = conv_bn_relu(inp = three_pad2, is_training = is_training, 
                       kernel_size = 3, scope =  'down3_f',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = False,
                       use_bias = False, batch_norm = True, csse = True, dropblock = False)
        
    
    if upsample == 'upconv' or 'bilinear':
        three_up = tf.keras.layers.UpSampling2D((2, 2), interpolation = 'bilinear')(three_f)
        if upsample == 'upconv':
            three_up = ReflectionPadding2D((1, 1,))(three_up)
            three_up = conv_bn_relu(inp = three_up, is_training = is_training, 
                       kernel_size = 3, scope =  'upconv1',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True,
                       csse = False, dropblock = False)
            
            # 4x4
            three_up = tf.nn.relu(tf.add(three_up, five_f))
    '''        
    
    if upsample == 'upconv' or "bilinear":
        five_up = tf.keras.layers.UpSampling2D((2, 2), interpolation = 'nearest')(five)
        if upsample == 'upconv':
            five_up = ReflectionPadding2D((1, 1,))(five_up)
            five_up = conv_bn_relu(inp = five_up, is_training = is_training, 
                       kernel_size = 3, scope =  'upconv2',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True, 
                       csse = False, dropblock = False)
            five_up = tf.nn.relu(tf.add(five_up, seven_f))
            
    if upsample == 'upconv' or "bilinear":
        seven_up = tf.keras.layers.UpSampling2D((2, 2), interpolation = 'nearest')(five_up)
        if upsample == 'upconv':
            seven_up = ReflectionPadding2D((1, 1,))(seven_up)
            seven_up = conv_bn_relu(inp = seven_up, is_training = is_training, 
                       kernel_size = 3, scope =  'upconv3',
                       filters = filter_count, clipping_params = clipping_params,
                       keep_rate = keep_rate, activation = True,
                       use_bias = False, batch_norm = True, 
                       csse = False, dropblock = False)
    
    print("One: {}".format(one.shape))
    print("Five_up: {}".format(five_up.shape))
    print("Seven_up: {}".format(seven_up.shape))
    
    # top block

    #pooled = tf.keras.layers.GlobalAveragePooling2D()(inp)
    #one_top = conv_bn_relu(inp = tf.reshape(pooled, (-1, 1, 1, pooled.shape[-1])),
    #                       is_training = is_training, 
    #                   kernel_size = 1, scope =  'topconv',
    #                   filters = filter_count, clipping_params = clipping_params,
    #                   keep_rate = keep_rate, activation = False,
    #                   use_bias = False, batch_norm = True, 
    #                   csse = False, dropblock = False)
    #one_top = conv_bn_relu(tf.reshape(pooled, (-1, 1, 1, pooled.shape[-1])),
    ##                      is_training, 1, 'top1', filter_count, pad = False)
    #four_top = tf.keras.layers.UpSampling2D((16, 16))(one_top)
    
    #seven_up = tf.multiply(one, seven_up)
    out = tf.nn.relu(tf.multiply(seven_up, one))
    return out


def conv_selu(inp, is_training, kernel_size, scope,
                filter_count = 16, pad = True, padding = 'valid', dilated = False,
                activation = True):
    '''Convolutional 2D layer with SELU activation and Lecun normal initialization
       with no batch norm. Only used if params['activation'] = 'selu'

         Parameters:
          inp (tf.Variable): (B, H, W, C) input layer
          is_training (str): flag to differentiate between train/test ops
          kernel_size (int): kernel size of convolution
          scope (str): tensorflow variable scope
          filter_count (int): number of convolution filters
          pad (bool): whether or not to reflect pad input
          padding (str): one of ['valid', 'same']
          dilated (bool): whether to perform atruous convolution
          activation (bool): whether to activate output

         Returns:
          conv (tf.Variable): output of Conv2D -> SELU
          
         References:
          https://arxiv.org/abs/1706.02515
    '''
    if activation:
        act = selu
    else:
        act = None
    if not dilated:
        padded = ReflectionPadding2D((1, 1,))(inp)
        conv = Conv2D(filters = filter_count, kernel_size = (kernel_size, kernel_size), activation = act,
                        padding = padding, kernel_initializer = lecun_normal())(padded)
    if not dilated and not pad:
        conv = Conv2D(filters = filter_count, kernel_size = (kernel_size, kernel_size), activation = act,
                        padding = padding, kernel_initializer = lecun_normal())(inp)
    if dilated:
        padded = ReflectionPadding2D((2, 2,))(inp)
        conv = Conv2D(filters = filter_count, kernel_size = (3, 3), activation = act, dilation_rate = (2, 2),
                        padding = padding, kernel_initializer = lecun_normal())(padded)
    return conv




def create_deconv_init(filter_size, num_channels):
    '''Initializes a kernel weight matrix with a bilinear deconvolution
    
         Parameters:
          filter_size (int): kernel size of convolution
          num_channels (int): number of filters for convolution

         Returns:
          bilinear_init (tf.Variable): [filter_size, filter_size, num_channels] kernel
    '''
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

def get_deconv2d(inp, filter_count, num_channels, scope, is_training, clipping_params):
    '''Creates a deconvolution layer with Conv2DTranspose. Following recent
       recommendations to use 4 kernel, 2 stride to avoid artifacts. 
       Initialize kernel with bilinear upsampling.

         Parameters:
          inp (tf.Variable): input tensorflow layer (B, X, Y, C) shape
          filter_count (int): number of filters for convolution
          num_channels (int): number of output channels
          scope (str): tensorflow variable scope
          is_training (str): flag to differentiate between train/test ops
          clipping_params (dict): specifies clipping of 
                                  rmax, dmax, rmin for renormalization

         Returns:
          x (tf.Variable): layer with (B, x * 2, y * 2, C) shape
          
         References:
          https://distill.pub/2016/deconv-checkerboard/
    '''
    bilinear_init = create_deconv_init(4, filter_count)
    x = tf.keras.layers.Conv2DTranspose(filters = filter_count, kernel_size = (4, 4),
                                        strides=(2, 2), padding='same', 
                                        use_bias = False,
                                        kernel_initializer = bilinear_init)(inp)
    #x = ELU()(x)
    #x = tf.nn.relu(x)
    x = Batch_Normalization(x, training=is_training, scope = scope + "bn", clipping_params = clipping_params)
    return x
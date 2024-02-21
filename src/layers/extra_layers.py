import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    #tf.disable_v2_behavior()
    #tf.logging.set_verbosity(tf.logging.ERROR)
import itertools
#from tensorflow.compat.v1.layers import *
from tensorflow.compat.v1.keras.layers import *
from dropblock import DropBlock2D, DropBlockMask, DoDropBlock

def group_norm(x, scope, G=8, esp=1e-5, window_size = 32):
    with tf.variable_scope('{}_norm'.format(scope)):
        # normalize
        # transpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        x = tf.transpose(x, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + esp)
        # per channel gamma and beta
        zeros = lambda: tf.zeros([C], dtype=tf.float32)
        ones = lambda: tf.ones([C], dtype=tf.float32)
        gamma = tf.Variable(initial_value = ones, dtype=tf.float32, name=f'gamma_{scope}')
        beta = tf.Variable(initial_value = zeros, dtype=tf.float32, name=f'beta_{scope}')
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])

        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        # tranpose: [bs, c, h, w, c] to [bs, h, w, c] following the paper
        output = tf.transpose(output, [0, 2, 3, 1])
    return output

def convGRU(x, cell_fw, cell_bw, ln):
        output, final = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, ln, dtype=tf.float32)
        #output = tf.concat(output, -1)
        #final = tf.concat(final, -1)
        return output, final

def fconvGRU(x, cell_fw, ln):
        output, final = tf.nn.dynamic_rnn(
            cell_fw, x, ln, dtype=tf.float32)
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



def cse_block(prevlayer, prefix):
    '''Channel excitation and spatial squeeze layer. 
       Calculates the mean of the spatial dimensions and then learns
       two dense layers, one with relu, and one with sigmoid, to rerank the
       input channels
       
         Parameters:
          prevlayer (tf.Variable): input layer
          prefix (str): prefix for tensorflow scope

         Returns:
          x (tf.Variable): output of the cse_block
    '''
    mean = Lambda(lambda xin: K.mean(xin, axis=[1, 2]))(prevlayer)
    lin1 = Dense(K.int_shape(prevlayer)[3] // 2, name=prefix + 'cse_lin1', activation='relu')(mean)
    lin2 = Dense(K.int_shape(prevlayer)[3], name=prefix + 'cse_lin2', activation='sigmoid')(lin1)
    x = Multiply()([prevlayer, lin2])
    return x


def sse_block(prevlayer, prefix):
    '''Spatial excitation and channel squeeze layer.
       Calculates a 1x1 convolution with sigmoid activation to create a 
       spatial map that is multiplied by the input layer

         Parameters:
          prevlayer (tf.Variable): input layer
          prefix (str): prefix for tensorflow scope

         Returns:
          x (tf.Variable): output of the sse_block
    '''
    conv = Conv2D(1, (1, 1), padding="same", kernel_initializer=tf.keras.initializers.he_normal(),
                  activation='sigmoid', strides=(1, 1),
                  name=prefix + "_conv")(prevlayer)
    conv = Multiply(name=prefix + "_mul")([prevlayer, conv])
    return conv


def csse_block(x, prefix):
    '''Implementation of Concurrent Spatial and Channel 
       ‘Squeeze & Excitation’ in Fully Convolutional Networks
    
        Parameters:
          prevlayer (tf.Variable): input layer
          prefix (str): prefix for tensorflow scope

         Returns:
          x (tf.Variable): added output of cse and sse block
          
         References:
          https://arxiv.org/abs/1803.02579
    '''
    #cse = cse_block(x, prefix)
    sse = sse_block(x, prefix)
    #x = Add(name=prefix + "_csse_mul")([cse, sse])

    return sse

class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        print("ZERO PADDING")
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
    
class ReflectionPadding5D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding5D, self).__init__(**kwargs)

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        print("ZERO PADDING")
        return tf.pad(x, [[0,0], [0, 0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')





class WSConv2D(tf.keras.layers.Conv2D):
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(kernel_initializer="he_normal", *args, **kwargs)

    def standardize_weight(self, weight, eps):

        mean = tf.math.reduce_mean(weight, axis=(0, 1, 2), keepdims=True)
        weight = weight - mean
        var = tf.keras.backend.std(weight, axis=[0, 1, 2], keepdims=True)
        weight = weight / (var + 1e-5)
        return weight

    def call(self, inputs, eps=1e-4):
        self.kernel.assign(self.standardize_weight(self.kernel, eps))
        return super().call(inputs)

def partial_conv(x, channels, kernel=3, stride=1, norm = True, use_bias=False, padding='SAME', scope='conv_0'):
    

    
    with tf.variable_scope(scope):
        if padding.lower() == 'SAME'.lower() :
            with tf.variable_scope('mask'):
                _, h, w, _ = x.get_shape().as_list()

                slide_window = kernel * kernel
                mask = tf.ones(shape=[1, h, w, 1])

                update_mask = tf.layers.conv2d(mask, filters=1,
                                               kernel_size=kernel, kernel_initializer=tf.constant_initializer(1.0),
                                               strides=stride, padding=padding, use_bias=False, trainable=False)

                mask_ratio = slide_window / (update_mask + 1e-8)
                update_mask = tf.clip_by_value(update_mask, 0.0, 1.0)
                mask_ratio = mask_ratio * update_mask

            with tf.variable_scope('x'):
                if 4 > 3:
                    x = WSConv2D(filters=channels, kernel_regularizer = None,
                                     kernel_size=kernel,
                                     strides=stride, padding=padding, use_bias=False)(x)
                else:
                    x = tf.layers.conv2d(x, filters=channels, kernel_regularizer = None,
                                     kernel_size=kernel, kernel_initializer=tf.keras.initializers.he_normal(),
                                     strides=stride, padding=padding, use_bias=False)
    
                x = x * mask_ratio
                
                

                if use_bias:
                    bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

                    x = tf.nn.bias_add(x, bias)
                    x = x * update_mask

        else :
            if 4 > 3:
                x = WSConv2D(filters=channels,kernel_regularizer = None,
                                 kernel_size=kernel,
                                 strides=stride, padding=padding, use_bias=False)(x)
            else:
                x = tf.layers.conv2d(x, filters=channels,kernel_regularizer = None,
                                 kernel_size=kernel, kernel_initializer=tf.keras.initializers.he_normal(),
                                 strides=stride, padding=padding, use_bias=use_bias)

        return x
    

def conv_swish_gn(inp, 
                 is_training, 
                 kernel_size,
                 scope,
                 filters, 
                 keep_rate,
                 stride = (1, 1),
                 activation = True,
                 use_bias = False,
                 norm = True,
                 dropblock = True,
                 csse = True,
                 weight_decay = None,
                 block_size = 5,
                 padding = "SAME",
                 window_size = 48):
    '''2D convolution, batch renorm, relu block, 3x3 drop block. 
       Use_bias must be set to False for batch normalization to work. 
       He normal initialization is used with batch normalization.
       RELU is better applied after the batch norm.
       DropBlock performs best when applied last, according to original paper.

         Parameters:
          inp (tf.Variable): input layer
          is_training (str): flag to differentiate between train/test ops
          kernel_size (int): size of convolution
          scope (str): tensorflow variable scope
          filters (int): number of filters for convolution
          clipping_params (dict): specifies clipping of 
                                  rmax, dmax, rmin for renormalization
          activation (bool): whether to apply RELU
          use_bias (str): whether to use bias. Should always be false

         Returns:
          bn (tf.Variable): output of Conv2D -> Batch Norm -> RELU
        
         References:
          http://papers.nips.cc/paper/8271-dropblock-a-regularization-
              method-for-convolutional-networks.pdf
          https://arxiv.org/abs/1702.03275
          
    '''
    
    bn_flag = "Group Norm" if norm else ""
    activation_flag = "RELU" if activation else "Linear"
    csse_flag = "CSSE" if csse else "No CSSE"
    bias_flag = "Bias" if use_bias else "NoBias"
    drop_flag = "DropBlock" if dropblock else "NoDrop"
        
    
    print("{} {} Conv 2D {} {} {} {} {}".format(scope, kernel_size,
                                                   bn_flag, activation_flag,
                                                   csse_flag, bias_flag, drop_flag))
    
    with tf.variable_scope(scope + "_conv"):
        #conv = Conv2D(filters = filters, kernel_size = (kernel_size, kernel_size),  strides = stride,
        #              activation = None, padding = 'valid', use_bias = use_bias,
                      #kernel_regularizer = weight_decay,
        #              kernel_initializer = tf.keras.initializers.he_normal())(inp)
        conv = partial_conv(inp, filters, kernel=kernel_size, stride=1, 
                            use_bias=False, padding=padding, scope = scope)
    print(f"The non normalized feats are {conv}")    
    if activation:
        conv = tf.nn.swish(conv)
    print(f"The non normalized feats are {conv}")
    # if dropblock:
    #.    mask = make_mask()
    
    if norm:
        conv = group_norm(x = conv, scope = scope, G = 8,#(filters // 4), 
                          window_size = window_size) #mask = mask
    if csse:
        conv = csse_block(conv, "csse_" + scope)
    
    if dropblock: 
        with tf.variable_scope(scope + "_drop"):
            drop_block = DropBlock2D(keep_prob=keep_rate, block_size=block_size)
            conv = drop_block(conv, is_training)
    return conv
import tensorflow as tf
if tf.__version__[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.ERROR)
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import ELU
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.layers import Conv2D, Lambda, Dense, Multiply, Add
from tensorflow.python.util import deprecation as deprecation
from keras.regularizers import l1
from tensorflow.compat.v1.initializers import orthogonal
import math
import os

deprecation._PRINT_DEPRECATION_WARNINGS = False
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import keras
from keras import backend as K



# ### Utility blocks (Batch norm, cSSE, etc.)
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


def weighted_group_norm(x, mask, scope, G=8, esp=1e-5, window_size = 32):
    with tf.variable_scope('{}_norm'.format(scope)):
        # normalize
        # transpose: [bs, h, w, c] to [bs, c, h, w] following the paper
        #x, mask = x
        x = tf.transpose(x, [0, 3, 1, 2])
        mask = tf.transpose(mask, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(G, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mask = tf.reshape(mask, [-1, G, C // G, H, W])
        mean, var = tf.nn.weighted_moments(x, [2, 3, 4], mask, keep_dims=True)
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


# ### Conv GRU Block
def gru_block(inp, length, size, flt, scope, train, normalize = False,
  zoneout = 0.75):
    '''Bidirectional convolutional GRU block with 
       zoneout and CSSE blocks in each time step

         Parameters:
          inp (tf.Variable): (B, T, H, W, C) layer
          length (tf.Variable): (B, T) layer denoting number of
                                steps per sample
          size (int): kernel size of convolution
          flt (int): number of convolution filters
          scope (str): tensorflow variable scope
          train (tf.Bool): flag to differentiate between train/test ops
          normalize (bool): whether to compute layer normalization

         Returns:
          gru (tf.Variable): (B, H, W, flt*2) bi-gru output
          steps (tf.Variable): (B, T, H, W, flt*2) output of each step
    '''
    with tf.variable_scope(scope):
        print(f"GRU input shape {inp.shape}, zoneout: {0.5}")
        """
        cell_fw = ConvLSTMCell(shape = size, filters = flt,
                               kernel = [3, 3], forget_bias=1.0, 
                               activation=tf.tanh, normalize=True, 
                               peephole=False, data_format='channels_last', reuse=None)
        cell_bw = ConvLSTMCell(shape = size, filters = flt,
                               kernel = [3, 3], forget_bias=1.0, 
                               activation=tf.tanh, normalize=True, 
                               peephole=False, data_format='channels_last', reuse=None)
        """
        cell_fw = ConvGRUCell(shape = size, filters = flt,
                           kernel = [3, 3], padding = 'VALID', normalize = True, sse = True)
        cell_bw = ConvGRUCell(shape = size, filters = flt,
                           kernel = [3, 3], padding = 'VALID', normalize = True, sse = True)
        cell_fw = ZoneoutWrapper(
           cell_fw, zoneout_drop_prob = zoneout, is_training = train)
        cell_bw = ZoneoutWrapper(
            cell_bw, zoneout_drop_prob = zoneout, is_training = train)
        #print(inp.shape)
        steps, out = convGRU(inp, cell_fw, cell_bw, length)#cell_bw, length)
        print(f"Zoneout: {zoneout}")
        gru = tf.concat(out, axis = -1)
        steps = tf.concat(steps, axis = -1)
        print(f"Down block output shape {gru.shape}")
    return gru, steps


def convGRU(x, cell_fw, cell_bw, ln):
        output, final = tf.nn.bidirectional_dynamic_rnn(
            cell_fw, cell_bw, x, ln, dtype=tf.float32)
        #output = tf.concat(output, -1)
        #final = tf.concat(final, -1)
        return output, final


class ConvGRUCell(tf.nn.rnn_cell.RNNCell):
  """A GRU cell with convolutions instead of multiplications."""

  def __init__(self, shape, filters, kernel, initializer = tf.initializers.orthogonal, sse = False, padding = 'VALID', pad_input = True, activation=tf.tanh, normalize=False, data_format='channels_last', reuse=None):
    super(ConvGRUCell, self).__init__(_reuse=reuse)
    self._filters = filters
    self._kernel = kernel
    self._activation = activation
    self._normalize = normalize
    self._padding = padding
    self._initializer = initializer
    self._pad_input = pad_input
    self._sse = sse
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return self._size

  @property
  def output_size(self):
    return self._size

  def call(self, x, h):
    channels = x.shape[self._feature_axis].value

    with tf.variable_scope('gates'):
      inputs = tf.concat([x, h], axis=self._feature_axis)
      n = channels + self._filters
      m = 2 * self._filters if self._filters > 1 else 2
      W = tf.get_variable('kernel', self._kernel + [n, m]) # [3, 3, C, 2C]
      print(W.shape)
      if self._pad_input:
        inputs_pad = tf.pad(inputs, [[0, 0], [1, 1], [1, 1] ,[0,0] ], 'REFLECT')
      y = tf.nn.convolution(inputs_pad, W, self._padding, data_format=self._data_format)
      #if self._sse:
      #  W_1 = tf.get_variable("kernel_1", [1, 1, m, 1]) # [1, 1, C, 1]
      #  y_1 = tf.nn.convolution(y, W_1, 'VALID')
       # y_1 = tf.nn.sigmoid(y _1)
       # print(y_1.shape)
       # y = y * y_1
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = group_norm(r, "gates_r", G = 8, esp = 1e-5, window_size = 104)
        u = group_norm(u, "gates_u", G = 8, esp = 1e-5, window_size = 104)
        #r = tf.contrib.layers.layer_norm(r)
        #u = tf.contrib.layers.layer_norm(u)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.ones_initializer())
        r, u = tf.split(y, 2, axis=self._feature_axis)
    r, u = tf.sigmoid(r), tf.sigmoid(u)

    with tf.variable_scope('candidate'):
      inputs = tf.concat([x, r * h], axis=self._feature_axis)
      n = channels + self._filters
      m = self._filters
      if self._pad_input:
        inputs_pad = tf.pad(inputs, [[0, 0], [1, 1], [1, 1] ,[0,0] ], 'REFLECT') #
      W = tf.get_variable('kernel', self._kernel + [n, m]) # [3, 3, C, 2C]
      y = tf.nn.convolution(inputs_pad, W, self._padding, data_format=self._data_format)
        # This is the one we want though
      if self._sse:
        W_1 = tf.get_variable("kernel_1", [1, 1, m, 1]) #[1, 1, C, 1]
        y_1 = tf.nn.convolution(y, W_1, 'VALID')
        y_1 = tf.nn.sigmoid(y_1)
        y = y * y_1
      if self._normalize:
        #y = tf.contrib.layers.layer_norm(y)
         y = group_norm(y, "candidate_y", G =8, esp = 1e-5, window_size = 104)
      else:
        y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    h = u * h + (1 - u) * self._activation(y)

    return h, h



class ConvLSTMCell(tf.nn.rnn_cell.RNNCell):
  """A LSTM cell with convolutions instead of multiplications.
  Reference:
    Xingjian, S. H. I., et al. "Convolutional LSTM network: A machine learning approach for precipitation nowcasting." Advances in Neural Information Processing Systems. 2015.
  """

  def __init__(self, shape, filters, kernel, forget_bias=1.0, activation=tf.tanh, normalize=True, peephole=True, data_format='channels_last', reuse=None):
    super(ConvLSTMCell, self).__init__(_reuse=reuse)
    self._kernel = kernel
    self._filters = filters
    self._forget_bias = forget_bias
    self._activation = activation
    self._normalize = normalize
    self._peephole = peephole
    if data_format == 'channels_last':
        self._size = tf.TensorShape(shape + [self._filters])
        self._feature_axis = self._size.ndims
        self._data_format = None
    elif data_format == 'channels_first':
        self._size = tf.TensorShape([self._filters] + shape)
        self._feature_axis = 0
        self._data_format = 'NC'
    else:
        raise ValueError('Unknown data_format')

  @property
  def state_size(self):
    return tf.nn.rnn_cell.LSTMStateTuple(self._size, self._size)

  @property
  def output_size(self):
    return self._size

  def call(self, x, state):
    c, h = state

    x = tf.concat([x, h], axis=self._feature_axis)
    n = x.shape[-1].value
    m = 4 * self._filters if self._filters > 1 else 4
    W = tf.get_variable('kernel', self._kernel + [n, m])
    y = tf.nn.convolution(x, W, 'SAME', data_format=self._data_format)
    if not self._normalize:
      y += tf.get_variable('bias', [m], initializer=tf.zeros_initializer())
    j, i, f, o = tf.split(y, 4, axis=self._feature_axis)

    if self._peephole:
      i += tf.get_variable('W_ci', c.shape[1:]) * c
      f += tf.get_variable('W_cf', c.shape[1:]) * c

    if self._normalize:
      j = tf.contrib.layers.layer_norm(j)
      i = tf.contrib.layers.layer_norm(i)
      f = tf.contrib.layers.layer_norm(f)

    f = tf.sigmoid(f + self._forget_bias)
    i = tf.sigmoid(i)
    c = c * f + i * self._activation(j)

    if self._peephole:
      o += tf.get_variable('W_co', c.shape[1:]) * c

    if self._normalize:
      o = tf.contrib.layers.layer_norm(o)
      c = tf.contrib.layers.layer_norm(c)

    o = tf.sigmoid(o)
    h = o * self._activation(c)

    state = tf.nn.rnn_cell.LSTMStateTuple(c, h)
    #state = state.h

    return h, state


def attention(inp, units):
    weighted = TimeDistributed(Conv2D(units, (1, 1), padding = 'same', kernel_initializer = tf.keras.initializers.Ones(),
                            activation = 'sigmoid', strides = (1, 1), use_bias = False, ))(inp) 
    alphas = tf.reduce_sum(weighted, axis = 1, keep_dims = True)
    alphas = weighted / alphas
    multiplied = tf.reduce_sum(alphas * inp, axis = 1)
    print(multiplied.shape)
    return multiplied


# ### Conv blocks
# Partial Conv with weight standardization
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
                if 3 > 2:
                    x = WSConv2D(filters=channels, kernel_regularizer = None,
                                     kernel_size=kernel,
                                     strides=stride, padding=padding, use_bias=False).apply(x)
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
            if 3 > 2:
                x = WSConv2D(filters=channels,kernel_regularizer = None,
                                 kernel_size=kernel,
                                 strides=stride, padding=padding, use_bias=use_bias).apply(x)
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
                 partial = True, window_size = 5):
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
        if not partial:
            conv = Conv2D(filters = filters, kernel_size = (kernel_size, kernel_size),  strides = stride,
                          activation = None, padding = 'valid', use_bias = use_bias,
                          #kernel_regularizer = weight_decay,
                          kernel_initializer = tf.keras.initializers.he_normal())(inp)
        if partial:
            conv = partial_conv(inp, filters, kernel=kernel_size, stride=1, norm = norm,
                                use_bias= use_bias, padding=padding, scope = scope)
    if activation:
        conv = tf.nn.swish(conv)
    if dropblock:
        _mask = DropBlockMask(keep_prob=keep_rate, block_size= block_size)
        mask = _mask(conv, is_training)
    else:
        _mask = DropBlockMask(keep_prob=1., block_size= block_size)
        mask = _mask(conv, is_training)
    if norm:
        if filters > 80:
            G = 40
        if filters == 80:
            G = 20
        if filters == 40:
            G = 10
        #conv = tf.layers.batch_normalization(conv, training=is_training)
        conv = weighted_group_norm(x = conv, mask = mask, scope = scope, G = 8, window_size = window_size)
        #conv = group_norm(x = conv, scope = scope, G = 8, window_size = window_size)
        
    if csse:
        conv = csse_block(conv, "csse_" + scope)
    if dropblock: 
        with tf.variable_scope(scope + "_drop"):
            print("MASK", mask)
            drop_block = DoDropBlock(keep_prob=keep_rate, block_size= block_size)
           # drop_block = DropBlock2D(keep_prob=keep_rate, block_size= block_size)
            conv = drop_block([conv, mask], is_training)
    return conv

class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
  """Add Zoneout to a RNN cell."""

  def __init__(self, cell, zoneout_drop_prob, is_training=True):
    self._cell = cell
    self._zoneout_prob = zoneout_drop_prob
    self._is_training = is_training

  @property
  def state_size(self):
    return self._cell.state_size

  @property
  def output_size(self):
    return self._cell.output_size

  def __call__(self, inputs, state, scope=None):
    output, new_state = self._cell(inputs, state, scope)
    if not isinstance(self._cell.state_size, tuple):
      new_state = tf.split(value=new_state, num_or_size_splits=2, axis=1)
      state = tf.split(value=state, num_or_size_splits=2, axis=1)
    final_new_state = [new_state[0], new_state[1]]
    if self._is_training is True:
      for i, state_element in enumerate(state):
        random_tensor = 1 - self._zoneout_prob  # keep probability
        random_tensor += tf.random_uniform(tf.shape(state_element))
        # 0. if [zoneout_prob, 1.0) and 1. if [1.0, 1.0 + zoneout_prob)
        binary_tensor = tf.floor(random_tensor)
        final_new_state[
            i] = (new_state[i] - state_element) * binary_tensor + state_element
    else:
      for i, state_element in enumerate(state):
        final_new_state[
            i] = state_element * self._zoneout_prob + new_state[i] * (
                1 - self._zoneout_prob)
    if isinstance(self._cell.state_size, tuple):
      return output, tf.contrib.rnn.LSTMStateTuple(
          final_new_state[0], final_new_state[1])

    return output, tf.concat([final_new_state[0], final_new_state[1]], 1)

def print_trainable_params():
  total_parameters = 0
  for variable in tf.trainable_variables():
      shape = variable.get_shape()
      variable_parameters = 1
      for dim in shape:
          variable_parameters *= dim.value
      total_parameters += variable_parameters
  print(f"This model has {total_parameters} parameters")


def get_tensors_in_checkpoint_file(file_name,all_tensors=True,tensor_name=None):
    varlist=[]
    var_value =[]
    reader = tf.compat.v1.train.NewCheckpointReader(file_name)
    if all_tensors:
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in sorted(var_to_shape_map):
            varlist.append(key)
            var_value.append(reader.get_tensor(key))
    else:
        varlist.append(tensor_name)
        var_value.append(reader.get_tensor(tensor_name))
    return (varlist, var_value)

def build_tensors_in_checkpoint_file(loaded_tensors):
    full_var_list = list()
    # Loop all loaded tensors
    for i, tensor_name in enumerate(loaded_tensors[0]):
        # Extract tensor
        try:
            tensor_aux = tf.get_default_graph().get_tensor_by_name(tensor_name+":0")
            full_var_list.append(tensor_aux)
        except:
            print('Not found: '+tensor_name)
        
    return full_var_list

def initialize_uninitialized(sess):
    global_vars = tf.global_variables()
    is_not_initialized = sess.run([tf.is_variable_initialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    # for i in not_initialized_vars: # only for testing
    #    print(i.name)

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

def _bernoulli(shape, mean):
    return tf.nn.relu(tf.sign(mean - tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float32)))

class DropBlockMask(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlockMask, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        print("SHAPE1", input_shape)
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlockMask, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def _mask():
            mask = self._create_mask(tf.shape(inputs))
            #output = inputs * mask
            #output = tf.cond(self.scale,
            #                 true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
            #                 false_fn=lambda: output)
            return mask

        if training is None:
            training = K.learning_phase()
        mask = _mask()
        return mask

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DoDropBlock(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DoDropBlock, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        input_shape, _ = input_shape
        print("SHAPE", input_shape)
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        #self.set_keep_prob()
        super(DoDropBlock, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            inp, mask = inputs
            output = inp * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = drop()#tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                 #        true_fn=lambda: inputs,
                  #       false_fn=drop)
        return output


class DropBlock2D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock2D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 4
        _, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0 = (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock2D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        w, h = tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = (1. - self.keep_prob) * (w * h) / (self.block_size ** 2) / \
                     ((w - self.block_size + 1) * (h - self.block_size + 1))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                       self.h - self.block_size + 1,
                                       self.w - self.block_size + 1,
                                       self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool(mask, [1, self.block_size, self.block_size, 1], [1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask


class DropBlock3D(tf.keras.layers.Layer):
    def __init__(self, keep_prob, block_size, scale=True, **kwargs):
        super(DropBlock3D, self).__init__(**kwargs)
        self.keep_prob = float(keep_prob) if isinstance(keep_prob, int) else keep_prob
        self.block_size = int(block_size)
        self.scale = tf.constant(scale, dtype=tf.bool) if isinstance(scale, bool) else scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def build(self, input_shape):
        assert len(input_shape) == 5
        _, self.d, self.h, self.w, self.channel = input_shape.as_list()
        # pad the mask
        p1 = (self.block_size - 1) // 2
        p0= (self.block_size - 1) - p1
        self.padding = [[0, 0], [p0, p1], [p0, p1], [p0, p1], [0, 0]]
        self.set_keep_prob()
        super(DropBlock3D, self).build(input_shape)

    def call(self, inputs, training=None, **kwargs):
        def drop():
            mask = self._create_mask(tf.shape(inputs))
            output = inputs * mask
            output = tf.cond(self.scale,
                             true_fn=lambda: output * tf.to_float(tf.size(mask)) / tf.reduce_sum(mask),
                             false_fn=lambda: output)
            return output

        if training is None:
            training = K.learning_phase()
        output = tf.cond(tf.logical_or(tf.logical_not(training), tf.equal(self.keep_prob, 1.0)),
                         true_fn=lambda: inputs,
                         false_fn=drop)
        return output

    def set_keep_prob(self, keep_prob=None):
        """This method only supports Eager Execution"""
        if keep_prob is not None:
            self.keep_prob = keep_prob
        d, w, h = tf.to_float(self.d), tf.to_float(self.w), tf.to_float(self.h)
        self.gamma = ((1. - self.keep_prob) * (d * w * h) / (self.block_size ** 3) /
                      ((d - self.block_size + 1) * (w - self.block_size + 1) * (h - self.block_size + 1)))

    def _create_mask(self, input_shape):
        sampling_mask_shape = tf.stack([input_shape[0],
                                        self.d - self.block_size + 1,
                                        self.h - self.block_size + 1,
                                        self.w - self.block_size + 1,
                                        self.channel])
        mask = _bernoulli(sampling_mask_shape, self.gamma)
        mask = tf.pad(mask, self.padding)
        mask = tf.nn.max_pool3d(mask, [1, self.block_size, self.block_size, self.block_size, 1], [1, 1, 1, 1, 1], 'SAME')
        mask = 1 - mask
        return mask
import tensorflow as tf
from tensorflow.initializers import orthogonal

def group_norm(x, scope, G=8, esp=1e-5):
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
       # y_1 = tf.nn.sigmoid(y_1)
       # print(y_1.shape)
       # y = y * y_1
      if self._normalize:
        r, u = tf.split(y, 2, axis=self._feature_axis)
        r = group_norm(r, "gates_r", G = 6, esp = 1e-5)
        u = group_norm(u, "gates_u", G = 6, esp = 1e-5)
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
         y = group_norm(y, "candidate_y", G = 6, esp = 1e-5)
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
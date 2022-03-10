import tensorflow as tf


class ZoneoutWrapper(tf.contrib.rnn.RNNCell):
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

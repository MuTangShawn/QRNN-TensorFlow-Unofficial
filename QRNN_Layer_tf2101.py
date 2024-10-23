
"""Quasi-Recurrent Neural Networks layer."""

from keras import activations
from keras import backend
from keras import constraints
from keras import initializers
from keras import regularizers
from keras.engine import base_layer
from keras.engine.input_spec import InputSpec
from keras.layers.rnn import rnn_utils
from keras.layers.rnn.base_rnn import RNN
from keras.layers.rnn.dropout_rnn_cell_mixin import DropoutRNNCellMixin
from keras.utils import tf_utils
import tensorflow as tf

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import keras_export

RECURRENT_DROPOUT_WARNING_MSG = (
    'RNN `implementation=2` is not supported when `recurrent_dropout` is set. '
    'Using `implementation=1`.')

@keras_export(v1=['keras.layers.QRNNCell'])
class QRNNCell(DropoutRNNCellMixin, base_layer.BaseRandomLayer):
  """Cell class for the QRNN layer.
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    bias_initializer: Initializer for the bias vector.
    unit_forget_bias: Boolean.
      If True, add 1 to the bias of the forget gate at initialization.
      Setting it to true will also force `bias_initializer="zeros"`.
      This is recommended in [Jozefowicz et al., 2015](
        http://www.jmlr.org/proceedings/papers/v37/jozefowicz15.pdf)
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
  Call arguments:
    inputs: A 2D tensor.
    states: List of state tensors corresponding to the previous timestep.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. Only relevant when `dropout` or
      `recurrent_dropout` is used.
  """

  def __init__(self,
               units,
               kernel_size=2,
               strides=1,
               pool_mode="fo",
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               reset_after=True,
               **kwargs):
    if units < 0:
      raise ValueError(f'Received an invalid value for argument `units`, '
                       f'expected a positive integer, got {units}.')
    # By default use cached variable under v2 mode, see b/143699808.
    if tf.compat.v1.executing_eagerly_outside_functions():
      self._enable_caching_device = kwargs.pop('enable_caching_device', True)
    else:
      self._enable_caching_device = kwargs.pop('enable_caching_device', False)
    super(QRNNCell, self).__init__(**kwargs)
    self.units = units
    self.kernel_size = kernel_size
    self.strides = strides
    self.pool_mode = pool_mode

    self.activation = activations.get(activation)
    self.recurrent_activation = activations.get(recurrent_activation)
    self.use_bias = use_bias

    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)

    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)

    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)

    self.dropout = min(1., max(0., dropout))
    self.recurrent_dropout = min(1., max(0., recurrent_dropout))
    implementation = kwargs.pop('implementation', 1)
    if self.recurrent_dropout != 0 and implementation != 1:
      logging.debug(RECURRENT_DROPOUT_WARNING_MSG)
      self.implementation = 1
    else:
      self.implementation = implementation
    self.reset_after = reset_after
    self.state_size = [self.units,]
    self.output_size = self.units
     
  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    return super().build(input_shape)

  def call(self, inputs, states, training=None):

    if(self.pool_mode == "f"):
      h_tm1 = states[0]

      z_t = inputs[:, :self.units]
      f_t = inputs[:, self.units:]

      z_t = self.activation(z_t)
      f_t = self.recurrent_activation(f_t)

      rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(f_t, training, count=1)
      if 0. < self.recurrent_dropout < 1.:
        f_t = 1 - (1 - f_t) * rec_dp_mask[0]
      else:
        f_t = f_t

      h_t = f_t * h_tm1 + (1 - f_t) * z_t
      
      out_states = h_t
      pass

    elif(self.pool_mode == "fo"):
      c_tm1 = states[0]

      z_t = inputs[:, :self.units]
      f_t = inputs[:, self.units:2 * self.units]
      o_t = inputs[:, 2 * self.units:]

      z_t = self.activation(z_t)
      f_t = self.recurrent_activation(f_t)
      o_t = self.recurrent_activation(o_t)

      rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(f_t, training, count=1)
      if 0. < self.recurrent_dropout < 1.:
        f_t = 1 - (1 - f_t) * rec_dp_mask[0]
      else:
        f_t = f_t

      c_t = f_t * c_tm1 + (1 - f_t) * z_t
      h_t = o_t * c_t
      
      out_states = c_t
      pass

    elif(self.pool_mode == "ifo"):
      c_tm1 = states[0]

      z_t = inputs[:, :self.units]
      i_t = inputs[:, self.units:2 * self.units]
      f_t = inputs[:, 2 * self.units:3 * self.units]
      o_t = inputs[:, 3 * self.units:]

      z_t = self.activation(z_t)
      i_t = self.recurrent_activation(i_t)
      f_t = self.recurrent_activation(f_t)
      o_t = self.recurrent_activation(o_t)

      rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(f_t, training, count=1)
      if 0. < self.recurrent_dropout < 1.:
        f_t = 1 - (1 - f_t) * rec_dp_mask[0]
      else:
        f_t = f_t

      c_t = f_t * c_tm1 + i_t * z_t
      h_t = o_t * c_t

      out_states = c_t
      pass
  
    new_state = [out_states] if tf.nest.is_nested(states) else out_states
    return h_t, new_state

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'pool_mode':
            self.pool_mode,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after,
    }
    config.update(rnn_utils.config_for_enable_caching_device(self))
    base_config = super(QRNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
    return rnn_utils.generate_zero_filled_state_for_cell(
        self, inputs, batch_size, dtype)

@keras_export(v1=['keras.layers.QRNN'])
class QRNN(DropoutRNNCellMixin, RNN, base_layer.BaseRandomLayer):
  """
  Args:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      Default: hyperbolic tangent (`tanh`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    recurrent_activation: Activation function to use
      for the recurrent step.
      Default: hard sigmoid (`hard_sigmoid`).
      If you pass `None`, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix,
      used for the linear transformation of the inputs.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation")..
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
    dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the inputs.
    recurrent_dropout: Float between 0 and 1.
      Fraction of the units to drop for
      the linear transformation of the recurrent state.
    return_sequences: Boolean. Whether to return the last output
      in the output sequence, or the full sequence.
    return_state: Boolean. Whether to return the last state
      in addition to the output.
    go_backwards: Boolean (default False).
      If True, process the input sequence backwards and return the
      reversed sequence.
    stateful: Boolean (default False). If True, the last state
      for each sample at index i in a batch will be used as initial
      state for the sample of index i in the following batch.
    unroll: Boolean (default False).
      If True, the network will be unrolled,
      else a symbolic loop will be used.
      Unrolling can speed-up a RNN,
      although it tends to be more memory-intensive.
      Unrolling is only suitable for short sequences.
    time_major: The shape format of the `inputs` and `outputs` tensors.
      If True, the inputs and outputs will be in shape
      `(timesteps, batch, ...)`, whereas in the False case, it will be
      `(batch, timesteps, ...)`. Using `time_major = True` is a bit more
      efficient because it avoids transposes at the beginning and end of the
      RNN calculation. However, most TensorFlow data is batch-major, so by
      default this function accepts input and emits output in batch-major
      form.
    reset_after: QRNN convention (whether to apply reset gate after or
      before matrix multiplication). False = "before" (default),
      True = "after" (cuDNN compatible).
  Call arguments:
    inputs: A 3D tensor.
    mask: Binary tensor of shape `(samples, timesteps)` indicating whether
      a given timestep should be masked. An individual `True` entry indicates
      that the corresponding timestep should be utilized, while a `False`
      entry indicates that the corresponding timestep should be ignored.
    training: Python boolean indicating whether the layer should behave in
      training mode or in inference mode. This argument is passed to the cell
      when calling it. This is only relevant if `dropout` or
      `recurrent_dropout` is used.
    initial_state: List of initial state tensors to be passed to the first
      call of the cell.
  """

  def __init__(self,
               units,
               kernel_size=2,
               strides=1,
               pool_mode="fo",
               activation='tanh',
               recurrent_activation='sigmoid',
               use_bias=False,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               dropout=0.,
               recurrent_dropout=0.,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               time_major=False,
               reset_after=True,
               **kwargs):
    # return_runtime is a flag for testing, which shows the real backend
    # implementation chosen by grappler in graph mode.
    self._return_runtime = kwargs.pop('return_runtime', False)
    implementation = kwargs.pop('implementation', 2)
    if implementation == 0:
      logging.warning('`implementation=0` has been deprecated, '
                      'and now defaults to `implementation=2`.'
                      'Please update your layer call.')
    if 'enable_caching_device' in kwargs:
      cell_kwargs = {'enable_caching_device':
                     kwargs.pop('enable_caching_device')}
    else:
      cell_kwargs = {}
    cell = QRNNCell(
        units,
        kernel_size=kernel_size,
        strides=strides,
        pool_mode=pool_mode,
        activation=activation,
        recurrent_activation=recurrent_activation,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
        kernel_regularizer=kernel_regularizer,
        bias_regularizer=bias_regularizer,
        kernel_constraint=kernel_constraint,
        bias_constraint=bias_constraint,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
        implementation=implementation,
        reset_after=reset_after,
        dtype=kwargs.get('dtype'),
        trainable=kwargs.get('trainable', True),
        **cell_kwargs)
    super(QRNN, self).__init__(
        cell,
        return_sequences=return_sequences,
        return_state=return_state,
        go_backwards=go_backwards,
        stateful=stateful,
        unroll=unroll,
        time_major=time_major,
        **kwargs)

    if(self.pool_mode == "f"):
      self.channel_length = 2
    elif(self.pool_mode == "fo"):
      self.channel_length = 3
    elif(self.pool_mode == "ifo"):
      self.channel_length = 4
    else:
      raise AttributeError('"pool_mode" must be selected one from "f", "fo" and "ifo".')

    self.activity_regularizer = regularizers.get(activity_regularizer)
    self.input_spec = [InputSpec(ndim=3)]

  @tf_utils.shape_type_conversion
  def build(self, input_shape):
    self.input_dim = input_shape[-1]
    default_caching_device = rnn_utils.caching_device(self)

    kernel_shape = (self.kernel_size, self.input_dim, self.units * self.channel_length)
    self.kernel = self.add_weight(name='kernel',
                                  shape=kernel_shape,
                                  initializer=self.kernel_initializer,
                                  regularizer=self.kernel_regularizer,
                                  constraint=self.kernel_constraint)
    if self.use_bias:
      bias_shape = (self.channel_length * self.units,)
      self.bias = self.add_weight(name='bias',
                                  shape=bias_shape,
                                  initializer=self.bias_initializer,
                                  regularizer=self.bias_regularizer,
                                  constraint=self.bias_constraint)

    self.built = True
    pass
    
  def conv_module(self, inputs, training=None):
    dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=self.channel_length)

    if self.implementation == 1:
      if 0.0 < self.dropout < 1.0:
        input_list = [inputs * dp_mask[i] for i in range(self.channel_length)]
      else:
        input_list = [inputs for i in range(self.channel_length)]

      conv_list = []
      for i in range(self.channel_length):
        kernel_now = self.kernel[:, :, self.units * i: self.units * (i + 1)]
        conv_now = backend.conv1d(input_list[i], kernel_now, strides=self.strides, padding='causal', data_format='channels_last')
        if self.use_bias:
          conv_now = backend.bias_add(conv_now, self.bias[self.units * i: self.units * (i + 1)], data_format='channels_last')
        conv_list.append(conv_now)
      conv_out = backend.concatenate(conv_list, axis=-1)
    else:
      if 0.0 < self.dropout < 1.0:
        inputs = inputs * dp_mask[0]
      else:
        inputs = inputs

      conv_out = backend.conv1d(inputs, self.kernel, strides=self.strides, padding='causal', data_format='channels_last')
      if self.use_bias:
        conv_out = backend.bias_add(conv_out, self.bias, data_format='channels_last')
    return conv_out

  def call(self, inputs, mask=None, training=None, initial_state=None):
    conv_out = self.conv_module(inputs=inputs, training=training)
    return super(QRNN, self).call(
        inputs=conv_out, mask=mask, training=training, initial_state=initial_state)

  @property
  def units(self):
    return self.cell.units
  
  @property
  def kernel_size(self):
    return self.cell.kernel_size
  
  @property
  def strides(self):
    return self.cell.strides
  
  @property
  def pool_mode(self):
    return self.cell.pool_mode

  @property
  def activation(self):
    return self.cell.activation

  @property
  def recurrent_activation(self):
    return self.cell.recurrent_activation

  @property
  def use_bias(self):
    return self.cell.use_bias

  @property
  def kernel_initializer(self):
    return self.cell.kernel_initializer

  @property
  def bias_initializer(self):
    return self.cell.bias_initializer

  @property
  def kernel_regularizer(self):
    return self.cell.kernel_regularizer

  @property
  def bias_regularizer(self):
    return self.cell.bias_regularizer

  @property
  def kernel_constraint(self):
    return self.cell.kernel_constraint

  @property
  def bias_constraint(self):
    return self.cell.bias_constraint

  @property
  def dropout(self):
    return self.cell.dropout

  @property
  def recurrent_dropout(self):
    return self.cell.recurrent_dropout

  @property
  def implementation(self):
    return self.cell.implementation

  @property
  def reset_after(self):
    return self.cell.reset_after

  def get_config(self):
    config = {
        'units':
            self.units,
        'kernel_size':
            self.kernel_size,
        'strides':
            self.strides,
        'pool_mode':
            self.pool_mode,
        'activation':
            activations.serialize(self.activation),
        'recurrent_activation':
            activations.serialize(self.recurrent_activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint),
        'dropout':
            self.dropout,
        'recurrent_dropout':
            self.recurrent_dropout,
        'implementation':
            self.implementation,
        'reset_after':
            self.reset_after
    }
    config.update(rnn_utils.config_for_enable_caching_device(self.cell))
    base_config = super(QRNN, self).get_config()
    del base_config['cell']
    return dict(list(base_config.items()) + list(config.items()))

  @classmethod
  def from_config(cls, config):
    if 'implementation' in config and config['implementation'] == 0:
      config['implementation'] = 1
    return cls(**config)

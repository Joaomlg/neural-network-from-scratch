import numpy as np

class Layer:
  def __init__(self):
    self.prev_layer = None
    self.next_layer = None
    self.output = None
    self.gradient = None

  def set_previus_layer(self, layer: 'Layer'):
    if not isinstance(layer, Layer):
      raise TypeError(f'Layer should be "{type(Layer)}" type, not "{type(layer)}"')
    self.prev_layer = layer

  def set_next_layer(self, layer: 'Layer'):
    if not isinstance(layer, Layer):
      raise TypeError(f'Layer should be "{type(Layer)}" type, not "{type(layer)}"')
    self.next_layer = layer
  
  def setup(self):
    raise NotImplementedError
  
  @property
  def input_shape(self):
    if self.prev_layer is None:
      raise Exception('Previous layer not assigned.')
    else:
      return self.prev_layer.output_shape
  
  @property
  def output_shape(self):
    raise NotImplementedError
  
  @property
  def input_data(self):
    if self.prev_layer is None:
      raise Exception('Previous layer not assigned.')
    else:
      return self.prev_layer.output

  def foward(self):
    raise NotImplementedError
  
  def backward(self):
    raise NotImplementedError

class InputLayer(Layer):
  def __init__(self, shape: (int, tuple)):
    super().__init__()
    self.shape = shape
  
  def setup(self):
    pass

  @property
  def input_shape(self):
    if isinstance(self.shape, int):
      return (1, self.shape)
    else:
      return self.shape
  
  @property
  def output_shape(self):
    return self.input_shape

  def foward(self, input_data):
    self.output = input_data

class DenseLayer(Layer):
  def __init__(self, num_of_neurons, activation, drop_probability=0):
    super().__init__()
    self.num_of_neurons = num_of_neurons
    self.activation = activation
    self.drop_probability = drop_probability
    self.weights = None
    self.bias = None
    self.pre_activation = None
  
  @property
  def output_shape(self):
    return (1, self.num_of_neurons)
  
  def setup(self):
    self.initialize_weights()
    self.initialize_bias()

  def initialize_weights(self):
    input_rows, input_cols = self.input_shape
    weights_shape = (input_cols, self.num_of_neurons)
    self.weights = np.random.uniform(-0.5, 0.5, weights_shape)

  def initialize_bias(self):
    bias_size = self.num_of_neurons
    self.bias = np.random.uniform(-0.5, 0.5, bias_size)

  def foward(self):
    self.pre_activation = (self.input_data @ self.weights) + self.bias
    self.output = self.activation(self.pre_activation)
    if self.drop_probability:
      keep_prob = 1 - self.drop_probability
      self.drop_mask = np.random.binomial(1, keep_prob, size=self.output.shape)
      self.scale = 1 / keep_prob if keep_prob > 0 else 0
      self.output *= self.drop_mask * self.scale
  
  def backward(self):
    self.gradient = (self.next_layer.gradient @ self.next_layer.weights.T) * self.activation(self.pre_activation, derivative=True)
  
  def update_weights(self, learning_rate):
    self.weights -= learning_rate * (self.input_data.T @ self.gradient)
    self.bias -= learning_rate * self.gradient.mean(axis=0)

class OutputLayer(DenseLayer):
  def __init__(self, num_of_neurons, activation, cost):
    self.cost = cost
    self.loss = None
    super().__init__(num_of_neurons, activation)

  def backward(self, target):
    self.gradient = self.cost(self.output, target, derivative=True) * self.activation(self.pre_activation, derivative=True)
  
  def calculate_loss(self, target):
    return self.cost(self.output, target)
  
  def calculate_accuracy(self, target):
    predicted = self.output
    hits = np.sum(np.argmax(predicted, axis=1) == np.argmax(target, axis=1))
    accuracy = hits / len(target)
    return accuracy

class ConvolutionalLayer(Layer):
  def __init__(self, num_of_kernels: int, kernel_size: tuple, num_of_channels=1, stride=(1, 1)):
    super().__init__()
    self.num_of_kernels = num_of_kernels
    self.kernel_size = self.kernel_height, self.kernel_width = kernel_size
    self.num_of_channels = num_of_channels
    self.stride = self.vertical_stride, self.horizontal_stride = stride
    self.kernels = None
    self.bias = None
    self.kernel_gradient = None
    self.bias_gradient = None
  
  def setup(self):
    self.initialize_kernels()
    self.initialize_bias()
  
  def initialize_kernels(self):
    kernels_shape = np.array([self.num_of_kernels, self.num_of_channels, self.kernel_height, self.kernel_width])
    std_dev = 1 / np.sqrt(np.prod(kernels_shape))
    self.kernels = np.random.normal(loc=0, scale=std_dev, size=kernels_shape)

  def initialize_bias(self):
    bias_size = self.num_of_kernels
    self.bias = np.zeros(bias_size)
  
  @property
  def output_height(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_height - self.kernel_height) // self.vertical_stride

  @property
  def output_width(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_width - self.kernel_width) // self.horizontal_stride
  
  @property
  def output_shape(self):
    return (self.num_of_kernels, self.output_height, self.output_width)

  def foward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    if input_channels != self.num_of_channels:
      raise Exception(f'Input and kernel channels should be match: {input_channels} != {self.num_of_channels}')
    
    self.output = np.zeros((input_samples, self.num_of_kernels, self.output_height, self.output_width))

    for sample in range(input_samples):
      for kernel in range(self.num_of_kernels):
        vertical_steps = input_height - self.kernel_height + 1
        for j in range(0, vertical_steps, self.vertical_stride):
          horizontal_steps = input_width - self.kernel_width + 1
          for i in range(0, horizontal_steps, self.horizontal_stride):
            input_slice = self.input_data[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            convolution_result = np.sum(self.kernels[kernel] * input_slice) + self.bias[kernel]
            output_x_index, output_y_index = i // self.horizontal_stride, j // self.vertical_stride
            self.output[sample, kernel, output_y_index, output_x_index] = convolution_result

  def backward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    self.gradient = np.zeros_like(self.input_data)
    self.kernel_gradient = np.zeros_like(self.kernels)
    self.bias_gradient = np.zeros_like(self.bias)

    for sample in range(input_samples):
      for kernel in range(self.num_of_kernels):
        self.bias_gradient[kernel] += np.sum(self.next_layer.gradient[kernel])
        vertical_steps = input_height - self.kernel_height + 1
        for j in range(0, vertical_steps, self.vertical_stride):
          horizontal_steps = input_width - self.kernel_width + 1
          for i in range(0, horizontal_steps, self.horizontal_stride):
            input_slice = self.input_data[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            self.kernel_gradient[kernel, :, j, i] += np.sum(input_slice * self.next_layer.gradient[kernel])
            gradient_slice = self.gradient[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            gradient_slice += self.next_layer.gradient[kernel, j, i] * self.kernels[kernel]

class MaxPoolingLayer(Layer):
  def __init__(self, pool_size: tuple, stride=(1, 1)):
    super().__init__()
    self.pool_size = self.pool_height, self.pool_width = pool_size
    self.stride = self.vertical_stride, self.horizontal_stride = stride

  def setup(self):
    pass

  @property
  def output_height(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_height - self.pool_height) // self.vertical_stride
  
  @property
  def output_width(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_width - self.pool_width) // self.horizontal_stride

  @property
  def output_shape(self):
    input_channels, input_height, input_width = self.input_shape
    return (input_channels, self.output_height, self.output_width)

  def foward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    self.output = np.zeros((input_samples, input_channels, self.output_height, self.output_width))

    for sample in range(input_samples):
      for channel in range(input_channels):
        for j in range(self.output_height):
          for i in range(self.output_width):
            x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
            x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
            input_slice = self.input_data[sample, channel, y_min:y_max, x_min:x_max]
            pooling_result = np.max(input_slice)
            self.output[sample, channel, j, i] = pooling_result

  def backward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    self.gradient = np.zeros_like(self.input_data)

    for sample in range(input_samples):
      for channel in range(input_channels):
        for j in range(self.output_height):
          for i in range(self.output_width):
            x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
            x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
            input_slice = self.input_data[sample, channel, y_min:y_max, x_min:x_max]
            mask = (input_slice == self.output[sample, channel, j, i])
            gradient_slice = self.gradient[sample, channel, y_min:y_max, x_min:x_max]
            gradient_slice += mask * self.next_layer.gradient[sample, channel, j, i]

class FlattenLayer(Layer):
  def __init__(self):
    super().__init__()

  @property
  def input_width(self):
    return np.prod(self.input_shape)

  @property
  def output_shape(self):
    return (1, self.input_width)
  
  def setup(self):
    pass

  def foward(self):
    self.output = self.input_data.reshape((-1, self.input_width))
  
  def backward(self):
    self.gradient = self.next_layer.gradient.reshape((-1, *self.input_shape))
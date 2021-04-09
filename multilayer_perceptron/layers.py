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
  def __init__(self, size, activation, cost):
    self.cost = cost
    self.loss = None
    super().__init__(size, activation)

  def backward(self, target):
    self.gradient = self.cost(self.output, target, derivative=True) * self.activation(self.pre_activation, derivative=True)
  
  def calculate_loss(self, target):
    return self.cost(self.output, target)
  
  def calculate_accuracy(self, target):
    predicted = self.output
    hits = np.sum(np.argmax(predicted, axis=1) == np.argmax(target, axis=1))
    accuracy = hits / len(target)
    return accuracy

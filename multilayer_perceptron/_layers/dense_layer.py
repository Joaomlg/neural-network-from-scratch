import numpy as np

from multilayer_perceptron.layers import Layer

class DenseLayer(Layer):
  def __init__(self, num_of_neurons, activation, drop_probability=0):
    super().__init__()
    self.num_of_neurons = num_of_neurons
    self.activation = activation
    self.drop_probability = drop_probability
    self.weights = None
    self.bias = None
    self.weight_gradient = None
    self.bias_gradient = None
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
    pre_activation_gradient = self.next_layer.gradient * self.activation(self.pre_activation, derivative=True)
    self.weight_gradient = self.input_data.T @ pre_activation_gradient
    self.bias_gradient = pre_activation_gradient.sum(axis=0)
    self.gradient = pre_activation_gradient @ self.weights.T
  
  def update_weights(self, learning_rate):
    self.weights -= learning_rate * self.weight_gradient
    self.bias -= learning_rate * self.bias_gradient
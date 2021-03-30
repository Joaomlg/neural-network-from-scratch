import numpy as np

class Layer:
  def __init__(self):
    self.output = None

  def foward(self):
    raise NotImplementedError
  
  def backward(self):
    raise NotImplementedError

class InputLayer(Layer):
  def __init__(self, size):
    self.size = size
    super().__init__()
  
  def foward(self, input_data):
    self.output = input_data

class DenseLayer(Layer):
  def __init__(self, size, activation, dropout=None):
    self.size = size
    self.activation = activation
    self.gradient = None
    self.weights = None
    self.bias = None
    self.dropout = dropout
    self.pre_activation = None
    super().__init__()

  def foward(self, prev_layer):
    self.pre_activation = (prev_layer.output @ self.weights) + self.bias
    self.output = self.activation(self.pre_activation)
    if self.dropout:
      self.compute_dropout()
  
  def compute_dropout(self):
    keep_prob = 1 - self.dropout
    mask = np.random.binomial(1, keep_prob, size=self.output.shape)
    scale = 1 / keep_prob if keep_prob > 0 else 0
    self.output *= (mask * scale)
  
  def backward(self, next_layer):
    self.gradient = (next_layer.gradient @ next_layer.weights.T) * self.activation(self.pre_activation, derivative=True)
  
  def update_weights(self, prev_layer, learning_rate):
    self.weights -= learning_rate * (prev_layer.output.T @ self.gradient)
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
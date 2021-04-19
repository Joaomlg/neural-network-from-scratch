import numpy as np

from multilayer_perceptron.layers.dense_layer import DenseLayer

class OutputLayer(DenseLayer):
  def __init__(self, num_of_neurons, activation, cost):
    self.cost = cost
    self.loss = None
    super().__init__(num_of_neurons, activation)

  def backward(self, target):
    pre_activation_gradient = self.cost(self.output, target, derivative=True) * self.activation(self.pre_activation, derivative=True)
    self.weight_gradient = self.input_data.T @ pre_activation_gradient
    self.bias_gradient = pre_activation_gradient.sum(axis=0)
    self.gradient = pre_activation_gradient @ self.weights.T

  def calculate_loss(self, target):
    return self.cost(self.output, target)
  
  def calculate_accuracy(self, target):
    predicted = self.output
    hits = np.sum(np.argmax(predicted, axis=1) == np.argmax(target, axis=1))
    accuracy = hits / len(target)
    return accuracy
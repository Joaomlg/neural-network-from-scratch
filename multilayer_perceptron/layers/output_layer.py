import numpy as np

from multilayer_perceptron.layers.dense_layer import DenseLayer

class OutputLayer(DenseLayer):
  def __init__(self, num_of_neurons, activation, cost):
    self.cost = cost
    self.loss = None
    super().__init__(num_of_neurons, activation)

  def backward(self, target):
    self.output_gradient = self.cost(self.output, target, derivative=True) * self.activation(self.pre_activation, derivative=True)
    self.gradient = self.input_data.T @ self.output_gradient
    self.bias_gradient = self.output_gradient.mean(axis=0)

  def calculate_loss(self, target):
    return self.cost(self.output, target)
  
  def calculate_accuracy(self, target):
    predicted = self.output
    hits = np.sum(np.argmax(predicted, axis=1) == np.argmax(target, axis=1))
    accuracy = hits / len(target)
    return accuracy
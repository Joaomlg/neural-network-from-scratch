from abc import ABC, abstractmethod
from typing import List

from multilayer_perceptron.layers import AbstractLayer


class AbstractOptimizer(ABC):
  @abstractmethod
  def optimize(self, layers: List[AbstractLayer]):
    raise NotImplementedError


class GradientDescentOptmizer(AbstractOptimizer):
  def __init__(self, learning_rate: float):
    self.learning_rate = learning_rate
  
  def optimize(self, layers: List[AbstractLayer]):
    for layer in layers:
      if layer.has_weights:
        layer.update_weights(layer.weights - self.learning_rate * layer.weights_gradient)
      
      if layer.has_bias:
        layer.update_bias(layer.bias - self.learning_rate * layer.bias_gradient)

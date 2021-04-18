import numpy as np

from multilayer_perceptron.layers import Layer

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
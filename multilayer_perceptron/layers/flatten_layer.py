import numpy as np

from multilayer_perceptron.layers import Layer

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
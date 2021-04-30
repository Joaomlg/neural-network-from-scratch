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
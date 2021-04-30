import numpy as np

from multilayer_perceptron.layers import Layer

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

    for j in range(self.output_height):
      for i in range(self.output_width):
        x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
        x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
        input_slice = self.input_data[:, :, y_min:y_max, x_min:x_max]
        self.output[:, :, j, i] = np.max(input_slice, axis=(2, 3))

  def backward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    self.gradient = np.zeros_like(self.input_data)

    for j in range(self.output_height):
      for i in range(self.output_width):
        x_min, y_min = i * self.horizontal_stride, j * self.vertical_stride
        x_max, y_max = x_min + self.pool_width, y_min + self.pool_height
        input_slice = self.input_data[:, :, y_min:y_max, x_min:x_max]
        mask = (input_slice.reshape((input_samples, input_channels, self.pool_height*self.pool_width)) == self.output[:, :, j, i, np.newaxis])
        gradient_slice = self.gradient[:, :, y_min:y_max, x_min:x_max]
        gradient_slice += (mask * self.next_layer.gradient[:, :, j, i, np.newaxis]).reshape(gradient_slice.shape)

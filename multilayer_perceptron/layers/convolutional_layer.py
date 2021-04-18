import numpy as np

from multilayer_perceptron.layers import Layer

class ConvolutionalLayer(Layer):
  def __init__(self, num_of_kernels: int, kernel_size: tuple, num_of_channels=1, stride=(1, 1)):
    super().__init__()
    self.num_of_kernels = num_of_kernels
    self.kernel_size = self.kernel_height, self.kernel_width = kernel_size
    self.num_of_channels = num_of_channels
    self.stride = self.vertical_stride, self.horizontal_stride = stride
    self.kernels = None
    self.bias = None
    self.kernel_gradient = None
    self.bias_gradient = None
  
  def setup(self):
    self.initialize_kernels()
    self.initialize_bias()
  
  def initialize_kernels(self):
    kernels_shape = np.array([self.num_of_kernels, self.num_of_channels, self.kernel_height, self.kernel_width])
    std_dev = 1 / np.sqrt(np.prod(kernels_shape))
    self.kernels = np.random.normal(loc=0, scale=std_dev, size=kernels_shape)

  def initialize_bias(self):
    bias_size = self.num_of_kernels
    self.bias = np.zeros(bias_size)
  
  def update_weights(self, learning_rate):
    self.kernels -= learning_rate * self.kernel_gradient
    self.bias -= learning_rate * self.bias_gradient
  
  @property
  def output_height(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_height - self.kernel_height) // self.vertical_stride

  @property
  def output_width(self):
    input_channels, input_height, input_width = self.input_shape
    return 1 + (input_width - self.kernel_width) // self.horizontal_stride
  
  @property
  def output_shape(self):
    return (self.num_of_kernels, self.output_height, self.output_width)

  def foward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    if input_channels != self.num_of_channels:
      raise Exception(f'Input and kernel channels should be match: {input_channels} != {self.num_of_channels}')
    
    self.output = np.zeros((input_samples, self.num_of_kernels, self.output_height, self.output_width))

    for sample in range(input_samples):
      for kernel in range(self.num_of_kernels):
        vertical_steps = input_height - self.kernel_height + 1
        for j in range(0, vertical_steps, self.vertical_stride):
          horizontal_steps = input_width - self.kernel_width + 1
          for i in range(0, horizontal_steps, self.horizontal_stride):
            input_slice = self.input_data[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            convolution_result = np.sum(self.kernels[kernel] * input_slice) + self.bias[kernel]
            output_x_index, output_y_index = i // self.horizontal_stride, j // self.vertical_stride
            self.output[sample, kernel, output_y_index, output_x_index] = convolution_result

  def backward(self):
    input_samples, input_channels, input_height, input_width = self.input_data.shape

    self.gradient = np.zeros_like(self.input_data)
    self.kernel_gradient = np.zeros_like(self.kernels)
    self.bias_gradient = np.zeros_like(self.bias)

    for sample in range(input_samples):
      for kernel in range(self.num_of_kernels):
        self.bias_gradient[kernel] += np.sum(self.next_layer.gradient[sample, kernel])
        vertical_steps = input_height - self.kernel_height + 1
        for j in range(0, vertical_steps, self.vertical_stride):
          horizontal_steps = input_width - self.kernel_width + 1
          for i in range(0, horizontal_steps, self.horizontal_stride):
            input_slice = self.input_data[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            output_x_index, output_y_index = i // self.horizontal_stride, j // self.vertical_stride
            next_layer_gradient_slice = self.next_layer.gradient[sample, kernel, output_y_index, output_x_index]
            self.kernel_gradient[kernel, :] += input_slice * next_layer_gradient_slice
            gradient_slice = self.gradient[sample, :, j:j+self.kernel_height, i:i+self.kernel_width]
            gradient_slice += next_layer_gradient_slice * self.kernels[kernel]

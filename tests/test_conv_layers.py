import unittest

import numpy as np

from multilayer_perceptron.layers import *
from multilayer_perceptron.activations import *

class LayersTestCase(unittest.TestCase):
  def test_convolutional_layer_output_shape(self):
    input_layer = InputLayer((3, 5, 5))
    conv_layer = ConvolutionalLayer(
      num_of_kernels=12,
      kernel_size=(2, 2),
      num_of_channels=3,
      stride=(1, 1)
    )

    conv_layer.set_previus_layer(input_layer)

    expected_output_shape = (12, 4, 4)
    np.testing.assert_almost_equal(conv_layer.output_shape, expected_output_shape)
  
  def test_convolutional_layer_foward(self):
    input_layer = InputLayer((2, 3, 3))
    conv_layer = ConvolutionalLayer(
      num_of_kernels=2,
      kernel_size=(2, 2),
      num_of_channels=2,
      stride=(1, 1)
    )

    conv_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0.4, 0.3, 0.2],
                             [0.5, 0.7, 0.6],
                             [0.2, 0.4, 0.6]],

                            [[0.1, 0.4, 0.2],
                             [0.9, 0.8, 0.7],
                             [0.3, 0.3, 0.4]]]])
    
    input_layer.output = input_data

    conv_layer.kernels = np.ones((2, 2, 2, 2))
    conv_layer.bias = np.zeros(2)

    conv_layer.foward()

    expected_output = np.array([[[[4.1, 3.9],
                                  [4.1, 4.5]],

                                 [[4.1, 3.9],
                                  [4.1, 4.5]]]])

    np.testing.assert_almost_equal(conv_layer.output, expected_output)

  def test_convolutional_layer_backward(self):
    input_layer = InputLayer((2, 3, 3))

    conv_layer = ConvolutionalLayer(
      num_of_kernels=1,
      kernel_size=(2, 2),
      num_of_channels=2,
      stride=(1, 1)
    )

    output_layer = Layer()

    conv_layer.set_previus_layer(input_layer)
    conv_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0.4, 0.3, 0.2],
                             [0.5, 0.7, 0.6],
                             [0.2, 0.4, 0.6]],

                            [[0.1, 0.4, 0.2],
                             [0.9, 0.8, 0.7],
                             [0.3, 0.3, 0.4]]]])
    
    input_layer.output = input_data

    conv_layer.kernels = np.array([[[[1., 2.],
                                     [3., 4.]],

                                    [[5., 6.],
                                     [7., 8.]]]])

    conv_layer.bias = np.zeros(1)

    output_layer.gradient = np.array([[[0.1, 0.2],
                                       [0.3, 0.4]]])

    conv_layer.backward()

    expected_input_gradient = np.array([[[[0.1, 0.4, 0.4],
                                          [0.6, 2. , 1.6],
                                          [0.9, 2.4, 1.6]],

                                         [[0.5, 1.6, 1.2],
                                          [2.2, 6. , 4. ],
                                          [2.1, 5.2, 3.2]]]])

    expected_kernel_gradient = np.array([[[[1.21, 1.12],
                                           [0.87, 1.02]],

                                          [[1.21, 1.12],
                                           [0.87, 1.02]]]])

    expected_bias_gradient = np.array([1.0])

    np.testing.assert_almost_equal(conv_layer.gradient, expected_input_gradient)

    np.testing.assert_almost_equal(conv_layer.kernel_gradient, expected_kernel_gradient)

    np.testing.assert_almost_equal(conv_layer.bias_gradient, expected_bias_gradient)

if __name__ == '__main__':
  unittest.main()
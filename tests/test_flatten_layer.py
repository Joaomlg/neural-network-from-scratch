import unittest

import numpy as np

from multilayer_perceptron.layers import Layer
from multilayer_perceptron.layers.input_layer import InputLayer
from multilayer_perceptron.layers.flatten_layer import FlattenLayer
from multilayer_perceptron.activations import *

class FlattenLayerTestCase(unittest.TestCase):
  def test_flatten_layer_output_shape(self):
    input_layer = InputLayer((3, 2, 2))
    flatten_layer = FlattenLayer()

    flatten_layer.set_previus_layer(input_layer)

    expected_output_shape = (1, 12)
    np.testing.assert_almost_equal(flatten_layer.output_shape, expected_output_shape)
  
  def test_flatten_layer_foward(self):
    input_layer = InputLayer((3, 2, 2))
    flatten_layer = FlattenLayer()

    flatten_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0. , 0.1],
                             [0.2, 0.3]],

                            [[0.4, 0.5],
                             [0.6, 0.7]],

                            [[0.8, 0.9],
                             [1. , 1.1]]]])
    
    input_layer.output = input_data

    flatten_layer.foward()

    expected_output = np.array([[0. , 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. , 1.1]])

    np.testing.assert_almost_equal(flatten_layer.output, expected_output)

  def test_flatten_layer_backward(self):
    input_layer = InputLayer((3, 2, 2))

    flatten_layer = FlattenLayer()

    output_layer = Layer()

    flatten_layer.set_previus_layer(input_layer)
    flatten_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0. , 0.1],
                             [0.2, 0.3]],

                            [[0.4, 0.5],
                             [0.6, 0.7]],

                            [[0.8, 0.9],
                             [1. , 1.1]]]])
    
    input_layer.output = input_data

    output_layer.gradient = np.array([[0., 0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88, 0.99, 1.1, 1.21]])

    flatten_layer.backward()

    expected_input_gradient = np.array([[[[0.  , 0.11],
                                          [0.22, 0.33]],

                                         [[0.44, 0.55],
                                          [0.66, 0.77]],

                                         [[0.88, 0.99],
                                          [1.1 , 1.21]]]])

    np.testing.assert_almost_equal(flatten_layer.gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
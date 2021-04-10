import unittest

import numpy as np

from multilayer_perceptron.layers import *
from multilayer_perceptron.activations import *

class FlattenLayerTestCase(unittest.TestCase):
  def test_pooling_layer_output_shape(self):
    input_layer = InputLayer((3, 2, 2))
    flatten_layer = FlattenLayer()

    flatten_layer.set_previus_layer(input_layer)

    expected_output_shape = (1, 12)
    np.testing.assert_almost_equal(flatten_layer.output_shape, expected_output_shape)
  
  def test_pooling_layer_foward(self):
    input_layer = InputLayer((2, 3, 3))
    flatten_layer = FlattenLayer()

    flatten_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0.1, 0.2, 0.3],
                             [0.3, 0.2, 0.4],
                             [0.4, 0.3, 0.2]],

                            [[0.5, 0.3, 0.2],
                             [0.9, 0.5, 0.3],
                             [0.4, 0.3, 0.2]]]])
    
    input_layer.output = input_data

    flatten_layer.foward()

    expected_output = np.array([[0.1, 0.2, 0.3, 0.3, 0.2, 0.4, 0.4, 0.3, 0.2, 0.5, 0.3, 0.2, 0.9,
       0.5, 0.3, 0.4, 0.3, 0.2]])

    np.testing.assert_almost_equal(flatten_layer.output, expected_output)

  def test_pooling_layer_backward(self):
    input_layer = InputLayer((2, 3, 3))

    flatten_layer = FlattenLayer()

    output_layer = Layer()

    flatten_layer.set_previus_layer(input_layer)
    flatten_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0.1, 0.2, 0.3],
                             [0.3, 0.2, 0.4],
                             [0.4, 0.3, 0.2]],

                            [[0.5, 0.3, 0.2],
                             [0.9, 0.5, 0.3],
                             [0.4, 0.3, 0.2]]]])
    
    input_layer.output = input_data

    output_layer.gradient = np.array([[0.2 , 0.4 , 0.5 , 0.23, 0.65, 0.37, 0.12, 0.21, 0.85, 0.37, 0.99,
       0.1 , 0.54, 0.23, 0.98, 0.5 , 0.23, 0.7 ]])

    flatten_layer.backward()

    expected_input_gradient = np.array([[[[0.2 , 0.4 , 0.5 ],
                                          [0.23, 0.65, 0.37],
                                          [0.12, 0.21, 0.85]],

                                         [[0.37, 0.99, 0.1 ],
                                          [0.54, 0.23, 0.98],
                                          [0.5 , 0.23, 0.7 ]]]])

    np.testing.assert_almost_equal(flatten_layer.gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
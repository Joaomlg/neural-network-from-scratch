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
    input_layer = InputLayer((1, 2, 2))
    flatten_layer = FlattenLayer()

    flatten_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0.89738036, 0.98673454],
                             [0.55129726, 0.8030491 ]]],


                           [[[0.62816706, 0.05484155],
                             [0.18796328, 0.98746524]]],


                           [[[0.63243453, 0.88908661],
                             [0.48268783, 0.88405604]]]])
    
    input_layer.output = input_data

    flatten_layer.foward()

    expected_output = np.array([[0.89738036, 0.98673454, 0.55129726, 0.8030491 ],
                                [0.62816706, 0.05484155, 0.18796328, 0.98746524],
                                [0.63243453, 0.88908661, 0.48268783, 0.88405604]])

    np.testing.assert_almost_equal(flatten_layer.output, expected_output)

  def test_flatten_layer_backward(self):
    input_layer = InputLayer((1, 2, 2))

    flatten_layer = FlattenLayer()

    output_layer = Layer()

    flatten_layer.set_previus_layer(input_layer)
    flatten_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0.89738036, 0.98673454],
                             [0.55129726, 0.8030491 ]]],


                           [[[0.62816706, 0.05484155],
                             [0.18796328, 0.98746524]]],


                           [[[0.63243453, 0.88908661],
                             [0.48268783, 0.88405604]]]])
    
    input_layer.output = input_data

    output_layer.gradient = np.array([[0.06494434, 0.1065572 , 0.61667899, 0.69901242],
                                      [0.57754459, 0.39518983, 0.52322838, 0.34086176],
                                      [0.6649398 , 0.4820081 , 0.14890696, 0.66733853]])

    flatten_layer.backward()

    expected_input_gradient = np.array([[[[0.06494434, 0.1065572 ],
                                          [0.61667899, 0.69901242]]],


                                        [[[0.57754459, 0.39518983],
                                          [0.52322838, 0.34086176]]],


                                        [[[0.6649398 , 0.4820081 ],
                                          [0.14890696, 0.66733853]]]])

    np.testing.assert_almost_equal(flatten_layer.gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
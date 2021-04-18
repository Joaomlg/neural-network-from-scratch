import unittest

import numpy as np

from multilayer_perceptron.layers import Layer
from multilayer_perceptron.layers.input_layer import InputLayer
from multilayer_perceptron.layers.dense_layer import DenseLayer
from multilayer_perceptron.activations import *

class DenseLayerTestCase(unittest.TestCase):
  def test_dense_layer_output_shape(self):
    input_layer = InputLayer((1, 3))

    dense_layer = DenseLayer(num_of_neurons=4, activation=tanh)
    dense_layer.set_previus_layer(input_layer)

    expected_output_shape = (1, 4)

    np.testing.assert_almost_equal(dense_layer.output_shape, expected_output_shape)
  
  def test_dense_layer_foward(self):
    input_layer = InputLayer((1, 3))
    input_data = np.array([[0.1, 0.2, 0.3]])
    input_layer.output = input_data

    dense_layer = DenseLayer(num_of_neurons=4, activation=tanh)
    dense_layer.set_previus_layer(input_layer)
    dense_layer.weights = np.array([[0.1, 0.3, 0.5, 0.7],
                                    [0.9, 1.1, 1.3, 1.5],
                                    [1.7, 1.9, 2.1, 2.3]])
    dense_layer.bias = np.array([0.1, 1.2, 2.3, 3.4])

    dense_layer.foward()

    expected_pre_activation = np.array([[0.8, 2.02, 3.24, 4.46]])
    expected_pos_activation = np.array([[0.66403677, 0.96541369, 0.99693708, 0.99973266]])

    np.testing.assert_almost_equal(dense_layer.pre_activation, expected_pre_activation)
    np.testing.assert_almost_equal(dense_layer.output, expected_pos_activation)

  def test_dense_layer_backward(self):
    input_layer = InputLayer((1, 3))
    input_data = np.array([[0.1, 0.2, 0.3]])
    input_layer.output = input_data

    output_layer = Layer()
    output_layer.gradient = np.array([0.1, 0.6, 1.1, 1.6])

    dense_layer = DenseLayer(num_of_neurons=4, activation=tanh)
    dense_layer.set_previus_layer(input_layer)
    dense_layer.set_next_layer(output_layer)
    dense_layer.weights = np.array([[0.1, 0.3, 0.5, 0.7],
                                    [0.9, 1.1, 1.3, 1.5],
                                    [1.7, 1.9, 2.1, 2.3]])
    dense_layer.bias = np.array([0.1, 1.2, 2.3, 3.4])
    dense_layer.pre_activation = np.array([[0.8, 2.02, 3.24, 4.46]])

    dense_layer.backward()

    expected_weights_gradient = np.array([[5.59055168e-03, 4.07858479e-03, 6.72811186e-04, 8.55376044e-05],
                                          [1.11811034e-02, 8.15716959e-03, 1.34562237e-03, 1.71075209e-04],
                                          [1.67716550e-02, 1.22357544e-02, 2.01843356e-03, 2.56612813e-04]])
    expected_bias_gradient = np.array([0.05590552, 0.04078585, 0.00672811, 0.00085538])
    expected_input_gradient = np.array([[0.02178913, 0.10520901, 0.18862889]])

    np.testing.assert_almost_equal(dense_layer.weight_gradient, expected_weights_gradient)
    np.testing.assert_almost_equal(dense_layer.bias_gradient, expected_bias_gradient)
    np.testing.assert_almost_equal(dense_layer.gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
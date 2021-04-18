import unittest

import numpy as np

from multilayer_perceptron import MLP
from multilayer_perceptron.layers.input_layer import InputLayer
from multilayer_perceptron.layers.dense_layer import DenseLayer
from multilayer_perceptron.layers.output_layer import OutputLayer
from multilayer_perceptron.activations import *
from multilayer_perceptron.costs import *

class MLPTestCase(unittest.TestCase):
  def test_simple_perceptron(self):
    mlp = MLP()
    
    input_layer = InputLayer(3)
    mlp.add(input_layer)

    output_layer = OutputLayer(1, identity, mean_square_cost)
    mlp.add(output_layer)

    mlp.build()

    output_layer.weights = np.array([[0.1], [0.2], [0.3]])
    output_layer.bias = np.array([0.5])

    x = np.array([[1, 2, 3]])
    y = np.array([[2.0]])

    mlp.fit(
      train_data=(x, y),
      epochs=1,
      learning_rate=0.5,
      batch_size=1
    )

    np.testing.assert_almost_equal(output_layer.output, np.array([[1.9]]))

    loss = output_layer.calculate_loss(y)
    np.testing.assert_almost_equal(loss, 0.005)

    np.testing.assert_almost_equal(output_layer.output_gradient, np.array([[-0.1]]))

    np.testing.assert_almost_equal(output_layer.weights, np.array([[0.15], [0.3], [0.45]]))
    np.testing.assert_almost_equal(output_layer.bias, np.array([0.55]))

  def test_multilayer_perceptron(self):
    mlp = MLP()
    
    input_layer = InputLayer(2)
    mlp.add(input_layer)

    hidden_layer = DenseLayer(2, tanh)
    mlp.add(hidden_layer)

    output_layer = OutputLayer(1, identity, mean_square_cost)
    mlp.add(output_layer)

    mlp.build()

    hidden_layer.weights = np.array([[0.5, 0.1], [-0.5, -0.3]])
    hidden_layer.bias = np.array([-0.2, 0.2])

    output_layer.weights = np.array([[0.5], [-0.1]])
    output_layer.bias = np.array([0.3])

    x = np.array([[1.2, 2.5]])
    y = np.array([[3.0]])

    mlp.fit(
      train_data=(x, y),
      epochs=1,
      learning_rate=0.1,
      batch_size=1
    )

    np.testing.assert_almost_equal(output_layer.output, np.array([[-0.005002604047518955]]))

    loss = output_layer.calculate_loss(y)
    np.testing.assert_almost_equal(loss, 4.5150203251661845)

    np.testing.assert_almost_equal(output_layer.output_gradient, np.array([[-3.005002604047519]]))
    np.testing.assert_almost_equal(hidden_layer.output_gradient, np.array([[-0.7849412194740529, 0.25113246595910205]]))

    np.testing.assert_almost_equal(hidden_layer.weights, np.array([[0.5941929463368864, 0.06986410408490776], [-0.30376469513148674, -0.3627831164897755]]))
    np.testing.assert_almost_equal(hidden_layer.bias, np.array([-0.12150587805259472, 0.1748867534040898]))

    np.testing.assert_almost_equal(output_layer.weights, np.array([[0.2923334443574305], [-0.22179915880877843]]))
    np.testing.assert_almost_equal(output_layer.bias, np.array([0.6005002604047519]))

if __name__ == '__main__':
  unittest.main()
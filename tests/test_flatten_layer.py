import unittest
import numpy as np

from neural_network.layers import FlattenLayer

class FlattenLayerTestCase(unittest.TestCase):
  def test_flatten_layer_forward(self):
    flatten_layer = FlattenLayer()
    flatten_layer.input_shape = (1, 2, 2)

    input_data = np.array([[[[0.89738036, 0.98673454],
                             [0.55129726, 0.8030491 ]]],


                           [[[0.62816706, 0.05484155],
                             [0.18796328, 0.98746524]]],


                           [[[0.63243453, 0.88908661],
                             [0.48268783, 0.88405604]]]])

    output = flatten_layer.forward(input_data)

    expected_output = np.array([[0.89738036, 0.98673454, 0.55129726, 0.8030491 ],
                                [0.62816706, 0.05484155, 0.18796328, 0.98746524],
                                [0.63243453, 0.88908661, 0.48268783, 0.88405604]])

    np.testing.assert_almost_equal(output, expected_output)

  def test_flatten_layer_backward(self):
    flatten_layer = FlattenLayer()
    flatten_layer.input_shape = (1, 2, 2)

    output_gradient = np.array([[0.06494434, 0.1065572 , 0.61667899, 0.69901242],
                                [0.57754459, 0.39518983, 0.52322838, 0.34086176],
                                [0.6649398 , 0.4820081 , 0.14890696, 0.66733853]])

    input_gradient = flatten_layer.backward(output_gradient)

    expected_input_gradient = np.array([[[[0.06494434, 0.1065572 ],
                                          [0.61667899, 0.69901242]]],


                                        [[[0.57754459, 0.39518983],
                                          [0.52322838, 0.34086176]]],


                                        [[[0.6649398 , 0.4820081 ],
                                          [0.14890696, 0.66733853]]]])

    np.testing.assert_almost_equal(input_gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
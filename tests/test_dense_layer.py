import unittest
import numpy as np

from neural_network.layers import DenseLayer

class DenseLayerTestCase(unittest.TestCase):  
  def test_dense_layer_forward(self):
    dense_layer = DenseLayer(units=4)
    dense_layer.weights = np.array([[-0.04653779, -0.0469091 , -0.25623124,  0.16194522, -0.07224174],
                                    [-0.01354412,  0.07100885,  0.10992199, -0.03207688,  0.02484984],
                                    [-0.01371569,  0.03104802,  0.15018086,  0.08298718, -0.04885989],
                                    [ 0.02374537, -0.02832615,  0.12901608,  0.02934764,  0.10403082]])
    dense_layer.bias = np.array([0.04222229,  0.12840084,  0.04837644, -0.01436746, -0.07240851])

    input_data = np.array([[0.03460588, 0.89022329, 0.75139759, 0.19370216],
                           [0.02684922, 0.50230407, 0.39663982, 0.39070667],
                           [0.37704716, 0.59068836, 0.72832059, 0.34944772]])

    output = dense_layer.forward(input_data)

    expected_output = np.array([[0.02284811, 0.20783381, 0.27520068, 0.03072228, -0.0693488 ],
                                        [0.0380068 , 0.16405706, 0.20668625, 0.01825064, -0.04060021],
                                        [0.01498334, 0.1653724 , 0.17115901, 0.09844282, -0.08420087]])

    np.testing.assert_almost_equal(output, expected_output)

  def test_dense_layer_backward(self):
    dense_layer = DenseLayer(units=4)
    dense_layer.weights = np.array([[-0.04653779, -0.0469091 , -0.25623124,  0.16194522, -0.07224174],
                                    [-0.01354412,  0.07100885,  0.10992199, -0.03207688,  0.02484984],
                                    [-0.01371569,  0.03104802,  0.15018086,  0.08298718, -0.04885989],
                                    [ 0.02374537, -0.02832615,  0.12901608,  0.02934764,  0.10403082]])
    dense_layer.bias = np.array([0.04222229,  0.12840084,  0.04837644, -0.01436746, -0.07240851])
    dense_layer.prev_input = np.array([[0.03460588, 0.89022329, 0.75139759, 0.19370216],
                                       [0.02684922, 0.50230407, 0.39663982, 0.39070667],
                                       [0.37704716, 0.59068836, 0.72832059, 0.34944772]])
 
    output_gradient = np.array([[0.58065229, 0.07073203, 0.99459365, 0.95053096, 0.67950329],
                                [0.61324957, 0.74274262, 0.10816378, 0.90538395, 0.86251981],
                                [0.42439558, 0.27020356, 0.08093633, 0.05554409, 0.33081295]])

    input_gradient = dense_layer.backward(output_gradient)

    expected_weights_gradient = np.array([[0.19657641, 0.12426929, 0.06783972, 0.07814556, 0.17140488],
                                          [1.07563348, 0.59565604, 0.9875497 , 1.3337721 , 1.23356423],
                                          [0.98863597, 0.544544  , 0.84918494, 1.11379191, 1.09362472],
                                          [0.50037837, 0.39831747, 0.26319826, 0.5572692 , 0.58421533]])

    expected_bias_gradient = np.array([1.61829744, 1.08367822, 1.18369377, 1.91145901, 1.87283605])

    expected_input_gradient = np.array([[-0.18034076,  0.09288137,  0.18928241,  0.23868794],
                                        [-0.00678294,  0.04871654,  0.06388632,  0.12377719],
                                        [-0.06806726,  0.02877443,  0.00316946,  0.04891052]])

    np.testing.assert_almost_equal(dense_layer.weights_gradient, expected_weights_gradient)
    np.testing.assert_almost_equal(dense_layer.bias_gradient, expected_bias_gradient)
    np.testing.assert_almost_equal(input_gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
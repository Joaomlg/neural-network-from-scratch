import unittest

import numpy as np

from multilayer_perceptron.layers import Layer
from multilayer_perceptron.layers.input_layer import InputLayer
from multilayer_perceptron.layers.convolutional_layer import ConvolutionalLayer
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
    input_layer = InputLayer((3, 3, 3))
    conv_layer = ConvolutionalLayer(
      num_of_kernels=2,
      kernel_size=(2, 2),
      num_of_channels=3,
      stride=(1, 1)
    )

    conv_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0.41382909, 0.35215405, 0.22447339],
                             [0.28362417, 0.8411642 , 0.87428741],
                             [0.32789206, 0.15391448, 0.41023217]],

                            [[0.09930137, 0.60255387, 0.01367108],
                             [0.98175134, 0.66947525, 0.59237594],
                             [0.10619066, 0.12703427, 0.51139995]],

                            [[0.7419589 , 0.77035624, 0.52306009],
                             [0.5854722 , 0.29715111, 0.13123254],
                             [0.80512239, 0.33665096, 0.60685376]]]])
    
    input_layer.output = input_data

    conv_layer.kernels = np.array([[[[0.84801409, 0.95214852],
                                     [0.50387677, 0.82924158]],

                                    [[0.24384244, 0.81228219],
                                     [0.71585909, 0.65924244]],

                                    [[0.97075681, 0.67433967],
                                     [0.49533248, 0.10578075]]],


                                   [[[0.23239533, 0.80963279],
                                     [0.71252993, 0.88484524]],

                                    [[0.98454472, 0.74437492],
                                     [0.88673769, 0.43311433]],

                                    [[0.40947502, 0.24171889],
                                     [0.53967182, 0.63845918]]]])
    conv_layer.bias = np.array([0.1877753 , 0.55639552])

    conv_layer.foward()

    expected_output = np.array([[[[4.93343058, 4.13839897],
                                  [3.66816108, 3.83168382]],

                                 [[4.58658664, 4.33259815],
                                  [4.24827132, 4.08913912]]]])

    np.testing.assert_almost_equal(conv_layer.output, expected_output)

  def test_convolutional_layer_backward(self):
    input_layer = InputLayer((3, 3, 3))

    conv_layer = ConvolutionalLayer(
      num_of_kernels=2,
      kernel_size=(2, 2),
      num_of_channels=3,
      stride=(1, 1)
    )

    output_layer = Layer()

    conv_layer.set_previus_layer(input_layer)
    conv_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0.41382909, 0.35215405, 0.22447339],
                             [0.28362417, 0.8411642 , 0.87428741],
                             [0.32789206, 0.15391448, 0.41023217]],

                            [[0.09930137, 0.60255387, 0.01367108],
                             [0.98175134, 0.66947525, 0.59237594],
                             [0.10619066, 0.12703427, 0.51139995]],

                            [[0.7419589 , 0.77035624, 0.52306009],
                             [0.5854722 , 0.29715111, 0.13123254],
                             [0.80512239, 0.33665096, 0.60685376]]]])
    
    input_layer.output = input_data

    conv_layer.kernels = np.array([[[[0.84801409, 0.95214852],
                                     [0.50387677, 0.82924158]],

                                    [[0.24384244, 0.81228219],
                                     [0.71585909, 0.65924244]],

                                    [[0.97075681, 0.67433967],
                                     [0.49533248, 0.10578075]]],


                                   [[[0.23239533, 0.80963279],
                                     [0.71252993, 0.88484524]],

                                    [[0.98454472, 0.74437492],
                                     [0.88673769, 0.43311433]],
 
                                    [[0.40947502, 0.24171889],
                                     [0.53967182, 0.63845918]]]])

    conv_layer.bias = np.array([0.1877753 , 0.55639552])

    output_layer.gradient = np.array([[[[0.79545941, 0.48125974],
                                        [0.23329675, 0.29189185]],

                                       [[0.48145531, 0.58811836],
                                        [0.75841116, 0.27388268]]]])

    conv_layer.backward()

    expected_input_gradient = np.array([[[[0.78644875, 1.6919885 , 0.93439066],
                                          [1.11795497, 2.89453414, 1.41914312],
                                          [0.65794346, 1.206763  , 0.48439265]],

                                         [[0.66798104, 1.70090114, 0.82869927],
                                          [1.79993876, 2.69381707, 1.01295929],
                                          [0.83951935, 0.93409339, 0.31105001]],

                                         [[0.96934156, 1.36079263, 0.46669185],
                                          [1.19086957, 1.68345602, 0.68943446],
                                          [0.52485259, 0.80128315, 0.20573945]]]])

    expected_kernel_gradient = np.array([[[[0.81035938, 0.8395925 ],
                                           [0.75185251, 1.24552249]],

                                          [[0.7934289 , 0.8149826 ],
                                           [1.16498903, 0.99653724]],

                                          [[1.18426438, 0.97214498],
                                           [0.89482435, 0.55520383]]],


                                         [[[0.85183251, 1.17896385],
                                           [0.92208798, 1.14825339]],

                                          [[1.33011102, 0.96812198],
                                           [0.98172875, 0.90711737]],

                                          [[1.33569389, 0.93981838],
                                           [1.1594554 , 0.64177183]]]])

    expected_bias_gradient = np.array([1.80190775, 2.1018675 ])

    np.testing.assert_almost_equal(conv_layer.gradient, expected_input_gradient)

    np.testing.assert_almost_equal(conv_layer.kernel_gradient, expected_kernel_gradient)

    np.testing.assert_almost_equal(conv_layer.bias_gradient, expected_bias_gradient)

if __name__ == '__main__':
  unittest.main()
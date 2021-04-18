import unittest

import numpy as np

from multilayer_perceptron.layers import Layer
from multilayer_perceptron.layers.input_layer import InputLayer
from multilayer_perceptron.layers.max_pooling_layer import MaxPoolingLayer
from multilayer_perceptron.activations import *

class PoolingLayerTestCase(unittest.TestCase):
  def test_pooling_layer_output_shape(self):
    input_layer = InputLayer((3, 4, 4))
    pool_layer = MaxPoolingLayer(
      pool_size=(2, 2),
      stride=(2, 2)
    )

    pool_layer.set_previus_layer(input_layer)

    expected_output_shape = (3, 2, 2)
    np.testing.assert_almost_equal(pool_layer.output_shape, expected_output_shape)
  
  def test_pooling_layer_foward(self):
    input_layer = InputLayer((2, 4, 4))
    pool_layer = MaxPoolingLayer(
      pool_size=(2, 2),
      stride=(2, 2)
    )

    pool_layer.set_previus_layer(input_layer)

    input_data = np.array([[[[0.1 , 0.2 , 0.3 , 0.5 ],
                             [0.3 , 0.2 , 0.4 , 0.1 ],
                             [0.4 , 0.3 , 0.2 , 1.  ],
                             [0.1 , 0.1 , 0.2 , 0.5 ]],

                            [[0.5 , 0.3 , 0.2 , 0.3 ],
                             [0.9 , 0.5 , 0.3 , 0.98],
                             [0.4 , 0.3 , 0.2 , 0.1 ],
                             [0.5 , 0.55, 0.43, 0.21]]]])
    
    input_layer.output = input_data

    pool_layer.foward()

    expected_output = np.array([[[[0.3 , 0.5 ],
                                  [0.4 , 1.  ]],

                                 [[0.9 , 0.98],
                                  [0.55, 0.43]]]])

    np.testing.assert_almost_equal(pool_layer.output, expected_output)

  def test_pooling_layer_backward(self):
    input_layer = InputLayer((2, 4, 4))

    pool_layer = MaxPoolingLayer(
      pool_size=(2, 2),
      stride=(2, 2)
    )

    output_layer = Layer()

    pool_layer.set_previus_layer(input_layer)
    pool_layer.set_next_layer(output_layer)

    input_data = np.array([[[[0.1 , 0.2 , 0.3 , 0.5 ],
                             [0.3 , 0.2 , 0.4 , 0.1 ],
                             [0.4 , 0.3 , 0.2 , 1.  ],
                             [0.1 , 0.1 , 0.2 , 0.5 ]],

                            [[0.5 , 0.3 , 0.2 , 0.3 ],
                             [0.9 , 0.5 , 0.3 , 0.98],
                             [0.4 , 0.3 , 0.2 , 0.1 ],
                             [0.5 , 0.55, 0.43, 0.21]]]])
    
    input_layer.output = input_data

    pool_layer.output = np.array([[[[0.3 , 0.5 ],
                                    [0.4 , 1.  ]],

                                   [[0.9 , 0.98],
                                    [0.55, 0.43]]]])

    output_layer.gradient = np.array([[[[0.1, 0.2],
                                        [0.3, 0.4]],

                                       [[0.5, 0.6],
                                        [0.7, 0.8]]]])

    pool_layer.backward()

    expected_input_gradient = np.array([[[[0.0, 0.0, 0.0, 0.2],
                                          [0.1, 0.0, 0.0, 0.0],
                                          [0.3, 0.0, 0.0, 0.4],
                                          [0.0, 0.0, 0.0, 0.0]],

                                         [[0.0, 0.0, 0.0, 0.0],
                                          [0.5, 0.0, 0.0, 0.6],
                                          [0.0, 0.0, 0.0, 0.0],
                                          [0.0, 0.7, 0.8, 0.0]]]])

    np.testing.assert_almost_equal(pool_layer.gradient, expected_input_gradient)

if __name__ == '__main__':
  unittest.main()
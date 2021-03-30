import numpy as np

from data import MNIST

from multilayer_perceptron import MLP
from multilayer_perceptron.layers import InputLayer, DenseLayer, OutputLayer
from multilayer_perceptron.activations import tanh, softmax, identity
from multilayer_perceptron.costs import mean_square_cost

def one_hot_encode(data):
  encoded = np.zeros([len(data), 10])
  for i in range(len(data)):
    encoded[i, data[i]] = 1
  return encoded

mnist_dataset = MNIST()
train_data, validation_data, test_data = mnist_dataset.load()

train_data = (
  train_data[0] / 255,
  one_hot_encode(train_data[1])
)

validation_data = (
  validation_data[0] / 255,
  one_hot_encode(validation_data[1])
)

test_data = (
  test_data[0] / 255,
  one_hot_encode(test_data[1])
)

mlp = MLP()
mlp.add(InputLayer(784))
mlp.add(DenseLayer(64, tanh))
mlp.add(OutputLayer(10, softmax, mean_square_cost))
mlp.initialize_random_weights()
mlp.fit(
  train_data=train_data,
  epochs=20,
  learning_rate=0.01,
  batch_size=100,
  validation_data=validation_data
)
import numpy as np
from time import time

from multilayer_perceptron.layers import *

class MLP:
  def __init__(self):
    self.layers = []
    self.train_loss_per_epoch = []
    self.train_accu_per_epoch = []
    self.valid_loss_per_epoch = []
    self.valid_accu_per_epoch = []

  def add(self, layer: Layer):
    if not isinstance(layer, Layer):
      raise TypeError(f'layer param should be \'{type(Layer)}\' type, not \'{type(layer)}\'')
    self.layers.append(layer)
  
  def initialize_random_weights(self):
    num_of_layers = len(self.layers)
    for i in range(1, num_of_layers):
      prev_layer = self.layers[i - 1]
      curr_layer = self.layers[i]

      weights_shape = prev_layer.size, curr_layer.size
      random_weights = np.random.uniform(-0.5, 0.5, weights_shape)

      bias_size = curr_layer.size
      random_bias = np.random.uniform(-0.5, 0.5, bias_size)

      curr_layer.weights = random_weights
      curr_layer.bias = random_bias

  def fit(self, train_data, epochs, learning_rate, batch_size, validation_data=None):
    xtrain, ytrain = train_data
    train_size = len(xtrain)
    try:
      for epoch in range(epochs):
        batchs = self.split_data_in_batchs(train_size, batch_size)
        for batch in batchs:
          x, y = xtrain[batch], ytrain[batch]

          self.feedfoward(x)
          self.backpropagation(y)
          self.update_weights(learning_rate)

        output_layer = self.layers[-1]

        train_loss = output_layer.calculate_loss(y)
        train_accuracy = output_layer.calculate_accuracy(y)
        
        self.train_loss_per_epoch.append(train_loss)
        self.train_accu_per_epoch.append(train_accuracy)

        print('\rEpoch: {epoch}/{epochs}\tTrain: {train_loss:.3f} | {train_accuracy:.3f}'.format(**locals()), end='')

        if validation_data:
          xvalid, yvalid = validation_data
          
          self.feedfoward(xvalid)

          valid_loss = output_layer.calculate_loss(yvalid)
          valid_accuracy = output_layer.calculate_accuracy(yvalid)
          
          self.valid_loss_per_epoch.append(valid_loss)
          self.valid_accu_per_epoch.append(valid_accuracy)

          print('\tValid: {valid_loss:.3f} | {valid_accuracy:.3f}'.format(**locals()), end='')
    except KeyboardInterrupt:
      print('\nStoped!')

  @staticmethod
  def split_data_in_batchs(data_size, batch_size, random=True):
    samples = np.arange(data_size)
    if random:
      np.random.shuffle(samples)
    num_of_batchs = np.floor_divide(data_size, batch_size)
    batchs = np.array_split(samples, num_of_batchs)
    return batchs

  def feedfoward(self, input_data):
    prev_layer = None
    for layer in self.layers:
      if isinstance(layer, InputLayer):
        layer.foward(input_data)
      elif isinstance(layer, (DenseLayer, OutputLayer)):
        layer.foward(prev_layer)
      else:
        raise TypeError(f'Unexpected layer type: \'{type(layer)}\'')
      prev_layer = layer
  
  def backpropagation(self, target):
    next_layer = None
    for layer in reversed(self.layers):
      if isinstance(layer, OutputLayer):
        layer.backward(target)
      elif isinstance(layer, DenseLayer):
        layer.backward(next_layer)
      elif isinstance(layer, InputLayer):
        break
      else:
        raise TypeError(f'Unexpected layer type: \'{type(layer)}\'')
      next_layer = layer
  
  def update_weights(self, learning_rate):
    prev_layer = None
    for layer in self.layers:
      if isinstance(layer, InputLayer):
        pass
      elif isinstance(layer, (DenseLayer, OutputLayer)):
        layer.update_weights(prev_layer, learning_rate)
      else:
        raise TypeError(f'Unexpected layer type: \'{type(layer)}\'')
      prev_layer = layer

  def predict(self, input_data):
    pass

  def test(self, input_data, target):
    pass
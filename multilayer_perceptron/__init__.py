from typing import List, Tuple
import numpy as np
from time import time
from datetime import timedelta

from multilayer_perceptron.layers import *
from multilayer_perceptron.optimizers import AbstractOptimizer
from multilayer_perceptron.costs import AbstractCost
from multilayer_perceptron.metrics import AbstractMetric
from multilayer_perceptron.utils import generate_batches

class MLP:
  def __init__(self, optimizer: AbstractOptimizer, cost: AbstractCost, metric: AbstractMetric):
    self.layers : List[AbstractLayer] = []

    self.optimizer = optimizer
    self.cost = cost
    self.metric = metric

    self.train_loss_per_iter = []
    self.train_accu_per_iter = []
    self.valid_loss_per_epoch = []
    self.valid_accu_per_epoch = []

  def add(self, layer: AbstractLayer):
    if not isinstance(layer, AbstractLayer):
      raise TypeError(f'layer should be Layer(AbstractLayer), not \'{type(layer).__name__}\'')

    if len(self.layers) == 0:
      if not isinstance(layer, InputLayer):
        raise TypeError(f'first layer should be InputLayer(AbstractLayer), not \'{type(layer).__name__}\'')
    else:
      layer.input_shape = self.layers[-1].output_shape

    self.layers.append(layer)
  
  def compile(self):
    for layer in self.layers:
      layer.initialize()

  def fit(self, train_data: Tuple[np.array, np.array], epochs: int, batch_size: int, validation_data: Tuple[np.array, np.array]=None):
    xtrain, ytrain = train_data
    train_size = len(xtrain)
    try:
      for epoch in range(epochs):
        t0 = time()
        for i, (x, y) in enumerate(generate_batches(train_data, batch_size)):
          predict = self.feedforward(x)

          train_loss = self.cost.loss(predict, y)
          train_accuracy = self.metric.compare(predict, y)

          output_gradient = self.cost.gradient(predict, y)
          self.backpropagation(output_gradient)
          self.update_weights()
          
          self.train_loss_per_iter.append(train_loss)
          self.train_accu_per_iter.append(train_accuracy)
        t1 = time()
        passed = timedelta(seconds=t1-t0)

        print('\nEpoch: {epoch}/{epochs}\tTrain: {train_loss:.3f} | {train_accuracy:.3f}\tTime: {passed}'.format(**locals()), end='')

        if validation_data:
          xvalid, yvalid = validation_data
          
          valid_predict = self.feedforward(xvalid)

          valid_loss = self.cost.loss(valid_predict, yvalid)
          valid_accuracy = self.metric.compare(valid_predict, yvalid)
          
          self.valid_loss_per_epoch.append(valid_loss)
          self.valid_accu_per_epoch.append(valid_accuracy)

          print('\tValid: {valid_loss:.3f} | {valid_accuracy:.3f}'.format(**locals()), end='')
    except KeyboardInterrupt:
      print('\nStoped!')
    else:
      print('\nDone.')

  def feedforward(self, input_data: np.array) -> np.array:
    activation = input_data
    for layer in self.layers:
      activation = layer.forward(activation)
    return activation
  
  def backpropagation(self, output_gradient: np.array) -> np.array:
    gradient = output_gradient
    for layer in reversed(self.layers):
      gradient = layer.backward(gradient)
    return gradient
  
  def update_weights(self):
    self.optimizer.optimize(self.layers)

  def predict(self, input_data: np.array) -> np.array:
    return self.feedforward(input_data)

  def test(self, test_data: Tuple[np.array, np.array]) -> np.array:
    input_data, target = test_data
    predict = self.feedforward(input_data)
    accuracy = self.metric.compare(predict, target)
    loss = self.cost.loss(predict, target)
    return accuracy, loss
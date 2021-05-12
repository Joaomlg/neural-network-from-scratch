import numpy as np
import webbrowser

from data import MNIST

from multilayer_perceptron import MLP
from multilayer_perceptron.layers import *
from multilayer_perceptron.activations import *
from multilayer_perceptron.optimizers import *
from multilayer_perceptron.costs import *
from multilayer_perceptron.metrics import *
from multilayer_perceptron.utils import format_data

from app import WebApp


mnist_dataset = MNIST()
train_data, validation_data, test_data = mnist_dataset.load()

NUM_OF_SAMPLES = 1000

train_data = format_data(train_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))
validation_data = format_data(validation_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))
test_data = format_data(test_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))

mlp = MLP(
  optimizer=GradientDescentOptmizer(learning_rate=0.01),
  cost=BinaryCrossEntropyCost(),
  metric=CategoricalAccuracyMetric()
)

mlp.add(InputLayer(shape=(1, 28, 28)))
mlp.add(Conv2DLayer(num_of_kernels=32, kernel_shape=(5, 5), num_of_channels=1, stride=(1, 1)))
mlp.add(ActivationLayer(function=relu))
mlp.add(MaxPooling2DLayer(pool_shape=(2, 2), stride=(2, 2)))
mlp.add(FlattenLayer())
mlp.add(DenseLayer(units=64))
mlp.add(ActivationLayer(function=relu))
mlp.add(DropoutLayer(drop_probability=0.5))
mlp.add(DenseLayer(units=10))
mlp.add(ActivationLayer(function=softmax))

mlp.compile()

mlp.fit(
  train_data=train_data,
  epochs=5,
  batch_size=32,
  validation_data=validation_data,
  test_data=test_data
)

webapp = WebApp(mlp)
webapp.run()

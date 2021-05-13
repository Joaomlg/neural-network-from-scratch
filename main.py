import numpy as np
import webbrowser

from data import MNIST

from neural_network import Network
from neural_network.layers import *
from neural_network.activations import *
from neural_network.optimizers import *
from neural_network.costs import *
from neural_network.metrics import *
from neural_network.utils import format_data

from app import WebApp


mnist_dataset = MNIST()
train_data, validation_data, test_data = mnist_dataset.load()

NUM_OF_SAMPLES = 1000

train_data = format_data(train_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))
validation_data = format_data(validation_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))
test_data = format_data(test_data, samples=NUM_OF_SAMPLES, input_shape=(1, 28, 28))

network = Network(
  optimizer=GradientDescentOptmizer(learning_rate=0.01),
  cost=BinaryCrossEntropyCost(),
  metric=CategoricalAccuracyMetric()
)

network.add(InputLayer(shape=(1, 28, 28)))
network.add(Conv2DLayer(num_of_kernels=32, kernel_shape=(5, 5), num_of_channels=1, stride=(1, 1)))
network.add(ActivationLayer(function=relu))
network.add(MaxPooling2DLayer(pool_shape=(2, 2), stride=(2, 2)))
network.add(FlattenLayer())
network.add(DenseLayer(units=64))
network.add(ActivationLayer(function=relu))
network.add(DropoutLayer(drop_probability=0.5))
network.add(DenseLayer(units=10))
network.add(ActivationLayer(function=softmax))

network.load_config('config.pkl.gz')

network.compile()

network.fit(
  train_data=train_data,
  epochs=5,
  batch_size=32,
  validation_data=validation_data,
  test_data=test_data
)

network.save_config()

webapp = WebApp(network)
webapp.run()

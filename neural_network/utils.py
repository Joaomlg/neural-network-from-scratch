import numpy as np


def categorical_to_onehot(x: np.array, n: int=-1) -> np.array:
  if n == -1:
    n = x.max() + 1
  onehot = np.zeros((x.size, n))
  onehot[np.arange(x.size), x] = 1
  return onehot

def probability_to_categorical(x: np.array) -> np.array:
  return np.argmax(x, axis=1)

def probability_to_onehot(x: np.array, n: int=-1) -> np.array:
  return categorical_to_onehot(probability_to_categorical(x), n)

def generate_batches(data, batch_size, random=True):
  x, y = data
  N = len(x)
  
  if random:
    n = np.arange(N)
    np.random.shuffle(n)
    x, y = x[n], y[n]

  for i in range(0, N, batch_size):
    yield (
      x[i:min(i + batch_size, N)],
      y[i:min(i + batch_size, N)]
    )

def format_data(data, input_shape: tuple):
  x, y = data
  x = x.reshape((-1, *input_shape)) / 255
  y = categorical_to_onehot(y)
  return x, y
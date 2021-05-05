import numpy as np


def categorical_to_onehot(x: np.array) -> np.array:
  onehot = np.zeros((x.size, x.max() + 1))
  onehot[np.arange(x.size), x] = 1
  return onehot

def probability_to_categorical(x: np.array) -> np.array:
  return np.argmax(x, axis=1)

def probability_to_onehot(x: np.array) -> np.array:
  return categorical_to_onehot(probability_to_categorical(x))

def generate_batches(data, batch_size, random=True):
  x, y = data
  N = len(x)
  
  if random:
    np.random.shuffle(x)
    np.random.shuffle(y)

  for i in range(0, N, batch_size):
    yield (
      x[i:min(i + batch_size, N)],
      y[i:min(i + batch_size, N)]
    )
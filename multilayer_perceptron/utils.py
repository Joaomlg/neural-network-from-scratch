import numpy as np


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
import numpy as np

def identity(x, derivative=False):
  if derivative:
    return 1
  else:
    return x

def step(x, derivative=False):
  if derivative:
    return 1.0
  else:
    return np.heaviside(x, 0)

def sigmoid(x, derivative=False):
  if derivative:
    return sigmoid(x) * (1 - sigmoid(x))
  else:
    return 1 / (1 + np.exp(-x))

def tanh(x, derivative=False):
  if derivative:
    return (1 / np.cosh(x)) ** 2
  else:
    return np.tanh(x)

def relu(x, derivative=False):
  if derivative:
    return np.heaviside(x, 0)
  else:
    return np.maximum(0, x)

def softmax(x, derivative=False):
  if derivative:
    return softmax(x) * (1 - softmax(x))
  else:
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    s = np.sum(e, axis=1, keepdims=True)
    return e / s

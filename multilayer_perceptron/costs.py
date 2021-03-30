import numpy as np

def mean_square_cost(predict, target, derivative=False):
  if derivative:
    return -1 * (target - predict)
  return (1/2) * np.sum((target - predict) ** 2) / len(target)

def cross_entropy_cost(predict, target, derivative=False):
  raise NotImplementedError
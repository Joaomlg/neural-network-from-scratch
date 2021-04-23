import numpy as np

def mean_square_cost(predict, target, derivative=False):
  if derivative:
    return -1 * (target - predict)
  return (1/2) * np.sum((target - predict) ** 2) / len(target)

def cross_entropy_cost(predict, target, derivative=False):
  clipped_predict = np.clip(predict, 1e-15, 1 - 1e-15)
  if derivative:
    return -1 * (target / clipped_predict)
  else:
    return -1 * np.sum(target * np.log(clipped_predict)) / len(target)

def binary_cross_entropy_cost(predict, target, derivative=False):
  clipped_predict = np.clip(predict, 1e-15, 1 - 1e-15)
  if derivative:
    return -1 * ((target / clipped_predict) - ((1 - target) / (1 - clipped_predict)))
  else:
    return -1 * np.sum(target * np.log(clipped_predict) + (1 - target) * np.log(1 - clipped_predict)) / len(target)
from abc import ABC, abstractmethod
import numpy as np

from multilayer_perceptron.utils import probability_to_onehot

class AbstractMetric(ABC):
  @abstractmethod
  def compare(self, predict: np.array, target: np.array) -> float:
    raise NotImplementedError


class AccuracyMetric(AbstractMetric):
  def __init__(self, tolerance: float):
    self.tolerance = tolerance

  def compare(self, predict: np.array, target: np.array) -> float:
    return (np.abs(predict - target) < self.tolerance).all(axis=1).mean()


class CategoricalAccuracyMetric(AbstractMetric):
  def compare(self, predict: np.array, target: np.array) -> float:
    predict_onehot = probability_to_onehot(predict, target.shape[1])
    return (predict_onehot == target).all(axis=1).mean()
import numpy as np
import gzip
import pickle
import os

class MNIST:
  filenames = {
    'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
  }

  decoded_filename = 'mnist.pkl.gz'

  train_samples = 50000
  validation_samples = 10000
  test_samples = 10000

  def __init__(self):
    self.current_dir = os.path.dirname(__file__)

  def download(self):
    pass

  def decode_and_save(self):
    data = {}
    
    for key, (images_filename, labels_filename) in self.filenames.items():
      with gzip.open(os.path.join(self.current_dir, images_filename), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16).reshape(-1, 28*28)
      
      with gzip.open(os.path.join(self.current_dir, labels_filename), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)

      data[key] = (images, labels)

    training = tuple(x[:self.train_samples] for x in data['train'])
    validation = tuple(x[self.train_samples:] for x in data['train'])
    test = tuple(data['test'])

    with gzip.open(os.path.join(self.current_dir, self.decoded_filename), 'wb') as file:
      pickle.dump((training, validation, test), file)

  def load(self):
    with gzip.open(os.path.join(self.current_dir, self.decoded_filename), 'rb') as file:
      training, validation, test = pickle.load(file)
    return training, validation, test

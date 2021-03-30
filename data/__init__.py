import numpy as np
import gzip
import pickle
import os
import urllib.request

class MNIST:
  host = 'http://yann.lecun.com/exdb/mnist/'

  filenames = {
    'train': ('train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz'),
    'test': ('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz'),
  }

  dataset_filename = 'mnist.pkl.gz'

  train_samples = 50000
  validation_samples = 10000
  test_samples = 10000

  def __init__(self):
    self.current_dir = os.path.dirname(__file__)

    if not self.is_dataset_available():
      print('Dataset not available! It will be downloaded and decoded, and can be take a while, please wait!')
      datasets = self.get_base_datasets_filenames()

      for dataset in datasets:
        if not self.is_base_dataset_downloaded(dataset):
          print(f'Downloading {dataset}...')
          self.download_dataset(dataset)

      print('Decoding files and saving it...')
      self.decode_and_save()

      print('Deleting base files (downloaded)...')
      for dataset in datasets:
        self.delete_dataset(dataset)
    
      print('Done.')

  def is_dataset_available(self):
    return os.path.exists(os.path.join(self.current_dir, self.dataset_filename))
  
  def get_base_datasets_filenames(self):
    return self.filenames['train'] + self.filenames['test']

  def is_base_dataset_downloaded(self, filename):
    return os.path.exists(os.path.join(self.current_dir, filename))

  def download_dataset(self, filename):
    url = self.host + filename
    dest = os.path.join(self.current_dir, filename)
    urllib.request.urlretrieve(url, dest)
  
  def delete_dataset(self, filename):
    os.remove(os.path.join(self.current_dir, filename))

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

    with gzip.open(os.path.join(self.current_dir, self.dataset_filename), 'wb') as file:
      pickle.dump((training, validation, test), file)

  def load(self):
    with gzip.open(os.path.join(self.current_dir, self.dataset_filename), 'rb') as file:
      training, validation, test = pickle.load(file)
    return training, validation, test

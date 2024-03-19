import numpy as np
from torch.utils.data import Dataset

from configs import TRAIN_BS, TEST_BS

class dataset(Dataset) :
  def __init__(self, root_dir='hacktrick/', type='train') :
    self.real = np.load(root_dir + 'real.npz')
    self.fake = np.load(root_dir + 'fake.npz')
    self.indices = np.ones(1500) * -1

    self.X = np.concatenate([self.real['x'].astype(np.float32), self.fake['x'].astype(np.float32)]).astype(np.float32)
    self.X[np.isinf(self.X)] = 1e3

    self.y = np.concatenate([np.ones(len(self.real['x']), dtype=np.float32).reshape(-1, 1), np.zeros(len(self.fake['x']), dtype=np.float32).reshape(-1, 1)], axis=0)
    self.vec = np.concatenate([self.real['y'], self.fake['y']])

    for i in range(len(self.y)) :
      if self.y[i]:
        self.indices[i] = np.where(self.vec[i])[0][0]

    self.X_train = np.concatenate([self.X[:600], self.X[-600:]])
    self.X_test = np.concatenate([self.X[600:750], self.X[-750:-600]])


    self.y_train = np.concatenate([self.y[:600], self.y[-600:]])
    self.y_test = np.concatenate([self.y[600:750], self.y[-750:-600]])

    self.ind_train = np.concatenate([self.indices[:600], self.indices[-600:]])
    self.ind_test = np.concatenate([self.indices[600:750], self.indices[-750:-600]])

    if type=='train':
      self.X = self.X_train
      self.y = self.y_train
      self.ind = self.ind_train
    else :
      self.X = self.X_test
      self.y = self.y_test
      self.ind = self.ind_test

  def __len__(self) :
    return len(self.X)

  def __getitem__(self, idx) :
    return np.log1p(self.X[idx]), self.y [idx], self.ind[idx]

class NoiseDataset(Dataset) :
    def __init__(self, root_dir='footprints/', type='train') :
        self.real = np.load(root_dir + 'real.npz')
        self.fake = np.load(root_dir + 'fake.npz')

        noise_mean = np.random.randint(25, 28, len(self.real['x'])).reshape(-1, 1, 1)
        noise_std = np.random.randint(50, 110, len(self.real['x'])).reshape(-1, 1, 1)
        
        noise = np.random.randn(*self.real['x'].shape) * noise_std + noise_mean

        real_X = self.real['x'].astype(np.float32) * 0.992 + noise * 0.008
        
        
        
        self.indices = np.ones(1500) * -1

        self.X = np.concatenate([real_X, self.fake['x'].astype(np.float32)]).astype(np.float32) + 10
        self.X[np.isinf(self.X)] = 1e5

        self.y = np.concatenate([np.ones(len(self.real['x']), dtype=np.float32).reshape(-1, 1), np.zeros(len(self.fake['x']), dtype=np.float32).reshape(-1, 1)], axis=0)
        self.vec = np.concatenate([self.real['y'], self.fake['y']])

        for i in range(len(self.y)) :
            if self.y[i]:
                self.indices[i] = np.where(self.vec[i])[0][0]

        self.X_train = np.concatenate([self.X[:600], self.X[-600:]])
        self.X_test = np.concatenate([self.X[600:750], self.X[-750:-600]])


        self.y_train = np.concatenate([self.y[:600], self.y[-600:]])
        self.y_test = np.concatenate([self.y[600:750], self.y[-750:-600]])

        self.ind_train = np.concatenate([self.indices[:600], self.indices[-600:]])
        self.ind_test = np.concatenate([self.indices[600:750], self.indices[-750:-600]])

        if type=='train':
            self.X = self.X_train
            self.y = self.y_train
            self.ind = self.ind_train
        else :
            self.X = self.X_test
            self.y = self.y_test
            self.ind = self.ind_test

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        return np.log1p(self.X[idx]), self.y [idx], self.ind[idx]
# Obsahuje nástroje pro normalizaci dat, generování datasetu a PyTorch dataset třídu

import numpy as np
import torch
from torch.utils.data import Dataset

class Normalizer:
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit(self, x):
        self.mu = np.mean(x)
        self.sd = np.std(x)

    def transform(self, x):
        return (x - self.mu) / self.sd

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        return (x * self.sd) + self.mu


def create_lstm_dataset(close_prices, window_size, reshape=True):
    x, y = [], []
    for i in range(len(close_prices) - window_size):
        x.append(close_prices[i:i+window_size])
        y.append(close_prices[i+window_size])
    x, y = np.array(x), np.array(y)
    if reshape:
        x = x.reshape((x.shape[0], x.shape[1], 1))
    return x, y


class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

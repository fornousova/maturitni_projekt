import numpy as np
from sklearn.preprocessing import MinMaxScaler

class Normalizer:
    def __init__(self):
        self.scaler = MinMaxScaler()

    def fit_transform(self, data):
        data = np.array(data).reshape(-1, 1) if data.ndim == 1 else np.array(data)
        return self.scaler.fit_transform(data)

    def transform(self, data):
        data = np.array(data).reshape(-1, 1) if data.ndim == 1 else np.array(data)
        return self.scaler.transform(data)

    def inverse_transform(self, data):
        data = np.array(data).reshape(-1, 1) if data.ndim == 1 else np.array(data)
        return self.scaler.inverse_transform(data)

def create_multifeature_lstm_dataset(features, target, window_size):
    x, y = [], []
    for i in range(len(features) - window_size):
        x.append(features[i:i + window_size])
        y.append(target[i + window_size])
    return np.array(x), np.array(y)
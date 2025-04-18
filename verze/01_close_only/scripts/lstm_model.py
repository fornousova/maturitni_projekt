# 📦 Soubor: scripts/lstm_model.py
# Obsahuje definici LSTM modelu pro predikci cen akcií

import torch
from torch import nn

def get_model():
    class LSTMModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
            self.dropout = nn.Dropout(0.2)
            self.fc = nn.Linear(32 * 2, 1)  # 2 vrstvy -> 2x 32 neuronů

        def forward(self, x):
            out, (hn, cn) = self.lstm(x)            # výstup z LSTM
            hn = hn.permute(1, 0, 2).reshape(x.size(0), -1)  # spojení výstupů vrstev
            out = self.dropout(hn)
            return self.fc(out).squeeze()  # výstup je 1D

    return LSTMModel()

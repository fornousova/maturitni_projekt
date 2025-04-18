# scripts/prepare_data.py

import os
import pandas as pd
import numpy as np
from utils import Normalizer, create_lstm_dataset

# ⚙️ Parametry
WINDOW_SIZE = 20
TRAIN_SPLIT = 0.8

def load_datasets(base_path="../data/downloaded_stock_data"):
    datasets = []
    for sector in os.listdir(base_path):
        sector_path = os.path.join(base_path, sector)
        if not os.path.isdir(sector_path):
            continue

        for file in os.listdir(sector_path):
            if file.endswith(".csv"):
                ticker = file.replace(".csv", "")
                file_path = os.path.join(sector_path, file)

                try:
                    df_raw = pd.read_csv(file_path, skiprows=3, header=None)
                    df = df_raw[[1, 2]].copy()
                    df.columns = ["Date", "Close"]
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
                    df = df.dropna().sort_values("Date")

                    prices = df["Close"].values
                    normalizer = Normalizer()
                    normalized = normalizer.fit_transform(prices)

                    x, y = create_lstm_dataset(normalized, WINDOW_SIZE)
                    split_idx = int(len(x) * TRAIN_SPLIT)

                    datasets.append({
                        "sector": sector,
                        "ticker": ticker,
                        "x_train": x[:split_idx],
                        "y_train": y[:split_idx],
                        "x_val": x[split_idx:],
                        "y_val": y[split_idx:],
                        "normalizer": normalizer,
                        "original_df": df
                    })

                except Exception as e:
                    print(f"⚠️ Chyba při zpracování {file}: {e}")

    return datasets
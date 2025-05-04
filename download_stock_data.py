# Skript pro stažení historických dat akcií podle sektorů
import yfinance as yf
import os
from datetime import datetime

# Vybrané sektory a tickery
TICKERS_BY_SECTOR = {
    "Technology":     ["AAPL", "MSFT", "NVDA"],
    "Communication":  ["META", "GOOGL", "DIS"],
    "Finance":        ["JPM", "GS", "BAC"],
    "Energy":         ["XOM", "CVX", "NEE"],
    "Consumer":       ["KO", "PG", "WMT"],
    "Industrial":     ["BA", "CAT", "UNP"]
}

# Parametry stahování
START_DATE = "2018-01-01"
END_DATE = datetime.today().strftime('%Y-%m-%d')
INTERVAL = "1d"  # denní data

# Cílové složky pro uložení dat
TARGET_DIRS = [
    "versions/01_close_only/data",
    "versions/02_multifeature/data"
]

# Vytvoření složek pokud neexistují
for base_path in TARGET_DIRS:
    os.makedirs(base_path, exist_ok=True)

# Stažení a uložení CSV souborů do obou složek
for sector, tickers in TICKERS_BY_SECTOR.items():
    for base_path in TARGET_DIRS:
        sector_path = os.path.join(base_path, sector)
        os.makedirs(sector_path, exist_ok=True)

    for ticker in tickers:
        print(f"Stahuji data pro {ticker} ({sector})")
        df = yf.download(ticker, start=START_DATE, end=END_DATE, interval=INTERVAL)

        if df.empty:
            print(f"Žádná data pro {ticker} nenalezena.")
            continue

        # Uložit do každé cílové složky
        for base_path in TARGET_DIRS:
            output_path = os.path.join(base_path, sector, f"{ticker}.csv")
            df.to_csv(output_path)
            print(f"Uloženo do {output_path}")

print("Všechna data byla stažena a uložena.")
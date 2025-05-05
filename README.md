# Predikce cen akcií pomocí neuronové sítě LSTM

Tento projekt vznikl jako maturitní práce z předmětu **Informační a komunikační technologie** v roce 2025. Jeho cílem je ověřit, zda lze pomocí neuronových sítí typu LSTM (Long Short-Term Memory) predikovat budoucí vývoj cen akcií na základě historických dat.

Zároveň je testována jednoduchá investiční strategie, která rozhoduje o nákupu akcie podle očekávaného růstu. Projekt je rozdělen do dvou verzí, které se liší množstvím a typem vstupních dat.

---

## Struktura projektu
```
projekt/
├── versions/
│   ├── 01_close_only/
│   │   ├── data/         # Historická data
│   │   ├── scripts/      # Skripty pro přípravu dat a model
│   │   ├── notebooks/    # Notebooky pro trénování a vyhodnocení
│   │   └── results/      # Výsledky modelů a grafy
│   │
│   └── 02_multifeature/
│       ├── data/
│       ├── scripts/
│       ├── notebooks/
│       └── results/
│
├── download_stock_data.py     # Skript pro stažení historických dat z Yahoo Finance
├── README.md                  # Dokumentace projektu
├── requirements.txt           # Přehled použitých knihoven
```

Každá verze je samostatná, s vlastním datasetem, notebooky, modely a výsledky.

---

## Spuštění a ovládání projektu

Projekt je navržen pro prostředí s Pythonem 3.11 a využívá knihovny jako `PyTorch`, `NumPy`, `Pandas`, `Matplotlib`, `Plotly` a `Scikit-learn`. Nejjednodušší způsob spuštění je pomocí **Jupyter Notebooku**.

### Doporučené prostředí

- Python 3.11
- Jupyter Notebook nebo JupyterLab
- Knihovny: `torch`, `numpy`, `pandas`, `matplotlib`, `plotly`, `scikit-learn`, `yfinance`

### Instalace závislostí

Pro instalaci všech potřebných knihoven spusťte: `pip install -r requirements.txt`

### Postup spuštění

V každé verzi je sada notebooků, které je doporučeno spouštět v tomto pořadí:

1. `01_prepare_data.ipynb`  
   → načtení, čištění a normalizace dat, uložení do `.pkl`.

2. `02_train_single_model.ipynb`  
   → trénink a vizualizace modelu pro jednu akcii.

3. `03_train_all_models.ipynb`  
   → trénink všech modelů napříč sektory a firmami.

4. `04_evaluation.ipynb`  
   → vyhodnocení výkonu modelů a uložení výsledků.

5. `05_profit_simulation.ipynb`  
   → simulace investiční strategie a výpočet zisků.

6. `06_visual_analysis.ipynb`  
   → interaktivní vizualizace predikcí, kumulativních zisků a sektorových rozdílů.

> **Poznámka:** Historická data akcií jsou již připravena ve složkách `data/`. Skript `download_stock_data.py` slouží pro případné stažení aktualizovaných dat.

---

## Výstupy projektu

- **Modely:** Uložené modely `.pth` ve složce `results/`.
- **CSV přehledy:** `evaluation_summary.csv`, `profit_simulation_summary.csv`.
- **PDF grafy:** Přehledy zisků a přesnosti po sektorech.
- **Interaktivní HTML grafy:** Predikce a kumulativní zisky pro vybrané akcie, sektorová srovnání (vytvořeno pomocí `Plotly`).
- **Notebooky:** Přehledně komentované kroky s vizualizacemi.

---

## Verze 1: Predikce na základě uzavírací ceny (`Close`)

### Popis
Model pracuje pouze s uzavírací cenou (`Close`) a predikuje hodnotu následující den na základě předchozích 20 dní.

### Parametry
- Vstup: `Close`
- Okno: 20 dní
- LSTM vrstvy: 2
- Skrytá vrstva: 32 neuronů
- Dropout: 0.2
- Ztrátová funkce: MSELoss
- Optimalizátor: Adam

---

## Verze 2: Predikce s více parametry (`Multifeature`)

### Popis
Tato verze využívá širší spektrum dat – `Open`, `High`, `Low`, `Volume` – k predikci ceny `Close`. Výsledky jsou stabilnější a přesnější.

### Parametry
- Vstupy: `Open`, `High`, `Low`, `Volume`
- Výstup: `Close`
- Okno: 20 dní
- LSTM vrstvy: 2
- Skrytá vrstva: 64 neuronů
- Dropout: 0.3
- Normalizace: `MinMaxScaler` (odděleně pro vstupy a cílové hodnoty)
- Trénování: `EarlyStopping`, ukládání nejlepších vah

---

## Investiční strategie

Použitá strategie testuje, zda je možné vydělat na základě predikce růstu:

- Pokud model predikuje růst → nákup
- Pokud predikuje pokles → žádná akce

Zisk je počítán jako rozdíl mezi skutečnou cenou další den a cenou při nákupu.

---

## Výsledky a porovnání

| Verze | Vstupy           | Nejlepší val_loss | Nejvýnosnější akcie | Poznámka                                |
|:------|:-----------------|:------------------|:--------------------|:----------------------------------------|
| 1     | `Close`           | –                  | UNP (~198 USD)       | Jednodušší model, překvapivě dobrý výsledek |
| 2     | OHLC + Volume     | BA (0.00027)          | GS (~216 USD)        | Vyšší přesnost, rozšířené vstupy         |

---

## Shrnutí a potenciál

Projekt ukazuje, že neuronové sítě typu LSTM lze úspěšně použít k predikci cen akcií. I jednoduchý model dokáže vytvořit ziskovou strategii. Rozšířením vstupních parametrů se podařilo zvýšit stabilitu výsledků.

Možné směry rozšíření:
- přidání sentimentální analýzy - zprávy a články
- nasazení predikce v reálném čase
- vyzkoušení jiných typů neuronových sítí
- rozšíření na další trhy

---

## Autor

**Jméno:** Fornousová Karla  
**Rok:** 2025  
**Předmět:** Maturitní projekt – Informační a komunikační technologie

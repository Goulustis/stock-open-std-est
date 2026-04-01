# stock open std est — Opening Window Volatility Prediction

Predicts the realized volatility (RV) of a stock's first 30 minutes (9:30–10:00 ET) using only data available before market open. Trained on NVDA 5-minute bars (2016–2026), 2,477 trading days, 67 features.

## What This Does

Given 5-minute OHLCV data, this pipeline:
1. Computes a **target**: realized volatility during 9:30–10:00 ET
2. Extracts **67 pre-market features**: lagged RV, premarket vol/range/volume, overnight gap, jump decomposition, semivariance, microstructure proxies, calendar effects
3. Trains **12 models** across baselines, statistical methods, and ML
4. Compares **full features vs. Lasso-selected features** for each model
5. Outputs position sizing recommendations for volatility-targeted trading

## Quick Start

```bash
conda activate lang
python3 main.py                          # Full pipeline, all 12 models
python3 example_position_sizing.py       # Position sizing demo
python3 main.py --train_end_date 2022-12-31   # Custom train/test split
```

## Results Summary

| Model | R² (Full) | R² (Selected) | MAE |
|-------|-----------|---------------|-----|
| **Lasso** | **0.337** | 0.338 | 0.0035 |
| LightGBM | 0.320 | 0.238 | 0.0034 |
| Premarket RV (baseline) | 0.280 | — | 0.0034 |
| XGBoost | 0.232 | 0.181 | 0.0037 |
| HAR-RV-X | 0.180 | -0.071 | 0.0034 |
| Random Forest | 0.140 | 0.258 | 0.0041 |
| Ridge | 0.034 | 0.301 | 0.0037 |

Lasso wins with R²=0.337, selecting 19 of 67 features. The premarket RV baseline (R²=0.280) is surprisingly strong — same-day premarket vol captures most of the signal. GARCH-family models underperform (negative R²) due to regime shifts between train (2016–2023) and test (2024–2026) periods.

## Documentation

- **[Methods & Results](docs/methods.md)** — Full model descriptions, feature list, detailed results, literature comparison
- [Code Structure](docs/code_structure.md) — Module descriptions and data flow

## Key Files

| File | Description |
|------|-------------|
| `main.py` | CLI entry point — runs full pipeline |
| `example_position_sizing.py` | Demo: how to use predictions for volatility-targeted position sizing |
| `pipeline.py` | Orchestration: data → features → train → predict → evaluate |
| `features.py` | 67 feature computations across 10 categories |
| `models/` | 12 model implementations, each in its own file |

## Data

Single CSV: `NVDA_5m.csv` — 5-minute OHLCV bars, 2016–2026, ~2.5M rows. Timestamps in UTC, converted to ET for feature computation.

## Setup

```bash
pip install -e .
python3 main.py
```

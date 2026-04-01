# Code Structure

## Overview

```
std_pred/
├── main.py                    # CLI entry point (tyro)
├── config.py                  # Base config classes, PrintableConfig
├── data.py                    # CSV loading, timezone parsing, time filtering
├── features.py                # 67 feature computations (10 categories)
├── pipeline.py                # PipelineConfig, Pipeline class — orchestration
├── evaluate.py                # Metrics (R², MAE, RMSE, MAPE), comparison tables
├── utils.py                   # Rich printing utilities
├── example_position_sizing.py # Position sizing demo
├── models/
│   ├── base.py                # BasePredictor interface
│   ├── naive.py               # NaivePrevDay baseline
│   ├── rolling_mean.py        # RollingMean baseline
│   ├── premarket.py           # PremarketRV baseline
│   ├── ewma.py                # EWMA volatility model
│   ├── har_rv.py              # HAR-RV-X model
│   ├── garch.py               # GARCH(1,1) + EGARCH
│   ├── lasso.py               # LassoCV predictor
│   ├── ridge.py               # RidgeCV predictor
│   ├── random_forest.py       # RandomForest predictor
│   ├── lightgbm.py            # LightGBM predictor
│   └── xgboost.py             # XGBoost predictor
└── docs/
    ├── methods.md             # Full methods, results, analysis
    └── code_structure.md      # This file
```

## Data Flow

```
CSV (5-min OHLCV, UTC)
    │
    ▼
data.py: load_and_filter()
    ├── Parse orig_ts → ET timezone
    ├── Filter to 08:00–16:00 ET
    └── Return DataFrame with date_et, time_et columns
    │
    ▼
features.py: build_feature_matrix()
    ├── compute_target()        → RV for 9:30–10:00 window
    ├── compute_lagged_rv()     → Lagged RV features
    ├── compute_full_ar_lags()  → AR lags 1–22
    ├── compute_premarket_features() → Premarket vol/range/volume
    ├── compute_overnight_features() → Overnight gap
    ├── compute_prev_day_features()  → Full-day prior stats
    ├── compute_jump_features()      → Bipower variation jumps
    ├── compute_higher_moments()     → Quarticity, skewness, kurtosis
    ├── compute_microstructure_features() → Amihud, spread proxy
    ├── compute_overnight_intraday_rv() → Overnight/intraday split
    ├── compute_semivariance_features() → Upside/downside RV
    └── compute_calendar_features()   → Day of week, month
    │
    ▼
pipeline.py: Pipeline
    ├── _time_split()           → Train/test by date
    ├── StandardScaler          → Fit on train, transform on test
    ├── populate_modules()      → Instantiate all 12 predictors
    ├── run()
    │   ├── Pass 1: Train all models on full features → predict
    │   ├── Lasso selects features (non-zero coefficients)
    │   └── Pass 2: Retrain all models on selected features → predict
    └── Returns results dict
    │
    ▼
evaluate.py: evaluate_all_results()
    ├── Compute R², MAE, RMSE, MAE for each model
    ├── Print comparison table (full vs selected)
    ├── Print feature importance (Lasso, RF, LightGBM)
    └── Save results.csv
```

## Module Details

### config.py
- `PrintableConfig` — Base class with rich-formatted `__str__`
- `BaseModelConfig` — Base for all model configs, has `setup()` method
- `BaseConfig` — Top-level config: data paths, time windows, feature settings
- `load_config_from_yaml()` — Load config overrides from YAML

### data.py
- `load_data()` — Read CSV, parse UTC timestamps, convert to ET
- `filter_hours()` — Filter to time range [start, end]
- `load_and_filter()` — Combined: load + filter to premarket-through-close

### features.py
- `log_returns()`, `realized_vol()` — Core computations
- `compute_daily_rv()` — Per-day RV for a time window
- Each feature category has its own function returning a DataFrame indexed by date
- `build_feature_matrix()` — Master function, concatenates all features + target

### pipeline.py
- `PipelineConfig` — Full config with all 12 model sub-configs, supports YAML override via `config_f`
- `Pipeline` class:
  - `populate_modules()` — Loads data, builds features, splits, standardizes, instantiates predictors
  - `_time_split()` — Date-based split with NaN handling
  - `_train_predict()` — Trains a set of predictors, returns predictions
  - `run()` — Two-pass: full features → Lasso selects → selected features

### evaluate.py
- `compute_metrics()` — R², MAE, RMSE, MAPE
- `evaluate_all_results()` — Evaluates both full and selected feature results
- `save_results_csv()` — Saves combined results to CSV

### utils.py
- Rich printing: `print_header`, `print_section`, `print_info`, `print_success`, `print_warning`, `print_error`
- Tables: `print_metrics_table`, `print_comparison_table`, `print_feature_importance`, `print_data_summary`

### models/
Each model follows the same pattern:
- `XxxConfig` dataclass with `_target` pointing to predictor class
- `XxxPredictor` inherits `BasePredictor`, implements `fit()`, `predict()`, `uses_features()`

| Model | Uses Features | Key Method |
|-------|--------------|------------|
| NaivePrevDay | No | Returns last training RV |
| RollingMean | No | Returns rolling mean of training RV |
| PremarketRV | Yes | Linear regression on pm_rv |
| EWMA | No | Exponential weighted moving average |
| HAR-RV-X | Yes | OLS with daily/weekly/monthly lags + exogenous |
| GARCH11 | No | arch.arch_model(vol='GARCH', p=1, q=1) |
| EGARCH | No | arch.arch_model(vol='EGARCH') |
| Lasso | Yes | LassoCV with feature selection |
| Ridge | Yes | RidgeCV |
| RandomForest | Yes | RandomForestRegressor |
| LightGBM | Yes | LGBMRegressor |
| XGBoost | Yes | XGBRegressor |

## Configuration

Run with defaults:
```bash
python3 main.py
```

Override via CLI:
```bash
python3 main.py --train_end_date 2022-12-31 --csv_path /path/to/data.csv
```

Override via YAML:
```bash
python3 main.py --config_f config.yaml
```

YAML example:
```yaml
train_end_date: "2022-12-31"
lasso_config:
  n_alphas: 200
  cv_folds: 10
lgbm_config:
  n_estimators: 500
  learning_rate: 0.01
```

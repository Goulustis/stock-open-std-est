# Methods & Results

## Problem Statement

Predict the realized volatility (RV) of NVDA's first 30 minutes of trading (9:30–10:00 ET) using only data available before 9:30. Target:

```
RV = sqrt( Σ r_i² )   where r_i = ln(P_i / P_{i-1})
```

Computed from 5-minute log returns within the window (typically 5–6 bars).

## Feature Set (67 features)

### Lagged RV (14 features)
- `rv_lag_{1,2,3,5,10,20}d` — Individual lagged RV values
- `rv_mean_{2,3,5,10,20}d` — Rolling mean RV
- `rv_std_{2,3,5,10,20}d` — Rolling std of RV
- `rv_ratio_5d_20d` — Short-term vs long-term vol ratio

### Full AR Lags (22 features) — Audrino & Knaus (2016)
- `ar_lag_{1..22}d` — Individual AR lags 1 through 22. Unlike HAR-RV which aggregates into daily/weekly/monthly buckets, these let Lasso select specific lags.

### Premarket (7 features)
- `pm_rv` — Realized vol from premarket bars (08:00–09:25)
- `pm_range` — (max high − min low) / first open
- `pm_volume_total`, `pm_volume_mean`, `pm_volume_std`
- `pm_volume_concentration` — std/mean of volume
- `pm_bar_count` — Number of premarket bars (data quality proxy)

### Overnight Gap (2 features)
- `overnight_gap_abs` — |ln(premarket open / prev close)|
- `close_to_open_return` — ln(premarket open / prev close)

### Prior Day Full (4 features)
- `prev_day_rv_full`, `prev_day_range`, `prev_day_volume`, `prev_day_return_abs`

### Jump Decomposition (4 features)
- `jump_t`, `continuous_rv_t`, `signed_jump_t`, `bpv_t` — Via bipower variation

### Higher Moments (3 features)
- `realized_quarticity`, `realized_skewness`, `realized_kurtosis`

### Microstructure (3 features)
- `amihud_illiquidity`, `spread_proxy`, `volume_concentration`

### Overnight/Intraday Split (2 features)
- `rv_overnight`, `rv_intraday`

### Semivariance (9 features) — New
- `rv_upside_lag1d`, `rv_upside_mean5d`, `rv_upside_mean22d`
- `rv_downside_lag1d`, `rv_downside_mean5d`, `rv_downside_mean22d`
- `rv_asymmetry_lag1d`, `rv_asymmetry_mean5d`, `rv_asymmetry_mean22d`

### Calendar (2 features)
- `day_of_week`, `month`

## Models (12)

### Baselines
| Model | Description |
|-------|-------------|
| **Naive** | Predicts yesterday's RV. Strong baseline due to vol clustering. |
| **Rolling Mean** | Predicts 10-day average RV. Smooths noise. |
| **Premarket RV** | Linear regression: target = slope × pm_rv + intercept. Uses same-day premarket vol. |

### Statistical
| Model | Description |
|-------|-------------|
| **EWMA** | σ²_t = λ·σ²_{t-1} + (1-λ)·r²_{t-1}, λ=0.94. No fitting required. |
| **HAR-RV-X** | OLS with daily/weekly/monthly RV lags + exogenous features (pm_rv, overnight_gap_abs). Recursive prediction for test period. |
| **GARCH(1,1)** | Standard GARCH via `arch` package. Univariate. |
| **EGARCH** | Asymmetric GARCH. Captures leverage effect. |

### Machine Learning
| Model | Description |
|-------|-------------|
| **Lasso** | L1-regularized regression with CV for alpha selection. Performs automatic feature selection. |
| **Ridge** | L2-regularized regression with CV for alpha selection. |
| **Random Forest** | 200 trees, max_depth=10. |
| **LightGBM** | 200 estimators, lr=0.05, max_depth=6. |
| **XGBoost** | 200 estimators, lr=0.05, max_depth=6. |

## Results

### R² Rankings (Full Features)

| Model | R² | MAE | RMSE | MAPE |
|-------|-----|-----|------|------|
| **Lasso** | **0.337** | 0.0035 | 0.0045 | 42.4% |
| **LightGBM** | **0.320** | 0.0034 | 0.0046 | 40.3% |
| **Premarket RV** | **0.280** | 0.0034 | 0.0047 | 39.5% |
| XGBoost | 0.232 | 0.0037 | 0.0048 | 45.5% |
| HAR-RV-X | 0.180 | 0.0034 | 0.0050 | 39.1% |
| Random Forest | 0.140 | 0.0041 | 0.0050 | 53.8% |
| Ridge | 0.034 | 0.0037 | 0.0054 | 42.9% |
| EGARCH | -0.766 | 0.0052 | 0.0074 | 41.4% |
| GARCH(1,1) | -0.850 | 0.0054 | 0.0075 | 42.9% |
| Rolling Mean | -1.127 | 0.0060 | 0.0081 | 48.4% |
| EWMA | -2.362 | 0.0085 | 0.0102 | 75.7% |
| Naive | -2.460 | 0.0087 | 0.0103 | 78.5% |

### Full vs Selected Features (19/67 selected by Lasso)

| Model | R² (Full) | R² (Selected) | Δ |
|-------|-----------|---------------|---|
| Lasso | 0.337 | 0.338 | +0.001 |
| Ridge | 0.034 | 0.301 | +0.267 |
| Random Forest | 0.140 | 0.258 | +0.118 |
| LightGBM | 0.320 | 0.238 | -0.082 |
| XGBoost | 0.232 | 0.181 | -0.051 |
| HAR-RV-X | 0.180 | -0.071 | -0.251 |

Tree models degrade with selected features because they rely on the full set of RV statistics. Linear models (Ridge, Lasso) benefit from dimensionality reduction.

### Top Selected Features (Lasso)

| Rank | Feature | Coefficient |
|------|---------|-------------|
| 1 | prev_day_range | 0.0015 |
| 2 | pm_range | 0.0011 |
| 3 | pm_volume_ratio | 0.0006 |
| 4 | bpv_t | 0.0006 |
| 5 | rv_lag_3d | 0.0003 |
| 6 | rv_lag_5d | 0.0003 |
| 7 | overnight_gap_abs | 0.0002 |
| 8 | ar_lag_7d | 0.0002 |

### Top Features (LightGBM)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | rv_std_10d | 277 |
| 2 | ar_lag_4d | 234 |
| 3 | ar_lag_6d | 230 |
| 4 | rv_lag_1d | 211 |
| 5 | rv_ratio_5d_20d | 208 |

## Analysis

### Why Lasso Wins

Lasso achieves the best R² (0.337) because it:
1. **Selects the right features** — 19 of 67, focusing on premarket range, prior day range, and short-term RV lags
2. **Handles collinearity** — AR lags are highly correlated; Lasso picks the most informative ones
3. **Avoids overfitting** — L1 regularization prevents the model from chasing noise in 67 features with only ~2,000 training samples

### Why GARCH Underperforms

GARCH(1,1) and EGARCH have negative R² because:
1. They are **univariate** — they only use past RV values, ignoring premarket signals
2. They **predict constants** — multi-step GARCH forecasts converge to the unconditional mean
3. **Regime shift** — the test period (2024–2026) has a different vol level than training (2016–2023)
4. This is consistent with the literature: Hansen & Lunde (2005) found GARCH often loses to realized measures out-of-sample

### Why Premarket RV Is Strong

The premarket RV baseline (R²=0.280) uses only one feature — same-day premarket realized volatility. It captures:
1. **Same-day vol regime** — premarket conditions reflect today's information flow
2. **Information discovery** — Lou & Shu (2017) showed premarket trading contains significant price/vol information
3. **No regime shift problem** — unlike lagged features, premarket vol adapts to current conditions

### Comparison to Literature

| Study | R² | Notes |
|-------|-----|-------|
| This project (Lasso) | 0.34 | 30-min window, premarket only, single stock |
| Audrino & Knaus (2016) | 0.45–0.55 | Full-day RV, AR lags only, index + large caps |
| Ding et al. (2021) | ~0.40 | Full-day RV, ordered Lasso, multiple assets |

Our R² of 0.34 is strong given the harder setting: predicting a 30-minute window (noisier than full-day) using only premarket data.

## Train/Test Split

- **Train**: 2016–2023 (~1,973 days)
- **Test**: 2024–2026 (~504 days)
- Features standardized on train, transformed on test
- NaN targets dropped (early days with insufficient history)

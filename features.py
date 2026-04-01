import numpy as np
import pandas as pd
from typing import Tuple
from config import BaseConfig


def log_returns(prices: pd.Series) -> pd.Series:
    """Compute log returns from a price series."""
    return np.log(prices / prices.shift(1))


def realized_vol(returns: pd.Series) -> float:
    """Compute realized volatility: sqrt(sum(r_i^2))."""
    returns = returns.dropna()
    if len(returns) == 0:
        return np.nan
    return np.sqrt(np.sum(returns**2))


def compute_daily_rv(df: pd.DataFrame, start: str, end: str) -> pd.Series:
    """Compute realized volatility for each day in a time window.

    Args:
        df: DataFrame with date_et, close columns
        start: Start time (inclusive)
        end: End time (inclusive)

    Returns:
        Series indexed by date_et with RV values
    """
    mask = df["time_et"].between(start, end)
    window = df[mask].copy()

    results = {}
    for date, group in window.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"])
        rv = realized_vol(returns)
        results[date] = rv

    return pd.Series(
        results, name=f"rv_{start.replace(':', '')}_{end.replace(':', '')}"
    )


def compute_target(df: pd.DataFrame, config: BaseConfig) -> pd.Series:
    """Compute target: RV during 9:30-10:00 window."""
    return compute_daily_rv(df, config.target_start, config.target_end)


# --- Lagged RV Features ---


def compute_lagged_rv(
    daily_rv: pd.Series, windows: Tuple[int, ...] = (1, 2, 3, 5, 10, 20)
) -> pd.DataFrame:
    """Compute lagged RV and rolling statistics.

    Args:
        daily_rv: Series of daily RV values, indexed by date
        windows: Lag windows for features

    Returns:
        DataFrame with lagged RV and rolling mean/std features
    """
    features = pd.DataFrame(index=daily_rv.index)

    for lag in windows:
        features[f"rv_lag_{lag}d"] = daily_rv.shift(lag)

    for window in [w for w in windows if w > 1]:
        features[f"rv_mean_{window}d"] = daily_rv.shift(1).rolling(window).mean()
        features[f"rv_std_{window}d"] = daily_rv.shift(1).rolling(window).std()

    if 5 in windows and 20 in windows:
        features["rv_ratio_5d_20d"] = features["rv_mean_5d"] / features["rv_mean_20d"]

    return features


def compute_full_ar_lags(daily_rv: pd.Series, max_lag: int = 22) -> pd.DataFrame:
    """Compute individual AR lags 1..max_lag (Audrino & Knaus, 2016).

    Unlike HAR-RV which aggregates into daily/weekly/monthly buckets,
    this creates a separate feature for each lag, letting Lasso decide
    which specific lags matter.
    """
    features = pd.DataFrame(index=daily_rv.index)
    for lag in range(1, max_lag + 1):
        features[f"ar_lag_{lag}d"] = daily_rv.shift(lag)
    return features


def compute_semivariance_features(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Compute realized semivariance (upside/downside) and their lags.

    RV+ = sum(max(r_i, 0)^2)  — upside volatility
    RV- = sum(min(r_i, 0)^2)  — downside volatility

    Downside vol is more persistent and more predictive of future total RV.
    """
    mask = df["time_et"].between(config.target_start, config.target_end)
    window = df[mask].copy()

    results = {}
    for date, group in window.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"]).dropna()

        if len(returns) == 0:
            continue

        rv_up = np.sqrt(np.sum(np.maximum(returns, 0) ** 2))
        rv_down = np.sqrt(np.sum(np.minimum(returns, 0) ** 2))

        results[date] = {
            "rv_upside": rv_up,
            "rv_downside": rv_down,
            "rv_asymmetry": rv_down - rv_up,
        }

    semi_df = pd.DataFrame.from_dict(results, orient="index")

    # Add lagged versions (1d, 5d, 22d means)
    for col in ["rv_upside", "rv_downside", "rv_asymmetry"]:
        semi_df[f"{col}_lag1d"] = semi_df[col].shift(1)
        semi_df[f"{col}_mean5d"] = semi_df[col].shift(1).rolling(5).mean()
        semi_df[f"{col}_mean22d"] = semi_df[col].shift(1).rolling(22).mean()

    return semi_df.drop(columns=["rv_upside", "rv_downside", "rv_asymmetry"])


# --- Premarket Features ---


def compute_premarket_features(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Compute features from premarket data (08:00-09:25)."""
    mask = df["time_et"].between(config.premarket_start, config.premarket_end)
    pm = df[mask].copy()

    results = {}
    for date, group in pm.groupby("date_et"):
        group = group.sort_values("ts_et")
        feats = {}

        returns = log_returns(group["close"])
        feats["pm_rv"] = realized_vol(returns)

        feats["pm_range"] = (group["high"].max() - group["low"].min()) / group[
            "open"
        ].iloc[0]

        feats["pm_volume_total"] = group["volume"].sum()

        feats["pm_bar_count"] = len(group)

        feats["pm_return_abs"] = abs(
            np.log(group["close"].iloc[-1] / group["close"].iloc[0])
        )

        if len(group) > 1:
            feats["pm_volume_std"] = group["volume"].std()
            feats["pm_volume_mean"] = group["volume"].mean()
            feats["pm_volume_concentration"] = (
                feats["pm_volume_std"] / feats["pm_volume_mean"]
            )
        else:
            feats["pm_volume_std"] = np.nan
            feats["pm_volume_mean"] = group["volume"].iloc[0]
            feats["pm_volume_concentration"] = np.nan

        results[date] = feats

    return pd.DataFrame.from_dict(results, orient="index")


# --- Overnight Gap Features ---


def compute_overnight_features(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Compute overnight gap and close-to-open return features."""
    mask = df["time_et"].between(config.premarket_start, config.premarket_end)
    pm = df[mask].copy()

    prev_close = {}
    for date, group in df.groupby("date_et"):
        group = group.sort_values("ts_et")
        regular = group[group["time_et"] >= "09:30"]
        if len(regular) > 0:
            prev_close[date] = regular["close"].iloc[-1]
        else:
            prev_close[date] = group["close"].iloc[-1]
    prev_close = pd.Series(prev_close)

    results = {}
    for date, group in pm.groupby("date_et"):
        group = group.sort_values("ts_et")
        if date not in prev_close.index:
            continue

        prev_date = prev_close.index[prev_close.index.get_loc(date) - 1]
        if prev_date not in prev_close:
            continue

        pm_first_open = group["open"].iloc[0]
        prev_day_close = prev_close[prev_date]

        feats = {}
        feats["overnight_gap_abs"] = abs(np.log(pm_first_open / prev_day_close))
        feats["close_to_open_return"] = np.log(pm_first_open / prev_day_close)

        results[date] = feats

    return pd.DataFrame.from_dict(results, orient="index")


# --- Prior Day Full Features ---


def compute_prev_day_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute full-day features from the previous trading day."""
    regular = df[df["time_et"] >= "09:30"].copy()

    daily_stats = {}
    for date, group in regular.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"])
        daily_stats[date] = {
            "prev_day_rv_full": realized_vol(returns),
            "prev_day_range": (group["high"].max() - group["low"].min())
            / group["open"].iloc[0],
            "prev_day_volume": group["volume"].sum(),
            "prev_day_return_abs": abs(
                np.log(group["close"].iloc[-1] / group["close"].iloc[0])
            ),
        }

    stats_df = pd.DataFrame.from_dict(daily_stats, orient="index")
    stats_df_shifted = stats_df.shift(1)
    stats_df_shifted.columns = [f"{c}" for c in stats_df_shifted.columns]
    return stats_df_shifted


# --- Jump Decomposition ---


def compute_jump_features(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Compute jump detection via bipower variation.

    Jump_t = max(RV_t - BPV_t, 0)
    BPV_t = (pi/2) * sum(|r_i| * |r_{i-1}|)
    """
    regular = df[df["time_et"] >= "09:30"].copy()

    results = {}
    for date, group in regular.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"]).dropna()

        if len(returns) < 2:
            continue

        rv = np.sqrt(np.sum(returns**2))

        abs_returns = returns.abs()
        bpv = (np.pi / 2) * np.sum(
            abs_returns.iloc[1:].values * abs_returns.iloc[:-1].values
        )
        bpv_vol = np.sqrt(bpv) if bpv > 0 else 0

        jump = max(rv - bpv_vol, 0)
        signed_jump = np.sign(returns.iloc[-1]) * jump if len(returns) > 0 else 0
        continuous_rv = rv - jump

        results[date] = {
            "jump_t": jump,
            "continuous_rv_t": continuous_rv,
            "signed_jump_t": signed_jump,
            "bpv_t": bpv_vol,
        }

    jump_df = pd.DataFrame.from_dict(results, orient="index")
    return jump_df.shift(1)


# --- Higher Moments ---


def compute_higher_moments(df: pd.DataFrame, config: BaseConfig) -> pd.DataFrame:
    """Compute realized quarticity, skewness, and kurtosis from intraday returns."""
    regular = df[df["time_et"] >= "09:30"].copy()

    results = {}
    for date, group in regular.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"]).dropna()

        if len(returns) < 4:
            continue

        rv2 = np.sum(returns**2)
        quarticity = np.sum(returns**4)

        if rv2 > 0:
            skewness = (np.sum(returns**3) / (rv2**1.5)) * np.sqrt(len(returns))
            kurtosis = (np.sum(returns**4) / (rv2**2)) * len(returns)
        else:
            skewness = 0
            kurtosis = 0

        results[date] = {
            "realized_quarticity": quarticity,
            "realized_skewness": skewness,
            "realized_kurtosis": kurtosis,
        }

    moments_df = pd.DataFrame.from_dict(results, orient="index")
    return moments_df.shift(1)


# --- Microstructure Features ---


def compute_microstructure_features(
    df: pd.DataFrame, config: BaseConfig
) -> pd.DataFrame:
    """Compute Amihud illiquidity, spread proxy, volume concentration."""
    regular = df[df["time_et"] >= "09:30"].copy()

    results = {}
    for date, group in regular.groupby("date_et"):
        group = group.sort_values("ts_et")
        returns = log_returns(group["close"]).dropna()

        if len(returns) == 0:
            continue

        total_volume = group["volume"].sum()

        feats = {}
        feats["amihud_illiquidity"] = (
            np.sum(returns.abs()) / total_volume if total_volume > 0 else np.nan
        )
        feats["spread_proxy"] = (group["high"].max() - group["low"].min()) / group[
            "close"
        ].mean()

        if len(group) > 1:
            feats["volume_concentration"] = (
                group["volume"].std() / group["volume"].mean()
            )
        else:
            feats["volume_concentration"] = 0

        results[date] = feats

    micro_df = pd.DataFrame.from_dict(results, orient="index")
    return micro_df.shift(1)


# --- Overnight/Intraday RV Split ---


def compute_overnight_intraday_rv(df: pd.DataFrame) -> pd.DataFrame:
    """Split RV into overnight (close-to-open) and intraday (open-to-close) components."""
    prev_close = {}
    for date, group in df.groupby("date_et"):
        group = group.sort_values("ts_et")
        regular = group[group["time_et"] >= "09:30"]
        if len(regular) > 0:
            prev_close[date] = regular["close"].iloc[-1]
        else:
            prev_close[date] = group["close"].iloc[-1]
    prev_close = pd.Series(prev_close)

    results = {}
    for date, group in df.groupby("date_et"):
        group = group.sort_values("ts_et")
        regular = group[group["time_et"] >= "09:30"]
        if len(regular) == 0:
            continue

        if date not in prev_close.index:
            continue

        prev_date_idx = prev_close.index.get_loc(date)
        if prev_date_idx == 0:
            continue

        prev_date = prev_close.index[prev_date_idx - 1]
        if prev_date not in prev_close:
            continue

        overnight_return = np.log(regular["open"].iloc[0] / prev_close[prev_date])
        intraday_return = np.log(regular["close"].iloc[-1] / regular["open"].iloc[0])

        results[date] = {
            "rv_overnight": abs(overnight_return),
            "rv_intraday": abs(intraday_return),
        }

    split_df = pd.DataFrame.from_dict(results, orient="index")
    return split_df.shift(1)


# --- Calendar Features ---


def compute_calendar_features(dates: pd.Index) -> pd.DataFrame:
    """Compute calendar-based features."""
    dates_dt = pd.to_datetime(dates)

    features = pd.DataFrame(index=dates)
    features["day_of_week"] = dates_dt.dayofweek
    features["month"] = dates_dt.month

    return features


# --- Master Feature Builder ---


def build_feature_matrix(
    df: pd.DataFrame, config: BaseConfig
) -> Tuple[pd.DataFrame, pd.Series]:
    """Build complete feature matrix and target vector.

    Returns:
        feature_df: DataFrame indexed by date, one row per trading day
        target: Series of target RV values indexed by date
    """
    target = compute_target(df, config)

    daily_rv = compute_daily_rv(df, config.target_start, config.target_end)

    lagged_features = compute_lagged_rv(daily_rv, config.rv_windows)

    full_ar_lags = compute_full_ar_lags(daily_rv, max_lag=22)

    semivariance = compute_semivariance_features(df, config)

    premarket_features = compute_premarket_features(df, config)

    overnight_features = compute_overnight_features(df, config)

    prev_day_features = compute_prev_day_features(df)

    jump_features = compute_jump_features(df, config)

    higher_moments = compute_higher_moments(df, config)

    microstructure = compute_microstructure_features(df, config)

    overnight_intraday = compute_overnight_intraday_rv(df)

    all_dates = target.index
    calendar = compute_calendar_features(all_dates)

    feature_df = pd.concat(
        [
            lagged_features,
            full_ar_lags,
            semivariance,
            premarket_features,
            overnight_features,
            prev_day_features,
            jump_features,
            higher_moments,
            microstructure,
            overnight_intraday,
            calendar,
        ],
        axis=1,
    )

    feature_df = feature_df.reindex(all_dates)

    feature_df["pm_volume_ratio"] = feature_df.get(
        "pm_volume_total", pd.Series(index=all_dates)
    ) / feature_df.get("prev_day_volume", pd.Series(index=all_dates))

    return feature_df, target

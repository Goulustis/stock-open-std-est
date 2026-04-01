import pandas as pd
from config import BaseConfig


def load_data(csv_path: str) -> pd.DataFrame:
    """Load NVDA 5-minute bar CSV and parse timestamps.

    Returns DataFrame with:
        - orig_ts: timezone-aware datetime (UTC)
        - ts_et: timezone-converted datetime (US/Eastern)
        - time_et: string time in HH:MM format (ET)
        - date_et: date in ET timezone
        - OHLCV columns preserved
    """
    df = pd.read_csv(csv_path)

    df["orig_ts"] = pd.to_datetime(df["orig_ts"], utc=True)
    df["ts_et"] = df["orig_ts"].dt.tz_convert("US/Eastern")
    df["time_et"] = df["ts_et"].dt.strftime("%H:%M")
    df["date_et"] = df["ts_et"].dt.date

    return df


def filter_hours(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Filter DataFrame to rows within [start, end] time window (ET).

    Args:
        df: DataFrame with 'time_et' column in HH:MM format
        start: Start time (inclusive), e.g. "08:00"
        end: End time (inclusive), e.g. "16:00"

    Returns:
        Filtered DataFrame
    """
    mask = df["time_et"].between(start, end)
    return df[mask].copy()


def load_and_filter(config: BaseConfig) -> pd.DataFrame:
    """Load data and filter to relevant hours (premarket through close).

    Returns data from premarket_start through 16:00 ET.
    """
    df = load_data(config.csv_path)
    df = filter_hours(df, config.premarket_start, "16:00")
    return df

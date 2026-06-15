# ============================================================
# src/data_preprocessing.py
# Load, validate, clean, and enrich Uber stock data
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple

from utils import get_logger, load_config, resolve_path


def load_raw_data(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Load raw CSV data from disk and perform initial type coercions.

    Args:
        config: Loaded project configuration.

    Returns:
        Raw DataFrame with Date as datetime index.

    Raises:
        FileNotFoundError: If CSV is missing.
        ValueError: If required columns are absent.
    """
    logger = get_logger(__name__, config)
    raw_path = resolve_path(config["data"]["raw_path"])

    if not raw_path.exists():
        raise FileNotFoundError(f"Raw data not found at: {raw_path}")

    logger.info(f"Loading raw data from: {raw_path}")
    df = pd.read_csv(raw_path)

    # ── Column validation ────────────────────────────────────────
    required = {"Date", "Open", "High", "Low", "Close", "Adj Close", "Volume"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # ── Date parsing & indexing ──────────────────────────────────
    date_col = config["data"]["date_column"]
    df[date_col] = pd.to_datetime(df[date_col])
    df.set_index(date_col, inplace=True)
    df.sort_index(inplace=True)

    logger.info(f"Raw data loaded: {len(df)} rows | {df.index.min()} → {df.index.max()}")
    return df


def clean_data(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Handle missing values, cast types, and remove invalid rows.

    Args:
        df: Raw DataFrame (Date as index).
        config: Loaded project configuration.

    Returns:
        Cleaned DataFrame.
    """
    logger = get_logger(__name__, config)
    fill_method = config["preprocessing"]["fill_method"]

    original_len = len(df)

    # ── Fill missing Volume (common in stock CSVs) ───────────────
    missing_vol = df["Volume"].isna().sum()
    if missing_vol > 0:
        logger.warning(f"Found {missing_vol} missing Volume values — applying {fill_method}")
        df["Volume"] = df["Volume"].fillna(method=fill_method)

    # ── Fill any other NaN with forward-fill then back-fill ──────
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ── Cast all price/volume columns to float ───────────────────
    for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── Drop rows where Close is still NaN after fills ───────────
    df.dropna(subset=["Close"], inplace=True)

    # ── Sanity check: High >= Low ────────────────────────────────
    invalid_hl = (df["High"] < df["Low"]).sum()
    if invalid_hl > 0:
        logger.warning(f"Found {invalid_hl} rows where High < Low — removing")
        df = df[df["High"] >= df["Low"]]

    # ── Remove duplicate index entries ───────────────────────────
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        logger.warning(f"Removing {dupes} duplicate date entries")
        df = df[~df.index.duplicated(keep="first")]

    logger.info(f"Cleaning complete: {original_len} → {len(df)} rows")
    return df


def engineer_features(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Add derived time-series features used in EDA and modelling.

    Features added:
        - Daily_Return         : % change in Close price
        - Daily_Range          : High − Low (intraday volatility)
        - Log_Return           : Log of daily return ratio
        - MA_30 / MA_90        : Rolling moving averages
        - Volume_MA_30         : 30-day rolling avg volume
        - Volatility_30        : 30-day rolling std of daily returns
        - Month / Year / DayOfWeek : Calendar decomposition

    Args:
        df: Cleaned DataFrame.
        config: Loaded project configuration.

    Returns:
        Feature-enriched DataFrame.
    """
    logger = get_logger(__name__, config)
    windows = config["preprocessing"]["rolling_windows"]  # [30, 90]

    df = df.copy()

    # ── Price-based features ─────────────────────────────────────
    df["Daily_Return"]   = df["Close"].pct_change() * 100          # % daily change
    df["Daily_Range"]    = df["High"] - df["Low"]                   # intraday range
    df["Log_Return"]     = np.log(df["Close"] / df["Close"].shift(1))  # log returns

    # ── Rolling moving averages ──────────────────────────────────
    for w in windows:
        df[f"MA_{w}"] = df["Close"].rolling(window=w, min_periods=1).mean()

    # ── Volume MA ────────────────────────────────────────────────
    df["Volume_MA_30"] = df["Volume"].rolling(window=30, min_periods=1).mean()

    # ── 30-day rolling volatility of returns ─────────────────────
    df["Volatility_30"] = df["Daily_Return"].rolling(window=30, min_periods=1).std()

    # ── Calendar features ────────────────────────────────────────
    df["Month"]      = df.index.month
    df["Year"]       = df.index.year
    df["DayOfWeek"]  = df.index.dayofweek   # 0=Monday … 4=Friday

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


def get_monthly_avg(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Resample daily Close to monthly averages for macro trend EDA.

    Args:
        df: Feature-enriched DataFrame.
        config: Loaded project configuration.

    Returns:
        Monthly-average DataFrame indexed by month-end dates.
    """
    freq = config["preprocessing"]["resample_freq"]   # "ME"
    monthly = df["Close"].resample(freq).mean().reset_index()
    monthly.columns = ["Month", "Avg_Close"]
    return monthly


def save_processed(df: pd.DataFrame, config: Dict[str, Any]) -> None:
    """
    Persist the processed DataFrame to CSV.

    Args:
        df: Processed, feature-enriched DataFrame.
        config: Loaded project configuration.
    """
    logger = get_logger(__name__, config)
    out_path = resolve_path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    logger.info(f"Processed data saved to: {out_path}")


def run_preprocessing(config: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full preprocessing pipeline: load → clean → engineer → save.

    Args:
        config: Loaded project configuration.

    Returns:
        Tuple of (daily_df, monthly_avg_df).
    """
    logger = get_logger(__name__, config)
    logger.info("═" * 50)
    logger.info("  PREPROCESSING PIPELINE START")
    logger.info("═" * 50)

    df      = load_raw_data(config)
    df      = clean_data(df, config)
    df      = engineer_features(df, config)
    monthly = get_monthly_avg(df, config)

    save_processed(df, config)

    logger.info("═" * 50)
    logger.info("  PREPROCESSING PIPELINE COMPLETE")
    logger.info("═" * 50)
    return df, monthly


# ─── Standalone run ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    cfg = load_config("config.yaml")
    daily_df, monthly_df = run_preprocessing(cfg)
    print(daily_df.tail())
    print(monthly_df.tail())

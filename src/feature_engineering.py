# ============================================================
# src/feature_engineering.py
# Train/test split, stationarity checks, Prophet data prep
# ============================================================

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from pathlib import Path
from statsmodels.tsa.stattools import adfuller

from utils import get_logger, load_config


# ─── Train / Test split ───────────────────────────────────────

def train_test_split_ts(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[pd.Series, pd.Series]:
    """
    Chronological train/test split preserving time order.

    Args:
        df: Feature-enriched DataFrame with DatetimeIndex.
        config: Loaded project configuration.

    Returns:
        (train_series, test_series) of the Close price column.
    """
    logger = get_logger(__name__, config)
    target = config["data"]["target_column"]
    test_size = config["evaluation"]["test_size"]

    series = df[target].dropna()
    split_idx = int(len(series) * (1 - test_size))

    train = series.iloc[:split_idx]
    test  = series.iloc[split_idx:]

    logger.info(
        f"Train/Test split — Train: {len(train)} rows "
        f"({train.index.min().date()} → {train.index.max().date()}) | "
        f"Test: {len(test)} rows "
        f"({test.index.min().date()} → {test.index.max().date()})"
    )
    return train, test


# ─── Stationarity check ───────────────────────────────────────

def check_stationarity(
    series: pd.Series,
    config: Dict[str, Any],
    label: str = "Series"
) -> Dict[str, Any]:
    """
    Run the Augmented Dickey-Fuller (ADF) test for stationarity.

    Null hypothesis H0: Series has a unit root (non-stationary).
    Reject H0 if p-value < 0.05 → series is stationary.

    Args:
        series: Time series to test.
        config: Loaded project configuration.
        label: Descriptive label for logging.

    Returns:
        Dict with adf_statistic, p_value, is_stationary, critical_values.
    """
    logger = get_logger(__name__, config)
    result = adfuller(series.dropna(), autolag="AIC")

    adf_stat    = result[0]
    p_value     = result[1]
    crit_values = result[4]
    is_stat     = p_value < 0.05

    logger.info(f"ADF Test — {label}")
    logger.info(f"  ADF Statistic : {adf_stat:.4f}")
    logger.info(f"  p-value       : {p_value:.4f}")
    logger.info(f"  Stationary    : {is_stat}")
    for pct, val in crit_values.items():
        logger.info(f"  Critical [{pct}]: {val:.4f}")

    return {
        "adf_statistic": adf_stat,
        "p_value": p_value,
        "is_stationary": is_stat,
        "critical_values": crit_values,
    }


def make_stationary(
    series: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.Series, int]:
    """
    Apply first-order differencing until series is stationary (max 2 diffs).

    Args:
        series: Original Close price series.
        config: Loaded project configuration.

    Returns:
        (differenced_series, d) where d is the differencing order applied.
    """
    logger = get_logger(__name__, config)
    d = 0
    s = series.copy()

    for _ in range(config["forecasting"]["arima"]["max_d"]):
        result = check_stationarity(s, config, label=f"After d={d}")
        if result["is_stationary"]:
            break
        s = s.diff().dropna()
        d += 1
        logger.info(f"Applied differencing → d={d}")

    return s, d


# ─── Prophet formatter ────────────────────────────────────────

def prepare_prophet_df(
    series: pd.Series,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Convert a DatetimeIndex Series to the ds/y format required by Prophet.

    Args:
        series: Close price series with DatetimeIndex.
        config: Loaded project configuration.

    Returns:
        DataFrame with columns ['ds', 'y'].
    """
    logger = get_logger(__name__, config)
    prophet_df = series.reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
    logger.info(f"Prophet DataFrame prepared: {len(prophet_df)} rows")
    return prophet_df


# ─── Lag feature builder (bonus for ML extensions) ────────────

def build_lag_features(
    df: pd.DataFrame,
    lags: list = [1, 5, 10, 21],
    target: str = "Close"
) -> pd.DataFrame:
    """
    Add lagged Close price features — useful if extending to XGBoost/LSTM.

    Args:
        df: DataFrame with Close column.
        lags: List of lag periods in trading days.
        target: Column name to lag.

    Returns:
        DataFrame with additional lag columns.
    """
    df = df.copy()
    for lag in lags:
        df[f"{target}_Lag_{lag}"] = df[target].shift(lag)
    df.dropna(inplace=True)
    return df


# ─── Standalone run ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_preprocessing import run_preprocessing

    cfg = load_config("config.yaml")
    daily_df, _ = run_preprocessing(cfg)

    train, test = train_test_split_ts(daily_df, cfg)
    print(f"Train size: {len(train)} | Test size: {len(test)}")

    stat_result = check_stationarity(train, cfg, label="Close (Train)")
    print(stat_result)

    prophet_df = prepare_prophet_df(daily_df["Close"], cfg)
    print(prophet_df.head())

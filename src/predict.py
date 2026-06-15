# ============================================================
# src/predict.py
# Generate and save future forecasts from trained models
# ============================================================

import pandas as pd
import numpy as np
import pickle
import joblib
import json
from pathlib import Path
from typing import Dict, Any, Tuple

from utils import get_logger, load_config, resolve_path


# ─── Load persisted models ────────────────────────────────────

def load_arima_model(config: Dict[str, Any]):
    """
    Load the saved ARIMA model from disk.

    Args:
        config: Loaded project configuration.

    Returns:
        Fitted pmdarima ARIMA model.
    """
    logger = get_logger(__name__, config)
    model_path = resolve_path(config["outputs"]["models_dir"]) / "arima_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"ARIMA model not found at: {model_path}")
    model = joblib.load(model_path)
    logger.info(f"ARIMA model loaded from {model_path}")
    return model


def load_prophet_model(config: Dict[str, Any]):
    """
    Load the saved Prophet model from disk.

    Args:
        config: Loaded project configuration.

    Returns:
        Fitted Prophet model.
    """
    logger = get_logger(__name__, config)
    model_path = resolve_path(config["outputs"]["models_dir"]) / "prophet_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Prophet model not found at: {model_path}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    logger.info(f"Prophet model loaded from {model_path}")
    return model


# ─── Forecast generation ──────────────────────────────────────

def forecast_arima(
    arima_model,
    horizon_days: int,
    config: Dict[str, Any],
    last_date: pd.Timestamp
) -> pd.DataFrame:
    """
    Generate ARIMA forecast with 95% confidence intervals.

    Args:
        arima_model: Fitted pmdarima model.
        horizon_days: Number of future trading days to forecast.
        config: Loaded project configuration.
        last_date: Last date in the training data for index generation.

    Returns:
        DataFrame with columns [Date, Forecast, Lower_95, Upper_95].
    """
    logger = get_logger(__name__, config)
    logger.info(f"Generating ARIMA forecast for {horizon_days} periods …")

    preds, conf_int = arima_model.predict(
        n_periods=horizon_days,
        return_conf_int=True,
        alpha=0.05,
    )

    # Build future business-day date index
    future_dates = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1),
        periods=horizon_days,
    )

    forecast_df = pd.DataFrame({
        "Date":     future_dates,
        "Forecast": np.round(preds, 4),
        "Lower_95": np.round(conf_int[:, 0], 4),
        "Upper_95": np.round(conf_int[:, 1], 4),
        "Model":    "ARIMA",
    }).set_index("Date")

    logger.info(f"ARIMA forecast: {forecast_df.index.min().date()} → {forecast_df.index.max().date()}")
    return forecast_df


def forecast_prophet(
    prophet_model,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Generate Prophet 180-day forecast with uncertainty intervals.

    Args:
        prophet_model: Fitted Prophet model.
        config: Loaded project configuration.

    Returns:
        DataFrame with Prophet forecast and components.
    """
    logger = get_logger(__name__, config)
    horizon = config["forecasting"]["prophet"]["forecast_horizon_days"]
    logger.info(f"Generating Prophet forecast for {horizon} days …")

    future = prophet_model.make_future_dataframe(periods=horizon)
    forecast = prophet_model.predict(future)

    # Trim to future-only rows
    forecast_future = forecast[["ds", "yhat", "yhat_lower", "yhat_upper", "trend"]].copy()
    forecast_future.rename(columns={
        "ds": "Date",
        "yhat": "Forecast",
        "yhat_lower": "Lower_95",
        "yhat_upper": "Upper_95",
    }, inplace=True)
    forecast_future["Model"] = "Prophet"
    forecast_future.set_index("Date", inplace=True)

    logger.info(f"Prophet forecast: {forecast_future.index.min().date()} → {forecast_future.index.max().date()}")
    return forecast_future, forecast


# ─── Save forecasts ───────────────────────────────────────────

def save_forecasts(
    arima_fc: pd.DataFrame,
    prophet_fc: pd.DataFrame,
    config: Dict[str, Any]
) -> None:
    """
    Persist both model forecasts as CSV files.

    Args:
        arima_fc: ARIMA forecast DataFrame.
        prophet_fc: Prophet forecast DataFrame.
        config: Loaded project configuration.
    """
    logger = get_logger(__name__, config)
    reports_dir = resolve_path(config["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    arima_path   = reports_dir / "arima_forecast.csv"
    prophet_path = reports_dir / "prophet_forecast.csv"

    arima_fc.to_csv(arima_path)
    prophet_fc.to_csv(prophet_path)

    logger.info(f"ARIMA forecast saved   → {arima_path}")
    logger.info(f"Prophet forecast saved → {prophet_path}")


# ─── Standalone run ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_preprocessing import run_preprocessing
    from feature_engineering import train_test_split_ts
    from train import train_arima, train_prophet

    cfg = load_config("config.yaml")
    daily_df, _ = run_preprocessing(cfg)
    train, test  = train_test_split_ts(daily_df, cfg)

    arima_model,   _ = train_arima(train, cfg)
    prophet_model, _ = train_prophet(daily_df, cfg)

    horizon = cfg["forecasting"]["prophet"]["forecast_horizon_days"]
    last_date = daily_df.index.max()

    arima_fc              = forecast_arima(arima_model, horizon, cfg, last_date)
    prophet_fc, full_fc   = forecast_prophet(prophet_model, cfg)

    save_forecasts(arima_fc, prophet_fc, cfg)
    print(arima_fc.head())
    print(prophet_fc.tail())

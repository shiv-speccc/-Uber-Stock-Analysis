# ============================================================
# src/train.py
# Train ARIMA (via auto_arima) and Facebook Prophet models
# ============================================================

import pandas as pd
import numpy as np
import joblib
import json
import pickle
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple

import pmdarima as pm
from prophet import Prophet

from utils import get_logger, load_config, resolve_path
from feature_engineering import prepare_prophet_df

warnings.filterwarnings("ignore")


# ─── ARIMA Training ───────────────────────────────────────────

def train_arima(
    train_series: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pm.arima.ARIMA, Dict[str, Any]]:
    """
    Fit an ARIMA model using pmdarima's auto_arima with AIC minimisation.

    Steps:
        1. Confirm stationarity expectation (d will be auto-selected)
        2. Stepwise AIC search over (p, d, q) parameter space
        3. Ljung-Box diagnostic on residuals
        4. Persist model to disk

    Args:
        train_series: Chronological training Close price series.
        config: Loaded project configuration.

    Returns:
        Tuple of (fitted_model, summary_dict).
    """
    logger = get_logger(__name__, config)
    cfg = config["forecasting"]["arima"]

    logger.info("─" * 45)
    logger.info("  Training ARIMA via auto_arima …")
    logger.info("─" * 45)
    logger.info(f"  max_p={cfg['max_p']} | max_q={cfg['max_q']} | max_d={cfg['max_d']}")
    logger.info(f"  Stepwise={cfg['stepwise']} | IC={cfg['information_criterion']}")

    model = pm.auto_arima(
        train_series,
        max_p=cfg["max_p"],
        max_q=cfg["max_q"],
        max_d=cfg["max_d"],
        stepwise=cfg["stepwise"],
        information_criterion=cfg["information_criterion"],
        seasonal=cfg["seasonal"],
        error_action="ignore",
        suppress_warnings=True,
        trace=False,
    )

    order = model.order
    aic   = model.aic()

    logger.info(f"  Best order selected: ARIMA{order}")
    logger.info(f"  AIC: {aic:.4f}")
    logger.info(f"  Observations fitted: {len(train_series)}")

    # ── Ljung-Box residual diagnostics ──────────────────────────
    from statsmodels.stats.diagnostic import acorr_ljungbox
    lb_result = acorr_ljungbox(model.resid(), lags=[10], return_df=True)
    lb_pvalue = lb_result["lb_pvalue"].iloc[0]
    logger.info(f"  Ljung-Box p-value (lag=10): {lb_pvalue:.4f} "
                f"({'white noise ✓' if lb_pvalue > 0.05 else 'autocorrelation detected'})")

    summary = {
        "model": "ARIMA",
        "order": list(order),
        "aic": round(aic, 4),
        "n_train": len(train_series),
        "ljung_box_pvalue": round(lb_pvalue, 4),
        "residual_kurtosis": round(float(pd.Series(model.resid()).kurtosis()), 4),
    }

    # ── Save model ───────────────────────────────────────────────
    models_dir = resolve_path(config["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "arima_model.pkl"
    joblib.dump(model, model_path)
    logger.info(f"  ARIMA model saved → {model_path}")

    return model, summary


# ─── Prophet Training ─────────────────────────────────────────

def train_prophet(
    daily_df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[Prophet, Dict[str, Any]]:
    """
    Fit a Facebook Prophet model on the full Close price series.

    Prophet handles:
        - Trend changepoints automatically
        - Weekly + yearly seasonality
        - Missing data gracefully

    Args:
        daily_df: Feature-enriched DataFrame with DatetimeIndex.
        config: Loaded project configuration.

    Returns:
        Tuple of (fitted_prophet, summary_dict).
    """
    logger = get_logger(__name__, config)
    cfg = config["forecasting"]["prophet"]

    logger.info("─" * 45)
    logger.info("  Training Facebook Prophet …")
    logger.info("─" * 45)

    prophet_df = prepare_prophet_df(daily_df["Close"], config)

    model = Prophet(
        daily_seasonality=cfg["daily_seasonality"],
        weekly_seasonality=cfg["weekly_seasonality"],
        yearly_seasonality=cfg["yearly_seasonality"],
        uncertainty_samples=cfg["uncertainty_samples"],
        changepoint_prior_scale=0.05,   # regularise trend flexibility
    )

    model.fit(prophet_df)

    # ── Generate in-sample fitted values for diagnostics ────────
    in_sample = model.predict(prophet_df)
    y_true    = prophet_df["y"].values
    y_pred    = in_sample["yhat"].values

    rmse_train = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae_train  = np.mean(np.abs(y_true - y_pred))

    logger.info(f"  Prophet in-sample RMSE : {rmse_train:.4f}")
    logger.info(f"  Prophet in-sample MAE  : {mae_train:.4f}")

    summary = {
        "model": "Prophet",
        "n_train": len(prophet_df),
        "in_sample_rmse": round(rmse_train, 4),
        "in_sample_mae": round(mae_train, 4),
        "daily_seasonality": cfg["daily_seasonality"],
        "yearly_seasonality": cfg["yearly_seasonality"],
        "forecast_horizon_days": cfg["forecast_horizon_days"],
    }

    # ── Save model ───────────────────────────────────────────────
    models_dir = resolve_path(config["outputs"]["models_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    prophet_path = models_dir / "prophet_model.pkl"
    with open(prophet_path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"  Prophet model saved → {prophet_path}")

    return model, summary


# ─── Save summaries ───────────────────────────────────────────

def save_training_summaries(
    arima_summary: Dict[str, Any],
    prophet_summary: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Persist training metadata to JSON for reproducibility.

    Args:
        arima_summary: ARIMA training metadata dict.
        prophet_summary: Prophet training metadata dict.
        config: Loaded project configuration.
    """
    logger = get_logger(__name__, config)
    reports_dir = resolve_path(config["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)

    combined = {
        "arima": arima_summary,
        "prophet": prophet_summary,
    }
    out_path = reports_dir / "training_summary.json"
    with open(out_path, "w") as f:
        json.dump(combined, f, indent=2)
    logger.info(f"Training summaries saved → {out_path}")


# ─── Standalone run ───────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    from data_preprocessing import run_preprocessing
    from feature_engineering import train_test_split_ts

    cfg = load_config("config.yaml")
    daily_df, _ = run_preprocessing(cfg)
    train, test  = train_test_split_ts(daily_df, cfg)

    arima_model,   arima_sum   = train_arima(train, cfg)
    prophet_model, prophet_sum = train_prophet(daily_df, cfg)

    save_training_summaries(arima_sum, prophet_sum, cfg)
    print("\nARIMA summary:", arima_sum)
    print("Prophet summary:", prophet_sum)

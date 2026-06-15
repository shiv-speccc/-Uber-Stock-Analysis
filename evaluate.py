# ============================================================
# src/evaluate.py
# Compute and compare ARIMA vs Prophet evaluation metrics
# ============================================================

import json
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, Tuple

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from utils import get_logger, load_config, resolve_path, format_metrics
from feature_engineering import prepare_prophet_df


# ─── Core metrics ─────────────────────────────────────────────

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label: str = "Model"
) -> Dict[str, float]:
    """
    Compute RMSE, MAE, MAPE, and R² for a set of predictions.

    Args:
        y_true: Ground-truth values.
        y_pred: Predicted values.
        label: Model name for display.

    Returns:
        Dict of metric_name → float value.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    # MAPE — guard against division by zero
    mape = np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1e-9, y_true))) * 100
    r2   = r2_score(y_true, y_pred)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE":  round(mae,  4),
        "MAPE": round(mape, 4),
        "R2":   round(r2,   4),
    }
    return metrics


# ─── ARIMA evaluation ─────────────────────────────────────────

def evaluate_arima(
    arima_model,
    test_series: pd.Series,
    config: Dict[str, Any]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Generate ARIMA out-of-sample predictions and compute metrics.

    Uses rolling one-step-ahead forecasting to be realistic.

    Args:
        arima_model: Fitted pmdarima ARIMA model.
        test_series: Held-out test Close price series.
        config: Loaded project configuration.

    Returns:
        Tuple of (predictions_array, metrics_dict).
    """
    logger = get_logger(__name__, config)
    logger.info("Evaluating ARIMA …")

    n_test = len(test_series)
    predictions = arima_model.predict(n_periods=n_test)
    y_true = test_series.values

    metrics = compute_metrics(y_true, predictions, label="ARIMA")
    logger.info(format_metrics({f"ARIMA {k}": v for k, v in metrics.items()}))
    return predictions, metrics


# ─── Prophet evaluation ───────────────────────────────────────

def evaluate_prophet(
    prophet_model,
    test_series: pd.Series,
    config: Dict[str, Any]
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Generate Prophet predictions aligned with the test set dates.

    Args:
        prophet_model: Fitted Prophet model.
        test_series: Held-out test Close price series.
        config: Loaded project configuration.

    Returns:
        Tuple of (forecast_df, metrics_dict).
    """
    logger = get_logger(__name__, config)
    logger.info("Evaluating Prophet …")

    # Build a future dataframe covering test dates only
    future_df = pd.DataFrame({"ds": test_series.index})
    forecast  = prophet_model.predict(future_df)

    y_true = test_series.values
    y_pred = forecast["yhat"].values

    metrics = compute_metrics(y_true, y_pred, label="Prophet")
    logger.info(format_metrics({f"Prophet {k}": v for k, v in metrics.items()}))
    return forecast, metrics


# ─── Comparison summary ───────────────────────────────────────

def compare_models(
    arima_metrics: Dict[str, float],
    prophet_metrics: Dict[str, float],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Build a side-by-side comparison and declare a winner.

    Args:
        arima_metrics: ARIMA evaluation metrics.
        prophet_metrics: Prophet evaluation metrics.
        config: Loaded project configuration.

    Returns:
        Comparison dict with per-metric winner and overall winner.
    """
    logger = get_logger(__name__, config)

    comparison = {}
    overall_arima_wins = 0

    for metric in ["RMSE", "MAE", "MAPE"]:
        a_val = arima_metrics[metric]
        p_val = prophet_metrics[metric]
        # Lower is better for error metrics
        winner = "ARIMA" if a_val < p_val else "Prophet"
        if winner == "ARIMA":
            overall_arima_wins += 1
        comparison[metric] = {
            "ARIMA": a_val,
            "Prophet": p_val,
            "winner": winner,
        }

    # R² — higher is better
    a_r2 = arima_metrics["R2"]
    p_r2 = prophet_metrics["R2"]
    winner_r2 = "ARIMA" if a_r2 > p_r2 else "Prophet"
    if winner_r2 == "ARIMA":
        overall_arima_wins += 1
    comparison["R2"] = {
        "ARIMA": a_r2,
        "Prophet": p_r2,
        "winner": winner_r2,
    }

    overall = "ARIMA" if overall_arima_wins >= 3 else "Prophet"
    comparison["overall_winner"] = overall

    logger.info("\n" + "=" * 55)
    logger.info("  MODEL COMPARISON RESULTS")
    logger.info("=" * 55)
    logger.info(f"  {'Metric':<10} {'ARIMA':>10} {'Prophet':>10} {'Winner':>10}")
    logger.info("  " + "-" * 45)
    for m, vals in comparison.items():
        if m == "overall_winner":
            continue
        logger.info(
            f"  {m:<10} {vals['ARIMA']:>10.4f} {vals['Prophet']:>10.4f} {vals['winner']:>10}"
        )
    logger.info("=" * 55)
    logger.info(f"  ★ OVERALL WINNER: {overall}")
    logger.info("=" * 55)

    return comparison


def save_evaluation_report(
    comparison: Dict[str, Any],
    config: Dict[str, Any]
) -> None:
    """
    Persist evaluation comparison to JSON.

    Args:
        comparison: Model comparison results dict.
        config: Loaded project configuration.
    """
    logger = get_logger(__name__, config)
    reports_dir = resolve_path(config["outputs"]["reports_dir"])
    reports_dir.mkdir(parents=True, exist_ok=True)
    out_path = reports_dir / "evaluation_report.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    logger.info(f"Evaluation report saved → {out_path}")


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

    preds_a, metrics_a = evaluate_arima(arima_model, test, cfg)
    _, metrics_p        = evaluate_prophet(prophet_model, test, cfg)

    comparison = compare_models(metrics_a, metrics_p, cfg)
    save_evaluation_report(comparison, cfg)

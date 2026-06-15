# ============================================================
# main.py — Uber Stock Analysis & Forecasting Pipeline
# Run this file to execute the complete end-to-end pipeline
# ============================================================
# Usage:
#   python main.py                    (full pipeline)
#   python main.py --skip-train       (skip re-training, load saved models)
#   python main.py --plots-only       (regenerate plots from saved outputs)
# ============================================================

import sys
import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add src/ to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils import load_config, get_logger, ensure_dirs, format_metrics
from data_preprocessing import run_preprocessing
from feature_engineering import train_test_split_ts, check_stationarity
from train import train_arima, train_prophet, save_training_summaries
from evaluate import evaluate_arima, evaluate_prophet, compare_models, save_evaluation_report
from predict import forecast_arima, forecast_prophet, save_forecasts, load_arima_model, load_prophet_model
from visualise import generate_all_plots


def parse_args():
    parser = argparse.ArgumentParser(
        description="Uber Stock Price Analysis & Forecasting Pipeline"
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Load previously saved models instead of retraining"
    )
    parser.add_argument(
        "--plots-only", action="store_true",
        help="Only regenerate plots (requires saved models and data)"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config YAML file (default: config.yaml)"
    )
    return parser.parse_args()


def main():
    # ── Setup ──────────────────────────────────────────────────
    args   = parse_args()
    config = load_config(args.config)
    logger = get_logger("main", config)
    ensure_dirs(config)

    logger.info("╔" + "═" * 54 + "╗")
    logger.info("║  UBER STOCK PRICE ANALYSIS & FORECASTING PIPELINE  ║")
    logger.info("║  B.Tech CSE — Data Science | JAIN University        ║")
    logger.info("║  Author: Shivarchan C  |  ID: 23BTRDC040            ║")
    logger.info("╚" + "═" * 54 + "╝")

    # ── Step 1: Preprocessing ──────────────────────────────────
    logger.info("\n[STEP 1/6] DATA PREPROCESSING")
    daily_df, monthly_df = run_preprocessing(config)

    logger.info(f"  Dataset shape : {daily_df.shape}")
    logger.info(f"  Date range    : {daily_df.index.min().date()} → {daily_df.index.max().date()}")
    logger.info(f"  All-time High : ${daily_df['Close'].max():.2f} on {daily_df['Close'].idxmax().date()}")
    logger.info(f"  All-time Low  : ${daily_df['Close'].min():.2f} on {daily_df['Close'].idxmin().date()}")

    # ── Step 2: Feature engineering / split ───────────────────
    logger.info("\n[STEP 2/6] FEATURE ENGINEERING & STATIONARITY")
    train, test = train_test_split_ts(daily_df, config)
    adf_result  = check_stationarity(train, config, label="Close Price (Train Set)")

    if adf_result["is_stationary"]:
        logger.info("  Series is stationary — d=0 is acceptable")
    else:
        logger.info("  Series is non-stationary — auto_arima will apply differencing")

    # ── Step 3: Model training ─────────────────────────────────
    logger.info("\n[STEP 3/6] MODEL TRAINING")

    if args.skip_train or args.plots_only:
        logger.info("  --skip-train: Loading saved models from disk …")
        arima_model   = load_arima_model(config)
        prophet_model = load_prophet_model(config)
        arima_sum, prophet_sum = {}, {}
    else:
        arima_model,   arima_sum   = train_arima(train, config)
        prophet_model, prophet_sum = train_prophet(daily_df, config)
        save_training_summaries(arima_sum, prophet_sum, config)

    # ── Step 4: Evaluation ─────────────────────────────────────
    logger.info("\n[STEP 4/6] MODEL EVALUATION")
    arima_preds,    arima_metrics   = evaluate_arima(arima_model, test, config)
    prophet_fc_test, prophet_metrics = evaluate_prophet(prophet_model, test, config)
    comparison = compare_models(arima_metrics, prophet_metrics, config)
    save_evaluation_report(comparison, config)

    # ── Step 5: Future forecasting ────────────────────────────
    logger.info("\n[STEP 5/6] FUTURE FORECASTING")
    horizon   = config["forecasting"]["prophet"]["forecast_horizon_days"]
    last_date = daily_df.index.max()

    arima_future_fc               = forecast_arima(arima_model, horizon, config, last_date)
    prophet_future_fc, full_fc    = forecast_prophet(prophet_model, config)
    save_forecasts(arima_future_fc, prophet_future_fc, config)

    logger.info(f"  ARIMA 180-day forecast range: "
                f"${arima_future_fc['Forecast'].min():.2f} – ${arima_future_fc['Forecast'].max():.2f}")
    logger.info(f"  Prophet 180-day forecast end: "
                f"${prophet_future_fc['Forecast'].iloc[-1]:.2f}")

    # ── Step 6: Visualisations ────────────────────────────────
    if not args.plots_only or True:
        logger.info("\n[STEP 6/6] GENERATING VISUALISATIONS")
        generate_all_plots(
            df              = daily_df,
            monthly         = monthly_df,
            train           = train,
            test            = test,
            arima_preds     = arima_preds,
            full_prophet_fc = full_fc,
            arima_metrics   = arima_metrics,
            prophet_metrics = prophet_metrics,
            config          = config,
        )

    # ── Final summary ─────────────────────────────────────────
    logger.info("\n" + "═" * 56)
    logger.info("  PIPELINE COMPLETE — FINAL SUMMARY")
    logger.info("═" * 56)
    logger.info(f"  Best model    : {comparison['overall_winner']}")
    logger.info(format_metrics({"ARIMA RMSE":   arima_metrics["RMSE"],
                                 "Prophet RMSE": prophet_metrics["RMSE"],
                                 "ARIMA MAE":    arima_metrics["MAE"],
                                 "Prophet MAE":  prophet_metrics["MAE"]}))
    logger.info("  Outputs saved to: outputs/plots/ and outputs/reports/")
    logger.info("═" * 56)


if __name__ == "__main__":
    main()

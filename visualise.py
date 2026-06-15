# ============================================================
# src/visualise.py
# Generate all EDA and forecasting visualisations
# ============================================================

import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from utils import get_logger, load_config, resolve_path

# ── Global style ──────────────────────────────────────────────
sns.set_theme(style="darkgrid", palette="muted", font="DejaVu Sans")
COLORS = {
    "close":    "#1f77b4",
    "ma30":     "#ff7f0e",
    "ma90":     "#2ca02c",
    "volume":   "#9467bd",
    "arima":    "#d62728",
    "prophet":  "#17becf",
    "ci":       "#aec7e8",
    "highlight":"#e377c2",
}
FIGSIZE_WIDE  = (14, 5)
FIGSIZE_TALL  = (14, 7)
FIGSIZE_SQ    = (10, 6)
DPI           = 150


def _save(fig: plt.Figure, name: str, config: Dict[str, Any]) -> Path:
    """Save figure to outputs/plots/ and close it."""
    plots_dir = resolve_path(config["outputs"]["plots_dir"])
    plots_dir.mkdir(parents=True, exist_ok=True)
    path = plots_dir / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    get_logger(__name__, config).info(f"Plot saved → {path}")
    return path


# ─── Plot 1: Full closing price trend ─────────────────────────

def plot_closing_price(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Line chart of Uber's full closing price history with peak/trough annotations."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.plot(df.index, df["Close"], color=COLORS["close"], lw=1.4, label="Close Price")

    # Annotate all-time high and all-time low
    ath_idx = df["Close"].idxmax()
    atl_idx = df["Close"].idxmin()

    ax.annotate(
        f"ATH: ${df['Close'].max():.2f}\n{ath_idx.date()}",
        xy=(ath_idx, df["Close"].max()),
        xytext=(ath_idx - pd.DateOffset(months=8), df["Close"].max() - 5),
        arrowprops=dict(arrowstyle="->", color="red"),
        fontsize=9, color="red",
    )
    ax.annotate(
        f"ATL: ${df['Close'].min():.2f}\n{atl_idx.date()}",
        xy=(atl_idx, df["Close"].min()),
        xytext=(atl_idx + pd.DateOffset(months=6), df["Close"].min() + 8),
        arrowprops=dict(arrowstyle="->", color="green"),
        fontsize=9, color="green",
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_title("Uber (UBER) — Closing Price History (May 2019 – Feb 2025)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "01_closing_price_trend.png", config)


# ─── Plot 2: Moving averages overlay ──────────────────────────

def plot_moving_averages(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Close price with 30-day and 90-day moving average overlays."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.plot(df.index, df["Close"], color=COLORS["close"],  lw=1.0, alpha=0.6, label="Close")
    ax.plot(df.index, df["MA_30"], color=COLORS["ma30"],   lw=1.6, label="30-Day MA")
    ax.plot(df.index, df["MA_90"], color=COLORS["ma90"],   lw=1.6, label="90-Day MA")

    # Crossover markers (MA_30 crosses above MA_90 → bullish)
    crossovers = (
        (df["MA_30"] > df["MA_90"]) &
        (df["MA_30"].shift(1) <= df["MA_90"].shift(1))
    )
    ax.scatter(
        df.index[crossovers], df["Close"][crossovers],
        marker="^", color="gold", s=60, zorder=5, label="Bullish Crossover"
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_title("Uber — 30-Day & 90-Day Moving Averages", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "02_moving_averages.png", config)


# ─── Plot 3: Trading volume ───────────────────────────────────

def plot_volume(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Daily volume bars with 30-day average line."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE_TALL, sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    ax1.plot(df.index, df["Close"], color=COLORS["close"], lw=1.4)
    ax1.set_ylabel("Close Price (USD)")
    ax1.set_title("Uber — Price vs Trading Volume", fontsize=14, fontweight="bold")

    ax2.bar(df.index, df["Volume"] / 1e6, color=COLORS["volume"], alpha=0.5, label="Volume (M)")
    ax2.plot(df.index, df["Volume_MA_30"] / 1e6, color="orange", lw=1.4, label="30-Day MA")
    ax2.set_ylabel("Volume (M shares)")
    ax2.set_xlabel("Date")
    ax2.legend(fontsize=8)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    fig.tight_layout()

    return _save(fig, "03_volume_analysis.png", config)


# ─── Plot 4: Monthly average closing prices ───────────────────

def plot_monthly_avg(monthly: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Bar chart of monthly average closing prices."""
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    colours = [
        "#2ca02c" if v >= monthly["Avg_Close"].mean() else "#d62728"
        for v in monthly["Avg_Close"]
    ]

    ax.bar(monthly["Month"], monthly["Avg_Close"], color=colours, alpha=0.85, edgecolor="white")
    ax.axhline(monthly["Avg_Close"].mean(), color="black", lw=1.5, ls="--",
               label=f"Overall Mean: ${monthly['Avg_Close'].mean():.2f}")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_title("Uber — Monthly Average Closing Price", fontsize=14, fontweight="bold")
    ax.set_xlabel("Month")
    ax.set_ylabel("Avg Close Price (USD)")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "04_monthly_avg_close.png", config)


# ─── Plot 5: Daily returns distribution ───────────────────────

def plot_return_distribution(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Histogram + KDE of daily percentage returns."""
    fig, ax = plt.subplots(figsize=FIGSIZE_SQ)
    returns = df["Daily_Return"].dropna()

    sns.histplot(returns, bins=80, kde=True, ax=ax,
                 color=COLORS["close"], edgecolor="white", alpha=0.7)
    ax.axvline(0, color="black", lw=1.2, ls="--", label="Zero Return")
    ax.axvline(returns.mean(), color="red", lw=1.5, ls="-",
               label=f"Mean: {returns.mean():.3f}%")

    ax.set_title("Uber — Daily Return Distribution", fontsize=14, fontweight="bold")
    ax.set_xlabel("Daily Return (%)")
    ax.set_ylabel("Frequency")
    ax.legend()

    # Add stats box
    stats_text = (
        f"Mean:  {returns.mean():.3f}%\n"
        f"Std:   {returns.std():.3f}%\n"
        f"Skew:  {returns.skew():.3f}\n"
        f"Kurt:  {returns.kurtosis():.3f}"
    )
    ax.text(0.97, 0.97, stats_text, transform=ax.transAxes,
            fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    fig.tight_layout()

    return _save(fig, "05_daily_return_distribution.png", config)


# ─── Plot 6: 30-day rolling volatility ───────────────────────

def plot_volatility(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Rolling 30-day volatility of daily returns."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    ax.fill_between(df.index, df["Volatility_30"], alpha=0.35, color=COLORS["ma90"])
    ax.plot(df.index, df["Volatility_30"], color=COLORS["ma90"], lw=1.2)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_title("Uber — 30-Day Rolling Volatility (Std of Daily Returns)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Volatility (%)")
    fig.tight_layout()

    return _save(fig, "06_rolling_volatility.png", config)


# ─── Plot 7: Correlation heatmap ──────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame, config: Dict[str, Any]) -> Path:
    """Pearson correlation heatmap of all numeric features."""
    numeric_cols = ["Open", "High", "Low", "Close", "Volume",
                    "Daily_Return", "Daily_Range", "MA_30", "MA_90", "Volatility_30"]
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.5, square=True, cbar_kws={"shrink": 0.8})
    ax.set_title("Uber — Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()

    return _save(fig, "07_correlation_heatmap.png", config)


# ─── Plot 8: ARIMA forecast vs actuals ───────────────────────

def plot_arima_forecast(
    train: pd.Series,
    test: pd.Series,
    arima_preds: np.ndarray,
    config: Dict[str, Any],
) -> Path:
    """ARIMA out-of-sample predictions vs actual test prices."""
    fig, ax = plt.subplots(figsize=FIGSIZE_WIDE)

    # Show last 200 train days for context
    ax.plot(train.index[-200:], train.values[-200:],
            color=COLORS["close"], lw=1.2, label="Train (last 200 days)")
    ax.plot(test.index, test.values,
            color="black", lw=1.5, label="Actual (Test)")
    ax.plot(test.index, arima_preds,
            color=COLORS["arima"], lw=1.5, ls="--", label="ARIMA Forecast")

    ax.axvline(test.index[0], color="grey", ls=":", lw=1.5, label="Train/Test Split")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    ax.set_title("Uber — ARIMA Forecast vs Actual (Test Set)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "08_arima_forecast.png", config)


# ─── Plot 9: Prophet forecast with CI ────────────────────────

def plot_prophet_forecast(
    full_forecast: pd.DataFrame,
    daily_df: pd.DataFrame,
    config: Dict[str, Any],
) -> Path:
    """Prophet 180-day future forecast with confidence interval."""
    fig, ax = plt.subplots(figsize=FIGSIZE_TALL)

    cutoff = daily_df.index.max()

    # Historical actuals
    ax.plot(daily_df.index, daily_df["Close"],
            color=COLORS["close"], lw=1.2, alpha=0.7, label="Historical Close")

    # Prophet in-sample fit
    historical_fc = full_forecast[full_forecast["ds"] <= cutoff]
    ax.plot(historical_fc["ds"], historical_fc["yhat"],
            color=COLORS["prophet"], lw=1.2, alpha=0.7, label="Prophet Fit")

    # Prophet future forecast
    future_fc = full_forecast[full_forecast["ds"] > cutoff]
    ax.plot(future_fc["ds"], future_fc["yhat"],
            color=COLORS["prophet"], lw=2.0, ls="--", label="Prophet Forecast")
    ax.fill_between(
        future_fc["ds"], future_fc["yhat_lower"], future_fc["yhat_upper"],
        alpha=0.25, color=COLORS["prophet"], label="95% CI"
    )

    ax.axvline(cutoff, color="grey", ls=":", lw=1.5, label="Forecast Start")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.xticks(rotation=45)
    ax.set_title("Uber — Prophet 180-Day Forecast with Confidence Intervals",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price (USD)")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "09_prophet_forecast.png", config)


# ─── Plot 10: Model comparison bar chart ─────────────────────

def plot_model_comparison(
    arima_metrics: Dict[str, float],
    prophet_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Path:
    """Side-by-side bar chart comparing ARIMA and Prophet on all metrics."""
    metrics = ["RMSE", "MAE", "MAPE"]
    x = np.arange(len(metrics))
    width = 0.35

    arima_vals   = [arima_metrics[m]   for m in metrics]
    prophet_vals = [prophet_metrics[m] for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars_a = ax.bar(x - width/2, arima_vals,   width, label="ARIMA",   color=COLORS["arima"],   alpha=0.85)
    bars_p = ax.bar(x + width/2, prophet_vals, width, label="Prophet", color=COLORS["prophet"], alpha=0.85)

    # Value labels
    for bar in bars_a + bars_p:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9
        )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_title("ARIMA vs Prophet — Model Comparison (Lower is Better)",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Error Value")
    ax.legend()
    fig.tight_layout()

    return _save(fig, "10_model_comparison.png", config)


# ─── Run all plots ────────────────────────────────────────────

def generate_all_plots(
    df: pd.DataFrame,
    monthly: pd.DataFrame,
    train: pd.Series,
    test: pd.Series,
    arima_preds: np.ndarray,
    full_prophet_fc: pd.DataFrame,
    arima_metrics: Dict[str, float],
    prophet_metrics: Dict[str, float],
    config: Dict[str, Any],
) -> None:
    """
    Generate and save all 10 project visualisations.

    Args:
        df: Feature-enriched daily DataFrame.
        monthly: Monthly average DataFrame.
        train: Training Close series.
        test: Test Close series.
        arima_preds: ARIMA test-set predictions array.
        full_prophet_fc: Prophet full forecast DataFrame (with ds/yhat columns).
        arima_metrics: ARIMA evaluation metrics dict.
        prophet_metrics: Prophet evaluation metrics dict.
        config: Loaded project configuration.
    """
    logger = get_logger(__name__, config)
    logger.info("Generating all visualisations …")

    plot_closing_price(df, config)
    plot_moving_averages(df, config)
    plot_volume(df, config)
    plot_monthly_avg(monthly, config)
    plot_return_distribution(df, config)
    plot_volatility(df, config)
    plot_correlation_heatmap(df, config)
    plot_arima_forecast(train, test, arima_preds, config)
    plot_prophet_forecast(full_prophet_fc, df, config)
    plot_model_comparison(arima_metrics, prophet_metrics, config)

    logger.info("All 10 plots saved to outputs/plots/")

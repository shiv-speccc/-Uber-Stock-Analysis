# ============================================================
# src/utils.py — Shared utilities: logging, config, helpers
# ============================================================

import logging
import os
import yaml
from pathlib import Path
from typing import Any, Dict


# ─── Config loader ────────────────────────────────────────────────────────────

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file and return as nested dict.

    Args:
        config_path: Relative or absolute path to config.yaml.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config


# ─── Logger factory ───────────────────────────────────────────────────────────

def get_logger(name: str, config: Dict[str, Any] = None) -> logging.Logger:
    """
    Create and return a named logger with console + file handlers.

    Args:
        name: Logger name (typically __name__ of calling module).
        config: Optional loaded config dict (uses default values if None).

    Returns:
        Configured logging.Logger instance.
    """
    # Defaults if config not provided
    log_level = logging.INFO
    log_format = "%(asctime)s — %(name)s — %(levelname)s — %(message)s"
    log_file = "logs/pipeline.log"

    if config:
        log_cfg = config.get("logging", {})
        log_level = getattr(logging, log_cfg.get("level", "INFO"), logging.INFO)
        log_format = log_cfg.get("format", log_format)
        log_file = log_cfg.get("log_file", log_file)

    # Ensure log directory exists
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if not logger.handlers:
        formatter = logging.Formatter(log_format)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(log_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


# ─── Directory helpers ────────────────────────────────────────────────────────

def ensure_dirs(config: Dict[str, Any]) -> None:
    """
    Create all output directories defined in config if they don't exist.

    Args:
        config: Loaded configuration dictionary.
    """
    dirs = [
        config["data"]["processed_path"].rsplit("/", 1)[0],
        config["outputs"]["plots_dir"],
        config["outputs"]["reports_dir"],
        config["outputs"]["models_dir"],
        config["outputs"]["logs_dir"],
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)


# ─── Metric formatter ─────────────────────────────────────────────────────────

def format_metrics(metrics: Dict[str, float]) -> str:
    """
    Pretty-print a metrics dictionary for logging/display.

    Args:
        metrics: Dict of metric_name → value.

    Returns:
        Formatted multi-line string.
    """
    lines = ["\n" + "=" * 45, "  MODEL EVALUATION METRICS", "=" * 45]
    for k, v in metrics.items():
        lines.append(f"  {k:<10}: {v:.4f}")
    lines.append("=" * 45)
    return "\n".join(lines)


# ─── Path resolver ────────────────────────────────────────────────────────────

def resolve_path(relative_path: str) -> Path:
    """
    Resolve a path relative to the project root (where config.yaml lives).

    Args:
        relative_path: Relative path string from config.

    Returns:
        Absolute Path object.
    """
    root = Path(__file__).resolve().parent.parent
    return root / relative_path

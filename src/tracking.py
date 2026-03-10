"""Experiment tracking integration via MLflow (Phase 5).

Provides a context manager and helper functions for logging hyperparameters,
metrics, and artifacts to MLflow. When MLflow is unavailable or disabled,
all operations gracefully become no-ops.
"""

import logging
from contextlib import contextmanager
from pathlib import Path

from src.config import (
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    USE_MLFLOW,
)

logger = logging.getLogger(__name__)

_mlflow = None


def _get_mlflow():
    """Lazy-import MLflow. Returns the module or None."""
    global _mlflow
    if _mlflow is not None:
        return _mlflow
    try:
        import mlflow
        _mlflow = mlflow
        return mlflow
    except ImportError:
        logger.info("MLflow not installed — experiment tracking disabled.")
        _mlflow = False  # sentinel: tried and failed
        return None


def is_tracking_enabled() -> bool:
    """Return True if MLflow tracking is both enabled and available."""
    if not USE_MLFLOW:
        return False
    mlflow = _get_mlflow()
    return mlflow is not None and mlflow is not False


@contextmanager
def start_run(
    run_name: str | None = None,
    tags: dict[str, str] | None = None,
):
    """Context manager that wraps an MLflow run.

    If MLflow is disabled or unavailable, yields a no-op context.

    Parameters
    ----------
    run_name : str, optional
        Human-readable name for the run.
    tags : dict, optional
        Key-value tags to attach to the run.
    """
    if not is_tracking_enabled():
        yield None
        return

    mlflow = _get_mlflow()
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    with mlflow.start_run(run_name=run_name) as run:
        if tags:
            mlflow.set_tags(tags)
        logger.info("MLflow run started: %s (id=%s)", run_name, run.info.run_id)
        yield run

    logger.info("MLflow run ended: %s", run_name)


def log_params(params: dict) -> None:
    """Log a dictionary of parameters to the active MLflow run."""
    if not is_tracking_enabled():
        return
    mlflow = _get_mlflow()
    # MLflow params must be strings; truncate long values
    for k, v in params.items():
        try:
            mlflow.log_param(k, str(v)[:500])
        except Exception:
            pass


def log_metrics(metrics: dict, step: int | None = None) -> None:
    """Log a dictionary of numeric metrics to the active MLflow run."""
    if not is_tracking_enabled():
        return
    mlflow = _get_mlflow()
    for k, v in metrics.items():
        if isinstance(v, (int, float)):
            try:
                mlflow.log_metric(k, v, step=step)
            except Exception:
                pass


def log_artifact(filepath: str | Path) -> None:
    """Log a file as an artifact to the active MLflow run."""
    if not is_tracking_enabled():
        return
    mlflow = _get_mlflow()
    filepath = str(filepath)
    try:
        mlflow.log_artifact(filepath)
    except Exception as exc:
        logger.debug("Failed to log artifact %s: %s", filepath, exc)


def log_artifacts_dir(dirpath: str | Path) -> None:
    """Log all files in a directory as artifacts."""
    if not is_tracking_enabled():
        return
    mlflow = _get_mlflow()
    dirpath = str(dirpath)
    try:
        mlflow.log_artifacts(dirpath)
    except Exception as exc:
        logger.debug("Failed to log artifacts from %s: %s", dirpath, exc)

"""Shared fixtures — bootstraps a trained toy model into a temp dir for API tests."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def _toy_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from scripts.generate_sample import generate
    from src.preprocess import preprocess_data
    from src.train import train_model

    workspace = tmp_path_factory.mktemp("toy")
    raw = workspace / "raw.csv"
    processed = workspace / "processed"
    models_dir = workspace / "models"
    mlruns_dir = workspace / "mlruns"

    df = generate(n=400, seed=7)
    df.to_csv(raw, index=False)
    train_csv, test_csv = preprocess_data(str(raw), str(processed))

    os.environ["MLFLOW_TRACKING_URI"] = f"file:{mlruns_dir}"
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "test-experiment"

    from src import config as config_module

    config_module.get_settings.cache_clear()

    train_model(
        str(train_csv),
        str(test_csv),
        model_out=str(models_dir / "xgb_model.joblib"),
        scaler_out=str(models_dir / "scaler.joblib"),
        params={"n_estimators": 30, "max_depth": 3},
    )
    return workspace


@pytest.fixture(scope="session")
def api_client(_toy_workspace: Path) -> Iterator:
    from fastapi.testclient import TestClient

    models_dir = _toy_workspace / "models"
    os.environ["MODEL_PATH"] = str(models_dir / "xgb_model.joblib")
    os.environ["SCALER_PATH"] = str(models_dir / "scaler.joblib")

    from src import config as config_module

    config_module.get_settings.cache_clear()

    import src.api as api_module

    with TestClient(api_module.app) as client:
        yield client

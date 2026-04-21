"""Verify the API still starts and /healthz responds when no model artifacts exist."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SAMPLE = json.loads((FIXTURE_DIR / "sample_request.json").read_text())


def test_healthz_ok_without_model(monkeypatch, tmp_path):
    missing_model = tmp_path / "nope_model.joblib"
    missing_scaler = tmp_path / "nope_scaler.joblib"
    monkeypatch.setenv("MODEL_PATH", str(missing_model))
    monkeypatch.setenv("SCALER_PATH", str(missing_scaler))

    from src import config

    config.get_settings.cache_clear()

    import src.api as api_module

    saved_model = api_module.bundle.model
    saved_scaler = api_module.bundle.scaler
    try:
        with TestClient(api_module.app) as client:
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json()["model_loaded"] is False

            pred = client.post("/predict", json=SAMPLE)
            assert pred.status_code == 503
            assert "Model artifacts not loaded" in pred.json()["detail"]
    finally:
        api_module.bundle.model = saved_model
        api_module.bundle.scaler = saved_scaler

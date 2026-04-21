from __future__ import annotations

import importlib


def test_defaults(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for var in (
        "MODEL_PATH",
        "SCALER_PATH",
        "API_PORT",
        "LOG_LEVEL",
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
    ):
        monkeypatch.delenv(var, raising=False)
    from src import config

    importlib.reload(config)
    s = config.get_settings()
    assert s.model_path.endswith("xgb_model.joblib")
    assert s.api_port == 8000
    assert s.log_level == "INFO"
    assert s.mlflow_experiment_name == "anomaly-detection"


def test_env_override(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("API_PORT", "9999")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    from src import config

    importlib.reload(config)
    s = config.get_settings()
    assert s.api_port == 9999
    assert s.log_level == "DEBUG"

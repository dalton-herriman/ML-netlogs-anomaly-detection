from __future__ import annotations

import joblib

from scripts.generate_sample import generate
from src.preprocess import preprocess_data
from src.train import train_model


def test_train_produces_artifacts_and_metrics(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"file:{tmp_path / 'mlruns'}")
    monkeypatch.setenv("MLFLOW_EXPERIMENT_NAME", "test-train")
    from src import config

    config.get_settings.cache_clear()

    raw = tmp_path / "raw.csv"
    processed = tmp_path / "processed"
    generate(n=300, seed=11).to_csv(raw, index=False)
    train_csv, test_csv = preprocess_data(str(raw), str(processed))

    model_out = tmp_path / "models" / "xgb.joblib"
    scaler_out = tmp_path / "models" / "scaler.joblib"

    metrics = train_model(
        str(train_csv),
        str(test_csv),
        model_out=str(model_out),
        scaler_out=str(scaler_out),
        params={"n_estimators": 20, "max_depth": 3},
    )

    assert model_out.exists()
    assert scaler_out.exists()
    assert set(metrics) == {"precision", "recall", "f1", "roc_auc"}
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert 0.0 <= metrics["f1"] <= 1.0

    loaded = joblib.load(model_out)
    assert hasattr(loaded, "predict")

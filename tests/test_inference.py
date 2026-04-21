from __future__ import annotations

import pandas as pd
import pytest

from scripts.generate_sample import generate
from src.inference import load_features, run_inference
from src.preprocess import preprocess_data
from src.train import train_model


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("inference")
    raw = workspace / "raw.csv"
    processed = workspace / "processed"
    generate(n=250, seed=3).to_csv(raw, index=False)
    train_csv, test_csv = preprocess_data(str(raw), str(processed))
    model_out = workspace / "xgb.joblib"
    scaler_out = workspace / "scaler.joblib"
    train_model(
        str(train_csv),
        str(test_csv),
        model_out=str(model_out),
        scaler_out=str(scaler_out),
        params={"n_estimators": 20, "max_depth": 3},
    )
    return {
        "model": str(model_out),
        "scaler": str(scaler_out),
        "raw": str(raw),
    }


def test_load_features_selects_columns(trained):
    df = load_features(trained["raw"])
    assert list(df.columns) == [
        "duration",
        "protocol",
        "src_port",
        "dst_port",
        "packet_count",
        "byte_count",
    ]


def test_run_inference_returns_predictions(trained):
    results = run_inference(trained["model"], trained["scaler"], trained["raw"])
    assert list(results.columns) == ["prediction", "anomaly_score"]
    assert len(results) > 0
    assert set(results["prediction"].unique()) <= {0, 1}
    assert results["anomaly_score"].between(0, 1).all()


def test_load_features_missing_column_raises(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"duration": [1.0], "label": [0]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="missing required feature columns"):
        load_features(str(bad))

"""Train an XGBoost anomaly-detection model and log the run to MLflow."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.config import get_settings

log = logging.getLogger(__name__)

DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 100,
    "max_depth": 6,
    "learning_rate": 0.1,
    "scale_pos_weight": 5,
    "eval_metric": "auc",
}


def train_model(
    train_path: str,
    test_path: str,
    model_out: str,
    scaler_out: str | None = None,
    params: dict[str, Any] | None = None,
) -> dict[str, float]:
    settings = get_settings()
    scaler_out = scaler_out or settings.scaler_path
    params = {**DEFAULT_PARAMS, **(params or {})}

    log.info("Loading training data", extra={"train_path": train_path, "test_path": test_path})
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_df.columns = train_df.columns.str.strip().str.lower()
    test_df.columns = test_df.columns.str.strip().str.lower()

    X_train = train_df.drop(columns=["label"])
    y_train = train_df["label"]
    X_test = test_df.drop(columns=["label"])
    y_test = test_df["label"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = XGBClassifier(**params)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
    }

    Path(model_out).parent.mkdir(parents=True, exist_ok=True)
    Path(scaler_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_out)
    joblib.dump(scaler, scaler_out)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(model_out, artifact_path="model")
        mlflow.log_artifact(scaler_out, artifact_path="model")

    log.info("Training complete", extra={"metrics": metrics, "model_out": model_out})
    return metrics


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly-detection model.")
    parser.add_argument("--train", required=True, help="Path to train.csv")
    parser.add_argument("--test", required=True, help="Path to test.csv")
    parser.add_argument(
        "--model_out",
        default=os.environ.get("MODEL_PATH", "models/xgb_model.joblib"),
        help="Path to save trained model",
    )
    parser.add_argument(
        "--scaler_out",
        default=os.environ.get("SCALER_PATH", "models/scaler.joblib"),
        help="Path to save fitted scaler",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    args = _parse_args()
    metrics = train_model(args.train, args.test, args.model_out, args.scaler_out)
    print("=== Metrics ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

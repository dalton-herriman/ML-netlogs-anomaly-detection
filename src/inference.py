"""Batch CSV inference — loads the saved model + scaler and scores a CSV."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.api import FEATURE_COLUMNS
from src.config import get_settings

log = logging.getLogger(__name__)


def load_features(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(columns=["flow id", "timestamp", "label"], errors="ignore")
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    for col in df.select_dtypes(include="object").columns:
        df[col] = pd.factorize(df[col])[0]
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Input missing required feature columns: {missing}")
    return df[FEATURE_COLUMNS]


def run_inference(model_path: str, scaler_path: str, input_csv: str) -> pd.DataFrame:
    log.info("Loading artifacts", extra={"model_path": model_path, "scaler_path": scaler_path})
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    features = load_features(input_csv)
    scaled = scaler.transform(features)

    predictions = model.predict(scaled)
    probabilities = model.predict_proba(scaled)[:, 1]
    return pd.DataFrame(
        {
            "prediction": predictions.astype(int),
            "anomaly_score": np.round(probabilities, 4),
        }
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run batch inference on a CSV.")
    parser.add_argument("--model", default=settings.model_path)
    parser.add_argument("--scaler", default=settings.scaler_path)
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", default=None, help="Optional CSV path to write predictions")
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    args = _parse_args()
    results = run_inference(args.model, args.scaler, args.input)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"[+] Wrote {len(results)} predictions to {args.output}")
    else:
        print(results.head().to_string(index=False))

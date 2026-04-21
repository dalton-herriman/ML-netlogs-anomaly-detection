"""Preprocess raw CSV logs into train/test CSVs for model training."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

log = logging.getLogger(__name__)

DROP_COLUMNS = {"flow id", "timestamp"}
LABEL_COLUMN = "label"


def preprocess_data(input_path: str, output_path: str, test_size: float = 0.2, seed: int = 42) -> tuple[Path, Path]:
    log.info("Loading raw data", extra={"input_path": input_path})
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip().str.lower()
    df = df.drop(columns=[c for c in df.columns if c in DROP_COLUMNS], errors="ignore")

    df = df.replace([np.inf, -np.inf], 0).fillna(0)

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Expected a '{LABEL_COLUMN}' column in {input_path}")

    train, test = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL_COLUMN]
    )

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    train_csv = out / "train.csv"
    test_csv = out / "test.csv"
    train.to_csv(train_csv, index=False)
    test.to_csv(test_csv, index=False)

    log.info("Wrote processed splits", extra={"train": str(train_csv), "test": str(test_csv)})
    return train_csv, test_csv


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw CSV into train/test CSVs.")
    parser.add_argument("--input", required=True, help="Path to raw CSV file")
    parser.add_argument("--output", required=True, help="Folder to store processed files")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    args = _parse_args()
    preprocess_data(args.input, args.output, args.test_size, args.seed)

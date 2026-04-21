from __future__ import annotations

import pandas as pd
import pytest

from scripts.generate_sample import generate
from src.preprocess import preprocess_data


def test_preprocess_splits_and_saves(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "processed"
    df = generate(n=200, seed=5)
    df.to_csv(raw, index=False)

    train_csv, test_csv = preprocess_data(str(raw), str(out))
    assert train_csv.exists() and test_csv.exists()

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    assert "label" in train_df.columns
    assert "label" in test_df.columns
    assert len(train_df) + len(test_df) == 200


def test_preprocess_drops_irrelevant_columns(tmp_path):
    raw = tmp_path / "raw.csv"
    out = tmp_path / "processed"
    df = generate(n=100, seed=5)
    df.insert(0, "Flow ID", range(len(df)))
    df.insert(1, "Timestamp", "2024-01-01")
    df.to_csv(raw, index=False)

    train_csv, _ = preprocess_data(str(raw), str(out))
    saved = pd.read_csv(train_csv)
    assert "flow id" not in saved.columns
    assert "timestamp" not in saved.columns


def test_preprocess_missing_label_raises(tmp_path):
    raw = tmp_path / "raw.csv"
    df = generate(n=50, seed=5).drop(columns=["label"])
    df.to_csv(raw, index=False)
    with pytest.raises(ValueError, match="label"):
        preprocess_data(str(raw), str(tmp_path / "out"))

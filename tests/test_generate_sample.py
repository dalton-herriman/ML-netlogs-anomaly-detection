from __future__ import annotations

import pandas as pd

from scripts.generate_sample import generate, main


def test_generate_shape_and_labels():
    df = generate(n=100, seed=1)
    assert len(df) == 100
    assert set(["duration", "protocol", "src_port", "dst_port", "packet_count", "byte_count", "label"]).issubset(
        df.columns
    )
    assert set(df["label"].unique()) <= {0, 1}


def test_generate_deterministic():
    a = generate(n=50, seed=1)
    b = generate(n=50, seed=1)
    pd.testing.assert_frame_equal(a, b)


def test_main_writes_file(tmp_path):
    out = tmp_path / "sample.csv"
    main(["--output", str(out), "--rows", "25", "--seed", "3"])
    assert out.exists()
    df = pd.read_csv(out)
    assert len(df) == 25

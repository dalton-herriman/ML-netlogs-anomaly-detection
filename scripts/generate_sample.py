"""Emit a small synthetic dataset shaped like CICIDS 2017 for demo / tests.

Columns match the simplified schema the API/training pipeline uses:
    duration, protocol, src_port, dst_port, packet_count, byte_count, label
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_benign = int(n * 0.8)
    n_attack = n - n_benign

    benign = pd.DataFrame(
        {
            "duration": rng.gamma(2.0, 1.5, size=n_benign),
            "protocol": rng.choice([6, 17], size=n_benign, p=[0.85, 0.15]),
            "src_port": rng.integers(1024, 65535, size=n_benign),
            "dst_port": rng.choice([80, 443, 22, 53, 8080], size=n_benign),
            "packet_count": rng.integers(1, 200, size=n_benign),
            "byte_count": rng.integers(64, 200_000, size=n_benign),
            "label": 0,
        }
    )

    attack = pd.DataFrame(
        {
            "duration": rng.gamma(0.5, 0.2, size=n_attack),
            "protocol": rng.choice([6, 17], size=n_attack, p=[0.5, 0.5]),
            "src_port": rng.integers(1024, 65535, size=n_attack),
            "dst_port": rng.integers(1, 1024, size=n_attack),
            "packet_count": rng.integers(500, 5000, size=n_attack),
            "byte_count": rng.integers(500_000, 5_000_000, size=n_attack),
            "label": 1,
        }
    )

    df = pd.concat([benign, attack], ignore_index=True).sample(
        frac=1.0, random_state=seed
    ).reset_index(drop=True)
    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic CICIDS-like dataset.")
    parser.add_argument("--output", default="data/raw/sample.csv", help="Output CSV path")
    parser.add_argument("--rows", type=int, default=2000, help="Number of rows")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    df = generate(args.rows, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[+] Wrote {len(df)} rows to {out}")


if __name__ == "__main__":
    main()

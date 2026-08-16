"""
Stage 1 of the comparison: let other tools write their synthetic tables.

This runs in a *separate* environment from the rest of the repo, on purpose.
SDV requires newer numpy/pandas/scipy than this project pins, so the two
cannot share a virtualenv. Rather than loosen the pins to accommodate a
comparison, each generator runs where it is supported and writes plain CSVs;
stage 2 (`score_comparison.py`) then scores every file with one scorer under
one dependency set, so a difference between tools is a difference between
tools and not a difference between their numpy versions.

Setup (anywhere outside this project's virtualenv):

    python3 -m venv /tmp/sdv-env
    /tmp/sdv-env/bin/pip install sdv
    /tmp/sdv-env/bin/python generate_baselines.py --input <table.csv> --out <dir>

Only pandas and sdv are needed here — deliberately nothing from this repo,
so no import of ours can influence a competitor's output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd


def _metadata_for(df: pd.DataFrame):
    """Build SDV metadata across the several API shapes SDV has shipped."""
    try:                                     # SDV >= 1.17
        from sdv.metadata import Metadata
        return Metadata.detect_from_dataframe(df, table_name="table")
    except Exception:
        pass
    from sdv.metadata import SingleTableMetadata   # SDV 1.0 - 1.16
    md = SingleTableMetadata()
    md.detect_from_dataframe(df)
    return md


def gaussian_copula(df: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """SDV's Gaussian copula — the closest peer to what this repo does."""
    import numpy as np
    from sdv.single_table import GaussianCopulaSynthesizer
    np.random.seed(seed)
    s = GaussianCopulaSynthesizer(_metadata_for(df))
    s.fit(df)
    return s.sample(num_rows=n_rows)


def ctgan(df: pd.DataFrame, n_rows: int, seed: int, epochs: int) -> pd.DataFrame:
    """CTGAN — a GAN, a genuinely different approach. Notably it encodes
    categories one-hot rather than ranking them, so it is not subject to the
    alphabetical-order limitation this repo's copula has."""
    import numpy as np
    import torch
    from sdv.single_table import CTGANSynthesizer
    np.random.seed(seed)
    torch.manual_seed(seed)
    s = CTGANSynthesizer(_metadata_for(df), epochs=epochs, verbose=False)
    s.fit(df)
    return s.sample(num_rows=n_rows)


GENERATORS = {
    "sdv-copula": lambda df, n, seed, epochs: gaussian_copula(df, n, seed),
    "sdv-ctgan": ctgan,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Write baseline synthetic tables")
    ap.add_argument("--input", required=True, help="Real table (CSV)")
    ap.add_argument("--out", required=True, help="Directory to write baselines into")
    ap.add_argument("--n-rows", type=int, default=None, help="Default: same as input")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=100, help="CTGAN epochs (default 100)")
    ap.add_argument("--only", nargs="*", default=None, help="Subset of generators")
    args = ap.parse_args()

    real = pd.read_csv(args.input)
    n_rows = args.n_rows or len(real)
    stem = Path(args.input).stem
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    names = args.only or list(GENERATORS)
    for name in names:
        fn = GENERATORS[name]
        print(f"[{name}] fitting on {len(real):,} rows ...", flush=True)
        t0 = time.time()
        try:
            synth = fn(real, n_rows, args.seed, args.epochs)
        except Exception as e:                       # a tool failing is a result
            print(f"[{name}] FAILED: {type(e).__name__}: {e}")
            continue
        secs = time.time() - t0
        path = out_dir / f"{stem}__{name}.csv"
        synth.to_csv(path, index=False)
        # The timing is recorded because cost is part of the comparison: a
        # tool that is better but takes a thousand times longer is a different
        # trade, not a strictly better one.
        (out_dir / f"{stem}__{name}.seconds").write_text(f"{secs:.1f}")
        print(f"[{name}] wrote {len(synth):,} rows to {path}  ({secs:.1f}s)")


if __name__ == "__main__":
    main()

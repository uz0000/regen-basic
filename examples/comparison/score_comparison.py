"""
Stage 2 of the comparison: score every synthetic table the same way.

Runs in this project's own environment. It generates the baselines that need
no extra dependencies (a bootstrap, an independent-column sampler, and this
repo's simulator), picks up any CSVs stage 1 left behind from other tools,
and puts all of them through identical measurements.

Three questions get asked of each table, in increasing order of how much
they matter:

  columns      does each column on its own look right?
  structure    do the columns still move together — numerically, and across
               a category boundary?
  conclusion   if you ran a real analysis on it, would you get the real
               answer? (only where the table supports a declared analysis)

The last one is the one that decides whether the data was any use, and it is
the one every other quality score in common use leaves out.

    python examples/comparison/score_comparison.py --baselines <dir from stage 1>
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from contracts.types import EstimandSpec
from engine.auditor.fidelity import (
    AuditorConfig,
    _categorical_association_delta,
    _correlation_delta,
    _eval_column,
)
from engine.ingest.loader import _build_field_dict
from certify.certifier import certify_dataset
from simulate.generate import generate_table

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# Two tables, because they stress different things. The first supports a
# declared analysis and is all numeric; the second carries the categorical
# structure that the numeric checks cannot see.
DATASETS = {
    "credit_default": {
        "path": REPO / "examples" / "decision_check" / "credit_default.csv",
        "estimand": EstimandSpec(
            outcome="default",
            predictors=["pay_delay_1", "utilization", "log_limit", "age"],
            family="logit",
        ),
        "note": "real public table, 30k rows, all numeric — supports a declared analysis",
    },
    "readings": {
        "path": REPO / "examples" / "readings.csv",
        "estimand": None,
        "note": "generated demo table with a category-to-measurement relationship",
    },
}


# ── Baselines that need no extra dependencies ─────────────────────────────────

def gen_bootstrap(real: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Positive control: resample the real rows. Not usable as synthetic data
    (it *is* the real data) but it marks the ceiling — no generator can beat
    resampling the truth, so anything scoring worse than this has lost
    something real."""
    rng = np.random.default_rng(seed)
    return real.iloc[rng.integers(0, len(real), size=n_rows)].reset_index(drop=True)


def gen_independent(real: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    """Negative control: sample each column separately. Every column's own
    distribution is perfect and every relationship between columns is gone —
    the failure mode that motivates the whole approach."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        c: rng.choice(real[c].to_numpy(), size=n_rows, replace=True)
        for c in real.columns
    })


def gen_ours(real: pd.DataFrame, n_rows: int, seed: int) -> pd.DataFrame:
    return generate_table(real, n_rows=n_rows, seed=seed).synthetic_df


LOCAL_GENERATORS = {
    "bootstrap (control)": gen_bootstrap,
    "independent (control)": gen_independent,
    "regen-synthetic": gen_ours,
}


# ── Scoring ───────────────────────────────────────────────────────────────────

def score(real: pd.DataFrame, synth: pd.DataFrame,
          estimand: Optional[EstimandSpec]) -> Dict[str, object]:
    """Every measurement in this repo, applied to one synthetic table."""
    cfg = AuditorConfig()
    fd = _build_field_dict(real, label_col="")
    shared = [c for c in real.columns if c in synth.columns]

    col_fails = 0
    for c in shared:
        if fd[c].is_identifier:
            continue
        if not _eval_column(c, real[c], synth[c], fd[c].field_type, cfg).passed:
            col_fails += 1

    corr = _correlation_delta(real, synth[shared], fd, "")
    cat, cat_pair = _categorical_association_delta(real, synth[shared], fd, "", cfg)

    out: Dict[str, object] = {
        "columns_off": col_fails,
        "corr_delta": corr,
        "cat_delta": cat,
        "cat_pair": cat_pair,
        "coef_kept": None,
        "coef_total": None,
    }

    if estimand is not None:
        cert = certify_dataset(real, synth, estimand)
        targets = cert.get("targets", [])
        out["coef_kept"] = sum(1 for t in targets if t.get("preserved"))
        out["coef_total"] = len(targets)
    return out


def _fmt(v, spec="{:.3f}", none="—"):
    return none if v is None else spec.format(v)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score synthetic tables side by side")
    ap.add_argument("--baselines", default=None,
                    help="Directory of CSVs written by generate_baselines.py")
    ap.add_argument("--n-rows", type=int, default=None)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    baseline_dir = Path(args.baselines) if args.baselines else None

    for ds_name, ds in DATASETS.items():
        path = Path(ds["path"])
        if not path.exists():
            print(f"\n[{ds_name}] skipped — {path} not present "
                  f"(run examples/make_sample_data.py first?)")
            continue

        real = pd.read_csv(path)
        n_rows = args.n_rows or len(real)
        est = ds["estimand"]

        print(f"\n{'=' * 100}")
        print(f"{ds_name}  —  {ds['note']}")
        print(f"{len(real):,} real rows, {len(real.columns)} columns")
        if est is not None:
            print(f"declared analysis: {est.family}  {est.outcome} ~ "
                  f"{' + '.join(est.predictors)}")
        print("=" * 100)

        tables: Dict[str, pd.DataFrame] = {
            name: fn(real, n_rows, args.seed) for name, fn in LOCAL_GENERATORS.items()
        }
        if baseline_dir and baseline_dir.exists():
            for f in sorted(baseline_dir.glob(f"{path.stem}__*.csv")):
                tables[f.stem.split("__", 1)[1]] = pd.read_csv(f)

        hdr = (f"{'generator':<24}{'cols off':>9}{'corr Δ':>9}{'cat Δ':>8}"
               f"{'conclusion kept':>18}   worst category pair")
        print(hdr)
        print("-" * len(hdr))
        for name, synth in tables.items():
            s = score(real, synth, est)
            concl = ("—" if s["coef_total"] is None
                     else f"{s['coef_kept']}/{s['coef_total']}")
            print(f"{name:<24}{s['columns_off']:>9}{_fmt(s['corr_delta']):>9}"
                  f"{_fmt(s['cat_delta']):>8}{concl:>18}   {s['cat_pair'] or ''}")

        print(f"\n  cols off = columns whose own distribution drifted too far"
              f"   (limit: TVD {AuditorConfig().tvd_threshold}, "
              f"Wasserstein {AuditorConfig().wasserstein_threshold})")
        print(f"  corr Δ   = change in numeric correlation structure "
              f"(limit {AuditorConfig().correlation_threshold}, lower is better)")
        print(f"  cat Δ    = worst change in a category-to-measurement relationship "
              f"(limit {AuditorConfig().categorical_association_threshold})")
        if est is not None:
            print("  conclusion kept = declared coefficients still statistically "
                  "indistinguishable from the truth")


if __name__ == "__main__":
    main()

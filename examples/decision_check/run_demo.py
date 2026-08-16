"""
If you make a decision from simulated data, do you reach the real answer?

That is the whole question this repo exists to ask, and this is where it gets
asked concretely. The setup needs a real dataset and a real decision drawn
from it; the one used here is a public dataset of 30,000 people, where the
decision is a regression estimating which of their recorded traits actually
predict an outcome. Nothing about the method is specific to that subject
matter — substitute any table where somebody fits a model and acts on the
coefficients, and the question is identical.

The measured quantity is the coefficient itself, because that *is* the
decision: it says which factors matter and how much. If it shifts between
real and simulated data, anyone acting on the simulation acts on a different
conclusion than the truth would have given them, without any signal that
something went wrong.

Three sources, run against the same declared analysis:

  bootstrap    positive control — a resample of the real data. Should
               certify. If it doesn't, the checker is broken, not the data.
  independent  negative control — this repo's own simulated table, with
               every column then shuffled on its own. Identical marginals,
               joint structure deliberately destroyed. Should be refused.
  simulator    the open question — the simulator's unmodified output. Does
               keeping every column's distribution *and* the correlations
               between columns suffice to keep the conclusion? Not assumed
               here; read what it prints.

Run from the repo root: python examples/decision_check/run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from simulate.generate import generate_table
from certify.certifier import certify_many
from contracts.types import EstimandSpec

CSV = Path(__file__).resolve().parent / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
ESTIMAND = EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")


def _bootstrap(real: pd.DataFrame, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(real), size=len(real))
    return real.iloc[idx].reset_index(drop=True)


def _independent_shuffle(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Same exact marginals, correlation destroyed: shuffle each column on
    its own, independently of every other column."""
    rng = np.random.default_rng(seed)
    out = df.copy()
    for col in out.columns:
        out[col] = rng.permutation(out[col].to_numpy())
    return out


def main() -> None:
    real = pd.read_csv(CSV)
    print(f"Real data:  {len(real):,} rows")
    print(f"Decision:   fit {ESTIMAND.family}  {ESTIMAND.outcome} ~ {' + '.join(PREDICTORS)}")
    print("            and act on the coefficients\n")

    sim = generate_table(real, n_rows=len(real), seed=1, table_name="real_table")
    print(f"Simulator's own verdict on its output: "
          f"{'PASS' if sim.fidelity_passed else 'FAIL'}"
          f"  (correlation delta {sim.correlation_delta:.3f} — lower is closer)\n")

    sources = {
        "bootstrap   (positive control)": _bootstrap(real, seed=2),
        "independent (negative control)": _independent_shuffle(sim.synthetic_df, seed=3),
        "simulator   (this repo's output)": sim.synthetic_df,
    }
    certs = certify_many(real, sources, ESTIMAND)

    truth = {t["coefficient"]: t["theta_real"] for t in next(iter(certs.values()))["targets"]}
    print("Each cell is the coefficient recovered from that source.")
    print("A ✗ means it is far enough from the truth to be a different conclusion.\n")

    header = f"{'source':<34} {'verdict':<10}" + "".join(f"{p:>14}" for p in PREDICTORS)
    print(header)
    print("-" * len(header))
    print(f"{'TRUTH (the real data)':<34} {'—':<10}"
          + "".join(f"  {truth[p]:+.3f}  " for p in PREDICTORS))
    for name, cert in certs.items():
        verdict = "matches" if cert["certified"] else "DIVERGES"
        row = f"{name:<34} {verdict:<10}"
        by_coef = {t["coefficient"]: t for t in cert["targets"]}
        for p in PREDICTORS:
            t = by_coef.get(p)
            mark = "✓" if t and t["preserved"] else "✗"
            row += f"  {t['theta_synth']:+.3f} {mark}" if t else f"{'n/a':>14}"
        print(row)

    n_ok = sum(c["certified"] for c in certs.values())
    print(f"\n{n_ok}/{len(certs)} sources reproduce the real conclusion.")


if __name__ == "__main__":
    main()

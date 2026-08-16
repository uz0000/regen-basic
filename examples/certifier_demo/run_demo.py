"""
Does the basic generator's own output preserve a declared conclusion?

One real dataset (UCI credit-card default), one declared analysis (a logistic
regression a credit analyst would actually run), and three sources:

  bootstrap    positive control — a bootstrap resample of the real data.
               Should always certify; if it doesn't, the certifier itself is
               broken, not the data.
  independent  negative control — this repo's own generate_table() output,
               with every column independently shuffled afterward. Same
               exact marginals as the joint-copula output, correlation
               deliberately destroyed. Should be refused.
  generator    the actual test — generate_table()'s unmodified output. This
               is the open question the demo exists to answer: does
               preserving marginals + cross-column correlation (what
               basic/generate.py checks) also preserve a downstream
               regression coefficient (what the certifier checks)? Not
               assumed — read the printed result.

Run from the repo root: python examples/certifier_demo/run_demo.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from basic.generate import generate_table
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
    print(f"Real data: {len(real):,} rows, default rate {real['default'].mean():.1%}")
    print(f"Analysis:  {ESTIMAND.family}  {ESTIMAND.outcome} ~ {' + '.join(PREDICTORS)}\n")

    generator_result = generate_table(real, n_rows=len(real), seed=1, table_name="credit_default")
    generator_out = generator_result.synthetic_df
    print(f"Generator fidelity: {'PASS' if generator_result.fidelity_passed else 'FAIL'}"
          f"  (correlation delta {generator_result.correlation_delta:.3f})\n")

    sources = {
        "bootstrap   (positive control)": _bootstrap(real, seed=2),
        "independent (negative control)": _independent_shuffle(generator_out, seed=3),
        "generator   (this repo's output)": generator_out,
    }
    certs = certify_many(real, sources, ESTIMAND)

    header = f"{'source':<34} {'certified':<10}" + "".join(f"{p:>14}" for p in PREDICTORS)
    print(header)
    print("-" * len(header))
    for name, cert in certs.items():
        verdict = "CERTIFIED" if cert["certified"] else "refused"
        row = f"{name:<34} {verdict:<10}"
        by_coef = {t["coefficient"]: t for t in cert["targets"]}
        for p in PREDICTORS:
            t = by_coef.get(p)
            mark = "✓" if t and t["preserved"] else "✗"
            row += f"  {t['theta_synth']:+.3f} {mark}" if t else f"{'n/a':>14}"
        print(row)

    print(f"\n{sum(c['certified'] for c in certs.values())}/{len(certs)} sources certified.")


if __name__ == "__main__":
    main()

"""How often the two controls behave, measured rather than asserted.

The README and the demo README used to say the positive control "must match" and
that a divergence meant the checker was broken. That is false, and misleading in
a way that matters: matching requires all four coefficients to agree at once,
each compared at 95%, so a resample of the real rows diverges on a small fraction
of seeds by chance. A reader who re-ran with a different seed, saw the control
diverge, and followed that instruction would have concluded the tool was broken
when it was working as designed. CORRECTIONS.md entry 4.

**These assert properties, not individual verdicts.** Pinning "seed 28 diverges"
would be pinning a z of 1.999 against a 1.96 cutoff — a margin of 0.04, which a
different numeric build moves. The sibling repo learned this by failing on Python
3.10 while passing on 3.11 and 3.12. So the sweeps below assert that *some* seeds
diverge and that most do not, which no build difference can flip.

Full measured rate: 11/300 = 3.67% (95% CI 1.5-5.8%). The real fit is computed
once and reused, so the sweep costs well under a second.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from certify.estimand import certify, fit_estimand
from contracts.types import EstimandSpec

DATA = Path(__file__).resolve().parent.parent / "examples" / "decision_check" / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
DEMO_SEED = 2      # what run_demo.py uses
SWEEP = 300


@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("credit_default.csv not present")
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def spec():
    return EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")


@pytest.fixture(scope="module")
def real_fit(real, spec):
    return fit_estimand(real, spec)


def _bootstrap(real, seed):
    """Exactly run_demo.py's _bootstrap."""
    rng = np.random.default_rng(seed)
    return real.iloc[rng.integers(0, len(real), size=len(real))].reset_index(drop=True)


def _independent(real, seed):
    """Every column shuffled on its own: marginals intact, joint destroyed."""
    rng = np.random.default_rng(seed)
    out = real.copy()
    for c in out.columns:
        out[c] = rng.permutation(out[c].to_numpy())
    return out.reset_index(drop=True)


def _verdicts(real, real_fit, spec, builder, n=SWEEP):
    return [certify(real_fit, fit_estimand(builder(real, s), spec), spec) for s in range(n)]


class TestThePositiveControl:

    def test_the_committed_demo_seed_matches(self, real, real_fit, spec):
        """The published table has to be reproducible, so this one IS pinned."""
        assert certify(real_fit, fit_estimand(_bootstrap(real, DEMO_SEED), spec), spec)["certified"] is True

    def test_it_diverges_on_some_seeds(self, real, real_fit, spec):
        """The "must match" claim is false. Measured 3.67% over 300 seeds."""
        diverged = sum(1 for c in _verdicts(real, real_fit, spec, _bootstrap)
                       if not c["certified"])
        assert diverged > 0, "expected chance divergences; the 'must match' claim would be true"

    def test_but_only_a_small_minority_of_the_time(self, real, real_fit, spec):
        """It is still a working positive control."""
        diverged = sum(1 for c in _verdicts(real, real_fit, spec, _bootstrap)
                       if not c["certified"])
        assert diverged < 0.20 * SWEEP, f"{diverged}/{SWEEP} diverged — too high for a positive control"

    def test_chance_divergences_are_near_misses(self, real, real_fit, spec):
        """They sit just past the cutoff, unlike the simulator's real failures."""
        worst = []
        for cert in _verdicts(real, real_fit, spec, _bootstrap):
            flagged = [t for t in cert["targets"] if t["preserved"] is False]
            if flagged:
                worst.append(max(abs(t["z"]) for t in flagged))
        assert worst, "expected at least one divergence in the sweep"
        assert max(worst) < 3.4, f"a control divergence reached z={max(worst):.2f}"


class TestTheNegativeControl:

    def test_shuffled_columns_never_match(self, real, real_fit, spec):
        """The one absolute claim about the controls that actually holds."""
        matched = sum(1 for c in _verdicts(real, real_fit, spec, _independent, n=40)
                      if c["certified"])
        assert matched == 0

    def test_the_coefficients_the_data_pinned_down_always_fail(self, real, real_fit, spec):
        for seed, cert in enumerate(_verdicts(real, real_fit, spec, _independent, n=20)):
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            for name in ("pay_delay_1", "utilization", "log_limit"):
                assert by_name[name]["preserved"] is False, f"{name} survived at seed {seed}"

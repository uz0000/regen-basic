"""How often the two controls behave, measured rather than asserted.

The README and the demo README used to say the positive control "must match" and
that a divergence meant the checker was broken. That is false, and misleading in
a way that matters: matching requires all four coefficients to agree at once,
each compared at 95%, so a resample of the real rows diverges on a small
fraction of seeds by chance. A reader who re-ran with a different seed, saw the
control diverge, and followed that instruction would have concluded the tool was
broken when it was working as designed. CORRECTIONS.md entry 4.

These pin what is actually true so the prose cannot drift back: the committed
demo seed matches, some seeds do not, and the negative control always diverges.

Specific seeds are used rather than a sweep to keep the suite fast. The full rate
(3.67%, 11 of 300) comes from sweeping `seed` in the parametrised test below.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from certify.certifier import certify_dataset
from contracts.types import EstimandSpec

DATA = Path(__file__).resolve().parent.parent / "examples" / "decision_check" / "credit_default.csv"
PREDICTORS = ["pay_delay_1", "utilization", "log_limit", "age"]
DEMO_SEED = 2                       # what run_demo.py uses
DIVERGING_SEEDS = [0, 25, 28, 59]   # found by sweeping 0..59


@pytest.fixture(scope="module")
def real():
    if not DATA.exists():
        pytest.skip("credit_default.csv not present")
    return pd.read_csv(DATA)


@pytest.fixture(scope="module")
def spec():
    return EstimandSpec(outcome="default", predictors=PREDICTORS, family="logit")


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


class TestThePositiveControl:

    def test_the_committed_demo_seed_matches(self, real, spec):
        """The published table has to be reproducible."""
        assert certify_dataset(real, _bootstrap(real, DEMO_SEED), spec)["certified"] is True

    @pytest.mark.parametrize("seed", DIVERGING_SEEDS)
    def test_some_seeds_diverge_by_chance(self, real, spec, seed):
        """Four coefficients, each a 95% test, so divergence happens without a bug."""
        assert certify_dataset(real, _bootstrap(real, seed), spec)["certified"] is False

    def test_a_chance_divergence_is_a_near_miss(self, real, spec):
        """It sits near the threshold, unlike the simulator's real failures."""
        cert = certify_dataset(real, _bootstrap(real, DIVERGING_SEEDS[0]), spec)
        flagged = [t for t in cert["targets"] if t["preserved"] is False]
        assert flagged
        assert max(abs(t["z"]) for t in flagged) < 3.0


class TestTheNegativeControl:

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_shuffled_columns_always_diverge(self, real, spec, seed):
        """The one absolute claim about the controls that actually holds."""
        assert certify_dataset(real, _independent(real, seed), spec)["certified"] is False

    def test_the_coefficients_the_data_pinned_down_always_fail(self, real, spec):
        for seed in range(3):
            cert = certify_dataset(real, _independent(real, seed), spec)
            by_name = {t["coefficient"]: t for t in cert["targets"]}
            for name in ("pay_delay_1", "utilization", "log_limit"):
                assert by_name[name]["preserved"] is False, f"{name} survived at seed {seed}"

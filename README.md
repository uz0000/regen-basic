# regen-synthetic

**A basic, generic synthetic table generator.** Give it a real table (or
several), get back a synthetic version — no label column, no rare-event
definition, no declared analysis required.

```bash
synth generate table1.csv table2.csv --n-rows 500 --out synth-output/
```

```python
from basic.generate import generate_table
result = generate_table(real_df, n_rows=500, seed=42)
result.synthetic_df          # the generated table
result.fidelity_passed       # did marginals + cross-column correlation come out right?
result.n_duplicates_guarded  # near-real-row copies caught and nudged away
```

Deterministic given a seed. No LLM, no network.

Also ships a **certifier** (`certify/`) — a separate, generator-agnostic
check for a sharper question than "does this look real": if someone runs a
real statistical analysis on this synthetic data, do they get the real
answer? See **The certifier**, below — including the honest result of
running it against this repo's own generator.

---

## What it does

1. **Infers each column's type** — continuous, categorical, binary, or
   identifier (a near-unique key like a customer ID) — from the data alone.
2. **Fits one joint Gaussian copula over the whole table**, continuous and
   categorical columns together, sharing a single latent correlation. This
   is the part that matters most: sampling each column independently would
   reproduce every column's own distribution while silently erasing the
   *relationships between columns* — the copula preserves both at once.
3. **Samples new rows from it.** Every value is drawn from a fitted
   distribution — never by perturbing a real row — so near-copies of real
   individuals are already rare by construction.
4. **Re-mints identifier columns** as fresh unique values rather than
   sampling them (a copula over a near-unique key would just reproduce
   noise), and **checks for verbatim duplicates** as a safety net on top of
   the by-construction privacy.
5. **Gates the result on fidelity**: per-column distribution match (TVD for
   categorical/binary, normalized Wasserstein for continuous) and the
   cross-column correlation delta — the check that catches a batch with
   right-looking columns but a scrambled relationship between them.

## Why the correlation check matters

A generator that only matches each column's own distribution can still be
badly wrong. Two columns that move together in the real data — income and
debt, say — need to move together in the synthetic data too, or anything
built on their relationship (a ratio, a model, a join) will be misleading
even though every individual column "looks right." This is the one check
worth understanding even if you skip everything else: **marginals matching
is not the same as the data being structurally right.**

## The certifier — does it actually hold up?

The correlation check above answers "does the synthetic data look
structurally right." It doesn't answer a sharper question: if someone runs
a real regression on this synthetic data, do they get the real answer? A
dataset can pass every fidelity and correlation check and still silently
shift the one coefficient a downstream model or decision depends on.

`certify/` answers that question directly, for any synthetic data —
generator-agnostic, doesn't ask who made it:

```python
from certify.certifier import certify_dataset
from contracts.types import EstimandSpec

estimand = EstimandSpec(outcome="default", predictors=["pay_delay_1", "utilization"], family="logit")
cert = certify_dataset(real_df, synthetic_df, estimand)
cert["certified"]  # True iff every declared coefficient is preserved
```

**Run the demo** (`python examples/certifier_demo/run_demo.py`) — real UCI
credit-default data, one logistic regression, three sources:

```
source                             certified    pay_delay_1   utilization     log_limit           age
-----------------------------------------------------------------------------------------------------
bootstrap   (positive control)     CERTIFIED   +0.714 ✓  -0.367 ✓  -0.329 ✓  +0.010 ✓
independent (negative control)     refused     -0.017 ✗  -0.031 ✗  +0.006 ✗  +0.001 ✗
generator   (this repo's output)   refused     +0.422 ✗  -0.224 ✗  -0.208 ✗  +0.009 ✓
```

The positive control (a plain bootstrap of the real data) certifies, and the
negative control (this generator's output with every column independently
shuffled — same marginals, correlation deliberately destroyed) is refused
across the board. That's the certifier working correctly: it isn't a rubber
stamp, and it isn't broken.

**The generator's own output is refused too — on 3 of 4 coefficients.** This
is the honest result, not the one that makes the best headline. Preserving
marginals and cross-column correlation (what `basic/generate.py` checks)
turned out not to be enough to preserve this particular downstream
regression. That's a real limitation of a joint-Gaussian-copula approach,
not a bug in either the generator or the certifier — a copula fits a
specific (Gaussian-latent) shape to the joint distribution, and a logistic
regression's coefficients can be sensitive to structure a Gaussian copula
doesn't capture. Fixing it would mean a different generation strategy tailored to preserving
a *declared* estimand specifically (modeling the predictor joint more
richly, and the outcome from a fitted conditional model rather than a
shared latent correlation) — a different, narrower tool than a basic,
generic generator, and out of scope here.

## What it's built from

Not a from-scratch sampler. The copula core and the fidelity/duplicate
checks were pulled from a larger rare-event synthetic-data project and
adapted to run against a whole table instead of a rare/normal split — reusing
code whose edge cases had already been found and fixed, rather than
rediscovering them from zero.

One edge case got rediscovered anyway, by testing against real data instead
of only toy fixtures: a binary column whose two real values weren't `{0, 1}`
(e.g. `{1, 2}`) collapsed to a single constant value in the synthetic
output, because the frequency-table code assumed 0/1-coded input. Fixed by
remapping binary columns to `{0, 1}` before encoding (`basic/generate.py`).

**What it deliberately does *not* do**: a δ-distance privacy floor (pushing
every synthetic row at least some distance from every real row) was tried
and removed. That mechanism is designed for a sparse reference set; applied
to a dense whole-table population, its fallback path for "no valid position
exists" resamples each dimension independently — which was measured to turn
a real 0.91 correlation into -0.29 in the synthetic output. The duplicate
guard (checked, not assumed) is the privacy mechanism here instead. Full
account in `basic/generate.py`'s docstring.

## Scope (v1)

- Single table at a time (call `generate_table` once per table, or
  `generate_tables` for several at once).
- No missing values — fails loudly with a clear error rather than silently
  corrupting the copula fit. Impute or drop nulls first.
- Not differential privacy: the duplicate guard prevents exact/near-exact
  reproduction of a real row's attributes; it does not bound aggregate or
  membership-inference attacks.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q
```

Three pinned dependencies: numpy, pandas, scipy. Pinned as a general
reproducibility discipline — determinism claims only hold within a fixed
dependency set, not across major versions. Not independently verified for
this code specifically: the one confirmed case of a version bump flipping a
result was in a different (since-removed) module of the source project that
depended on scikit-learn, which this codebase no longer uses at all.

## License

MIT

# regen-synthetic

**Synthetic tabular data generation.** Two layers, in order of how basic they are:

1. **A basic generic table generator** (`basic/`, the `synth` CLI) — give it
   any table (or several), get back a synthetic version that preserves each
   column's distribution *and* the relationships between columns. No label
   column, no declared analysis, no financial framing required.
2. **Advanced: rare-event generation + estimand certification for financial
   risk/scoring data** (`engine/`, `regen/`, the `regen` CLI) — built for
   credit/fraud/underwriting specifically, including a certifier that checks
   whether a *declared conclusion* (a regression coefficient) survives.

Start with the basic generator below unless you specifically need the
financial-specific machinery — the advanced layer solves a narrower problem
(rare-event imbalance, a declared coefficient) that most table-generation
needs don't actually have.

Deterministic throughout — no LLM in the value or verification path.
Single-table only (see **Scope**, below).

---

## The basic generator

```bash
synth generate table1.csv table2.csv --n-rows 500 --out synth-output/
```

or from Python:

```python
from basic.generate import generate_table
result = generate_table(real_df, n_rows=500, seed=42)
result.synthetic_df       # the generated table
result.fidelity_passed    # did marginals + cross-column correlation come out right?
result.n_duplicates_guarded  # near-real-row copies caught and nudged away
```

What it does: infers each column's type (continuous / categorical / binary /
identifier), fits a joint Gaussian copula over the whole table — continuous
and categorical columns together, sharing one latent correlation — and
samples new rows from it. Identifier columns (near-unique keys) are detected
and re-minted as fresh values rather than sampled, since a copula over a
near-unique column would just reproduce noise. Every value is drawn from a
fitted distribution, never from perturbing a real row, and a duplicate guard
checks no synthetic row ends up a verbatim copy of a real one anyway.

This isn't a from-scratch sampler: the copula core and the fidelity/duplicate
checks are the same functions the advanced (financial) engine below uses
internally, applied to the whole table instead of a rare/normal split —
reusing code that already had its edge cases found and fixed, rather than
rediscovering them. One of those edge cases got rediscovered anyway during
testing: a binary column whose two real values weren't `{0, 1}` (e.g. `{1, 2}`)
collapsed to a single constant value in the synthetic output, because the
frequency-table code assumed 0/1-coded input. Fixed in `basic/generate.py`
by remapping binary columns to `{0, 1}` before encoding — worth knowing about
if a generated binary column ever looks flat.

**What it deliberately does *not* do**: apply the advanced engine's
δ-distance privacy floor. That floor is designed for a sparse rare-event
reference set; applying it to a dense whole-table population was tested and
found to *corrupt* the cross-column correlation this generator exists to
preserve (its "saturated box" fallback respawns rows by sampling each
dimension independently — measured turning a real 0.91 correlation into
-0.29 in the synthetic output). The duplicate guard is the privacy mechanism
here; see the docstring in `basic/generate.py` for the full account.

**Scope (v1):** no missing values (fails loudly, doesn't silently corrupt the
fit — impute or drop nulls first), single table only.

---

## Advanced: financial risk/scoring data

Everything below is the narrower, financial-specific layer: rare-event
generation and a certifier for whether a *declared* regression/logit
coefficient survives. Skip this section unless that's specifically what you
need.

## What "preserves" means here — read this before the rest

**"Preserving the relationships in the real data" is not REGEN's claim, and it's
not enough — that's the entire reason this project exists.** The Gaussian copula
in the demo below *is* a preserve-the-correlations method — marginals plus a
covariance matrix, which is what "relationships" usually means — and it still
gets refused, because a correlation matrix looking right in aggregate doesn't
mean the one coefficient a credit model actually depends on survives. General
fidelity and specific analytical utility turned out to be different things.

**What REGEN generates and certifies instead is narrower and verified, not
broad and assumed: a specific, declared conclusion — one regression or logit
coefficient you name up front — either provably survives in the synthetic data,
or the certifier tells you it doesn't.** It makes no claim about relationships
you didn't declare. That's a deliberately smaller promise than "this data is
generally useful," and it's smaller on purpose: a verified narrow claim is worth
more than an assumed broad one, especially for data a financial decision gets
built on.

---

## The certifier

```python
from regen.certifier import certify_dataset
from contracts.scenario import EstimandSpec

estimand = EstimandSpec(outcome="default",
                        predictors=["pay_delay_1", "utilization", "log_limit", "age"],
                        family="logit")

cert = certify_dataset(real_df, synthetic_df, estimand)
cert["certified"]     # True iff every declared coefficient is preserved
cert["targets"]       # per-coefficient: θ_real vs θ_synth, the two-sample test, preserved?
```

It's **generator-agnostic** — it never asks who made the data — **per-coefficient**,
and **portable**: the certificate carries θ_real ± SE, so a third party can
recompute θ_synth from the synthetic data alone and re-check the verdict *without
the real rows*. This is the "attach a trust certificate to synthetic data you
share" model, and it's what a compliance or model-risk reviewer actually needs:
proof that's recomputable independently of who produced the data.

### The demo — real credit data, seven producers

`python examples/certifier_demo/run_demo.py`, one logistic regression, real UCI
credit-default data, verified on this repo's pinned dependencies:

```
source                              certified     pay_delay_1   utilization     log_limit           age
bootstrap_real  (positive control)  CERTIFIED        +0.649 ✓      -0.395 ✓      -0.369 ✓      +0.010 ✓
independent_cols(negative control)  refused          -0.007 ✗      -0.010 ✗      -0.010 ✗      -0.001 ✗
noised_real     (0.5σ anonymise)    refused          +0.521 ✗      -0.140 ✗      -0.294 ✓      +0.007 ✓
gaussian_copula (marginals+corr)    refused          +0.464 ✗      -0.314 ✓      -0.225 ✗      +0.011 ✓
SMOTE           (imblearn)          refused          +0.608 ✗      -0.469 ✓      -0.353 ✓      +0.015 ✓
REGEN           (this repo, v1)     refused          +0.932 ✗      -0.194 ✓      -0.339 ✓      +0.009 ✓
estimand_preserving (v2)            CERTIFIED        +0.660 ✓      -0.297 ✓      -0.258 ✓      +0.009 ✓
```

Every generic method — including this repo's own privacy-first generator —
silently breaks the strongest predictor while fidelity and prediction checks
flag none of it. Only a faithful source and the estimand-preserving generator
certify. Full write-up: [`examples/certifier_demo/README.md`](examples/certifier_demo/README.md).

---

## The generator — a privacy ↔ conclusion-fidelity dial, not a single answer

There is no free lunch between preserving a financial model's conclusions and
protecting the privacy of the people in the training data — the two modes here
sit at different points on that tradeoff, and the certifier tells you exactly
where:

- **`engine/` (v1, privacy-first)** — a rare-event active-learning campaign:
  Scout (targeting) → Prior (grounded sampling) → Amplifier (tail correction) →
  Auditor (fidelity gate) → Examiner (detection lift). Privacy is on by default
  (δ-distance floor + verbatim guard + k-anonymity — **not** differential
  privacy). Strong on rare-event fidelity; **its own coefficient gets refused by
  the certifier above** — treat it as fidelity-grade, not conclusion-grade.

  ```bash
  regen generate my_data.csv --label is_fraud     # generate a synthetic dataset
  regen doctor   my_data.csv --label is_fraud     # preflight: fits the envelope?
  regen verify   regen-output/                    # independently recompute a batch's stats
  ```

- **`regen/estimand_preserving.py` (v2, conclusion-first)** — models the
  predictor joint with a Gaussian mixture (novel rows, not perturbed real ones)
  and draws the outcome from a calibrated model of the real conditional P(y|x)
  — never the declared coefficient, so nothing is injected. Certifies where
  every generic method above doesn't. The honest cost: staying faithful to the
  real joint trades away privacy *distance* — novel rows, but nearer real than a
  strong δ-floor would allow.

  ```python
  from regen.estimand_preserving import generate_estimand_preserving
  synth = generate_estimand_preserving(real_df, estimand, n_rows=6000)
  ```

Every batch — either mode — ships a manifest (seed + config + code version +
SHA-256 of every artifact) and an `explanation.json` (per-gate verdicts,
per-column provenance/mechanism, the privacy account), so a reviewer gets a
recomputable account of what they're holding, not just a claim.

---

## Scope & honesty

- **Single-table, cross-sectional** tabular only. **Not** time-series or
  relational — rows are treated as exchangeable; temporal structure (a
  transaction sequence, a price series) is not modeled. Scope this to
  point-in-time risk/scoring data (credit underwriting, fraud snapshots,
  actuarial tables), not transaction streams.
- **Estimand certification v1 is numeric predictors + OLS/logit only.**
  Financial data is full of categoricals (industry code, loan type, state) —
  not yet certifiable, only generatable.
- **Privacy is a δ-distance floor + k-anonymity + verbatim guard, not
  differential privacy.** It prevents near-copy re-identification, not
  membership-inference or aggregate attacks.
- A declared analysis is required for certification — you certify what you tell
  it matters. That's what makes the guarantee precise instead of a vague "looks
  fine."

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python examples/make_sample_data.py   # generates examples/transactions.csv used by the test suite
pytest tests/ -q
```

Dependencies are pinned (`requirements.txt`) — determinism and the recomputable
certificate/manifest claims hold within a fixed dependency set. A newer
scikit-learn/numpy shifts GMM/GBM outputs enough to flip borderline
certification results; this was caught by CI-equivalent testing, not assumed.

## License

MIT

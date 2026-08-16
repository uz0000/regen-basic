# regen-financial

**Synthetic tabular data generation + certification for cross-sectional financial
risk and scoring data — credit, fraud, underwriting.**

Synthetic data can pass every fidelity and prediction check while silently
breaking the conclusion you'd draw from it. A credit or fraud model *is* a set of
regression coefficients; data that "looks real" but shifts those coefficients is
worse than useless. This repo ships two things: a generator built for the rare,
imbalanced events financial data is full of, and a certifier that tells you —
per coefficient — whether a declared analysis survives on whatever synthetic
data you actually used.

Deterministic and recomputable throughout — no LLM in the value or verification
path. Single-table, cross-sectional tabular only (see **Scope**, below).

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

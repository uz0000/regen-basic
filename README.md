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

Three pinned dependencies: numpy, pandas, scipy. Pinned because determinism
holds within a fixed dependency set, not across major versions — a
newer numpy/scikit-learn was previously found (on an earlier version of this
codebase) to shift results enough to flip a borderline test.

## License

MIT

# How this simulator compares

Run 2026-08-16. Every number below was produced by the two scripts in this
directory and can be reproduced with them; nothing is quoted from a paper or
a vendor page.

**Independently re-run 2026-08-16** against a freshly created SDV environment
(SDV 1.38.0) and a freshly created project virtualenv. Every figure in both
tables reproduced exactly — same correlation deltas, same category deltas, same
conclusion-kept counts. That is the claim the pinning discipline makes, checked
rather than assumed.

Each generator ran in whatever environment it is supported in — SDV needs
newer numpy than this project pins, so it gets its own — and then **one
scorer graded every output under one dependency set.** That matters: if the
scoring itself ran under different numpy versions per tool, a difference
between generators could really be a difference between their libraries.

## The generators

| | what it is |
|---|---|
| `bootstrap` | **positive control.** Resamples the real rows. Not usable as synthetic data — it *is* the real data — but it marks the ceiling. |
| `independent` | **negative control.** Samples each column separately. Every column's own distribution is perfect and every relationship between columns is destroyed. |
| `regen-basic` | this repo — one joint Gaussian copula over the whole table. |
| `sdv-copula` | [SDV](https://sdv.dev)'s `GaussianCopulaSynthesizer`. The closest peer, and the most widely used synthetic-tabular library there is. |
| `sdv-ctgan` | SDV's `CTGANSynthesizer` — a GAN. A genuinely different approach, and it encodes categories one-hot rather than ranking them. |

## Table 1 — does a real conclusion survive?

Real public table, 30,000 rows, all numeric. The declared analysis is
`logit  default ~ pay_delay_1 + utilization + log_limit + age`, and the
question is whether its coefficients come back the same.

| generator | columns off | correlation Δ | **conclusion kept** | time |
|---|---|---|---|---|
| bootstrap *(control)* | 0 | 0.006 | **4/4** | — |
| independent *(control)* | 0 | 0.177 | **0/4** | — |
| **regen-basic** | 0 | **0.037** | **1/4** | **0.4s** |
| sdv-copula | 0 | 0.064 | **1/4** | 4.1s |
| sdv-ctgan *(60 epochs)* | 0 | 0.091 | **0/4** | 91s |

**No practical generator preserves the conclusion.** The best any of them
manages is one coefficient out of four. The industry-standard copula scores
exactly what this repo scores. CTGAN, the fundamentally different approach,
does worse here and takes about 200× longer.

Notice also that every generator passes the checks people usually rely on —
zero columns off, correlation deltas comfortably inside any reasonable
threshold — while getting the actual analysis wrong. That gap between "looks
right" and "is right" is the entire subject of this repo, and it reproduces
across every tool tested, not just this one.

## Table 2 — do relationships across a category survive?

2,000-row demo table where regions genuinely differ in temperature.
`category Δ` is the worst single category-to-measurement relationship, on a
0–1 scale where 0 is perfect. The gate is 0.10.

| generator | columns off | correlation Δ | **category Δ** | time |
|---|---|---|---|---|
| bootstrap *(control)* | 0 | 0.014 | **0.014** | — |
| independent *(control)* | 0 | 0.218 | **0.657** | — |
| **regen-basic** | 0 | **0.039** | **0.575** | **0.02s** |
| sdv-copula | 0 | 0.060 | **0.569** | 1.7s |
| sdv-ctgan (100 epochs) | 3 | 0.186 | 0.646 | 18s |
| sdv-ctgan (300 epochs) | 1 | 0.072 | 0.465 | 30s |
| sdv-ctgan (600 epochs) | 0 | 0.093 | **0.370** | 50s |

Two things worth separating here.

**Both copulas are barely better than the control that destroys everything.**
0.575 and 0.569 against 0.657 — they recover roughly a tenth of the gap.
This is the limitation documented in the main README: to put a category into
a correlation-based model it has to become a number, the numbering is
alphabetical, and a relationship that doesn't happen to run in alphabetical
order can't be expressed. SDV's copula has it for the same structural reason
ours does. It is a property of the approach, not of either implementation.

**CTGAN, trained properly, is clearly better here** — 0.370, roughly halving
the error, with no columns off. It earns that by one-hot encoding categories
instead of ranking them, so the alphabetical trap doesn't apply. The result
only appears with enough training: at 100 epochs it is worse than everything
including the control, and it would have been easy (and wrong) to stop there
and report that GANs handle categories badly. It costs roughly 2,500× the
time (50s against 0.02s).

And still nobody passes. 0.370 is far outside the 0.10 gate. A relationship
this simple — three regions with clearly different temperatures — is not
faithfully reproduced by any general-purpose tool tested.

## What this says about this repo

Honestly read, three things:

1. **It is competitive.** Against the most widely used library in this space
   it ties on preserving the conclusion, is better on numeric correlation
   structure (0.037 vs 0.064), and runs about ten times faster. Not
   revolutionary — it is the same family of method — but not a toy either.
2. **Its weaknesses are shared, not unique.** The categorical limitation and
   the conclusion failure both reproduce in SDV. That reframes them: not
   "this implementation is weak" but "this is what this class of method
   does," which is worth knowing before trusting any of them.
3. **The difference is that these numbers exist at all.** Every failure above
   was found by checks this repo ships and runs by default. SDV reports no
   verdict on whether a declared analysis survives, because measuring that
   requires being told which analysis matters. The contribution here is not
   a better generator — it is refusing to report success without checking
   the thing that decides whether the data was any use.

## Limits of this comparison

- Two tables and one declared analysis. Enough to show the pattern is not
  unique to this implementation; not enough to call it a general law.
- CTGAN was given up to 600 epochs on the small table and only 60 on the
  large one. Since more training measurably improved it on the small table,
  its large-table result is likely understated — that epoch budget was set
  by patience, not by convergence. Treat its 0/4 as "not shown to do better
  here", not as a ceiling.
- Default settings throughout. Both SDV synthesizers can be tuned, and no
  tuning was attempted for any tool, including this one.
- One seed per generator. The differences between the copulas are small
  enough that seed variation could reorder them; the gap between any
  generator and the controls is far too large for that to matter.

## Reproducing

```bash
# stage 1 — other tools, in their own environment (SDV needs newer numpy)
python3 -m venv /tmp/sdv-env && /tmp/sdv-env/bin/pip install sdv
/tmp/sdv-env/bin/python examples/comparison/generate_baselines.py \
    --input examples/decision_check/credit_default.csv --out /tmp/baselines
/tmp/sdv-env/bin/python examples/comparison/generate_baselines.py \
    --input examples/readings.csv --out /tmp/baselines

# stage 2 — score everything identically, in this project's environment
python examples/comparison/score_comparison.py --baselines /tmp/baselines
```

Stage 2 runs on its own too, without stage 1 — it just compares this repo
against the two controls.

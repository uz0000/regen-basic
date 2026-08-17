# regen-basic

[![tests](https://github.com/uz0000/regen-basic/actions/workflows/tests.yml/badge.svg)](https://github.com/uz0000/regen-basic/actions/workflows/tests.yml)

**If you simulate data instead of using the real thing, and then make a
decision from it — do you reach the same conclusion the truth would have
given you?**

That question is the whole point of this repo. Synthetic data is usually
justified by necessity: the real records are private, restricted, expensive,
or simply not allowed to leave the building, so you work with a stand-in.
The stand-in is normally judged by whether it *looks* like the real data.
But looking right and *supporting the same conclusion* are different
properties, and the second one is what actually matters the moment anybody
acts on the result.

So this repo is built as two halves that argue with each other:

| | | |
|---|---|---|
| **`simulate/`** | build the stand-in | model a real table's joint distribution and draw new rows from it |
| **`certify/`** | check the conclusion | fit the same analysis on real and simulated data, and see whether it lands in the same place |

The second half exists to hold the first half to account. It is deliberately
independent of it — it never asks who produced the data, so it works just as
well on output from any other tool.

Deterministic given a seed. No LLM, no network. Three dependencies.

**Relationship to [`regen-synth`](https://github.com/uz0000/regen-synth).** That
repo is the larger system the same question came out of: rare-event
amplification, a privacy floor, an audit bundle, and a certifier that extends to
a second generator built specifically to preserve a declared analysis. This repo
is the compact version — the simulator and the check, three dependencies, no
model call anywhere.

Install them into **separate environments**. Both claim the top-level package
names `engine`, `contracts` and `cli`, so a shared environment resolves those
imports to whichever was installed last and the CLIs shadow each other.

---

## Simulating a table

```bash
synth generate mytable.csv --n-rows 500 --out synth-output/
```

```python
from simulate.generate import generate_table

result = generate_table(real_df, n_rows=500, seed=42)
result.synthetic_df                   # the stand-in
result.fidelity_passed                # did every check pass?
result.correlation_delta              # how far the numeric relationships drifted
result.categorical_association_delta  # the worst category-to-measurement drift
result.categorical_worst_pair         # which pair that was
result.n_duplicates_guarded           # rows that landed on a real record, moved off
```

The simulator infers what each column *is* (a measurement, a category, a
yes/no flag, or an identifier), fits **one joint model over the whole
table**, and samples fresh rows from it.

The word *joint* is carrying the weight there. The easy way to build
synthetic data is column by column: learn what values `age` takes, learn
what values `income` takes, then draw each independently. Every column comes
out with a perfect distribution and the table is nonsense — because in the
real data those two columns move together, and nothing in that procedure
ever recorded that they did. Anything downstream that depends on the
relationship (a ratio, a model, a filter, a join) is now quietly wrong while
every column-level check reports success.

A copula avoids this by separating the two things being learned: what each
column looks like on its own, and how the columns move together. Both are
fit, and both are reproduced — for numeric columns, very well. For
categorical ones, only partly, which is documented honestly below rather
than glossed.

Identifier columns get special treatment — they're re-minted as fresh unique
values rather than sampled, since drawing from a column where every value
appears once just reproduces noise, and reusing a real ID is the single
strongest way to re-identify someone.

### What the output checks tell you

```
$ synth generate examples/readings.csv --n-rows 2000 --out synth-output/
readings: 2000 real rows -> 2000 synthetic rows
  fidelity: FAIL
    numeric correlations   delta 0.037  (limit 0.25)  ok
    category vs. measure   delta 0.595  (limit 0.1)  TOO FAR [worst: temp_c by region]
  privacy: 0 row(s) landed on a real record and were moved off
```

That is the sample table failing, on purpose and correctly. The numeric
relationships came through almost exactly — temperature and humidity really
do correlate at -0.87 in the real table and -0.88 in the simulated one. But
the real table's regions have distinctly different temperatures, and in the
simulated table they mostly don't, so the check refuses to call it good.

Two things are worth noticing. First, **the check reports the worst single
pair and names it**, rather than averaging across all of them — an average is
how one badly broken relationship disappears behind nine intact ones.
Second, the tool tells you this instead of reporting success, which is the
entire point; a version of this that only checked numeric correlations
passed the exact same table cleanly, because a correlation needs two numbers
and a category isn't one.

**The underlying limitation, stated plainly**: to fit a category into a
correlation-based model it has to be turned into a number, and the numbering
is alphabetical, which is arbitrary. So the model can only express a
category-to-measurement relationship that happens to run in alphabetical
order. In the sample table it doesn't — the coldest region is `north`, third
alphabetically — so the relationship cannot be represented, only detected
and reported. If your table has meaningful differences between categories,
expect this check to fire, and treat it as real.

### Exit codes

A failed check is not a crash: the tool ran correctly and is telling you the
result is bad. A pipeline watching only for crashes would sail straight past
that, so the two are separate codes.

| code | meaning |
|---|---|
| `0` | every table simulated and passed every check |
| `1` | at least one table **failed** a check — the data was still written, so you can inspect it |
| `2` | could not run (missing file, or input the simulator refuses); nothing written |

```bash
synth generate mytable.csv --out synth-output/ || echo "don't trust that output"
```

Pass `--allow-fail` to get exit `0` regardless when you want the data
anyway. Note that `1` still writes the output — a verdict of "this isn't
faithful" is more useful with the evidence attached than without it.

**On privacy**: no synthetic row is built by perturbing a real row — every
value is drawn from a fitted distribution — so near-copies are already rare
by construction, and a duplicate guard checks for them anyway. This is *not*
differential privacy; it prevents reproducing a real record's attributes, and
makes no claim about what can be inferred in aggregate.

## Checking the conclusion

Fidelity checks answer "does this look right." They cannot answer "would
someone acting on this be right," because that depends on a specific
analysis, and no generator knows which analysis you have in mind. So you
declare it:

```python
from certify.certifier import certify_dataset
from contracts.types import EstimandSpec

# The conclusion you actually care about, whatever your table is about.
analysis = EstimandSpec(outcome="recovered", predictors=["dose", "age"], family="logit")

cert = certify_dataset(real_df, synthetic_df, analysis)
cert["certified"]   # did every coefficient survive the round trip?
cert["targets"]     # per coefficient: truth vs. simulated, and whether they agree
```

It reports **per coefficient**, not one overall score, because that is where
the useful information is. A dataset can preserve three relationships and
break the fourth — and if the broken one is the one driving your decision,
an aggregate "87% similar" would have hidden exactly the thing you needed to
know.

The certificate carries the real estimate and its uncertainty — aggregate
numbers, not rows — so someone holding only the synthetic data can re-run
the check themselves and confirm the verdict, without ever seeing the real
records.

## What happened when the check was pointed at the simulator

`python examples/decision_check/run_demo.py` — a real public dataset of
30,000 people, one regression, three sources. The subject matter is
incidental; substitute any table where someone fits a model and acts on the
coefficients.

```
source                             verdict      pay_delay_1   utilization     log_limit           age
-----------------------------------------------------------------------------------------------------
TRUTH (the real data)              —           +0.714    -0.369    -0.314    +0.010
bootstrap   (positive control)     matches     +0.714 ✓  -0.367 ✓  -0.329 ✓  +0.010 ✓
independent (negative control)     DIVERGES    -0.017 ✗  -0.031 ✗  +0.006 ✗  +0.001 ✗
simulator   (this repo's output)   DIVERGES    +0.422 ✗  -0.224 ✗  -0.208 ✗  +0.009 ✓
```

The two controls confirm the checker works. A plain resample of the real
data reproduces the conclusion, as it must — if that failed, the checker
would be broken. The negative control is the simulator's own output with
each column then shuffled on its own, so the distributions are *identical*
and only the joint structure is destroyed; it diverges on everything, which
is precisely the failure the correlation check was built to catch.

**And the simulator's own output diverges too — on three of four
coefficients.** Its own fidelity check passed: distributions matched,
correlation delta 0.037, comfortably inside threshold. Anyone trusting that
verdict would have shipped it. The independent check says a decision made
from that data lands somewhere different from the truth — `pay_delay_1`
comes back at +0.42 against a real +0.71, an effect understated by 41%.

This is the honest result rather than the flattering one, and it is the most
useful thing in the repo. It is not a bug in either half; it is a real
property of this class of generator. A copula reproduces a particular
*shape* of dependence, and a logistic regression's coefficients depend on
structure that shape doesn't fully capture. Matching distributions and
matching correlations is genuinely not sufficient for matching conclusions.

The narrower generator that *would* close this gap — one that models the
predictors more richly and draws the outcome from a fitted conditional model,
built against a declared analysis rather than general resemblance — is a
different tool than a general-purpose simulator, and isn't in this repo.

**What to take from it**: if you simulate data and something downstream
depends on a specific relationship in it, verify that relationship
explicitly. Do not infer it from the fact that the data looks right.

## How it compares to other tools

Full numbers and method: [`examples/comparison/RESULTS.md`](examples/comparison/RESULTS.md).
Measured against SDV — the most widely used synthetic-tabular library —
using both its Gaussian copula and CTGAN, plus the two controls.

On the real 30k-row table with a declared analysis:

| generator | correlation Δ | **conclusion kept** | time |
|---|---|---|---|
| bootstrap *(control)* | 0.006 | **4/4** | — |
| independent *(control)* | 0.177 | **0/4** | — |
| **regen-basic** | **0.037** | **1/4** | **0.4s** |
| sdv-copula | 0.064 | **1/4** | 4.1s |
| sdv-ctgan | 0.091 | **0/4** | 91s |

Three things this says, none of them flattering by default:

**No practical generator preserves the conclusion** — the best result from
any of them is one coefficient out of four, and every one of them passes the
ordinary quality checks while doing it. The gap between "looks right" and
"is right" is not a quirk of this implementation; it reproduces across the
field.

**This repo is competitive but not special.** It ties the industry standard
on the conclusion, edges it on correlation structure, and runs ten times
faster. It is the same family of method, and it inherits the same
weaknesses — including the categorical one, which SDV's copula has for
exactly the same structural reason.

**CTGAN genuinely beats both copulas on categorical structure** once given
enough training (0.370 against 0.575, on the second table in the results),
because it doesn't rank categories. It still fails the threshold, and costs
~2,500× the time. Worth knowing rather than hiding: the approach used here
is not the best available at that particular thing.

What isn't available elsewhere is the middle column. SDV ships no verdict on
whether a declared analysis survived — measuring that requires being told
which analysis matters. That is the contribution: not a better generator, but
refusing to report success without checking the thing that decides whether
the data was any use.

## Reading the code

```
simulate/generate.py     the simulator: infer column kinds, fit, sample, check
certify/certifier.py     the verdict: same analysis on both, per-coefficient
certify/estimand.py      the statistics: OLS and logistic fits, from scratch
engine/prior/grounded.py the copula — marginals and dependence, fit separately
engine/auditor/          the fidelity checks (per-column, correlation, category)
engine/privacy.py        the duplicate guard
engine/ingest/loader.py  column-kind and identifier inference
examples/decision_check/ the demo above
examples/comparison/     the head-to-head against SDV and CTGAN
```

The regression fits in `certify/estimand.py` are written directly against
numpy/scipy rather than pulled from a stats library, so the check has no
dependency on a solver whose behavior could drift between versions.

## Scope and limits

- **One table at a time.** `generate_table` for one, `generate_tables` for
  several; relationships *between* tables aren't modeled.
- **Category-to-measurement relationships survive only partly**, and only
  when they happen to run in alphabetical order of the category labels. The
  check detects the shortfall and names the pair; the generator cannot
  currently fix it. Detected, reported, not solved.
- **No missing values.** Fails with a clear error rather than silently
  distorting the fit — impute or drop first.
- **The check covers numeric predictors, OLS and logistic regression.** Other
  analyses are not certifiable here.
- **Rows are treated as independent.** Nothing sequential — no time series,
  no ordering, no per-entity history.
- **Not differential privacy.** See the privacy note above.
- **One dataset's worth of evidence.** The finding above is a single real
  dataset and a single analysis. It is a real result, not a general law; the
  honest next step is running the same check across more tables and more
  analyses to see how far the pattern holds.

## Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest tests/ -q                        # 35 tests
python examples/make_sample_data.py     # a demo table to try the CLI on
synth generate examples/readings.csv --n-rows 500 --out synth-output/
```

That last command exits `1` on purpose — the demo table contains a
relationship the simulator can't reproduce, and the checks say so. See
**Exit codes** above.

Dependencies are pinned (numpy, pandas, scipy) as reproducibility discipline:
identical seeds only guarantee identical output within a fixed dependency
set. That is not just asserted here — CI runs the suite and both examples on
Linux across Python 3.10, 3.11 and 3.12, none of which is the machine this
was written on. It matters more than it might sound: the tests assert numeric
thresholds on seeded draws, and those are exactly the assertions that can
drift with a different platform or numerical backend. They currently don't.

Worth knowing what that does *not* cover: only these pinned versions are
tested, so the pins are known-good rather than known-necessary. The one
confirmed case of a version bump changing a result was in an earlier,
since-removed module that used scikit-learn, which this codebase no longer
depends on at all.

## License

MIT

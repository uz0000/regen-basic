# The decision check

One script. It asks the question the whole repository is built around: if you
simulate data and then make a decision from it, do you reach the same conclusion
the truth would have given you?

The interpretation lives in [the README](../../README.md#the-finding-no-generator-preserves-the-conclusion--this-one-included).
This file is about running it and reading the output.

| Script | Question | Writes |
|---|---|---|
| `run_demo.py` | Does a declared regression survive the round trip through simulated data? | [`RESULTS.md`](RESULTS.md) |
| `mechanism_check.py` | Is the explanation for *why* it fails actually true? | [`MECHANISM.md`](MECHANISM.md) |

```bash
python examples/decision_check/run_demo.py
python examples/decision_check/mechanism_check.py
```

Run from the repository root. Both are deterministic given the seed, and both
write their table with a generated-file header, so the prose that links to them
cannot drift from a run.

## The data

`credit_default.csv` is the UCI *Default of Credit Card Clients* table — 30,000
accounts, a 22.1% default rate. The subject matter is incidental. Substitute any
table where someone fits a model and acts on the coefficients.

## The analysis under test

```
logit    default ~ pay_delay_1 + utilization + log_limit + age
```

This is declared up front, and it has to be. No simulator can guess which
relationship in a table you plan to act on, and "check everything" is not a
question anyone can answer — a table supports an unlimited number of analyses.

## The three sources

**Two controls, which exist to show the check discriminates.** `bootstrap`
resamples the real rows and should match. `independent` takes the simulator's own
output and shuffles each column on its own, which leaves every column's
distribution *identical* and destroys only the joint structure — it must diverge.

Neither is absolute. Matching needs all four coefficients to agree at once and
each is a 95% test, so `bootstrap` is refused on about **3.7% of seeds** (11 of
300) from chance alone. The committed table uses seed 2, which matches; a
divergence on another seed is not a broken checker, it is four 95% tests doing
what four 95% tests do. Which seeds diverge is not stable across machines and is
deliberately not listed — a chance divergence sits within about 0.1 of the
cutoff, and a different numeric build moves a borderline score by that much. The
arithmetic is in
[`../../MATH.md`](../../MATH.md), the correction in
[`../../CORRECTIONS.md`](../../CORRECTIONS.md) entry 4.

**One real generator.** `simulator` is this repo's output, from `simulate/`.

## Reading the output

One row per source, one column per coefficient. A check mark means the estimate
is statistically consistent with the real one; a cross means it is far enough
away to be a different conclusion. A source only "matches" when every declared
coefficient survives, so a row can be mostly check marks and still diverge. That
is intended: a conclusion is not partly true.

The thing to notice is that the simulator's *own* fidelity check passes first —
correlation delta 0.037, comfortably inside the limit — and then three of four
coefficients move anyway. Anyone trusting the fidelity verdict would have
shipped it.

`mechanism_check.py` then measures each factual claim the explanation rests on,
rather than arguing them: that the predictors overlap, that the strongest
predictor is a cliff rather than a slope, and that a copula flattens that cliff.
It reports whether each one holds.

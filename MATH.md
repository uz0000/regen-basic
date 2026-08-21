# The math behind the finding

This repo asks one question:

> **If you simulate data instead of using the real thing, and then make a
> decision from it, do you reach the same conclusion the truth would have given
> you?**

Answering it needs four things: a number that a decision actually rests on, a way
to say how sure you are of it, a way to judge whether two such numbers are the
same answer, and a way to check whether the simulated table looks right in the
first place. This file covers each — what problem it solves, how it works, and
what its output means.

**How to read this.** Every symbol is named in words before it's used, and every
formula is followed by what it means in practice. Skip the formulas and read only
the prose and the argument still holds. Sections 2 and 4 carry the finding.

---

## 1. Building the stand-in: why one joint model, not one model per column

**The problem.** You need a table that behaves like the real one. The easy way is
to go column by column: learn what values `age` takes, learn what values `income`
takes, then draw each independently.

**Why that fails.** Every column comes out with a perfect distribution and the
table is nonsense, because in the real data those two columns move together and
nothing in that procedure ever recorded that they did. Every column-level check
reports success while anything depending on the relationship — a ratio, a model, a
filter — is quietly wrong.

**How the math fixes it.** A **copula** separates two things that are usually
tangled together, and fits them independently:

1. **What each column looks like on its own** — its distribution.
2. **How the columns move together** — their dependence structure.

The mechanics, in three steps ([`engine/prior/grounded.py`](engine/prior/grounded.py)):

- **Rank each column and convert to a common scale.** Replace every value by its
  rank, divide by the row count, and map that through the bell curve. Every column
  now lives on the same standard scale regardless of its original units, so their
  relationships can be compared directly.
- **Measure and reproduce the dependence.** Compute how strongly the transformed
  columns move together, then draw fresh rows sharing that same structure.
- **Map back.** Send each drawn value back through the real column's own
  distribution, so every synthetic value lands on the real support — no invented
  categories, no impossible amounts.

**What this buys.** Relationships survive. On the sample table, temperature and
humidity correlate at −0.87 in the real data and −0.88 in the simulated one.

**What it cannot do, and why.** To fit a category like `region` into a
correlation-based model, it has to become a number — and the numbering is
alphabetical, which is arbitrary. So the model can only express a
category-to-measurement relationship that happens to run in alphabetical order. In
the sample table the coldest region is `north`, third alphabetically, so the
relationship cannot be represented at all. Section 3 is how the tool detects that
rather than shipping it silently.

---

## 2. The number a decision rests on, and how sure we are of it

**The problem.** "Does this table look right?" and "would someone acting on it be
right?" are different questions. The second depends on a specific analysis, and no
simulator can guess which one you have in mind. So you declare it.

**What gets declared.** A regression: a rule predicting one outcome from several
inputs at once. It produces one number per input, a **coefficient**, and each one
answers a narrow question:

> If this one input goes up by one unit, and every other input stays exactly where
> it is, how much does the outcome move?

For a yes-or-no outcome the same idea applies to the **odds** of yes. In the demo,
the coefficient on `pay_delay_1` — periods already behind — is **+0.714**.
Exponentiating turns that into plain language: e^0.714 ≈ 2.04, so one more period
behind roughly **doubles the odds** of default, holding the other inputs fixed.

That is what a lender acts on. Hold on to **"holding everything else fixed"** —
section 4 is about why that phrase is where simulation breaks.

**How sure are we?** Every coefficient comes with a **standard error**: if you
collected a fresh sample of the same size and refit, how much would this number
bounce? You need it because section 3 compares two coefficients that are *both*
uncertain.

Both come from [`certify/estimand.py`](certify/estimand.py), written directly
against numpy and scipy rather than pulled from a statistics library, so the check
does not depend on a solver whose behaviour could change between versions. For a
quantity outcome it's one calculation with no iteration. For a yes-or-no outcome
there is no one-shot formula, so the fit starts from a guess and improves it until
a round changes nothing meaningful.

The uncertainty then has a plain reading: **how sharply peaked the fit is**.
Picture the quality of fit as a hill with the best coefficients at the summit. A
sharp spike means only nearby coefficients fit well, so the answer is pinned down
and the standard error is small. A broad plateau means many coefficients fit about
as well, and the standard error is large.

---

## 3. Deciding whether two numbers are the same answer

**The problem.** The real data gives +0.714. The simulator gives +0.422. Those
differ — but *any* two estimates differ a little, just from sampling. Are they far
enough apart to be **different answers** rather than the same answer measured
twice?

**The rule that looks obvious, and is wrong.** You could ask whether the simulated
number falls inside the plausible range of the real one. That quietly rewards bad
simulators: one producing a *noisy* estimate has a wide range, so it's more likely
to overlap and pass. Being imprecise would help you.

**The rule used.** Both numbers are uncertain, and the two fits run on separate
datasets. When two independent uncertain quantities are subtracted, their
uncertainties combine like the sides of a right triangle — the squares add:

```
combined uncertainty  =  √( (real SE)²  +  (simulated SE)² )
```

The coefficient counts as **preserved** when the gap is no bigger than about two of
those combined units:

```
| real − simulated |   ≤   1.96 × combined uncertainty
```

Dividing one side by the other gives a single score, `z` — the gap measured in
units of "how much these numbers wobble anyway." `z = 1` means they differ by about
as much as noise alone explains, which is agreement. A large `z` means a genuinely
different answer.

**Why this shape is right.** It accounts for the uncertainty in *both* numbers, so
it doesn't wrongly fail data that genuinely preserved the estimate. And it contains
the naive rule as a special case: as the simulated dataset grows, its own
uncertainty shrinks toward zero and the rule collapses back into "is it inside the
real range?" The naive check isn't a different philosophy — it just applies that
limit before it's earned.

**Why every coefficient has to pass.** A source passes only if **all** declared
coefficients are preserved. An analysis whose third number is wrong is a wrong
analysis, and averaging four coefficients into one score would let a broken one
hide behind three intact ones. A conclusion is not partly true.

**What that costs, and it is not zero.** Each coefficient is its own 95% test, so
a genuinely faithful simulator is refused whenever any one of them lands in its 5%
tail — and the chance of that compounds with how many coefficients you declare.
Measured on the positive control, 300 seeds:

```
positive control refused   11 / 300  = 3.67%   (95% CI 1.5% - 5.8%)
```

Two things follow. A refused positive control is normal rather than a sign of a
broken checker — this repo previously documented it the other way round, which is
[`CORRECTIONS.md`](CORRECTIONS.md) entry 4. And a pass rate only means something
against a fixed declared analysis, since a two-predictor analysis is easier to
match than a ten-predictor one at the same underlying fidelity.

**One assumption worth naming.** The formula above adds the two uncertainties as
though the estimates were unrelated. They are not — the simulated table is built
from the real one, so the two move together, and the exact expression subtracts a
covariance term this code does not compute. Leaving it out makes the combined
uncertainty **too large**, the score **too small**, and the check biased toward
saying "preserved." Since the finding here is that simulators **fail**, that bias
understates the problem rather than inventing it — but it does mean the tool's
failure mode is false reassurance, which is worth stating plainly.

**Why a stranger can check it.** The certificate carries the real coefficient and
its standard error — two summary numbers, not any rows. Anyone holding only the
simulated table can refit, get their own two numbers, and recompute `z`. The
verdict is checkable by someone who never sees a real record.

---

## 4. Why the coefficient moves anyway

**The problem.** What does a coefficient actually depend on? Get those things right
and it survives simulation. Get either wrong and it doesn't.

**The answer, in two parts.** A coefficient is built from two ingredients:

```
coefficients  =  (how the inputs relate to each other)⁻¹  ×  (how each input relates to the outcome)
```

**Ingredient 1 — how the inputs relate to each other.** This is where "holding
everything else fixed" lives. Holding one input fixed while moving another only
means something if you know how those two move together in reality. The `⁻¹`
matters: it mixes every input's relationships into every coefficient, so an error
anywhere spreads into all of them.

**Ingredient 2 — how each input relates to the outcome.** Rows can sit in exactly
the right places and still carry the wrong outcome rate at each place.

**Why a copula breaks ingredient 1.** Section 1 described the copula's dependence
step as a single number per pair for how strongly two columns move together. That
works when the relationship is a steady trend. Being behind on payments isn't one.
Measured on the real table
([`examples/decision_check/MECHANISM.md`](examples/decision_check/MECHANISM.md)),
the share of accounts that default runs:

```
not behind      0.128
1 period behind 0.339
2 periods       0.691
```

Flat, then a **jump**. One number describing a straight-line relationship cannot
express a jump, so the copula renders the cliff as a gentle slope. Its correlation
with the outcome comes out at +0.21 against a real +0.33, and the coefficient
shrinks from +0.714 to +0.422 — the strongest warning sign in the data,
**understated by 41%**.

The simulator's own fidelity check passed first, at a correlation delta of 0.037.
Anyone trusting that verdict would have shipped it.

**The point.** Matching distributions and matching correlations is genuinely not
sufficient for matching conclusions. There is no rule of the form "if the
correlation delta is under 0.05, the coefficient moved less than X" — the
quantities aren't connected.

---

## 5. What the fidelity checks measure, and what they can't

These run automatically on every generated table. Each measures something real;
none of them measures the coefficient, which is why section 3 exists as a separate
step. All are in [`engine/auditor/fidelity.py`](engine/auditor/fidelity.py).

**Does each column look right?** Two measures, depending on the column type. For a
quantity, the distance between the real and synthetic distributions, divided by the
real spread so one threshold works whatever the units. For a category, half the
total gap between the two sets of proportions — 0 when identical, 1 when they share
nothing.

**Do the numeric columns move together correctly?** The **average** absolute change
across every pair of numeric columns:

```
correlation delta  =  average over pairs of | real correlation − synthetic correlation |
```

Threshold 0.25.

**Do the categories still differ from each other correctly?** This is the check a
correlation matrix structurally cannot perform, because a correlation needs two
numbers and a category isn't one. Instead, for every (category, measurement) pair,
measure the share of the measurement's variation explained by the category:

```
                 variation between category averages
association  =  ─────────────────────────────────────
                        total variation
```

0 means knowing the category tells you nothing about the measurement. 1 means it
tells you everything. This is the categorical counterpart of a squared correlation,
and — the reason it's used here — **it needs no ordering of the categories**, which
matters because a nominal category has none. Section 1's alphabetical-ordering
limitation is exactly what it is built to catch. The score is the largest change
across all pairs, with a threshold of 0.10.

**Why the worst pair, not the average.** Averaging is how a real problem
disappears: one badly broken relationship among ten intact ones divides down to a
comfortable-looking number, and the broken one is exactly what you needed told
about. So this check reports the worst pair **and names it**.

On the sample table that produces:

```
numeric correlations   delta 0.037  (limit 0.25)  ok
category vs. measure   delta 0.595  (limit 0.1)  TOO FAR [worst: temp_c by region]
```

which is the tool correctly refusing to call its own output good.

---

## 6. The whole argument on one page

1. Building a stand-in column by column destroys the relationships between
   columns, so one joint model is fitted instead (§1).
2. Decisions are made from a coefficient, which comes with a standard error saying
   how much it would wobble on a fresh sample (§2).
3. Two coefficients are the same answer when they differ by less than their
   combined wobble — one division, and a stranger holding only the simulated data
   can redo it (§3).
4. A coefficient depends on how the inputs relate to each other and on how they
   relate to the outcome. A copula gets the first one wrong wherever a relationship
   jumps rather than trends (§4).
5. The fidelity checks measure distributions and relationships, not coefficients,
   and nothing connects the two (§4, §5). So a table can pass every check and still
   move the number you act on — and this one does.

**What to take from it:** if you simulate data and something downstream depends on
a specific relationship in it, verify that relationship explicitly. Do not infer it
from the fact that the data looks right.

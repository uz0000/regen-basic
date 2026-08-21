# Corrections

Claims this repo published and then revised. Each entry states what was claimed,
what was wrong with it, what the re-check showed, and what changed.

The point of keeping this is that a repo which only shows the claims that
survived is not showing its method. All four below were found the same way: by
going back to a statement and asking what evidence was actually behind it.

---

## 1. The dependency-pinning rationale cited evidence from code that is not here

**Claimed.** Dependencies were pinned because "a newer numpy/scikit-learn was
previously found to shift results enough to flip a borderline test."

**What was wrong.** That did happen, but it happened in a different module of an
earlier project, one that depended on scikit-learn. This codebase does not depend
on scikit-learn at all. The evidence was real and belonged to something else, so
it did not support the claim it was attached to.

**What changed.** The rationale was rewritten as what it actually is: a general
reproducibility discipline, since determinism claims hold only within a fixed
dependency set. The specific incident is still cited, and now says where it
happened and that this codebase no longer uses the library involved.

---

## 2. "Not independently verified" stayed in the README after it became verified

**Claimed.** The install section said reproducibility across dependency versions
was "not independently verified for this code specifically."

**What was wrong.** It was true when written and went stale. Continuous
integration now runs the full suite and both examples on Linux across Python
3.10, 3.11, and 3.12, none of which is the machine this was developed on, and
they pass. The README was understating what was known.

**What changed.** The claim was updated to what is now actually established, and
the remaining limit was stated rather than left implied: only the pinned versions
are exercised, so the pins are known-good rather than known-necessary.

---

## 3. The comparison scorer labelled this repo's generator by an old name

**Claimed.** `examples/comparison/RESULTS.md` reported figures for
`regen-basic`.

**What was wrong.** The scorer still emitted the label `regen-synthetic`, a name
that predated the repository rename. Its output therefore disagreed with the
table that reported it. The numbers were right and the identification was not,
which is the kind of mismatch that makes a reader distrust both.

**Re-checked.** Both comparison tables were re-run against a freshly created SDV
1.38.0 environment and a freshly created project virtualenv. Every figure
reproduced exactly: correlation deltas, category deltas, and conclusion-kept
counts, on both tables.

**What changed.** The label was fixed, and the re-run was recorded in
[`examples/comparison/RESULTS.md`](examples/comparison/RESULTS.md) so the
reproducibility claim the pinning discipline makes is checked rather than
asserted.

---

## 4. "If the positive control ever diverges, the checker is broken"

**Claimed.** The README and the demo's own README both said a resample of the
real rows *must* match, and that a divergence meant the checker was broken rather
than the data. The demo README added that the negative control "must diverge on
everything."

**What was wrong.** Both are false, and the first is the damaging one. Matching
requires all four coefficients to agree at once, and each is a 95% test, so four
of them produce an occasional flag with nothing wrong. A reader re-running with a
different seed would see the control diverge and follow this repo's own
instruction to conclude the tool was broken — a false alarm about the instrument,
invited by the documentation. Nothing here measured the rate.

**Re-measured.** Under the demo's exact configuration — resample 30,000 rows,
compare against the same real fit, 300 seeds:

```
positive control refused   11 / 300  = 3.67%   (95% CI 1.5% - 5.8%)
committed demo seed (2)    matches
```

The rate is lower here than in [`regen-synth`](https://github.com/uz0000/regen-synth),
where the same control is refused about 12% of seeds, and the difference is
instructive rather than incidental: that repo resamples 6,000 rows against a
30,000-row real fit, so its synthetic estimate is much noisier and the comparison
has more room to land in a tail. Same rule, same data, different sample size,
three times the false-refusal rate.

**What changed.** Both documents now say what the control actually does, with the
rate and the diverging seeds named so a reader can reproduce one deliberately
rather than meet one by accident. The committed table is unaffected — seed 2 still
matches. [`MATH.md`](MATH.md) explains the compounding, and
`tests/test_control_rates.py` pins it.

**What did not change.** No reported result moves. The simulator's failures sit
far from the threshold, and this property biases the check toward refusing good
output, not toward passing bad output.

---

## Related

The larger system this repo was extracted from keeps the same record at
[`regen-synth/CORRECTIONS.md`](https://github.com/uz0000/regen-synth/blob/main/CORRECTIONS.md),
including a certification rate that fell from roughly 7 in 8 to 11 in 30 once it
was measured across enough seeds.

# Corrections

Claims this repo published and then revised. Each entry states what was claimed,
what was wrong with it, what the re-check showed, and what changed.

The point of keeping this is that a repo which only shows the claims that
survived is not showing its method. All three below were found the same way: by
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

## Related

The larger system this repo was extracted from keeps the same record at
[`regen-synth/CORRECTIONS.md`](https://github.com/uz0000/regen-synth/blob/main/CORRECTIONS.md),
including a certification rate that fell from roughly 7 in 8 to 11 in 30 once it
was measured across enough seeds.

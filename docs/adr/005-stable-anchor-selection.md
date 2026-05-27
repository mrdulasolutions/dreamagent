# 005. Stable (by-ID) anchor selection vs random sampling

- **Status:** Accepted
- **Date:** 2026-05-27

## Context

During the 16-run tuning loop on Qwen 3 0.6B (later abandoned for
Llama 3.2 1B), we observed that growing the anchor fixture pool from
40 to 105 entries made tuning results WORSE rather than better. This
was counterintuitive — more anchors should mean better protection
against catastrophic forgetting.

Investigation revealed the cause: anchors were being randomly sampled
from the pool (`random.sample(anchors, k)`). Growing the pool changed
*which* anchors got selected on a given run, introducing variance that
masked the hyperparameter effects we were actually trying to measure.

## Decision

Anchors are selected **deterministically by `source_memory_id` order**
via `_sample_stable(pool, k)`. The function takes the first `k`
anchors after sorting by ID. This guarantees:

1. Growing the anchor pool does NOT change which anchors are selected
   unless `k` exceeds the prior pool size.
2. Re-running with the same seed and the same anchor pool produces
   identical training data.
3. Anchor file ordering becomes meaningful: the most-critical-to-eval
   anchors should be early in the file.

Replay (sampled from prior memories) still uses random sampling
because variety in replay is desirable.

## Consequences

- **Easier:** Reproducibility. Comparing two runs that differ only in
  one hyperparameter is now actually possible.
- **Easier:** Adding new anchors is safe — they only enter the mix
  when the cap is raised.
- **Harder:** Anchor file ordering is now a contract. Re-shuffling
  the file changes which anchors get included at a given cap.
- **Documented:** [`docs/tuning/README.md`](../tuning/README.md) makes
  the ordering convention explicit (1-40 generalist core,
  41-105 targeted recoveries).

## Alternatives Considered

1. **Random sample with a fixed seed** — Deterministic per-run but
   sensitive to pool size. Same problem.
2. **Stratified sampling by anchor category** — Better in theory but
   requires categorizing anchors; we'd be optimizing for fairness
   inside the mix rather than reproducibility across runs.
3. **Take all anchors always** — Loses the budget control that
   `anchor_ratio` provides.

## Related

- [`src/dreamagent/compose/mix.py`](../../src/dreamagent/compose/mix.py)
- [`docs/tuning/qwen3-0.6b-4bit.md`](../tuning/qwen3-0.6b-4bit.md) (Runs 4-7 in particular)

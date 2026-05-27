# 004. Four-decision eval gate (PROMOTE / WARN / REJECT)

- **Status:** Accepted
- **Date:** 2026-05-26

## Context

Catastrophic forgetting is the central technical risk for any
continual-learning system. Without an automated safety net, a single
bad nightly run can corrupt the model and destroy weeks of work.

We need a gate that:
1. Accepts good runs without human review (otherwise nightly autonomy
   is impossible).
2. Rejects clearly-bad runs hard (preserve the prior live adapter).
3. Surfaces ambiguous runs for review without blocking promotion.
4. Is configurable per model class (validation-tier ≠ production-tier).

## Decision

The gate uses a four-decision matrix:

| Personal recall | General regression | Decision |
|---|---|---|
| ≥ min | ≤ max_regression | PROMOTE |
| ≥ min | (max..warn] regression | PROMOTE_WITH_WARNING |
| ≥ min | > warn_regression | REJECT |
| < min_personal_recall | (any) | REJECT (low recall) |

Thresholds:
- Production tier: min=70%, max=3pp, warn=8pp
- Validation tier: min=30%, max=5pp, warn=15pp

PROMOTE → adapter goes live, `live` symlink updates.
PROMOTE_WITH_WARNING → adapter goes live, gate.json flags for review.
REJECT → adapter saved under `snapshots/rejected/`, `live` untouched.

## Consequences

- **Easier:** Automated nightly runs are safe by default.
- **Easier:** Rollback is implicit (REJECT preserves prior `live`).
- **Easier:** The warning band catches drift that's not yet
  catastrophic but worth watching.
- **Harder:** Setting good thresholds requires per-model tuning (see
  `docs/tuning/`). This is the central honest tradeoff.
- **Accepted tradeoff:** The threshold values are themselves
  hyperparameters; tuning the gate too tight blocks improvements, too
  loose lets bad runs through.

## Alternatives Considered

1. **Binary gate (PROMOTE / REJECT)** — Simpler but loses the
   "this is suspicious, look at it" signal.
2. **Continuous score** — More information but less actionable; gates
   need a discrete decision.
3. **Human-review queue** — Removes autonomy, which is the point of
   nightly runs.

## Related

- [`src/dreamagent/promote/gate.py`](../../src/dreamagent/promote/gate.py)
- [`docs/METHODOLOGY.md`](../METHODOLOGY.md)

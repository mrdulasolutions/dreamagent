# OPLoRA k=32 Validation — Path A · Week 1

**Date:** 2026-05-27
**Status:** ❌ **Did not improve over vanilla LoRA at the locked recipe.**
**Base model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
**Snapshot:** `runs/oplora-validation/snapshots/2026-05-27T22-54-15Z/`

This is the first empirical result from Path A (improving fine-tuning
methods on Llama 3.1 8B) committed to after the V2.1 + V2.2 retractions.
Per the pre-registered falsifiability protocol in PAPER §10.5, we
publish the data as collected, including when it doesn't support the
hypothesis.

## What we tested

[OPLoRA](https://arxiv.org/abs/2407.08394) — Orthogonal Projection LoRA.
The hypothesis was: by projecting the LoRA update orthogonal to the top-k
singular subspaces of each base weight, we should reduce catastrophic
forgetting while preserving (or improving) personal recall.

Full implementation in `src/dreamagent/train/oplora.py`. Unit tests verify
the math: projections are idempotent, top-k subspaces are correctly
removed, forward-pass equivalence with direct projected ΔW (rel. err.
< 1e-4), `fuse()` equivalence, freeze mechanics.

## The single configuration tested

```bash
dreamagent dream \
  --validation-tier \
  --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --iters 90 \
  --num-layers 8 \
  --learning-rate 3e-5 \
  --anchor-ratio 0.30 --max-anchors 60 \
  --eval-max-tokens 48 \
  --use-oplora \
  --oplora-k-singular 32
```

Identical to the locked V1 production recipe, with `--use-oplora` and
`--oplora-k-singular 32` added. The k=32 default comes from the OPLoRA
paper.

## Results

### Eval-gate metrics (from `gate.json`)

| Metric | Vanilla LoRA baseline | OPLoRA k=32 | Δ |
|---|---|---|---|
| Personal recall (N=48) | 21/48 (43.75%) | 23/48 (47.92%) | +4.17pp |
| General base (N=30) | 29/30 (96.67%) | 29/30 (96.67%) | 0 |
| General adapter (N=30) | 29/30 (96.67%) | 27/30 (90.00%) | **−6.67pp** |
| General regression | 0.0pp | **−6.67pp** | — |
| Gate decision | PROMOTE (clean) | PROMOTE_WITH_WARNING | — |
| Wall time | ~10 min | ~18 min | +8 min (SVD overhead) |
| Trainable params | identical | 5.243M / 8030M (0.065%) | — |

The OPLoRA gate decision was PROMOTE_WITH_WARNING because general
regression (6.67pp) exceeded the 5pp warn threshold. Still well under
the 15pp reject limit.

### Head-to-head vs vector retrieval (from `vs_baselines`)

| Probe set | OPLoRA k=32 | Vector retrieval (top-5) | Δ |
|---|---|---|---|
| personal_recall (N=48) | 47.92% | 68.75% | **−20.83pp** |
| cross_memory_reasoning (N=10) | 40.00% | 90.00% | **−50.00pp** |

For comparison, the V2.1 vanilla-LoRA-7-night-drill adapter measured
75.0% personal / 90.0% cross-memory vs the same retrieval baseline —
i.e., parity on cross-memory and +6.25pp on personal.

## Read

OPLoRA at k=32 with the locked recipe did **not** deliver the promised
reduction in catastrophic forgetting. Specifically:

1. **Personal recall uplift is within noise.** σ for N=48 at p=0.45 is
   roughly 7pp; the observed +4.17pp difference is well inside that.
2. **General capability went the wrong direction.** −6.67pp regression
   where vanilla LoRA had 0pp. This is the opposite of the OPLoRA
   hypothesis. σ for N=30 at p=0.93 is roughly 4.6pp, so −6.67pp is
   ~1.4σ — borderline-significant in the wrong direction.
3. **Cross-memory at 40% is alarming.** The vanilla-LoRA 7-night-drill
   reached 90%; the locked-recipe single-night was not measured but
   would presumably be lower. Even so, 4/10 is a poor result on a
   reasoning task.

### Confounds

- **Training-budget confound.** The cross-memory comparison is most
  honestly read as OPLoRA-single-night (40%) vs vanilla-7-night (90%).
  A fair test would be OPLoRA-7-night vs vanilla-7-night. We haven't
  run OPLoRA for 7 nights and the single-night cross-memory number is
  bad enough that we want to understand it before spending the compute.
- **Hyperparameters were inherited from the vanilla-LoRA recipe.** It's
  possible OPLoRA needs different LR, more iters, or different
  layer selection.
- **k=32 may simply be too aggressive.** With 32 of the top singular
  subspaces removed, the LoRA update space is significantly constrained.
  Smaller k (e.g., 8 or 16) might preserve more update capacity.

## Decision

We are **not** declaring OPLoRA dead. The result above shows one
configuration (k=32, single night, locked recipe) doesn't help. Three
candidate next steps:

- **A. Try k=16 or k=8.** ~18 minutes of compute per run; cheap. If
  smaller k preserves general capability, OPLoRA is salvageable.
- **B. Skip to Path A · Week 2 (Sparse Memory Finetuning).** Treat this
  as one negative result on one technique and try the next candidate.
- **C. Stop Path A.** Accept that the V2.1/V2.2 retraction pattern is
  more fundamental than "we picked the wrong fine-tuning method," and
  rethink.

The research doc [`docs/research/2026-05-improving-memory.md`](../research/2026-05-improving-memory.md)
recommended OPLoRA as the lowest-risk first step; that recommendation
should be re-examined in light of this data.

## Reproducibility

- Snapshot: `runs/oplora-validation/snapshots/2026-05-27T22-54-15Z/`
- Adapter: `.../adapter/adapters.safetensors` (~210 MB; not committed
  to the repo per the .gitignore policy)
- Training log: `runs/oplora-validation/logs/dream-oplora-k32.log`
- vs-baselines log: `runs/oplora-validation/logs/vs-baselines-oplora.log`
- Code: commit `5da5525` ("feat: OPLoRA (Path A · Week 1)").

The full per-probe detail is in the snapshot's `eval_personal.json`,
`eval_general_*.json`, and the `vs_baselines-2026-05-27T*.json` result.

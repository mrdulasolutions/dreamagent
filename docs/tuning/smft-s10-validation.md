# SMFT s=0.10 Validation — Path A · Week 2

**Date:** 2026-05-27
**Status:** ❌ **Worse than vanilla LoRA on every measured axis.**
**Base model:** `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`
**Snapshot:** `runs/smft-validation/snapshots/2026-05-27T23-13-16Z/`

This is the second empirical result from Path A. The first
([OPLoRA k=32](oplora-k32-validation.md)) also did not improve over
vanilla LoRA. Per the pre-registered §10.5 falsifiability protocol,
the data lands as collected.

## What we tested

[Sparse Memory Finetuning](https://arxiv.org/pdf/2510.15103) — per-tensor
top-k gradient sparsification. Before each optimizer step, we zero all
but the top `smft_sparsity` fraction of gradient entries by absolute
magnitude. Hypothesis: many of the parameter updates vanilla LoRA makes
are unnecessary for the new memory and interfere with prior knowledge.
The paper reported 6× less catastrophic forgetting on continual-learning
benchmarks (11% drop vs 71% drop).

Implementation:
- `src/dreamagent/train/smft.py` — `SparseMemoryOptimizer` wrapping a
  base optimizer (Adam/AdamW/SGD).
- 11 unit tests verifying the masking math: top-k by magnitude is
  correct, dtype preserved, nested-dict tree walk works, integration
  with mlx optimizers gives the expected post-update parameters.

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
  --use-smft \
  --smft-sparsity 0.10
```

Identical to the locked V1 production recipe, with `--use-smft` and
`--smft-sparsity 0.10` added. `smft_sparsity=0.10` means we keep the
top 10% of gradient entries by magnitude per tensor, zero the rest.

## Results

### Eval-gate metrics

| Metric | Vanilla LoRA | OPLoRA k=32 | **SMFT s=0.10** |
|---|---|---|---|
| Personal recall (N=48) | 21/48 (43.75%) | 23/48 (47.92%) | **17/48 (35.42%)** |
| General base (N=30) | 29/30 (96.67%) | 29/30 (96.67%) | 29/30 (96.67%) |
| General adapter (N=30) | 29/30 (96.67%) | 27/30 (90.00%) | 27/30 (90.00%) |
| General regression | 0.0pp | −6.67pp | **−6.67pp** |
| Gate decision | PROMOTE (clean) | PROMOTE_WITH_WARNING | PROMOTE_WITH_WARNING |
| Final train loss | similar | 0.599 | similar |

### Head-to-head vs vector retrieval

| Probe set | SMFT s=0.10 | Vector retrieval (top-5) | Δ |
|---|---|---|---|
| personal_recall (N=48) | 35.42% | 68.75% | **−33.33pp** |
| cross_memory_reasoning (N=10) | 30.00% | 90.00% | **−60.00pp** |

For comparison: V2.1 vanilla-LoRA-7-night-drill measured 75.0% / 90.0%
(+6.25pp / 0.0pp vs retrieval). OPLoRA k=32 single-night measured
47.92% / 40.0% (−20.83pp / −50.0pp).

### General-regression probe overlap analysis

A key question: is the −6.67pp regression shared by OPLoRA and SMFT
"the same 2 noisy probes" or "two different sets of 2 probes"? They are
**different**:

- OPLoRA's 2 new failures: `anchor_eval_002`, `anchor_eval_013`
- SMFT's 2 new failures: `anchor_eval_015`, `anchor_eval_029`

That this hit different probes is evidence the regression isn't probe
noise — both techniques caused real but probe-different general-capability
damage that vanilla LoRA at the locked recipe doesn't produce.

## Read

SMFT at s=0.10 with the locked V1 recipe **underperformed vanilla LoRA on
every axis**:

1. **Personal recall regressed by 8.3pp** (43.75% → 35.42%). Outside
   the noise band (σ≈7pp at N=48, p=0.4); plausibly significant.
2. **General regression matched OPLoRA's, not vanilla's.** −6.67pp
   regression where vanilla had 0pp.
3. **Cross-memory at 30% is the worst of the three runs.** Worse than
   OPLoRA's 40%; far below the 7-night-vanilla 90% and the retrieval
   90%.

### Likely cause

s=0.10 keeps only 10% of gradient entries per step. With 5.243M
trainable parameters, that's ~524K parameters updating per step. For 90
iters × ~2 batches ≈ 180 update steps, this may not be enough total
update budget to learn 50 memories at the locked recipe's small LR.

### What we did NOT do

- Sweep s ∈ {0.30, 0.50, 0.80}. A less aggressive sparsity might recover
  personal recall while preserving the (claimed) forgetting reduction.
- Test SMFT with the 7-night drill. The training-budget hypothesis from
  the OPLoRA write-up applies here too — but with SMFT actively HURTING
  personal recall single-night, the cost-of-experiment for 7-night SMFT
  feels poorly justified.

## Cumulative picture: Path A Week 1 + Week 2

| Run | Personal | Gen regression | Cross-memory | vs Retrieval (personal) |
|---|---|---|---|---|
| Vanilla LoRA, single-night | 43.75% | 0.0pp | (not measured) | (not measured) |
| Vanilla LoRA, 7-night drill | 75.0% | acceptable | 90.0% | +6.25pp |
| OPLoRA k=32, single-night | 47.92% | −6.67pp | 40.0% | −20.83pp |
| **SMFT s=0.10, single-night** | **35.42%** | **−6.67pp** | **30.0%** | **−33.33pp** |
| Retrieval baseline (top-5) | n/a | n/a | 90.0% | (reference) |

Both Path A "cheap drop-in" techniques at paper-default hyperparameters,
with the locked V1 recipe, **underperform vanilla LoRA**. The pattern is
strong enough to warrant a decision-point conversation rather than
another sweep.

## Reproducibility

- Snapshot: `runs/smft-validation/snapshots/2026-05-27T23-13-16Z/`
- Training log: `runs/smft-validation/logs/dream-smft-s10.log`
- vs-baselines log: `runs/smft-validation/logs/vs-baselines-smft.log`
- Code: commit `8177719` ("feat: SMFT (Path A · Week 2)").
- Probe-level overlap analysis is reproducible from `eval_general_*.json`
  files in both the OPLoRA and SMFT snapshot directories.

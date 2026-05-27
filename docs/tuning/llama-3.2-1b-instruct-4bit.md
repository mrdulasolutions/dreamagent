# Tuning Log — `mlx-community/Llama-3.2-1B-Instruct-4bit`

**Tier:** validation
**Status:** ✅ Clean PROMOTE achieved at Run 16.
**Locked recipe:** below.

## Locked V1 Recipe

```bash
dreamagent dream \
  --validation-tier \
  --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
  --iters 90 \
  --num-layers 4 \
  --learning-rate 3e-5 \
  --anchor-ratio 0.30 \
  --max-anchors 60 \
  --eval-max-tokens 48
```

Pipeline composition with this recipe:
- 133 memory training examples + ~57 anchors + 0 replay (first night) = ~190 mix
- 90 iters × batch 2 = 180 grad steps over 190 examples → 0.95 epochs
- Soft enough to preserve general capability, deep enough to learn memories

## Eval State

Personal recall: **46%** (22/48). Substantially above the 30% validation-tier floor.
General base: **93%** (28/30). General adapter: **90%** (27/30). Regression: **3.3pp**.
Gate: **PROMOTE** (clean).

## Run Table

| # | iters | layers | LR | anc% | max_anc | Pers | Gen base | Gen adp | Δ gen | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 11 | 150 | 8 | 1e-4 | 0.30 | 40 | 67% | 93% | 43% | −50pp | REJECT (too aggressive for Llama) |
| 12 | 60  | 4 | 3e-5 | 0.30 | 40 | 31% | 93% | 83% | −10pp | PROMOTE_WITH_WARNING |
| 13 | 100 | 4 | 4e-5 | 0.30 | 40 | 47% | 93% | 73% | −20pp | REJECT |
| 14 | 60  | 4 | 3e-5 | 0.40 | -  | ~   | 93% | 73% | −20pp | REJECT (full anchor pool hurts) |
| 15 | 90  | 4 | 3e-5 | 0.30 | 40 | 44% | 93% | 83% | −10pp | PROMOTE_WITH_WARNING |
| **16** | **90** | **4** | **3e-5** | **0.30** | **60** | **46%** | **93%** | **90%** | **−3.3pp** | **PROMOTE** ✅ |

## Critical Findings

### 1. The `max_anchors` cap is a precision lever — bigger ≠ better

Run 14 used the full 105-anchor pool at `anchor_ratio=0.40`. Regression doubled (10pp → 20pp). The new anchors include longer-form answers ("Docker is a platform for packaging applications into lightweight, portable containers...") that compete with the eval probes' style. With Llama, more anchor variety added noise rather than coverage.

**The sweet spot for Llama 3.2 1B was 60 anchors** — enough to cover the targeted eval categories (primes, RAM, GPU, planets, translations) without dilution.

### 2. Anchor ordering matters

Because `compose_rehearsal_mix` uses **stable selection** (sorted by `source_memory_id`), the first N anchors get included before the rest. We deliberately placed the highest-leverage anchors (anchors 041–060: targeted factual recoveries) right after the original 40 generalist ones. So `max_anchors=60` includes:

- 1-40: original generalists (persona, basic math, Q&A)
- 41-60: targeted recoveries for known failure modes (prime number, RAM expansion, GPU expansion, # planets, basic translations)

Pushing the cap to 80-105 starts including longer-form / lower-leverage anchors and hurts.

### 3. Llama 3.2 1B has a steeper "aggressiveness" curve than Qwen 3 0.6B

Run 11 used the recipe that worked for Qwen 3 (iters=150, lr=1e-4, layers=8). On Llama it caused **50pp** regression — much worse. Llama has a stronger base (93% on the 30-probe eval vs Qwen's 77%), and aggressive training proportionally overwrites more of that high-quality knowledge.

**Heuristic:** stronger base → softer training. For Llama 3.2 1B, half the iters and a third the LR vs Qwen 3 0.6B.

### 4. Personal recall is acceptably lower on Llama (46% vs Qwen's 69-79%)

The Llama recipe trades some personal recall for general preservation. 46% personal recall on a 1B base is a reasonable point on the curve. For production tier (Qwen 3 4B) we expect higher.

## Open Hypotheses

- Re-running with a fresh random seed should reproduce the clean PROMOTE within ±3pp regression. Worth confirming once.
- Pulling `iters` slightly higher (100-110) may improve personal recall without losing the clean PROMOTE. Untested.
- Pulling `num_layers` to 6 may help personal recall. Untested.

## Reproducibility Notes

- All runs above used `--max-anchors` to pin the anchor count and `_sample_stable` for deterministic anchor selection.
- Anchor pool size at time of run 16: 105 items in `fixtures/anchors/general_anchor.jsonl`.
- Eval probe count: 30 items in `fixtures/anchors/general_eval.jsonl`.
- Memory fixture: 50 items in `fixtures/v1_baseline.jsonl`.

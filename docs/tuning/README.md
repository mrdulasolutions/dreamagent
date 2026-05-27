# DreamAgent — Tuning Playbook

Tuning is **per-model**. Hyperparameters that work for Qwen 3 0.6B will not work for Qwen 3 4B, and neither will work for Llama 4 70B. Catastrophic-forgetting dynamics, optimal LoRA rank, and the safe iteration count all change with model scale.

This directory captures the tuning history for each base model so that:
- Future runs can start from a known-good preset instead of from scratch.
- The reasons for each decision are recorded — so we don't relearn lessons.
- New contributors can see the shape of the tuning curve before they try to bend it.

## Files

| File | Purpose |
|---|---|
| `README.md` | This playbook |
| `llama-3.2-1b-instruct-4bit.md` | **Validation-tier model (locked recipe)** |
| `qwen3-0.6b-4bit.md` | Validation-tier attempt — abandoned due to reasoning conflict |
| `qwen3-4b.md` | Production-tier model tuning log (Pass 2 / future) |
| `phi4-14b.md` | Stretch model tuning log (future) |

## The Levers (ordered by effect size)

1. **Anchor ratio** (`--anchor-ratio`, default `0.10`).
   The strongest single lever for protecting general capability. Raising it forces the optimizer to spend more updates on the "you are still you" set, which keeps the model from overwriting persona and general knowledge. Effective ceiling is the size of the anchor pool — adding more anchor fixtures raises the ceiling.

2. **Iteration count** (`--iters`, default `200`).
   How many gradient steps the optimizer takes. Too few → memories don't stick. Too many → catastrophic overfit (train loss → 0, the model memorizes phrasings and forgets everything else). The sweet spot for small models is usually 100-150 with our default mix size.

3. **LoRA layers** (`--num-layers`, default `8`).
   How many transformer layers get a LoRA adapter attached. More layers = more capacity to learn AND more capacity to overwrite. For validation tier (0.6B), 4-8 is the working range. Production tier (8B) typically wants 8-16.

4. **Learning rate** (`--learning-rate`, default `1e-4`).
   How big each gradient step is. Higher → faster convergence + more overwriting. Lower → safer + slower. `1e-4` is the MLX-LM default; `5e-5` is the safety setting; `2e-5` is for fine-grained adjustment.

5. **Replay ratio** (`--replay-ratio`, default `0.15`).
   Fraction of mix that should be re-shown memories from prior nights. Only meaningful after night 2+. Defends against forgetting old memories when learning new ones.

## How to Tune a New Model

The first time DreamAgent meets a new base model, follow this loop:

1. **Bracket the failure modes** with two extreme runs:
   - Aggressive: `--iters 200 --num-layers 8 --learning-rate 1e-4 --anchor-ratio 0.10`
     — Expect personal recall high, general regression high.
   - Conservative: `--iters 80 --num-layers 4 --learning-rate 5e-5 --anchor-ratio 0.30`
     — Expect personal recall low, general regression near zero.

2. **Walk inward.** Pick the levers that produced the failure in each direction and split the difference. Always change at most 1-2 levers per run so you can attribute changes.

3. **Stop when the gate emits a clean PROMOTE.** Record the recipe in the per-model log.

4. **Lock the recipe** as a `configs/<model>.yaml` once stable, so future nightly runs use it automatically.

## Gate Decisions Cheat Sheet

| Personal recall | General regression | Decision |
|---|---|---|
| ≥ min | ≤ max_regression | PROMOTE |
| ≥ min | max..warn band | PROMOTE_WITH_WARNING |
| ≥ min | > warn | REJECT |
| < min | anything | REJECT |

Validation-tier defaults: `min_personal_recall=0.30`, `max_general_regression=0.05`, `warn_general_regression=0.15`.

Production-tier defaults: `min_personal_recall=0.70`, `max_general_regression=0.03`, `warn_general_regression=0.08`.

## Anchor-Pool Sizing

The mix composer caps anchor count at the pool size: if you ask for 60 anchors but only have 40 in the pool, you get 40. To raise the effective anchor share above ~25% with a 150-example today batch, you need 60+ anchors in the pool.

Anchor fixtures live in `fixtures/anchors/general_anchor.jsonl`. Anchors are taken **deterministically in `source_memory_id` order**, so the order in the file matters:

- IDs `anchor_001`–`anchor_040`: generalist core (persona, basic math, Q&A, hedging on unknowns)
- IDs `anchor_041`–`anchor_105`: targeted-fact recoveries (specific facts that addressed known eval failures during tuning)

Use `--max-anchors N` to cap how many anchors enter the mix. Smaller caps include just the core; larger caps add targeted recoveries. **Bigger is not always better** — see [llama-3.2-1b-instruct-4bit.md](./llama-3.2-1b-instruct-4bit.md) finding 1.

## Lessons Learned Across All Tuning So Far

These were learned the hard way in Pass 1. Read before tuning a new model:

1. **Eval probe count must match the threshold resolution.**
   With a 15-probe eval and `max_general_regression=0.05`, even 1 lost probe (6.7pp) breaches the warn threshold. The threshold is effectively unreachable. Use ≥20 probes, ideally 30+. See [qwen3-0.6b-4bit.md](./qwen3-0.6b-4bit.md) finding "Run 9's discovery."

2. **Reasoning models need anchor responses that preserve their reasoning structure.**
   Qwen 3, DeepSeek-R1, and other thinking-tag models will have their reasoning capability destroyed by anchors that go straight to the answer. Either rewrite anchors to include the model's reasoning style, or use a non-reasoning model for the validation tier.

3. **Stable anchor selection over random sampling.**
   `_sample_stable` is used for anchors; growing the anchor pool doesn't change which anchors get selected unless you raise the cap. This eliminates a variance source that masked our hyperparameter effects in Runs 4-7.

4. **Stronger base → softer training.**
   Llama 3.2 1B (base 93% on 30-probe eval) needed half the iters and a third the LR vs Qwen 3 0.6B (base 77%). Strong base models have more to lose.

5. **Anchor order matters because we use stable selection.**
   Most-critical-to-eval anchors should be early in the file. Use `--max-anchors N` to include only the first N. We discovered Llama's sweet spot was exactly the first 60 anchors — the core 40 plus 20 targeted recoveries.

6. **Bracket then bisect.**
   Always do one aggressive + one conservative run on a new model before tuning. The bracket reveals which lever crosses which failure threshold. Then change one lever at a time inward.

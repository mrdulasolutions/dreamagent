# Tuning Log — `mlx-community/Qwen3-0.6B-4bit`

**Tier:** validation (initially attempted)
**Status:** ❌ NOT VIABLE as the validation tier without anchor rewrites. See finding 1 below.
**Replaced by:** [`llama-3.2-1b-instruct-4bit.md`](./llama-3.2-1b-instruct-4bit.md) — which DID land a clean PROMOTE.

## Why Qwen 3 0.6B Was Abandoned

Qwen 3 0.6B is a **reasoning model** — its instruct tuning uses `<think>...</think>` blocks before answers. Our anchor fixtures train it on assistant responses that have no `<think>` block:

```jsonc
{"role": "assistant", "content": "4."}     // anchor format
```

vs. how Qwen 3 wants to answer natively:

```
<think>
Let me compute: 2 + 2 = 4.
</think>

4.
```

After even soft training, the model produces empty `<think></think>` blocks and jumps straight to an answer. **Training destroyed its reasoning capability**, and downstream the model fabricates wildly on factual questions because it skips the reasoning step that previously protected it.

Examples of the failure mode (Run 10):
- "25 / 5" → "12"
- "Mona Lisa painter" → "Salvador Dalí"
- "Largest ocean" → "Jupiter"

To make Qwen 3 0.6B viable, the anchor fixtures need rewrites that include `<think>` content — a significant authoring effort. We deferred this for V1 and switched to Llama 3.2 1B Instruct.

## Run Table (Qwen 3 attempts only)

| # | iters | layers | LR | anc% | anchor pool | max_anc | eval_n | Pers | Δ gen | Decision |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 200 | 8 | 1e-4 | 0.10 | 40  | -  | 15 | 79% | −40pp | REJECT |
| 2 | 80  | 4 | 5e-5 | 0.30 | 40  | -  | 15 | 27% |   0pp | REJECT |
| 3 | 150 | 8 | 1e-4 | 0.30 | 40  | -  | 15 | 69% |  −7pp | PROMOTE_WITH_WARNING |
| 4 | 150 | 8 | 1e-4 | 0.30 | 105 | -  | 15 | 67% | −13pp | PROMOTE_WITH_WARNING |
| 5 | 180 | 8 | 1e-4 | 0.40 | 105 | -  | 15 | 71% | −20pp | REJECT |
| 6 | 150 | 8 | 5e-5 | 0.30 | 105 | -  | 15 | 71% | −13pp | PROMOTE_WITH_WARNING |
| 7 | 150 | 8 | 1e-4 | 0.30 | 105 | -  | 15 | 73% | −13pp | PROMOTE_WITH_WARNING |
| 8 | 150 | 8 | 1e-4 | 0.30 | 105 | 40 | 15 | 69% |  −7pp | PROMOTE_WITH_WARNING (Run 3 reproduced) |
| 9 | 150 | 8 | 1e-4 | 0.30 | 105 | 40 | 30 | 73% | −20pp | REJECT (30-probe eval reveals truth) |
| 10| 100 | 8 | 5e-5 | 0.30 | 105 | 40 | 30 | 67% | −23pp | REJECT |

## Critical Finding (transferable across all future models)

**Run 9's discovery is the most important lesson of this tuning effort.**

Run 3 looked great with a 15-probe general eval (`−7pp`). When we expanded to 30 probes (Run 9), the SAME ADAPTER showed `−20pp` regression. The narrow eval set was hiding 5 additional failures.

**Implication:** the eval probe count must be large enough that the gate's regression threshold has meaningful resolution. With a 15-probe set, 1 lost probe = 6.7pp regression — already above a 5pp warn threshold. The warn band is effectively unreachable.

For a `max_general_regression=0.05` threshold to be meaningful, the eval set should have **at least 20 probes**, ideally 30+.

## Open: Reviving Qwen 3 0.6B

If a future contributor wants to revive Qwen 3 0.6B as a validation-tier model, the work is:

1. Rewrite all anchors in `general_anchor.jsonl` so the assistant response starts with a `<think>` block containing 1-2 sentences of reasoning, then `</think>`, then the answer.
2. Tune from a starting point of `iters=80, num_layers=4, lr=5e-5, anchor_ratio=0.30`.
3. Verify reasoning structure is preserved by inspecting raw adapter responses — `<think>` blocks should be non-empty.

This is tracked as a stretch task; not blocking V1.

# 006. Llama 3.2 1B Instruct as canonical validation-tier model

- **Status:** Accepted
- **Date:** 2026-05-27
- **Supersedes:** original choice of Qwen 3 0.6B-4bit

## Context

The plan called for Qwen 3 0.6B as the validation-tier base model:
small, fast to fine-tune, MLX-supported, Apache 2.0. Sixteen tuning
runs revealed a hard problem: Qwen 3 is a *reasoning* model — its
instruct tuning produces `<think>...</think>` blocks before answers.
Our anchor fixtures have no `<think>` content:

```jsonc
{"role": "assistant", "content": "4."}
```

After even soft training, Qwen 3's `<think>` blocks come out empty
and the model jumps to a guess. Training destroyed the model's
reasoning capability, causing wild fabrication on general-knowledge
probes (25/5 → "12", Mona Lisa → "Salvador Dalí").

## Decision

Switch the canonical validation-tier model to
`mlx-community/Llama-3.2-1B-Instruct-4bit`. Llama 3.2 1B has no
reasoning-tag structure, has higher base capability on our 30-probe
eval (93% vs Qwen's 77%), and tuned to a clean PROMOTE in Run 16
with a moderate hyperparameter sweep.

Qwen 3 0.6B is documented as "not viable without anchor rewrites"
in [`docs/tuning/qwen3-0.6b-4bit.md`](../tuning/qwen3-0.6b-4bit.md)
and is left as a future stretch task.

## Consequences

- **Easier:** Tuning, eval, and downstream verification all work
  cleanly with the new validation-tier model.
- **Easier:** New contributors get a working baseline out of the box.
- **Harder:** Smaller pool of explicitly Apache-2.0 options at the
  ≤1B size. Llama 3.2 1B's Community License is permissive but not
  Apache. Not a blocker for personal use; could be for some
  commercial use cases.
- **Documented:** The reasoning-conflict lesson is now in the tuning
  playbook as a transferable insight (any reasoning model — Qwen 3,
  R1-style — will need anchor rewrites with `<think>` content).

## Alternatives Considered

1. **Rewrite all 105 anchors to include `<think>` blocks** — Real
   long-term fix but ~3x the authoring effort. Deferred.
2. **Use Qwen 2.5 0.5B Instruct (non-reasoning)** — Smaller and faster
   than Llama 3.2 1B but lower base capability (76% on our eval).
3. **SmolLM2 1.7B** — Apache 2.0, no thinking, fast. Likely fine;
   chosen Llama as the larger ecosystem.
4. **Stay on Qwen 3 0.6B and just accept the reasoning destruction** —
   No. The general-capability eval results were catastrophic; the
   model became a fabrication machine.

## Related

- [`docs/tuning/llama-3.2-1b-instruct-4bit.md`](../tuning/llama-3.2-1b-instruct-4bit.md)
- [`docs/tuning/qwen3-0.6b-4bit.md`](../tuning/qwen3-0.6b-4bit.md)
- ADR-005 (stable anchor selection) — discovered during the same tuning loop

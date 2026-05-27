---
name: New tuning recipe
about: Share a tuning recipe for a base model we don't yet have
title: "[tuning] <model-id>"
labels: tuning
---

## Model

- Base model ID (HF / mlx-community):
- Param count:
- License:

## Best Recipe

```bash
dreamagent dream \
  --validation-tier \
  --base-model "..." \
  --iters ... \
  --num-layers ... \
  --learning-rate ... \
  --anchor-ratio ... \
  --max-anchors ...
```

## Result

- Personal recall: __%
- General base: __%
- General adapter: __%
- Regression: __pp
- Gate decision: PROMOTE / PROMOTE_WITH_WARNING

## Tuning Journey

<!-- Run table showing what you tried and why. Use the format in
docs/tuning/llama-3.2-1b-instruct-4bit.md as a template. -->

| # | iters | layers | LR | anc% | max_anc | Personal | Δ gen | Decision |
|---|---|---|---|---|---|---|---|---|

## Surprises / Learnings

<!-- What did you discover that's specific to this model and worth
capturing in the per-model tuning log? -->

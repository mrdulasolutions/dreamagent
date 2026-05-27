# 01 — Quickstart

The minimum-viable DreamAgent loop. End-to-end in about 5 minutes on
Apple Silicon (first run includes a ~700MB model download).

## What you'll do

1. Run the dream pipeline against the shipped fixture memories.
2. Inspect the resulting snapshot.
3. Ask the dreamed model a question that requires the trained memories.

## Steps

```bash
# Sync deps (one time)
uv sync

# Run the locked V1 recipe against the 50-memory fixture
uv run dreamagent dream \
    --validation-tier \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --source fixture:v1_baseline \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag quickstart

# Inspect the snapshot
uv run dreamagent snapshots
ls -la runs/snapshots/live/

# Check the gate decision
cat runs/snapshots/live/gate.json
```

## What good output looks like

```
                      Snapshots
┏━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━┓
┃ name                 ┃ decision  ┃ live ┃
┡━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━┩
│ 2026-MM-DDTHH-MM-SSZ │ promote   │  ★   │
└──────────────────────┴───────────┴──────┘
```

If `decision: promote` (no `_with_warning`), you've reproduced V1
Pass 1's clean PROMOTE.

If you got `promote_with_warning` or `reject`, see the troubleshooting
in [`docs/tuning/llama-3.2-1b-instruct-4bit.md`](../../docs/tuning/llama-3.2-1b-instruct-4bit.md).

## What it learned

The fixture memories include things like:
- The user's dog Otis (a golden retriever)
- The deploy command for the marketing site
- The user's preference for permissive licenses

You can verify recall via the personal-eval report:

```bash
cat runs/snapshots/live/eval_personal.json | head -50
```

Each entry shows the question, the adapter's response, and whether
it matched the expected substring.

## What's next

- **02-extract-from-chat** — try it with your own chat transcripts
- **03-nightly-cron** — schedule this to run every night
- **06-benchmark-suite** — run the full benchmark suite on your snapshot

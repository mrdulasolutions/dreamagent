# 06 — Run the benchmark suite

Reproduce the published numbers and run the comparison benchmarks
against your own snapshot.

## What you'll do

Run each benchmark and inspect the JSON report.

## Steps

```bash
# Get a live snapshot first (or use an existing one)
uv run dreamagent snapshots
# Note the live snapshot's name or use runs/snapshots/live symlink

# Personal recall — does the model know its training memories?
uv run python -m benchmarks.personal_recall \
    --memories fixture:v1_baseline \
    --snapshot runs/snapshots/live

# General capability — did we break the base model?
uv run python -m benchmarks.general_capability \
    --snapshot runs/snapshots/live

# Cross-memory reasoning — the parametric advantage benchmark
uv run python -m benchmarks.cross_memory_reasoning \
    --snapshot runs/snapshots/live

# Query latency — p50/p95/p99 on Mac
uv run python -m benchmarks.query_latency \
    --snapshot runs/snapshots/live \
    --queries 30

# Identity drift — persona preservation
uv run python -m benchmarks.identity_drift \
    --snapshot runs/snapshots/live
```

Each writes a JSON report to `benchmarks/results/`.

## What numbers to expect on V1 Pass 1

| Benchmark | Expected |
|---|---|
| personal_recall | ~46% on Llama 3.2 1B with the locked recipe |
| general_capability | base ~93%, adapter ~90%, regression ~3.3pp |
| cross_memory_reasoning | adapter ≥ base (the parametric advantage in microcosm) |
| query_latency | p50 ~500-900ms, p95 ~1-2s on Mac M-series |
| identity_drift | minimal on a fresh single-night run |

## When numbers are off

- **personal_recall < 30%** — training was too soft. Bump iters or LR.
- **general_capability regression > 5pp** — training was too aggressive. Drop iters, raise anchor ratio.
- **query_latency p95 > 5s** — likely on a smaller Mac; consider quantization or a smaller base.
- **identity_drift** showing alert — persona was overwritten; see [`docs/tuning/`](../../docs/tuning/) for mitigations.

See [`benchmarks/README.md`](../../benchmarks/README.md) for the full
benchmark methodology.

# DreamAgent Benchmarks

A reproducible validation suite for measuring DreamAgent against itself over
time, and against other memory solutions where comparable.

**Design philosophy.** We measure what we can defend. We won't publish a
LoCoMo number we haven't run; we won't claim wins on axes we can't reproduce.
Every benchmark in this directory is runnable from a fresh clone with a
single command.

## What lives here

| File | What it measures | Status |
|---|---|---|
| `personal_recall.py` | Does the dreamed model recall its training memories from weights alone? | ✅ runnable (uses V1 Pass 1 protocol) |
| `general_capability.py` | Did the nightly fine-tune break the base model? | ✅ runnable |
| `cross_memory_reasoning.py` | Can the model answer questions requiring synthesis across 3+ memories? | 🔬 protocol designed; probes shipped |
| `query_latency.py` | p50/p95/p99 query latency for the dreamed model | 🔬 protocol designed |
| `identity_drift.py` | Over many nights, does the model lose its assistant persona? | 🚧 protocol designed; needs multi-night runs |
| `vs_baselines/` | Same memories through DreamAgent vs. retrieval baselines | 🚧 mem0/Letta scaffolds |
| `locomo_compat.py` | DreamAgent on the canonical LoCoMo benchmark (two protocols) | 🚧 in progress (Pass 2 target) |

Legend:
- ✅ **runnable** — fully implemented, produces numbers
- 🔬 **protocol designed; probes shipped** — methodology documented, probes
  in `benchmarks/probes/`, runner sketched
- 🚧 **in progress** — designed; awaiting Pass 2 (production tier) work

## Reproduce V1 Pass 1 numbers

The published V1 numbers (46% personal recall, 90/93 general capability on
Llama 3.2 1B Instruct) can be reproduced in ~5 minutes on Apple Silicon:

```bash
dreamagent dream \
  --validation-tier \
  --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
  --source fixture:v1_baseline \
  --iters 90 --num-layers 4 --learning-rate 3e-5 \
  --anchor-ratio 0.30 --max-anchors 60
```

After it completes, the eval JSONs in `runs/snapshots/<timestamp>/` *are* the
benchmark output. No separate harness needed — every nightly run is a
benchmark run.

## Run an individual benchmark

```bash
# Personal recall against the current live adapter
python -m benchmarks.personal_recall \
    --memories fixture:v1_baseline \
    --snapshot runs/snapshots/live

# General capability
python -m benchmarks.general_capability \
    --snapshot runs/snapshots/live

# Cross-memory reasoning probes (the parametric-advantage benchmark)
python -m benchmarks.cross_memory_reasoning \
    --snapshot runs/snapshots/live

# Query latency
python -m benchmarks.query_latency --snapshot runs/snapshots/live
```

Each prints a JSON report to stdout and writes to `benchmarks/results/`.

## Add a new benchmark

1. Create `benchmarks/<your_benchmark>.py`.
2. Implement the standard signature: `def run(snapshot_path, **kwargs) -> dict`.
3. Document what it measures, the failure modes, and the publishable result
   format in this README.
4. Add a row to the table above.
5. Submit a PR.

See `CONTRIBUTING.md` for the bar on benchmark reproducibility.

## A note on LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) is the canonical agent-memory
benchmark in 2026. Per the [comparison docs](../docs/comparison/README.md),
LoCoMo measures *retrieval over conversation*, which is not what DreamAgent
does. We're committing to publishing LoCoMo numbers under two protocols once
Pass 2 (production-tier Qwen 3 4B) lands:

- **Protocol A — Per-conversation fine-tune:** train on each LoCoMo
  conversation, then evaluate. Maps DreamAgent's loop onto the benchmark
  honestly; expensive (one fine-tune per conversation).
- **Protocol B — DreamAgent as oracle:** use the dreamed model as a
  knowledge backend for a separate agent that does the LoCoMo task.
  Tests the V2 architecture more directly.

We will not cherry-pick the better number. Both will be published, with the
methodology documented in `benchmarks/locomo_compat.py`.

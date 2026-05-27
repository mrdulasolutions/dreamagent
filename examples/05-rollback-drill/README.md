# 05 — Rollback drill

Verify that the eval gate + snapshot system actually protects you
from a bad nightly run. This is the safety property V1 was designed
around, and you should run this drill before trusting it in
production.

## What you'll do

1. Establish a known-good baseline (PROMOTE).
2. Intentionally trigger a REJECT by passing extreme hyperparameters.
3. Verify `live` is untouched after the REJECT.
4. Practice manual rollback to an older snapshot.

## Steps

### 1. Baseline good run

```bash
uv run dreamagent dream \
    --validation-tier --source fixture:v1_baseline \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag baseline

uv run dreamagent snapshots
# Expect: one snapshot with decision=promote, ★ live
```

### 2. Trigger a REJECT

Force an over-aggressive training run that will blow general capability:

```bash
uv run dreamagent dream \
    --validation-tier --source fixture:v1_baseline \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 400 --num-layers 8 --learning-rate 5e-4 \
    --anchor-ratio 0.05 --max-anchors 20 \
    --tag intentional-bad-run \
    --notes "rollback drill: high iters + lr + minimal anchors"
```

Expect output to end with:

```
REJECT
  · general regression +0.XXX > reject threshold +0.150
```

### 3. Verify `live` is preserved

```bash
uv run dreamagent snapshots
# Expect: live STILL points at the baseline snapshot. The bad run
# landed in runs/snapshots/rejected/ instead.

ls runs/snapshots/rejected/
# Expect: one directory from the intentional-bad-run
```

### 4. Practice manual rollback

Now simulate "we promoted a bad run we shouldn't have" by doing
another baseline run and then rolling back to the first one:

```bash
uv run dreamagent dream \
    --validation-tier --source fixture:v1_baseline \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag baseline-v2

uv run dreamagent snapshots
# Now two PROMOTE snapshots; live points at the newest

# Identify the original baseline's name
BASELINE=$(uv run dreamagent snapshots 2>&1 | grep -v '★' | awk '/promote/ {print $2; exit}')

# Roll back
uv run dreamagent rollback "$BASELINE"

uv run dreamagent snapshots
# Expect: live now points at $BASELINE
```

## What this proves

- A bad night gets caught and rejected automatically.
- Rejected runs are archived for inspection, never replace `live`.
- Rollback is a single command; no manual symlink surgery needed.
- The system fails safe by default.

This is the property that makes nightly autonomy actually safe.

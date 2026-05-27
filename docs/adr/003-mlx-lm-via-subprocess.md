# 003. Invoke MLX-LM via subprocess, not Python module API

- **Status:** Accepted
- **Date:** 2026-05-26

## Context

MLX-LM is the local fine-tuning engine. It exposes both a CLI
(`python -m mlx_lm lora --train`) and a Python module API (importing
`mlx_lm.tuner.utils`). The Python API has changed signatures across
versions (0.22 → 0.30 → 0.31). The CLI surface has been more stable.

DreamAgent also needs to target Unsloth on cloud GPU for V3, which has
yet another API. A single training-stage abstraction is desirable.

## Decision

Invoke MLX-LM via subprocess (`python -m mlx_lm lora --train ...`). The
training-stage wrapper (`src/dreamagent/train/runner.py`) constructs
the argv list, runs the subprocess, captures exit code + stdout/stderr,
and verifies the adapter file exists at the expected path.

For V3 cloud path, we'll wrap Unsloth's CLI the same way. The Python
caller sees a uniform `train_adapter(mix, config, run_dir)` regardless
of backend.

## Consequences

- **Easier:** API stability across MLX-LM versions.
- **Easier:** Same wrapper handles local (MLX) and cloud (Unsloth, V3).
- **Easier:** Failures surface as exit codes + log files instead of
  partial Python state.
- **Harder:** ~30-100ms subprocess startup overhead per run (negligible
  for a multi-minute training job).
- **Harder:** No streaming of intermediate metrics into the Python
  caller (we'd have to parse stdout if we wanted that).

## Alternatives Considered

1. **Direct `from mlx_lm.tuner.utils import train`** — Tighter but
   binds us to internal API stability.
2. **Vendor MLX-LM as a submodule** — More control, but means we own
   tracking upstream fixes.
3. **Spawn a Python child process and exchange JSON via stdin/stdout**
   — More structured than parsing CLI stdout, but more complex.

## Related

- [`src/dreamagent/train/runner.py`](../../src/dreamagent/train/runner.py)
- [`docs/tuning/README.md`](../tuning/README.md)

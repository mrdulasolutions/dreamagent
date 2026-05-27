"""MLX-LM-LoRA train runner.

Takes a RehearsalMix and a TrainConfig, writes train.jsonl/valid.jsonl in
mlx-lm's expected chat format, invokes `python -m mlx_lm lora --train`,
and returns the adapter path plus a metadata.json describing the run.

We call mlx-lm via subprocess (rather than importing its internals) because
the CLI surface is more stable across versions than the Python module API.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dreamagent.compose.examples import TrainingExample
from dreamagent.compose.mix import RehearsalMix
from dreamagent.train.config import TrainConfig

ADAPTER_FILENAME = "adapters.safetensors"
METADATA_FILENAME = "metadata.json"


@dataclass(frozen=True, slots=True)
class TrainResult:
    """Output of a successful training run."""

    run_dir: Path
    adapter_path: Path
    metadata_path: Path
    metadata: dict


class TrainError(RuntimeError):
    """Raised when the underlying mlx-lm process fails."""


def _split_train_valid(
    examples: list[TrainingExample], val_fraction: float, seed: int
) -> tuple[list[TrainingExample], list[TrainingExample]]:
    """Split examples into (train, valid). Valid is always at least 1 example
    if the input has 2+."""
    import random

    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    n = len(shuffled)
    if n == 0:
        return [], []
    if n == 1:
        return shuffled, shuffled  # tiny edge case: duplicate the single example
    n_val = max(1, round(n * val_fraction))
    return shuffled[n_val:], shuffled[:n_val]


def _write_jsonl(examples: list[TrainingExample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps({"messages": ex.messages}) + "\n")


def _resolve_python() -> str:
    """Use the same interpreter we're running under to invoke mlx-lm."""
    return sys.executable


def _build_command(
    python: str,
    data_dir: Path,
    adapter_dir: Path,
    config: TrainConfig,
    resume_adapter_file: Path | None = None,
) -> list[str]:
    cmd = [
        python,
        "-m",
        "mlx_lm",
        "lora",
        "--train",
        "--model",
        config.base_model,
        "--data",
        str(data_dir),
        "--fine-tune-type",
        config.fine_tune_type,
        "--num-layers",
        str(config.num_layers),
        "--batch-size",
        str(config.batch_size),
        "--iters",
        str(config.iters),
        "--learning-rate",
        str(config.learning_rate),
        "--max-seq-length",
        str(config.max_seq_length),
        "--optimizer",
        config.optimizer,
        "--steps-per-report",
        str(config.steps_per_report),
        "--steps-per-eval",
        str(config.steps_per_eval),
        "--val-batches",
        str(config.val_batches),
        "--grad-accumulation-steps",
        str(config.grad_accumulation_steps),
        "--seed",
        str(config.seed),
        "--adapter-path",
        str(adapter_dir),
    ]
    if config.mask_prompt:
        cmd.append("--mask-prompt")
    if resume_adapter_file is not None:
        cmd.extend(["--resume-adapter-file", str(resume_adapter_file)])
    return cmd


def _versions() -> dict[str, str]:
    """Gather library versions for lineage / reproducibility."""
    out: dict[str, str] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }
    for pkg in ("mlx_lm", "mlx", "transformers", "pydantic"):
        try:
            mod = __import__(pkg)
            out[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[pkg] = "not_installed"
    return out


def _write_metadata(
    metadata_path: Path,
    *,
    config: TrainConfig,
    mix: RehearsalMix,
    train_count: int,
    valid_count: int,
    started_at: datetime,
    completed_at: datetime,
    source_memory_ids: list[str],
    tag: str | None = None,
    notes: str | None = None,
    invocation: list[str] | None = None,
    resume_adapter_file: Path | None = None,
) -> dict:
    metadata = {
        "schema_version": "1.0",
        "tag": tag,
        "notes": notes,
        "invocation": invocation,
        "resumed_from": str(resume_adapter_file) if resume_adapter_file else None,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_seconds": (completed_at - started_at).total_seconds(),
        "config": {
            "base_model": config.base_model,
            "fine_tune_type": config.fine_tune_type,
            "num_layers": config.num_layers,
            "iters": config.iters,
            "batch_size": config.batch_size,
            "learning_rate": config.learning_rate,
            "grad_accumulation_steps": config.grad_accumulation_steps,
            "max_seq_length": config.max_seq_length,
            "optimizer": config.optimizer,
            "mask_prompt": config.mask_prompt,
            "seed": config.seed,
            "val_fraction": config.val_fraction,
        },
        "mix_composition": dict(mix.composition),
        "dataset_sizes": {
            "train": train_count,
            "valid": valid_count,
            "total_examples_in_mix": len(mix.examples),
        },
        "source_memory_ids": source_memory_ids,
        "versions": _versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def train_adapter(
    mix: RehearsalMix,
    config: TrainConfig,
    run_dir: Path,
    *,
    log_stream=None,
    tag: str | None = None,
    notes: str | None = None,
    invocation: list[str] | None = None,
    resume_adapter_file: Path | None = None,
) -> TrainResult:
    """Train a LoRA adapter on the given mix, writing outputs under run_dir.

    Layout produced:
        run_dir/
            data/
                train.jsonl
                valid.jsonl
            adapter/
                adapters.safetensors
                adapter_config.json
            metadata.json

    Raises TrainError if mlx-lm exits non-zero.
    """
    if not mix.examples:
        raise ValueError("cannot train on empty mix")

    run_dir = Path(run_dir)
    data_dir = run_dir / "data"
    adapter_dir = run_dir / "adapter"
    data_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    train_examples, valid_examples = _split_train_valid(
        mix.examples, config.val_fraction, config.seed
    )
    _write_jsonl(train_examples, data_dir / "train.jsonl")
    _write_jsonl(valid_examples, data_dir / "valid.jsonl")

    started_at = datetime.now(UTC)
    cmd = _build_command(
        _resolve_python(), data_dir, adapter_dir, config, resume_adapter_file
    )

    stdout_target = log_stream if log_stream is not None else subprocess.PIPE
    completed = subprocess.run(
        cmd,
        stdout=stdout_target,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    completed_at = datetime.now(UTC)

    if completed.returncode != 0:
        captured = completed.stdout if log_stream is None else "(streamed)"
        raise TrainError(
            f"mlx-lm exited {completed.returncode}; output:\n{captured}"
        )

    adapter_path = adapter_dir / ADAPTER_FILENAME
    if not adapter_path.exists():
        raise TrainError(f"training completed but adapter not found at {adapter_path}")

    metadata_path = run_dir / METADATA_FILENAME
    metadata = _write_metadata(
        metadata_path,
        config=config,
        mix=mix,
        train_count=len(train_examples),
        valid_count=len(valid_examples),
        started_at=started_at,
        completed_at=completed_at,
        source_memory_ids=sorted({ex.source_memory_id for ex in mix.examples}),
        tag=tag,
        notes=notes,
        invocation=invocation,
        resume_adapter_file=resume_adapter_file,
    )
    return TrainResult(
        run_dir=run_dir,
        adapter_path=adapter_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

"""In-process OPLoRA training runner.

Standard (vanilla LoRA) training runs `python -m mlx_lm lora --train` as a
subprocess — see `dreamagent.train.runner`. That's the most stable path
because we don't depend on mlx_lm's internal Python API across versions.

OPLoRA can't use the subprocess path because we need to inject a custom
layer type into mlx_lm's `linear_to_lora_layers` step. So we replicate
mlx_lm.lora.train_model in process, but with `linear_to_oplora_layers`
instead of `linear_to_lora_layers`.

The TrainResult interface returned here matches `runner.train_adapter` so
the dream pipeline doesn't need to care which path produced the adapter.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types as _types_module
from datetime import UTC, datetime
from pathlib import Path

import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from dreamagent.compose.mix import RehearsalMix
from dreamagent.train.config import TrainConfig
from dreamagent.train.oplora import (
    freeze_oplora_projections,
    linear_to_oplora_layers,
)
from dreamagent.train.runner import (
    ADAPTER_FILENAME,
    METADATA_FILENAME,
    TrainError,
    TrainResult,
    _split_train_valid,
    _versions,
    _write_jsonl,
)


def _dataset_args(data_dir: Path, mask_prompt: bool) -> _types_module.SimpleNamespace:
    """Build the SimpleNamespace mlx_lm.tuner.datasets.load_dataset expects."""
    return _types_module.SimpleNamespace(
        data=str(data_dir),
        train=True,
        test=False,
        hf_dataset=False,
        mask_prompt=mask_prompt,
    )


def _build_adapter_config(
    config: TrainConfig, adapter_dir: Path
) -> dict:
    """Build the adapter_config.json contents. The format is a superset of
    mlx_lm's standard lora adapter_config — it adds use_oplora and
    oplora_k_singular so the loader can rebuild OPLoRA layers."""
    return {
        "model": config.base_model,
        "fine_tune_type": "lora",
        # OPLoRA-specific fields:
        "use_oplora": True,
        "oplora_k_singular": config.oplora_k_singular,
        # Standard lora_parameters block — what linear_to_lora_layers reads:
        "num_layers": config.num_layers,
        "lora_parameters": {
            "rank": config.lora_rank,
            "dropout": config.lora_dropout,
            "scale": config.lora_scale,
        },
        # Bookkeeping for human inspection:
        "iters": config.iters,
        "batch_size": config.batch_size,
        "learning_rate": config.learning_rate,
        "max_seq_length": config.max_seq_length,
        "optimizer": config.optimizer,
        "seed": config.seed,
        "grad_accumulation_steps": config.grad_accumulation_steps,
        "mask_prompt": config.mask_prompt,
        "adapter_path": str(adapter_dir),
    }


def _select_optimizer(name: str, lr: float):
    name = name.lower()
    if name == "adam":
        return optim.Adam(learning_rate=lr)
    if name == "adamw":
        return optim.AdamW(learning_rate=lr)
    if name == "sgd":
        return optim.SGD(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer for OPLoRA: {name}")


def train_adapter_oplora(
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
    """Train an OPLoRA adapter in-process.

    Same input/output contract as `runner.train_adapter` so the dream
    pipeline can dispatch transparently.
    """
    if not mix.examples:
        raise ValueError("cannot train on empty mix")
    if not config.use_oplora:
        raise ValueError(
            "train_adapter_oplora called with use_oplora=False — "
            "use dreamagent.train.runner.train_adapter instead"
        )

    # Imports kept inside the function so unit tests that don't need mlx_lm
    # don't pay the import cost.
    from mlx_lm.tuner.datasets import CacheDataset, load_dataset
    from mlx_lm.tuner.trainer import TrainingArgs, train as run_train
    from mlx_lm.tuner.utils import print_trainable_parameters
    from mlx_lm.utils import load

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

    # Redirect mlx_lm's progress prints to the user's log stream so they
    # appear in the same place as subprocess training would.
    sink = log_stream if log_stream is not None else sys.stdout
    ctx = contextlib.redirect_stdout(sink) if log_stream is not None else contextlib.nullcontext()

    with ctx:
        mx.random.seed(config.seed)

        print(f"[oplora] Loading base model: {config.base_model}")
        model, tokenizer = load(
            config.base_model, tokenizer_config={"trust_remote_code": True}
        )

        print(f"[oplora] Loading datasets from {data_dir}")
        train_set, valid_set, _ = load_dataset(
            _dataset_args(data_dir, config.mask_prompt), tokenizer
        )

        adapter_config_dict = _build_adapter_config(config, adapter_dir)
        (adapter_dir / "adapter_config.json").write_text(
            json.dumps(adapter_config_dict, indent=2) + "\n", encoding="utf-8"
        )

        if config.num_layers > len(model.layers):
            raise ValueError(
                f"Requested to train {config.num_layers} layers but model "
                f"only has {len(model.layers)} layers."
            )

        print("[oplora] Freezing base model")
        model.freeze()

        print(
            f"[oplora] Converting last {config.num_layers} layers to OPLoRA "
            f"(k_singular={config.oplora_k_singular}, rank={config.lora_rank}) — "
            f"this includes SVD on each wrapped weight, may take several minutes"
        )
        linear_to_oplora_layers(
            model,
            num_layers=config.num_layers,
            config={
                "rank": config.lora_rank,
                "scale": config.lora_scale,
                "dropout": config.lora_dropout,
            },
            k_singular=config.oplora_k_singular,
        )

        n_frozen = freeze_oplora_projections(model)
        print(f"[oplora] Froze U_k/V_k on {n_frozen} OPLoRA modules")

        if resume_adapter_file is not None:
            print(f"[oplora] Resuming from {resume_adapter_file}")
            model.load_weights(str(resume_adapter_file), strict=False)

        print_trainable_parameters(model)

        adapter_file = adapter_dir / ADAPTER_FILENAME
        training_args = TrainingArgs(
            batch_size=config.batch_size,
            iters=config.iters,
            val_batches=config.val_batches,
            steps_per_report=config.steps_per_report,
            steps_per_eval=config.steps_per_eval,
            steps_per_save=max(1, config.iters),  # final save only
            max_seq_length=config.max_seq_length,
            adapter_file=str(adapter_file),
            grad_checkpoint=False,
            grad_accumulation_steps=config.grad_accumulation_steps,
        )

        opt = _select_optimizer(config.optimizer, config.learning_rate)

        print("[oplora] Starting training loop")
        try:
            run_train(
                model=model,
                args=training_args,
                optimizer=opt,
                train_dataset=CacheDataset(train_set),
                val_dataset=CacheDataset(valid_set),
            )
        except Exception as e:
            raise TrainError(f"OPLoRA training failed: {e}") from e

    completed_at = datetime.now(UTC)

    adapter_path = adapter_dir / ADAPTER_FILENAME
    if not adapter_path.exists():
        raise TrainError(
            f"OPLoRA training completed but adapter not found at {adapter_path}"
        )

    metadata_path = run_dir / METADATA_FILENAME
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
            "fine_tune_type": "oplora",
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
            "lora_rank": config.lora_rank,
            "lora_scale": config.lora_scale,
            "lora_dropout": config.lora_dropout,
            "use_oplora": True,
            "oplora_k_singular": config.oplora_k_singular,
        },
        "mix_composition": dict(mix.composition),
        "dataset_sizes": {
            "train": len(train_examples),
            "valid": len(valid_examples),
            "total_examples_in_mix": len(mix.examples),
        },
        "source_memory_ids": sorted({ex.source_memory_id for ex in mix.examples}),
        "versions": _versions(),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    return TrainResult(
        run_dir=run_dir,
        adapter_path=adapter_path,
        metadata_path=metadata_path,
        metadata=metadata,
    )

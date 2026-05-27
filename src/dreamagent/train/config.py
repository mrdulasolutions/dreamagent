"""Training configuration — the knobs the user actually tunes."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """LoRA training hyperparameters.

    Defaults are tuned for the validation tier (Qwen 3 0.6B) — small, fast,
    conservative. The production tier configs in `configs/` override these.
    """

    base_model: str = "mlx-community/Qwen3-0.6B-4bit"

    # LoRA shape
    fine_tune_type: str = "lora"  # "lora" | "dora" | "full"
    num_layers: int = 8

    # Optimization
    iters: int = 200
    batch_size: int = 2
    learning_rate: float = 1.0e-4
    grad_accumulation_steps: int = 1
    max_seq_length: int = 1024
    optimizer: str = "adamw"
    mask_prompt: bool = True

    # Eval cadence inside MLX
    steps_per_report: int = 20
    steps_per_eval: int = 100
    val_batches: int = 4

    # Reproducibility
    seed: int = 42

    # Validation split as fraction of the mix; rest goes to train
    val_fraction: float = 0.1

    # ── OPLoRA (Path A · Week 1) ──
    # Orthogonal Projection LoRA: project the LoRA update orthogonal to the
    # top-k singular subspaces of each base weight. Reduces interference with
    # pretrained knowledge — see docs/research/2026-05-improving-memory.md
    # and src/dreamagent/train/oplora.py.
    use_oplora: bool = False
    oplora_k_singular: int = 32

    # ── LoRA shape knobs surfaced for tuning (also used by OPLoRA) ──
    lora_rank: int = 8
    lora_scale: float = 20.0
    lora_dropout: float = 0.0

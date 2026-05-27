"""Sparse Memory Finetuning (SMFT) — gradient sparsification wrapper.

The hypothesis: vanilla LoRA at rank 8 modifies ~5.2M trainable parameters
every step. Many of those modifications interfere with prior knowledge
without being necessary for the new memory. SMFT addresses this by
zeroing all but the top-k% of gradient entries before each optimizer
step — the model only updates the parameters most relevant to the
current batch.

Reference: "Sparse Memory Finetuning" (https://arxiv.org/pdf/2510.15103)
which reported 11% performance drop vs 71% for vanilla LoRA on
continual-learning benchmarks — a 6× reduction in catastrophic
forgetting.

Implementation: a thin optimizer wrapper that pre-processes the gradient
tree before delegating to a base optimizer (Adam, AdamW, etc.). The
sparsity is applied per-tensor: for each parameter tensor, keep only the
top `smft_sparsity` fraction of entries by absolute gradient magnitude.

Per-tensor (not global) sparsity is the v1 choice. Global top-k would
require flattening all gradients into one vector and a single percentile;
that's more expensive and slightly harder to make stable. Per-tensor
gives each LoRA matrix its own update budget proportional to its size.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
from mlx.utils import tree_map

__all__ = [
    "SparseMemoryOptimizer",
    "sparsify_gradient",
]


def sparsify_gradient(
    grad: mx.array,
    sparsity_fraction: float,
) -> mx.array:
    """Keep the top-`sparsity_fraction` of entries by |grad|; zero the rest.

    Args:
        grad: gradient tensor of any shape.
        sparsity_fraction: fraction of entries to KEEP. e.g. 0.10 keeps
            the top 10% by magnitude. Must be in (0, 1].

    Returns:
        A tensor of the same shape and dtype as `grad`, with entries below
        the (1 - sparsity_fraction) quantile of |grad| set to zero.

    Notes:
        - sparsity_fraction=1.0 returns the input unchanged.
        - For tensors with grad.size == 0, returns the input.
        - The kth-largest threshold is computed via mx.sort on the
          flattened |grad|. For large tensors this is O(n log n) but
          mlx's sort is fast on Metal.
    """
    if sparsity_fraction >= 1.0:
        return grad
    if sparsity_fraction <= 0.0:
        raise ValueError(
            f"sparsity_fraction must be in (0, 1], got {sparsity_fraction}"
        )
    if grad.size == 0:
        return grad

    abs_grad = mx.abs(grad)
    flat = abs_grad.reshape(-1)
    # k = number of entries to KEEP (top-k by magnitude).
    k = max(1, int(grad.size * sparsity_fraction))
    # Sort ascending; threshold is the (size - k)-th element (everything
    # >= threshold is in the top k).
    sorted_asc = mx.sort(flat)
    threshold = sorted_asc[-k]
    mask = abs_grad >= threshold
    return grad * mask.astype(grad.dtype)


class SparseMemoryOptimizer:
    """Wraps a base optimizer and applies per-tensor gradient sparsification
    before each update.

    Quacks like an mlx.optimizers.Optimizer (has `update`, `learning_rate`,
    `state`) so it can be substituted for one in mlx_lm's training loop.

    Usage:
        base = mx.optimizers.AdamW(learning_rate=3e-5)
        opt = SparseMemoryOptimizer(base, sparsity_fraction=0.10)
        opt.update(model, gradients)   # masks gradients, then calls base.update
    """

    def __init__(
        self,
        base_optimizer: Any,
        sparsity_fraction: float = 0.10,
    ):
        if not (0.0 < sparsity_fraction <= 1.0):
            raise ValueError(
                f"sparsity_fraction must be in (0, 1], got {sparsity_fraction}"
            )
        self._base = base_optimizer
        self.sparsity_fraction = float(sparsity_fraction)

    # ── Optimizer-protocol surface ──

    @property
    def learning_rate(self):
        return self._base.learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        self._base.learning_rate = value

    @property
    def state(self):
        return self._base.state

    @property
    def step(self):
        return self._base.step

    def init(self, parameters: dict) -> None:
        self._base.init(parameters)

    # ── The main hook ──

    def _sparsify_tree(self, gradients: dict) -> dict:
        """Apply per-tensor top-k sparsification across the gradient tree."""
        return tree_map(
            lambda g: sparsify_gradient(g, self.sparsity_fraction)
            if isinstance(g, mx.array)
            else g,
            gradients,
        )

    def update(self, model, gradients: dict) -> None:
        """Mask gradients, then delegate to the base optimizer's update."""
        masked = self._sparsify_tree(gradients)
        return self._base.update(model, masked)

    def apply_gradients(self, gradients: dict, parameters: dict) -> dict:
        """Mask gradients, then delegate to the base optimizer's apply_gradients."""
        masked = self._sparsify_tree(gradients)
        return self._base.apply_gradients(masked, parameters)

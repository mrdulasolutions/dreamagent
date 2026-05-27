"""OPLoRA — Orthogonal Projection LoRA.

LoRA's catastrophic forgetting comes from the low-rank update writing into
the same singular subspaces that encode the base model's pretrained
knowledge. OPLoRA mitigates this by projecting the LoRA update so it is
orthogonal to the top-k principal subspaces of each base weight matrix W.

    Standard LoRA:  W' = W + scale · (B.T @ A.T)
    OPLoRA:         W' = W + scale · P_L @ (B.T @ A.T) @ P_R

where W = U Σ V^T is the SVD of the (dequantized) base weight, and

    P_L = I - U_k @ U_k.T     (output-side projection, removes top-k left subspace)
    P_R = I - V_k @ V_k.T     (input-side projection,  removes top-k right subspace)

The projection components U_k = U[:, :k], V_k = V[:, :k] are precomputed once
per layer at adapter construction time and frozen during training. lora_a and
lora_b remain trainable exactly as in standard LoRA.

This is dreamagent's first attempt at a capability improvement after the V2.1
and V2.2 retractions; the research case for it is in
`docs/research/2026-05-improving-memory.md`.

Reference:
    "Pareto Low-Rank Adapters" (https://arxiv.org/abs/2407.08394)
    and the catastrophic-forgetting reduction analysis in
    "CL-LoRA: Continual Low-Rank Adaptation" (CVPR 2025).
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_unflatten

__all__ = [
    "OPLoRALinear",
    "compute_op_projections",
    "linear_to_oplora_layers",
    "freeze_oplora_projections",
]


# ─── helpers ────────────────────────────────────────────────────────────────


def _dequantize_if_needed(linear) -> mx.array:
    """Return the linear's weight as a dense matrix, dequantizing if quantized."""
    if isinstance(linear, nn.QuantizedLinear):
        return mx.dequantize(
            linear.weight,
            linear.scales,
            linear.biases,
            group_size=linear.group_size,
            bits=linear.bits,
            mode=linear.mode,
        )
    return linear.weight


def compute_op_projections(
    weight: mx.array, k_singular: int
) -> tuple[mx.array, mx.array]:
    """Compute the top-k left and right singular vectors of `weight`.

    Args:
        weight: shape (output_dims, input_dims), any dtype.
        k_singular: number of singular vectors to retain.

    Returns:
        (U_k, V_k) where:
          - U_k has shape (output_dims, k)  (top-k left singular vectors)
          - V_k has shape (input_dims, k)   (top-k right singular vectors)

    SVD is performed in float32 on CPU (via numpy) for numerical stability,
    then results are cast back to the original dtype.
    """
    if k_singular <= 0:
        raise ValueError(f"k_singular must be > 0, got {k_singular}")

    target_dtype = weight.dtype
    # Move to numpy float32 for SVD — mlx.linalg.svd lacks broad coverage.
    w_np = np.array(weight.astype(mx.float32))
    # full_matrices=False: U (m, min(m,n)), S (min,), Vh (min(m,n), n).
    U_np, _S_np, Vh_np = np.linalg.svd(w_np, full_matrices=False)
    k = min(k_singular, U_np.shape[1])
    U_k = mx.array(U_np[:, :k]).astype(target_dtype)
    V_k = mx.array(Vh_np[:k, :].T).astype(target_dtype)
    return U_k, V_k


# ─── the layer ──────────────────────────────────────────────────────────────


class OPLoRALinear(nn.Module):
    """LoRA whose low-rank update is projected orthogonal to top-k singular
    subspaces of the base weight.

    Forward pass:
        y     = base_linear(x)
        x_p   = x  -  (x @ V_k) @ V_k.T          # P_R @ x
        z     = (dropout(x_p) @ lora_a) @ lora_b
        z_p   = z  -  (z @ U_k) @ U_k.T          # P_L @ z
        return y + scale * z_p

    The projection arrays U_k and V_k are non-trainable (frozen via
    `freeze_oplora_projections` after layer conversion). lora_a and lora_b
    train as in standard LoRA.

    The arrays U_k, V_k are saved into the adapter safetensors alongside
    lora_a/lora_b so the adapter is self-describing. At reload time,
    `load_oplora_adapter` rebuilds the OPLoRA layers and loads all four
    arrays.
    """

    # ── construction ──

    @staticmethod
    def from_base(
        linear,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        k_singular: int = 32,
        skip_svd: bool = False,
    ) -> "OPLoRALinear":
        """Build an OPLoRALinear that wraps `linear`.

        When `skip_svd=False` (default), runs SVD on the base weight and
        populates U_k/V_k from the top-k singular vectors. Use this for
        fresh adapter creation (training).

        When `skip_svd=True`, leaves U_k/V_k as the zero placeholders
        allocated in __init__. Use this for adapter loading, where the real
        U_k/V_k will be loaded from the saved safetensors via load_weights.
        """
        output_dims, input_dims = linear.weight.shape
        if isinstance(linear, nn.QuantizedLinear):
            input_dims = input_dims * 32 // linear.bits

        op = OPLoRALinear(
            input_dims=input_dims,
            output_dims=output_dims,
            r=r,
            dropout=dropout,
            scale=scale,
            k_singular=k_singular,
        )
        op.linear = linear

        if not skip_svd:
            weight = _dequantize_if_needed(linear)
            U_k, V_k = compute_op_projections(weight, k_singular=k_singular)
            op.U_k = U_k
            op.V_k = V_k
        return op

    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        r: int = 8,
        dropout: float = 0.0,
        scale: float = 20.0,
        k_singular: int = 32,
        bias: bool = False,
    ):
        super().__init__()

        # Frozen base linear (replaced by from_base()).
        self.linear = nn.Linear(input_dims, output_dims, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.scale = scale
        # Keep these as Python ints for serialization clarity.
        self.k_singular = int(k_singular)
        self.r = int(r)
        self.input_dims = int(input_dims)
        self.output_dims = int(output_dims)

        # Trainable low-rank weights (same init scheme as mlx_lm.LoRALinear).
        init_scale = 1.0 / math.sqrt(input_dims)
        self.lora_a = mx.random.uniform(
            low=-init_scale,
            high=init_scale,
            shape=(input_dims, r),
        )
        self.lora_b = mx.zeros(shape=(r, output_dims))

        # Frozen orthogonal projection components (from_base() overwrites
        # these with real SVD values). Allocated as zeros so the Module is
        # well-formed even before from_base() runs.
        k_eff = min(k_singular, min(input_dims, output_dims))
        self.U_k = mx.zeros(shape=(output_dims, k_eff))
        self.V_k = mx.zeros(shape=(input_dims, k_eff))

    # ── forward ──

    def __call__(self, x):
        y = self.linear(x)

        # Stop gradient through projections — they're frozen but we add an
        # explicit stop_gradient to keep the autograd graph clean.
        Vk = mx.stop_gradient(self.V_k)
        Uk = mx.stop_gradient(self.U_k)

        # P_R @ x  =  x - (x @ V_k) @ V_k.T
        x_p = x - (x @ Vk) @ Vk.T
        z = (self.dropout(x_p) @ self.lora_a) @ self.lora_b
        # P_L @ z  =  z - (z @ U_k) @ U_k.T
        z_p = z - (z @ Uk) @ Uk.T

        return y + (self.scale * z_p).astype(x.dtype)

    # ── fusing (for inference deployment) ──

    def fuse(self, dequantize: bool = False):
        """Fuse the projected low-rank update into a single nn.Linear (or
        QuantizedLinear).

        Mirrors mlx_lm.tuner.lora.LoRALinear.fuse() so the rest of the
        toolchain (e.g. mlx_lm.fuse) can treat OPLoRA adapters uniformly.
        """
        linear = self.linear
        bias = "bias" in linear
        weight = _dequantize_if_needed(linear)
        is_quantized = isinstance(linear, nn.QuantizedLinear)

        output_dims, input_dims = weight.shape
        fused_linear = nn.Linear(input_dims, output_dims, bias=bias)

        # Unprojected ΔW = scale · (lora_b.T @ lora_a.T) — shape (out, in).
        delta = (self.scale * self.lora_b.T) @ self.lora_a.T
        # Apply P_L on the left: delta - U_k @ (U_k.T @ delta)
        delta = delta - self.U_k @ (self.U_k.T @ delta)
        # Apply P_R on the right: delta - (delta @ V_k) @ V_k.T
        delta = delta - (delta @ self.V_k) @ self.V_k.T
        delta = delta.astype(weight.dtype)

        fused_linear.weight = weight + delta
        if bias:
            fused_linear.bias = linear.bias
        if is_quantized and not dequantize:
            fused_linear = nn.QuantizedLinear.from_linear(
                fused_linear,
                linear.group_size,
                linear.bits,
                mode=linear.mode,
            )
        return fused_linear


# ─── model-level conversion ─────────────────────────────────────────────────


def linear_to_oplora_layers(
    model: nn.Module,
    num_layers: int,
    config: dict,
    k_singular: int = 32,
    skip_svd: bool = False,
) -> None:
    """Convert linear layers in the last `num_layers` blocks of `model` to
    OPLoRALinear, in place.

    Mirrors `mlx_lm.tuner.utils.linear_to_lora_layers` but with OPLoRA. Falls
    back to plain LoRA for embedding and switch-linear layers (those don't
    have a natural SVD-based projection in the same shape).

    Args:
        model: an mlx-lm-shaped model (must have `.layers`).
        num_layers: how many trailing transformer blocks to wrap.
        config: dict with keys "rank", "scale", "dropout" (matches mlx_lm's
            lora_parameters); may also include "keys" to restrict which
            sub-modules get wrapped.
        k_singular: how many top singular vectors to project away.
        skip_svd: when True, leave U_k/V_k as zero placeholders. Used by the
            adapter-loading path where U_k/V_k are loaded from disk anyway.
    """
    # Imported lazily — avoids forcing mlx_lm import at module load time
    # (helps unit tests that don't need the model).
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear
    from mlx_lm.tuner.lora import LoRAEmbedding, LoRASwitchLinear

    rank = config["rank"]
    scale = config["scale"]
    dropout = config["dropout"]

    def to_oplora(layer):
        if isinstance(layer, (nn.Linear, nn.QuantizedLinear)):
            return OPLoRALinear.from_base(
                layer,
                r=rank,
                scale=scale,
                dropout=dropout,
                k_singular=k_singular,
                skip_svd=skip_svd,
            )
        if isinstance(layer, (nn.Embedding, nn.QuantizedEmbedding)):
            return LoRAEmbedding.from_base(layer, r=rank, scale=scale, dropout=dropout)
        if isinstance(layer, (SwitchLinear, QuantizedSwitchLinear)):
            return LoRASwitchLinear.from_base(
                layer, r=rank, scale=scale, dropout=dropout
            )
        raise ValueError(
            f"Can't convert layer of type {type(layer).__name__} to OPLoRA"
        )

    keys = config.get("keys")
    if keys is None:
        keys = set()
        types_to_match = (
            nn.Linear,
            nn.QuantizedLinear,
            nn.Embedding,
            nn.QuantizedEmbedding,
            SwitchLinear,
            QuantizedSwitchLinear,
        )

        def collect(p, m):
            if hasattr(m, "to_lora") or isinstance(m, types_to_match):
                keys.add(p)

        for l in model.layers:
            l.apply_to_modules(collect)

    for l in model.layers[-max(num_layers, 0) :]:
        op_layers = [(k, to_oplora(m)) for k, m in l.named_modules() if k in keys]
        if op_layers:
            l.update_modules(tree_unflatten(op_layers))

    op_modules = [(k, to_oplora(m)) for k, m in model.named_modules() if k in keys]
    if op_modules:
        model.update_modules(tree_unflatten(op_modules))


def freeze_oplora_projections(model: nn.Module) -> int:
    """Freeze U_k and V_k on every OPLoRALinear in the model.

    Returns the number of OPLoRALinear modules touched.

    Must be called AFTER `linear_to_oplora_layers` (and after any prior
    `model.freeze()` of the whole base model — those freezes don't include
    the new lora_a, lora_b, U_k, V_k arrays created by OPLoRA construction).
    """
    n = 0
    for _name, mod in model.named_modules():
        if isinstance(mod, OPLoRALinear):
            mod.freeze(keys=["U_k", "V_k"], recurse=False)
            n += 1
    return n

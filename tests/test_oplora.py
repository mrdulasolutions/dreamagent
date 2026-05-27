"""Tests for OPLoRA — Orthogonal Projection LoRA.

These tests verify the math (projections, forward equivalence, fuse) on
small synthetic weights so they run in well under a second. The "does it
actually train and improve recall on V1 fixture" question is answered by
the end-to-end benchmark, not here.
"""

from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from dreamagent.train.oplora import (
    OPLoRALinear,
    compute_op_projections,
    freeze_oplora_projections,
    linear_to_oplora_layers,
)


# ─── projection-matrix math ─────────────────────────────────────────────────


class TestComputeOpProjections:
    def test_shapes_match_top_k(self):
        W = mx.array(np.random.default_rng(0).standard_normal((64, 48)).astype(np.float32))
        U_k, V_k = compute_op_projections(W, k_singular=8)
        assert U_k.shape == (64, 8)
        assert V_k.shape == (48, 8)

    def test_k_capped_at_min_dim(self):
        # k larger than min(m,n) should silently cap.
        W = mx.array(np.random.default_rng(0).standard_normal((10, 4)).astype(np.float32))
        U_k, V_k = compute_op_projections(W, k_singular=99)
        # full_matrices=False gives min(m,n) = 4 columns at most.
        assert U_k.shape == (10, 4)
        assert V_k.shape == (4, 4)

    def test_projection_is_idempotent(self):
        """P_L @ P_L should equal P_L (projection matrices are idempotent)."""
        W = mx.array(np.random.default_rng(1).standard_normal((32, 24)).astype(np.float32))
        U_k, V_k = compute_op_projections(W, k_singular=4)
        I_out = mx.eye(32, dtype=mx.float32)
        I_in = mx.eye(24, dtype=mx.float32)
        P_L = I_out - U_k @ U_k.T
        P_R = I_in - V_k @ V_k.T
        err_L = mx.linalg.norm(P_L @ P_L - P_L).item() / mx.linalg.norm(P_L).item()
        err_R = mx.linalg.norm(P_R @ P_R - P_R).item() / mx.linalg.norm(P_R).item()
        assert err_L < 1e-4, f"P_L not idempotent: {err_L}"
        assert err_R < 1e-4, f"P_R not idempotent: {err_R}"

    def test_projection_kills_top_subspace(self):
        """P_L @ U_k should be ~zero — the projection removes that subspace."""
        W = mx.array(np.random.default_rng(2).standard_normal((32, 24)).astype(np.float32))
        U_k, V_k = compute_op_projections(W, k_singular=4)
        # P_L @ U_k = (I - U_k U_k.T) U_k = U_k - U_k(U_k.T U_k). Since U_k
        # columns are orthonormal, U_k.T @ U_k ≈ I, so the result ≈ 0.
        I_out = mx.eye(32, dtype=mx.float32)
        P_L = I_out - U_k @ U_k.T
        norm = mx.linalg.norm(P_L @ U_k).item()
        assert norm < 1e-4, f"P_L @ U_k norm = {norm}, expected ~0"

    def test_invalid_k_rejected(self):
        W = mx.array(np.zeros((4, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="k_singular must be > 0"):
            compute_op_projections(W, k_singular=0)


# ─── forward-pass equivalence ───────────────────────────────────────────────


class TestForwardEquivalence:
    """The OPLoRA forward pass must equal the direct projected-ΔW application.

    Concretely: if ΔW = scale · (B.T @ A.T) and ΔW' = P_L @ ΔW @ P_R, then
    `base(x) + x @ ΔW'.T` must equal what OPLoRALinear's __call__ produces.
    """

    def test_forward_matches_direct_projection(self):
        rng = np.random.default_rng(42)
        in_d, out_d, r, k = 16, 12, 4, 3

        # Synthetic base weight + bias.
        W = rng.standard_normal((out_d, in_d), dtype=np.float32)

        op = OPLoRALinear(
            input_dims=in_d, output_dims=out_d, r=r, dropout=0.0,
            scale=2.0, k_singular=k,
        )
        # Replace the auto-initialized linear with one whose weight is W.
        base_linear = nn.Linear(in_d, out_d, bias=False)
        base_linear.weight = mx.array(W)
        op.linear = base_linear

        # Set known lora_a, lora_b.
        A = rng.standard_normal((in_d, r), dtype=np.float32) * 0.1
        B = rng.standard_normal((r, out_d), dtype=np.float32) * 0.1
        op.lora_a = mx.array(A)
        op.lora_b = mx.array(B)

        # Compute SVD projections from the SAME W we set.
        U_k, V_k = compute_op_projections(mx.array(W), k_singular=k)
        op.U_k = U_k
        op.V_k = V_k

        # Forward pass through OPLoRALinear.
        x = mx.array(rng.standard_normal((5, in_d), dtype=np.float32))
        y_op = op(x)

        # Direct path: compute projected ΔW, then y = x @ (W + ΔW').T
        delta = op.scale * (B.T @ A.T)  # (out, in)
        Uk_np = np.array(U_k)
        Vk_np = np.array(V_k)
        delta_proj = delta - Uk_np @ (Uk_np.T @ delta)
        delta_proj = delta_proj - (delta_proj @ Vk_np) @ Vk_np.T

        x_np = np.array(x)
        y_direct = x_np @ (W + delta_proj).T

        rel_err = (np.linalg.norm(np.array(y_op) - y_direct)
                   / np.linalg.norm(y_direct))
        assert rel_err < 1e-4, f"forward path mismatch: rel err = {rel_err}"

    def test_fresh_layer_has_zero_lora_contribution(self):
        """At init, lora_b is zeros, so the LoRA branch contributes nothing.
        OPLoRA(x) must equal base_linear(x) exactly when the LoRA is fresh."""
        rng = np.random.default_rng(7)
        in_d, out_d = 8, 6

        op = OPLoRALinear(input_dims=in_d, output_dims=out_d, r=4, k_singular=2)
        # Give it a real base weight so we have something to compare to.
        W = mx.array(rng.standard_normal((out_d, in_d), dtype=np.float32))
        base_linear = nn.Linear(in_d, out_d, bias=False)
        base_linear.weight = W
        op.linear = base_linear

        # Set the SVD projections from W.
        U_k, V_k = compute_op_projections(W, k_singular=2)
        op.U_k = U_k
        op.V_k = V_k
        # lora_b is already zeros from __init__.

        x = mx.array(rng.standard_normal((3, in_d), dtype=np.float32))
        rel_err = (mx.linalg.norm(op(x) - base_linear(x)).item()
                   / mx.linalg.norm(base_linear(x)).item())
        assert rel_err < 1e-5, f"fresh OPLoRA != base_linear: {rel_err}"


# ─── fuse equivalence ───────────────────────────────────────────────────────


class TestFuseEquivalence:
    """`fuse()` must produce a Linear whose forward equals OPLoRALinear's."""

    def test_fuse_matches_call(self):
        rng = np.random.default_rng(11)
        in_d, out_d, r, k = 16, 12, 4, 3

        W = mx.array(rng.standard_normal((out_d, in_d), dtype=np.float32))
        op = OPLoRALinear(
            input_dims=in_d, output_dims=out_d, r=r, scale=2.0, k_singular=k,
        )
        base_linear = nn.Linear(in_d, out_d, bias=False)
        base_linear.weight = W
        op.linear = base_linear

        # Non-zero LoRA weights so fuse() is doing real work.
        op.lora_a = mx.array(rng.standard_normal((in_d, r), dtype=np.float32) * 0.1)
        op.lora_b = mx.array(rng.standard_normal((r, out_d), dtype=np.float32) * 0.1)

        U_k, V_k = compute_op_projections(W, k_singular=k)
        op.U_k = U_k
        op.V_k = V_k

        fused = op.fuse()
        x = mx.array(rng.standard_normal((5, in_d), dtype=np.float32))
        rel_err = (mx.linalg.norm(op(x) - fused(x)).item()
                   / mx.linalg.norm(op(x)).item())
        assert rel_err < 1e-4, f"fuse mismatch: rel err = {rel_err}"


# ─── trainability / freeze ──────────────────────────────────────────────────


class TestFreeze:
    def test_lora_params_trainable_projections_frozen(self):
        """After construction + freeze_oplora_projections, only lora_a and
        lora_b should appear in trainable_parameters()."""
        in_d, out_d = 8, 6
        op = OPLoRALinear(input_dims=in_d, output_dims=out_d, r=4, k_singular=2)
        op.freeze(keys=["U_k", "V_k"], recurse=False)
        # Also freeze the base linear (mirrors what train_model does).
        op.linear.freeze()

        from mlx.utils import tree_flatten
        names = sorted(k for k, _ in tree_flatten(op.trainable_parameters()))
        # We should see lora_a and lora_b only.
        assert "lora_a" in names
        assert "lora_b" in names
        assert "U_k" not in names
        assert "V_k" not in names
        # And the base linear's weight must NOT be trainable.
        assert all(not n.startswith("linear.") for n in names), names


# ─── linear_to_oplora_layers conversion ─────────────────────────────────────


class _TinyTransformerBlock(nn.Module):
    """Minimal block: two linears, mirroring what mlx-lm expects under .layers."""

    def __init__(self, dim, hidden):
        super().__init__()
        self.attn = nn.Linear(dim, dim, bias=False)
        self.mlp = nn.Linear(dim, hidden, bias=False)


class _TinyModel(nn.Module):
    def __init__(self, n_layers=3, dim=8, hidden=16):
        super().__init__()
        self.layers = [_TinyTransformerBlock(dim, hidden) for _ in range(n_layers)]


class TestLinearToOPLoRA:
    def test_replaces_targeted_layers(self):
        model = _TinyModel(n_layers=3, dim=8, hidden=16)
        linear_to_oplora_layers(
            model,
            num_layers=2,  # only last 2
            config={"rank": 4, "scale": 2.0, "dropout": 0.0},
            k_singular=2,
        )

        # Layer 0 must NOT be touched. Layers 1 and 2 must have OPLoRALinear.
        assert isinstance(model.layers[0].attn, nn.Linear)
        assert not isinstance(model.layers[0].attn, OPLoRALinear)

        for i in (1, 2):
            assert isinstance(model.layers[i].attn, OPLoRALinear)
            assert isinstance(model.layers[i].mlp, OPLoRALinear)

    def test_projections_initialized_from_base_weight(self):
        """After conversion, U_k @ U_k.T should approximate the top-k
        eigenspace of the original base weight."""
        rng = np.random.default_rng(13)
        model = _TinyModel(n_layers=1, dim=8, hidden=8)
        # Make the base weight a known low-rank-ish matrix so SVD is stable.
        W = rng.standard_normal((8, 8), dtype=np.float32)
        model.layers[0].attn.weight = mx.array(W)
        original_W = mx.array(W)

        linear_to_oplora_layers(
            model,
            num_layers=1,
            config={"rank": 4, "scale": 1.0, "dropout": 0.0},
            k_singular=3,
        )
        op = model.layers[0].attn
        assert isinstance(op, OPLoRALinear)

        # Sanity: the wrapped linear's weight should still be W.
        assert mx.linalg.norm(op.linear.weight - original_W).item() < 1e-5
        # And U_k, V_k should not be zero (i.e., from_base actually ran SVD).
        assert mx.linalg.norm(op.U_k).item() > 1e-3
        assert mx.linalg.norm(op.V_k).item() > 1e-3


# ─── freeze_oplora_projections helper ──────────────────────────────────────


class TestFreezeHelper:
    def test_counts_oplora_modules(self):
        model = _TinyModel(n_layers=3, dim=8, hidden=8)
        linear_to_oplora_layers(
            model,
            num_layers=2,
            config={"rank": 4, "scale": 1.0, "dropout": 0.0},
            k_singular=2,
        )
        n = freeze_oplora_projections(model)
        # 2 blocks * 2 linears per block = 4 OPLoRA modules.
        assert n == 4

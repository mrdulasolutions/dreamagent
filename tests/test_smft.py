"""Tests for Sparse Memory Finetuning — gradient sparsification math.

End-to-end "does it actually help" is the benchmark's job (task #72).
These tests verify the sparsifier behaves as specified on synthetic
gradients.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.optimizers as optim
import mlx.nn as nn
import numpy as np
import pytest

from dreamagent.train.smft import SparseMemoryOptimizer, sparsify_gradient


# ─── sparsify_gradient ────────────────────────────────────────────────────


class TestSparsifyGradient:
    def test_keeps_top_k_by_magnitude(self):
        # Construct a tensor where we know exactly which entries are top-k.
        g = mx.array([1.0, -10.0, 0.1, 5.0, -3.0, 0.01, 7.0, -0.5])
        # |g| = [1, 10, 0.1, 5, 3, 0.01, 7, 0.5]
        # Top 3 by |g|: indices 1 (10), 6 (7), 3 (5). Sparsity = 3/8 = 0.375.
        out = sparsify_gradient(g, sparsity_fraction=3 / 8)
        # Expect indices 1, 3, 6 preserved; others zero.
        kept = (np.array(mx.abs(out)) > 0).astype(int)
        np.testing.assert_array_equal(kept, [0, 1, 0, 1, 0, 0, 1, 0])
        # Values at kept positions match the original.
        np.testing.assert_allclose(np.array(out)[[1, 3, 6]], [-10.0, 5.0, 7.0])

    def test_sparsity_one_is_identity(self):
        g = mx.array([0.1, 0.2, 0.3, 0.4])
        out = sparsify_gradient(g, sparsity_fraction=1.0)
        np.testing.assert_array_equal(np.array(out), np.array(g))

    def test_minimum_one_kept(self):
        """Even with sparsity_fraction=0.001 on an 8-entry tensor, at least 1
        entry must remain (otherwise we'd zero everything)."""
        g = mx.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        out = sparsify_gradient(g, sparsity_fraction=0.001)
        n_nonzero = int(mx.sum((mx.abs(out) > 0).astype(mx.int32)))
        assert n_nonzero >= 1

    def test_empty_tensor_returns_unchanged(self):
        g = mx.zeros((0,))
        out = sparsify_gradient(g, sparsity_fraction=0.5)
        assert out.shape == (0,)

    def test_invalid_sparsity_rejected(self):
        g = mx.array([1.0, 2.0])
        with pytest.raises(ValueError, match="sparsity_fraction"):
            sparsify_gradient(g, sparsity_fraction=0.0)
        with pytest.raises(ValueError, match="sparsity_fraction"):
            sparsify_gradient(g, sparsity_fraction=-0.1)

    def test_dtype_preserved(self):
        g = mx.array([1.0, 2.0, 3.0, 4.0], dtype=mx.float16)
        out = sparsify_gradient(g, sparsity_fraction=0.5)
        assert out.dtype == mx.float16

    def test_multidim_tensor(self):
        """Sparsification should treat the whole tensor as one pool, not
        sparsify per-row or per-column."""
        rng = np.random.default_rng(0)
        g_np = rng.standard_normal((4, 6)).astype(np.float32)
        g = mx.array(g_np)
        out = sparsify_gradient(g, sparsity_fraction=0.25)
        kept = int(mx.sum((mx.abs(out) > 0).astype(mx.int32)))
        # 24 entries * 0.25 = 6 entries kept.
        assert kept == 6


# ─── SparseMemoryOptimizer wrapper ────────────────────────────────────────


class _ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = mx.zeros((4, 4))

    def __call__(self, x):
        return x @ self.w


class TestSparseMemoryOptimizer:
    def test_rejects_invalid_sparsity(self):
        base = optim.AdamW(learning_rate=1e-3)
        with pytest.raises(ValueError, match="sparsity_fraction"):
            SparseMemoryOptimizer(base, sparsity_fraction=0.0)
        with pytest.raises(ValueError, match="sparsity_fraction"):
            SparseMemoryOptimizer(base, sparsity_fraction=1.5)

    def test_forwards_lr_to_base(self):
        base = optim.AdamW(learning_rate=3e-5)
        opt = SparseMemoryOptimizer(base, sparsity_fraction=0.5)
        # learning_rate is an mx.array under the hood.
        assert float(opt.learning_rate) == pytest.approx(3e-5)

    def test_update_applies_masked_gradient(self):
        """After one update step with sparsity=0.5, exactly half the
        parameter entries should have changed from their initial value."""
        base = optim.SGD(learning_rate=1.0)  # SGD so updates are simple
        opt = SparseMemoryOptimizer(base, sparsity_fraction=0.5)

        # Construct a parameter and a gradient where exactly 2 of 4 entries
        # have non-zero magnitude after masking.
        params = {"w": mx.zeros((4,))}
        grads = {"w": mx.array([10.0, 0.01, 5.0, 0.02])}
        # Top-2 by magnitude: index 0 (10) and 2 (5).
        updated = opt.apply_gradients(grads, params)
        # SGD update: param - lr * grad. lr=1.0. So updated.w = 0 - 1*masked_grad.
        # masked_grad = [-10, 0, -5, 0], so updated.w = [-10, 0, -5, 0]. Wait —
        # SGD subtracts: new_param = param - lr*grad. So new_w = -masked_grad.
        # With our grad [10, 0.01, 5, 0.02] masked to [10, 0, 5, 0],
        # new_w = 0 - 1*[10, 0, 5, 0] = [-10, 0, -5, 0].
        expected = np.array([-10.0, 0.0, -5.0, 0.0])
        np.testing.assert_allclose(np.array(updated["w"]), expected)


# ─── Integration: optimizer wrapper composes with a real LoRA-style tree ─


class TestOptimizerIntegration:
    def test_nested_dict_gradient_tree(self):
        """Real mlx_lm gradients are nested dicts (model.layer.attn.lora_a etc).
        Make sure the tree_map walk handles nesting correctly."""
        base = optim.SGD(learning_rate=0.1)
        opt = SparseMemoryOptimizer(base, sparsity_fraction=0.5)

        params = {
            "layer0": {"lora_a": mx.zeros((4,)), "lora_b": mx.zeros((4,))},
            "layer1": {"lora_a": mx.zeros((4,))},
        }
        # Distinct magnitudes so we can predict the mask.
        grads = {
            "layer0": {
                "lora_a": mx.array([10.0, 0.1, 5.0, 0.2]),  # keep idx 0, 2
                "lora_b": mx.array([0.1, 0.2, 8.0, 4.0]),  # keep idx 2, 3
            },
            "layer1": {
                "lora_a": mx.array([1.0, 0.01, 0.5, 0.001]),  # keep idx 0, 2
            },
        }
        updated = opt.apply_gradients(grads, params)

        # Each updated param should equal -lr * masked_grad.
        np.testing.assert_allclose(
            np.array(updated["layer0"]["lora_a"]),
            [-1.0, 0.0, -0.5, 0.0],
        )
        np.testing.assert_allclose(
            np.array(updated["layer0"]["lora_b"]),
            [0.0, 0.0, -0.8, -0.4],
        )
        np.testing.assert_allclose(
            np.array(updated["layer1"]["lora_a"]),
            [-0.1, 0.0, -0.05, 0.0],
        )

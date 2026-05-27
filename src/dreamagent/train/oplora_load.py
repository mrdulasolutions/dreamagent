"""OPLoRA adapter loading.

The eval and serve paths currently call `mlx_lm.load(model_repo,
adapter_path=...)` which internally invokes
`mlx_lm.tuner.utils.load_adapters`. That helper rebuilds standard LoRA
layers from `adapter_config.json` — it has no knowledge of OPLoRA.

This module provides `load_model_with_optional_oplora` as a drop-in
replacement that detects the OPLoRA case (`use_oplora: true` in the
adapter config) and rebuilds OPLoRA layers instead, then loads the saved
weights.

Crucially, when loading we set `skip_svd=True` because the U_k/V_k arrays
will be loaded from the safetensors file — running SVD again would waste
~10 minutes of work on Llama 3.1 8B.
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_adapter_config(adapter_dir: Path) -> dict:
    cfg_path = adapter_dir / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")
    return json.loads(cfg_path.read_text())


def _normalize_adapter_dir(adapter_path) -> Path:
    """Accept either a path to adapters.safetensors or to its parent dir."""
    p = Path(adapter_path)
    return p.parent if p.is_file() else p


def is_oplora_adapter(adapter_path) -> bool:
    """Return True if the adapter at `adapter_path` is an OPLoRA adapter."""
    try:
        cfg = _read_adapter_config(_normalize_adapter_dir(adapter_path))
    except FileNotFoundError:
        return False
    return bool(cfg.get("use_oplora", False))


def load_model_with_optional_oplora(
    model_repo: str,
    adapter_path: str | Path | None = None,
):
    """Load a model and optionally an adapter, dispatching to OPLoRA if the
    adapter config flags it.

    Returns (model, tokenizer) — same shape as `mlx_lm.load`.
    """
    from mlx_lm import load
    from mlx_lm.tuner.utils import load_adapters

    if adapter_path is None:
        return load(model_repo)

    adapter_dir = _normalize_adapter_dir(adapter_path)
    cfg = _read_adapter_config(adapter_dir)

    if not cfg.get("use_oplora", False):
        # Standard LoRA — let mlx_lm handle it as before.
        return load(model_repo, adapter_path=str(adapter_dir))

    # OPLoRA path: load base model without adapter, then attach OPLoRA layers
    # and load weights ourselves.
    from dreamagent.train.oplora import (
        freeze_oplora_projections,
        linear_to_oplora_layers,
    )

    model, tokenizer = load(model_repo)

    lora_params = cfg.get("lora_parameters") or {}
    rank = lora_params.get("rank", 8)
    scale = lora_params.get("scale", 20.0)
    dropout = lora_params.get("dropout", 0.0)
    num_layers = cfg.get("num_layers", 16)
    k_singular = cfg.get("oplora_k_singular", 32)

    model.freeze()
    # skip_svd=True: U_k/V_k will be loaded from the adapter, no need to
    # spend ~10 minutes recomputing the SVD.
    linear_to_oplora_layers(
        model,
        num_layers=num_layers,
        config={"rank": rank, "scale": scale, "dropout": dropout},
        k_singular=k_singular,
        skip_svd=True,
    )
    freeze_oplora_projections(model)
    model.load_weights(str(adapter_dir / "adapters.safetensors"), strict=False)
    return model, tokenizer

"""MCP server exposing the dreamed model as a memory backend (V2.0 alpha).

Wraps the live adapter (from `runs/snapshots/live`) as a Model Context
Protocol server. Any MCP-capable client (Claude Code, Cursor, Hermes,
OpenClaw) can install this and gain a `query_memory` tool — calls return
the dreamed model's answer from weights, no retrieval, no third-party API.

Configuration via environment variables (because MCP servers are launched
by the host with no useful CLI args):

  DREAMAGENT_BASE_MODEL     — MLX repo or local path. Default: locked
                              V1 Llama 3.1 8B Instruct.
  DREAMAGENT_SNAPSHOTS_DIR  — Path to the snapshots directory containing
                              `live`. Default: ./runs/snapshots
  DREAMAGENT_MAX_TOKENS     — Max tokens per query response. Default: 128.

Usage standalone:
    dreamagent serve

Usage from Claude Code (mcpServers config):
    {
      "dreamagent": {
        "command": "uv",
        "args": ["run", "--directory", "/path/to/dreamagent", "dreamagent", "serve"],
        "env": {
          "DREAMAGENT_SNAPSHOTS_DIR": "/Users/me/runs/snapshots"
        }
      }
    }
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dreamagent.compose.templates import SYSTEM_PROMPT

DEFAULT_BASE_MODEL = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
DEFAULT_SNAPSHOTS_DIR = Path("runs") / "snapshots"
DEFAULT_MAX_TOKENS = 128


class _ServerState:
    """Lazy-loaded model state. Loaded on first query, not at import time.

    MCP servers should be cheap to start (the host may launch them
    speculatively before the user issues a tool call). We defer the
    ~5-10s model load until it's actually needed.
    """

    def __init__(self) -> None:
        self.model = None
        self.tokenizer = None
        self.adapter_dir: Path | None = None
        self.base_model: str = ""
        self.max_tokens: int = DEFAULT_MAX_TOKENS

    def configure_from_env(self) -> None:
        self.base_model = os.environ.get("DREAMAGENT_BASE_MODEL", DEFAULT_BASE_MODEL)
        snapshots_dir = Path(
            os.environ.get("DREAMAGENT_SNAPSHOTS_DIR", str(DEFAULT_SNAPSHOTS_DIR))
        )
        self.max_tokens = int(
            os.environ.get("DREAMAGENT_MAX_TOKENS", str(DEFAULT_MAX_TOKENS))
        )
        live = snapshots_dir / "live"
        if live.exists() or live.is_symlink():
            self.adapter_dir = (snapshots_dir / live.readlink()).resolve() / "adapter"
        else:
            self.adapter_dir = None  # serve base model with a warning

    def ensure_loaded(self) -> None:
        if self.model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError(
                "mlx_lm not installed; run `uv sync` from the project root"
            ) from e
        adapter_arg = str(self.adapter_dir) if self.adapter_dir is not None else None
        self.model, self.tokenizer = load(self.base_model, adapter_path=adapter_arg)


# Module-level singleton. FastMCP tool functions read this.
_state = _ServerState()


def _generate(question: str) -> str:
    """Generate the dreamed-model response for a question."""
    from mlx_lm import generate

    _state.ensure_loaded()
    assert _state.model is not None and _state.tokenizer is not None

    prompt = _state.tokenizer.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(
        _state.model,
        _state.tokenizer,
        prompt=prompt,
        max_tokens=_state.max_tokens,
        verbose=False,
    )


def build_server() -> Any:
    """Build the FastMCP server with the dreamagent tools registered.

    Returns the FastMCP instance, ready for `.run()`. Factored as a
    function so tests can instantiate without starting stdio.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError as e:
        raise RuntimeError(
            "mcp SDK not installed — install with `uv sync --extra mcp`"
        ) from e

    mcp = FastMCP("dreamagent")

    @mcp.tool()
    def query_memory(question: str) -> str:
        """Ask the user's dreamed memory specialist a question.

        The dreamed model has been fine-tuned on the user's structured
        memories (preferences, facts, procedures, events). It answers
        from its weights — there is no retrieval layer. If it doesn't
        know, it says so explicitly.

        Use this when you need to know what the user has previously told
        you about themselves, their preferences, their workflows, or
        their projects. Do NOT use this for general knowledge questions
        unrelated to the user.
        """
        return _generate(question)

    @mcp.tool()
    def query_memory_with_lineage(question: str) -> dict[str, Any]:
        """Same as query_memory, but returns a dict with the answer plus
        metadata about the model and the adapter that produced it.

        Use this when you need provenance — to know which adapter version
        produced the answer, what base model was used, and whether the
        model was operating from a dreamed adapter or the bare base model.
        """
        answer = _generate(question)
        return {
            "answer": answer,
            "base_model": _state.base_model,
            "adapter": (
                str(_state.adapter_dir.parent.name)
                if _state.adapter_dir is not None
                else None
            ),
            "warning": (
                "No live adapter loaded — answering from base model only."
                if _state.adapter_dir is None
                else None
            ),
        }

    return mcp


def run_stdio() -> None:
    """Start the MCP server on stdio. The entry point Claude Code calls."""
    _state.configure_from_env()
    mcp = build_server()
    mcp.run()  # FastMCP defaults to stdio transport

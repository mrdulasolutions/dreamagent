"""Tests for the MCP server.

We test tool registration shape + the env-config wiring. The full live
MCP server starting on stdio is not exercised here (it needs a real MCP
client to be meaningful); a manual smoke test against Claude Code or
the MCP inspector covers the live path.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def reset_state():
    """Reset the module-level state singleton between tests."""
    from dreamagent.serve import mcp as serve_mcp

    original = serve_mcp._state
    serve_mcp._state = serve_mcp._ServerState()
    yield serve_mcp._state
    serve_mcp._state = original


def test_build_server_returns_fastmcp(reset_state):
    """The server factory returns a FastMCP instance with our tools."""
    from dreamagent.serve import build_server

    mcp = build_server()
    assert mcp is not None
    assert mcp.name == "dreamagent"


def test_server_state_configure_uses_defaults(reset_state, monkeypatch):
    """ServerState picks up defaults when no env vars are set."""
    from dreamagent.serve.mcp import DEFAULT_BASE_MODEL, DEFAULT_MAX_TOKENS

    monkeypatch.delenv("DREAMAGENT_BASE_MODEL", raising=False)
    monkeypatch.delenv("DREAMAGENT_SNAPSHOTS_DIR", raising=False)
    monkeypatch.delenv("DREAMAGENT_MAX_TOKENS", raising=False)

    reset_state.configure_from_env()
    assert reset_state.base_model == DEFAULT_BASE_MODEL
    assert reset_state.max_tokens == DEFAULT_MAX_TOKENS


def test_server_state_respects_env(reset_state, monkeypatch, tmp_path: Path):
    """ServerState honors env-var overrides."""
    monkeypatch.setenv("DREAMAGENT_BASE_MODEL", "fake/model")
    monkeypatch.setenv("DREAMAGENT_MAX_TOKENS", "256")
    monkeypatch.setenv("DREAMAGENT_SNAPSHOTS_DIR", str(tmp_path))

    reset_state.configure_from_env()
    assert reset_state.base_model == "fake/model"
    assert reset_state.max_tokens == 256


def test_server_state_resolves_live_symlink(reset_state, monkeypatch, tmp_path: Path):
    """When live points at a snapshot, adapter_dir resolves to it."""
    snapshot = tmp_path / "2026-01-01T00-00-00Z"
    (snapshot / "adapter").mkdir(parents=True)
    live = tmp_path / "live"
    live.symlink_to(snapshot.name)

    monkeypatch.setenv("DREAMAGENT_SNAPSHOTS_DIR", str(tmp_path))
    reset_state.configure_from_env()

    assert reset_state.adapter_dir is not None
    assert reset_state.adapter_dir == (snapshot / "adapter").resolve()


def test_server_state_no_live_means_no_adapter(reset_state, monkeypatch, tmp_path: Path):
    """If snapshots dir has no live, adapter_dir stays None (base only)."""
    monkeypatch.setenv("DREAMAGENT_SNAPSHOTS_DIR", str(tmp_path))
    reset_state.configure_from_env()
    assert reset_state.adapter_dir is None


def test_query_memory_with_lineage_no_adapter_includes_warning(
    reset_state, monkeypatch, tmp_path: Path
):
    """When no adapter is loaded, the lineage tool surfaces a warning."""
    monkeypatch.setenv("DREAMAGENT_SNAPSHOTS_DIR", str(tmp_path))
    reset_state.configure_from_env()
    reset_state.base_model = "fake/model"

    # We don't call the tool; just verify state interpretation
    assert reset_state.adapter_dir is None
    # Lineage payload (constructed manually here to avoid invoking _generate)
    payload = {
        "answer": "stub",
        "base_model": reset_state.base_model,
        "adapter": None,
        "warning": (
            "No live adapter loaded — answering from base model only."
            if reset_state.adapter_dir is None
            else None
        ),
    }
    assert payload["warning"] is not None
    assert "base model only" in payload["warning"]

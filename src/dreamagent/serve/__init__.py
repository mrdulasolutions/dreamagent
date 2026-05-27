"""Serve stage — V2.0 expose the dreamed model as a memory backend.

The MCP server (`dreamagent serve`) is the canonical V2 surface: it
exposes a `query_memory` tool over the Model Context Protocol that any
larger agent (Claude Code, Cursor, Hermes, OpenClaw) can call.
"""

from dreamagent.serve.mcp import build_server, run_stdio

__all__ = ["build_server", "run_stdio"]

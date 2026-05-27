# 07 — MCP memory backend (V2.0 alpha — live)

**Status: alpha.** Ships in V2.0. The MCP server works today; the
"production hardening" items (signing, multi-user, mem0 composition)
land in V2.1+.

## The vision (delivered)

Your daily-driver agent — Claude Code, Cursor, Hermes, OpenClaw, or any
MCP-capable client — gets a single tool:

```
query_memory(question: str) -> str
```

backed by the dreamed model. The agent never sees the underlying memories
— only the synthesized answer. Privacy is preserved; the agent's
reasoning gets long-memory grounding without giant context.

Empirical headline (from V1 Pass 2/3): **base model 30% → dreamed
adapter 90% on cross-memory reasoning probes**, a 3× improvement. See
[`docs/tuning/llama-3.1-8b-instruct-4bit.md`](../../docs/tuning/llama-3.1-8b-instruct-4bit.md).

## Install

```bash
uv sync --extra mcp
```

This adds the `mcp` SDK to the dev environment. The `dreamagent serve`
command becomes available.

## Verify it starts

```bash
dreamagent serve --help
```

You should see the FastMCP-style help describing `--base-model`,
`--snapshots-dir`, `--max-tokens` overrides.

## Connect from Claude Code

Add to your Claude Code MCP configuration (typically `~/.claude/claude_desktop_config.json` or via the Claude Code MCP UI):

```json
{
  "mcpServers": {
    "dreamagent": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/absolute/path/to/dreamagent",
        "dreamagent",
        "serve"
      ],
      "env": {
        "DREAMAGENT_BASE_MODEL": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
        "DREAMAGENT_SNAPSHOTS_DIR": "/absolute/path/to/dreamagent/runs/snapshots",
        "DREAMAGENT_MAX_TOKENS": "128"
      }
    }
  }
}
```

Restart Claude Code. You should see the `query_memory` and
`query_memory_with_lineage` tools listed under the `dreamagent` server.

## Connect from Cursor / Hermes / OpenClaw

Same pattern with their respective MCP config locations. The server
itself is transport-agnostic — it's stdio JSON-RPC per the MCP spec.

## Test queries (after connecting)

In your daily-driver agent, ask things like:

- "Hey Claude, use the dreamagent memory tool: what is the user's dog's name?"
- "Use query_memory to find out what command the user uses to run tests."
- "Ask the dreamed memory specialist what the user prefers for response formatting."

The first call has a ~10s cold-start while the model loads. Subsequent
calls are sub-second (see `benchmarks/query_latency` for the p50/p95/p99
numbers).

## Available tools

### `query_memory(question: str) -> str`

Ask the dreamed model. Returns a string with its answer. If the model
doesn't know, the system prompt instructs it to say so explicitly.

### `query_memory_with_lineage(question: str) -> dict`

Same as `query_memory` but returns a dict with metadata:

```json
{
  "answer": "...",
  "base_model": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
  "adapter": "2026-05-27T17-05-59Z",
  "warning": null
}
```

`adapter` is the snapshot name of the live adapter — useful for tracing
back to a specific nightly run. If `warning` is non-null, the server is
serving the base model only (no live adapter found).

## Configuration

Precedence: CLI flag → env var → default.

| Var | CLI | Default |
|---|---|---|
| Base model | `--base-model` / `$DREAMAGENT_BASE_MODEL` | `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` |
| Snapshots dir | `--snapshots-dir` / `$DREAMAGENT_SNAPSHOTS_DIR` | `./runs/snapshots` |
| Max tokens | `--max-tokens` / `$DREAMAGENT_MAX_TOKENS` | `128` |

For MCP server hosting (where the host launches the process), use env
vars in the `env` block of the MCP config — CLI args from the host are
not portable across clients.

## Failure modes

| What goes wrong | What you'll see |
|---|---|
| `mcp` SDK not installed | "mcp SDK not installed — install with `uv sync --extra mcp`" |
| No live adapter | Server starts; queries answer from base model. `query_memory_with_lineage` returns a `warning`. |
| Wrong snapshots dir | Same as above (no live found, fall back to base). |
| Model file missing | First query errors with the MLX-LM "model not found" message. |
| Cold start | ~5-10s on first query. Subsequent calls are fast. |

## Roadmap from here

- **V2.1**: HTTP transport, latency optimization (target p95 < 500ms),
  the `vs-mem0` head-to-head benchmark
- **V2.2**: Multi-user namespacing, adapter signing, `dreamagent serve`
  process supervision
- **V3**: Same MCP surface but the backend is a 70B-class dreamed model
  on rented GPU

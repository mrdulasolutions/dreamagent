# 07 — MCP memory backend (V2 preview)

> **Status: Preview.** Full MCP support lands in V2.0 — see [`ROADMAP.md`](../../ROADMAP.md). This recipe shows the *intent* and a current-day workaround using the CLI as a query interface.

## The vision

Your daily-driver agent (Claude Desktop, Cursor, Hermes, OpenClaw,
or any MCP-capable client) gets a single tool:

```
query_memory(question: str) -> { answer, confidence, sources }
```

Backed by the dreamed model. The agent never sees the underlying
memories — only the synthesized answer. Privacy is preserved; the
agent's reasoning gets long-memory grounding without giant context.

## Current-day workaround

Until the MCP server ships, you can invoke the dreamed model as a
subprocess from any agent that supports shell tools. This is a hack
but proves the concept.

```bash
# A simple wrapper script: ask-dreamagent.sh
#!/usr/bin/env bash
question="$1"
uv run python -c "
from mlx_lm import load, generate
from pathlib import Path
model, tok = load(
    'mlx-community/Llama-3.2-1B-Instruct-4bit',
    adapter_path=str(Path('runs/snapshots/live/adapter').resolve()),
)
prompt = tok.apply_chat_template(
    [{'role': 'system', 'content': 'You are the user\\'s personal assistant. Answer concisely from your knowledge of the user.'},
     {'role': 'user', 'content': '''$question'''}],
    tokenize=False, add_generation_prompt=True,
)
print(generate(model, tok, prompt=prompt, max_tokens=128, verbose=False))
"
```

Use it from Claude Code or Cursor by adding the script as an
allowed shell command, then asking your agent to call it when it
needs to know something about the user.

## V2 MCP design (in progress)

When `dreamagent serve` lands, it will expose an MCP server with:

```json
{
  "tools": [
    {
      "name": "query_memory",
      "description": "Ask the user's dreamed memory specialist a question.",
      "inputSchema": {"question": "string"}
    },
    {
      "name": "query_memory_with_confidence",
      "description": "Same as query_memory but includes confidence + source memory IDs.",
      "inputSchema": {"question": "string", "min_confidence": "number"}
    }
  ]
}
```

Configuration would be standard MCP — adding `dreamagent serve --mcp`
as an MCP server in your client's config.

See [`docs/comparison/vs-mem0.md`](../../docs/comparison/vs-mem0.md)
for how this composes with mem0 (hot retrieval + DreamAgent deep memory).

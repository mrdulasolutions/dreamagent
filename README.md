# DreamAgent

Nightly LoRA consolidation of agent memories into model weights.

By day, an upstream memory system (mem0, supermemory, Claude memory, OpenClaw, Hermes, or a JSONL stream) emits `MemoryItem` records. By night, DreamAgent consolidates those memories into the **weights** of a small open model via LoRA fine-tuning — so the next day the model recalls them parametrically instead of via RAG retrieval.

The trained small model is "the guy who knows" — a memory specialist any larger agent (Claude, GPT, Llama 70B) can query as a memory backend over MCP, HTTP, or in-process.

See the [plan](https://github.com/mrdulasolutions/dreamagent) for full design rationale, prior art, and the verification protocol.

## Status

V1 Phase 0 — project skeleton, MemoryItem schema, connectors, fixture memories.

## Quick start

```bash
uv sync
uv run dreamagent --help
```

## Architecture (V1 outline)

```
ingest/   →  compose/  →  train/  →  eval/  →  promote/  →  serve/
                                              ↓
                                          merge/ (weekly)
```

- **ingest** — `MemoryConnector` implementations (JSONL, mem0, fixtures, …)
- **compose** — `MemoryItem` → training examples + rehearsal mix
- **train** — MLX-Tune (local) or Unsloth (cloud), one config
- **eval** — personal-recall + general-capability runners
- **promote** — eval gate, adapter snapshot, rollback
- **merge** — weekly mergekit consolidation
- **serve** — MLX-LM / Ollama with adapter hot-swap

## License

MIT

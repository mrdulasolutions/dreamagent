# DreamAgent vs. mem0

[mem0](https://github.com/mem0ai/mem0) is the most popular agent-memory system in 2026 — 56.9k GitHub stars, Y Combinator S24, an active SaaS, deep integrations with LangGraph and CrewAI. Their April 2026 algorithm reports 92.5% on LoCoMo and 94.4% on LongMemEval with a published research paper.

DreamAgent is not trying to replace mem0. **They solve different problems.** This page is for the engineer deciding which one belongs in their architecture, or whether to use both.

## At a glance

| | mem0 | DreamAgent |
|---|---|---|
| Storage paradigm | Vector + graph + entity index | LoRA adapters on model weights |
| Where memories live at query time | External index | Inside the model's parameters |
| Query flow | Vector lookup → top-k chunks → LLM call | Direct forward pass on the dreamed model |
| Cross-memory reasoning | Within retrieved chunks only | Across *all* trained memories (model saw them together) |
| Hosted option | mem0 Cloud | None (intentional) |
| Self-host option | Yes (Postgres+pgvector+Next.js dashboard) | Yes (single Mac binary) |
| Privacy floor | Embeddings + indexes persist | Weights are the only persistent artifact |
| Setup time | Minutes (SDK install + API key) | Hours (first nightly run downloads model) |
| Cost model | Per-call (cloud) or self-host infra | Compute for one nightly fine-tune (free on Mac, ~$1-5 on cloud GPU) |
| Throughput at inference | Vector lookup ~10-50ms + LLM round-trip | One forward pass on a 4B model, sub-second on Mac |
| LoCoMo benchmark | 92.5% (self-reported, April 2026 algorithm) | Not yet measured — see [comparison/README](./README.md) |

## When mem0 is the right answer

- Your agent does multi-turn chat where users add information *during* the conversation and expect it to be available immediately on the next turn.
- You need integration with LangGraph, CrewAI, or other agent frameworks today.
- You don't want to manage a model.
- Memory volume is high (>100k entries) and you need vector-search-style nearest-neighbor lookup.

mem0 has done excellent work on the extraction-and-retrieval problem. If that's the problem you have, use mem0.

## When DreamAgent is the right answer

- You're building a personal AI assistant that should feel like it *knows* the user, not like it's looking them up.
- You need cross-memory reasoning: "Given everything you know about my work patterns, what would I prefer here?" — questions that no vector lookup can answer in one shot.
- You need true local privacy. Not "self-hosted but the data sits in a database" — actually local. Weights on disk, no index, no embeddings.
- You're already running a small local model and want it to learn from you over time.
- You're researching parametric vs. non-parametric memory and want a reproducible playground.

## When you should use both

This is the most interesting case. **mem0 and DreamAgent are complementary** in the V2 architecture.

```
                  ┌─────────────────────────────┐
                  │  Your daily-driver agent    │
                  │  (Claude, GPT, Llama 70B)   │
                  └──────────────┬──────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
            ┌──────────┐                 ┌──────────────┐
            │   mem0   │                 │  DreamAgent  │
            │ (hot     │                 │  (long       │
            │  memory, │                 │  memory,     │
            │  recent  │                 │  consolidated│
            │  facts)  │                 │  weights)    │
            └──────────┘                 └──────────────┘
              recent → retrieval         old → parametric
              high volume                stable, deep
              low latency on writes      low latency on reads
```

A typical query routes to *both*: mem0 returns the freshest chunks via retrieval, DreamAgent returns the consolidated, reasoned answer from weights. The agent reconciles.

This isn't speculative — it's the V2 design target. mem0 can ship as an upstream connector to DreamAgent today (we have a `Mem0Connector` stub). DreamAgent's MCP server (V2, in design) exposes the dreamed model as a memory backend any agent can query alongside mem0.

## Migration: from mem0 to DreamAgent

If you already have a mem0 deployment and want to try parametric consolidation:

```bash
# Export mem0 memories to the JSONL format DreamAgent reads
python -m mem0 export --output mem0-export.jsonl

# Run extraction (DreamAgent normalizes mem0's shape via the connector)
dreamagent ingest mem0-export.jsonl

# Compose, train, eval, promote
dreamagent dream --source mem0-export.jsonl --validation-tier
```

mem0's memory shape and DreamAgent's `MemoryItem` schema are close cousins. The `Mem0Connector` (planned, ~50 LoC) handles the bridge automatically.

## What we copied from mem0

We have no shame in admitting it. mem0's [extraction pipeline](https://docs.mem0.ai/core-concepts/memory-types) is the most thoroughly engineered piece of memory infrastructure in this space. DreamAgent's extraction prompt (in [`src/dreamagent/extract/prompt.py`](../../src/dreamagent/extract/prompt.py)) borrows the "extract durable memories, skip ephemeral" philosophy directly, with adaptations for our 5-kind taxonomy.

We are explicitly grateful for mem0's prior work; we're trying to solve a different layer of the same stack.

# DreamAgent vs. Letta / MemGPT

[Letta](https://github.com/letta-ai/letta) (formerly MemGPT) is the most architecturally ambitious agent-memory system in 2026 — 23k+ stars, hierarchical memory blocks (in-context / archival / recall), self-editing memory via tool calls, a sophisticated agent runtime. The Letta team also publishes the most methodologically careful [benchmarking work](https://www.letta.com/blog/benchmarking-ai-agent-memory) in the space.

## At a glance

| | Letta | DreamAgent |
|---|---|---|
| Storage paradigm | Hierarchical text blocks + filesystem | LoRA adapters on model weights |
| Self-editing memory | Yes (via tool calls) | No — memories are produced upstream |
| Agent runtime | Built-in, opinionated | None — DreamAgent is a backend |
| Hosted option | Letta Cloud | None |
| Stateful agents (persistent across sessions) | First-class | Out of scope (DreamAgent is the memory layer) |
| Memory block size limit | Bounded by context window | Bounded by training compute |
| Cross-memory reasoning | Limited to what fits in context | Across all trained memories |
| LoCoMo | 83.2% | Not yet measured |

## Where Letta is genuinely better

- **Stateful agent runtime.** Letta has spent years building a production-grade agent runtime. DreamAgent is intentionally just a backend; you bring your own agent.
- **Self-editing memory.** Letta's agents can curate their own memory blocks at runtime. DreamAgent treats memory production as a separate upstream concern.
- **Mature SDK surface.** Python + TypeScript SDKs, hundreds of releases, Discord community.

If your problem is "I want an agent runtime that handles memory natively," Letta is the answer. DreamAgent's V1 doesn't compete here.

## Where DreamAgent is genuinely different

- **Memories become parametric, not just retrieved.** Letta's memory blocks are read at runtime; DreamAgent's are *part of the model*.
- **Cross-memory reasoning by construction.** When the dreamed model answers a question, it has already seen all the memories together at training time. This is not something a retrieval layer can replicate — even with infinite context.
- **Honest local-only mode.** Letta supports local LLMs, but the memory blocks still live as text. DreamAgent's local mode is *structurally* local: there is no memory index, just weights on disk.
- **Eval-gated promotion.** Every nightly update has explicit pass/fail gates and one-command rollback. Letta's runtime self-edits don't have this safety machinery (they don't need it for retrieval, but they would for fine-tuning).

## Where Letta's research has helped DreamAgent

Two ideas from Letta's [benchmarking work](https://www.letta.com/blog/benchmarking-ai-agent-memory) are directly reflected in DreamAgent:

1. **Benchmarks measure what they measure.** LoCoMo is a *retrieval* benchmark; a sufficiently good agent with filesystem tools beats specialized memory systems on it. Our `benchmarks/` directory explicitly designs probes that resist this gaming (cross-memory reasoning, identity drift, query latency on a fixed model).

2. **Tool capability dominates index sophistication.** The model's ability to *use* the memory matters more than the index structure. DreamAgent leans into this by training the model itself — the memory and the agent are the same artifact.

## When to use both

In production, the most interesting architecture is:

- **Letta as the agent runtime** (its strength)
- **DreamAgent as the memory backend** (parametric consolidation, V2 MCP server)

A Letta agent could call DreamAgent for "deep memory" questions and use its own filesystem memory blocks for fresh, ephemeral context. We expect to ship a `letta-dreamagent` integration once V2 lands.

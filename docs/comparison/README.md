# DreamAgent vs. The Field

> **TL;DR** — Every other AI agent memory system on the market stores memories as **retrievable text** (vectors, graphs, files). DreamAgent stores them as **model weights**. This is a categorically different paradigm, so head-to-head benchmarks on retrieval-style tasks (LoCoMo, LongMemEval) are not apples-to-apples. The honest comparison is on the axes that *matter to the user*: privacy, latency, cross-memory reasoning, switching cost, and what happens when the retrieval index returns the wrong chunk.

## The Memory System Landscape (May 2026)

| System | Paradigm | Storage | Cross-memory reasoning | Local-only option | LoCoMo published |
|---|---|---|---|---|---|
| **[mem0](https://github.com/mem0ai/mem0)** | Non-parametric | Vector + graph + entity index | No (per-chunk retrieval) | Yes (self-hosted) | 92.5%* |
| **[Letta / MemGPT](https://github.com/letta-ai/letta)** | Non-parametric | Filesystem + hierarchical memory blocks | Limited (within context) | Yes | 83.2% |
| **[Supermemory](https://supermemory.ai/)** | Non-parametric | Vector index, cloud-hosted | No | No | ~70% |
| **[Zep](https://www.getzep.com/)** | Non-parametric | Knowledge graph + vector | Within-graph only | Yes (community) | ~85% |
| **[MemMachine](https://arxiv.org/pdf/2604.04853)** | Non-parametric | Ground-truth preserving graph | Limited | Research | 91.7% |
| **[EverMemOS](https://evermind.ai/)** | Non-parametric | Hybrid graph + retrieval | Within-system | No | 92.3% |
| **DreamAgent** | **Parametric** | **LoRA adapters on weights** | **Yes (model saw all memories together at train time)** | **Yes (Apple Silicon)** | Not yet measured — see below |

*\* Mem0's self-reported number on their April 2026 algorithm; [independent measures](https://www.letta.com/blog/benchmarking-ai-agent-memory) range lower (66%) depending on configuration.*

### Why no head-to-head LoCoMo number for DreamAgent yet?

LoCoMo measures **retrieval over long conversation histories**. DreamAgent does not retrieve — it embodies. Running DreamAgent on LoCoMo would either:

1. **Re-train the model on the LoCoMo conversation** before each question — possible but expensive and not how a real user would deploy it
2. **Use DreamAgent as a knowledge oracle** for a separate retrieval agent — meaningful but tests the combined system, not memory consolidation alone

We will publish LoCoMo numbers under both protocols, transparently labeled, in `benchmarks/` once Pass 2 (production tier, Qwen 3 4B) is complete. Until then, we won't claim a number we haven't measured.

---

## The Axes That Actually Matter

When you pick a memory system for a real product, the comparison usually comes down to these properties, *not* a benchmark percentage:

| Axis | mem0 / Letta / etc. | DreamAgent | Why this matters |
|---|---|---|---|
| **Storage paradigm** | Text/vectors/graphs | Model weights | Retrieval can return wrong chunks; weights can't "miss" the way a vector index can. |
| **Cross-memory reasoning** | Limited to retrieved chunks | Model has seen all memories together | "What contradictions exist in what I've told you?" — only parametric memory can answer this without re-running a separate aggregation pass. |
| **Privacy** | Often cloud-hosted; vectors persist | Fully local; weights stay on device | EU AI Act compliance, healthcare, finance, anyone who can't ship PII to a third party. |
| **Query latency** | Vector lookup + LLM call (~hundreds of ms) | One forward pass on a 4B model (~sub-second on Mac) | Tail-latency at scale; eliminates a network round trip. |
| **Switching cost from frontier model** | High (you're inside their SDK) | Zero — DreamAgent is a backend; keep using Claude/GPT | Users don't migrate off their daily-driver agent. |
| **Upgrade story** | Wait for vendor; re-index everything | Re-dream nightly; rollback in one command | You own the schedule; you own the safety net. |
| **Memory contradiction handling** | Manual dedup logic in the index | `kind: correction` + `supersedes` in the schema, surfaces in training | First-class versioning; auditable. |
| **Memory deletion (GDPR right-to-be-forgotten)** | Delete the vector + any cached representations | Re-train without the memory + invalidate adapters that saw it | Both are non-trivial; DreamAgent provides a clear protocol. |
| **Failure mode** | "Found nothing similar" (silent) | Eval gate REJECTs the nightly run, base model preserved | Observable, recoverable failure. |

---

## So Which Should You Use?

A blunt answer:

- **Building a chatbot that needs to remember recent conversations within a session?** Use mem0 or Letta. They're battle-tested and lower-effort.
- **Building a personal AI assistant whose memories should feel like part of *who the assistant is*?** Use DreamAgent.
- **Need both?** They're not exclusive. DreamAgent's V2 architecture explicitly proposes the dreamed model as a memory specialist that any larger agent (with its own mem0/Letta retrieval layer) can query. **Use mem0 for the hot path; use DreamAgent for the long memory.**

---

## Per-System Comparisons

- [vs. mem0](./vs-mem0.md)
- [vs. Letta / MemGPT](./vs-letta.md)
- [vs. Supermemory](./vs-supermemory.md)
- [vs. Zep](./vs-zep.md)
- [vs. "just stuff it all in a 1M context window"](./vs-giant-context.md)

---

## Benchmark Suite Roadmap

The `benchmarks/` directory contains our reproducible validation harness. We commit to publishing the following numbers:

| Benchmark | Status | What it measures |
|---|---|---|
| **personal_recall** | ✅ measured (46% on V1 Pass 1) | Can the dreamed model recall its training memories from weights? |
| **general_capability** | ✅ measured (90% on V1 Pass 1, vs 93% base) | Did we break the base model? |
| **identity_drift** | 🔬 protocol designed, not yet run | Over 30 nights, does the model lose its assistant persona? |
| **cross_memory_reasoning** | 🔬 protocol designed | Can the model answer questions that require synthesizing 3+ memories at once? Tests parametric advantage over retrieval. |
| **query_latency** | 🔬 protocol designed | p50, p95, p99 query latency on Mac Mini M4. |
| **LoCoMo compatibility** | 🚧 in progress (Pass 2) | DreamAgent on the standard agent-memory benchmark, both protocols. |
| **vs-mem0 head-to-head** | 🚧 in progress | Same memory set through both systems; compare on cross-memory + latency + privacy axes. |

Run any of these locally: see [`benchmarks/README.md`](../../benchmarks/README.md).

---

## A Note on Benchmark Gameability

[Letta's benchmarking post](https://www.letta.com/blog/benchmarking-ai-agent-memory) makes a critical methodological point: a generic agent with filesystem tools beat Mem0's specialized memory layer on LoCoMo. *The benchmark measures retrieval, but agent capability dominates.*

We take this seriously. The DreamAgent benchmark suite is designed to measure properties that **cannot** be gamed by giving a bigger model more context. Cross-memory reasoning, identity-drift, query latency on a fixed model, and personal recall on held-out probes all stay honest under that pressure.

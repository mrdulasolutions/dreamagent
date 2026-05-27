# Research Notes

This directory holds research investigations that inform but don't bind the core
project's roadmap. The main project picks one path and ships; the others live
here for contributors who want to take a different route.

| File | What it covers |
|---|---|
| [`2026-05-improving-memory.md`](./2026-05-improving-memory.md) | Comprehensive research into how to improve DreamAgent's memory after the V2.2 retractions. Surveys 5 candidate techniques (OPLoRA, Sparse Memory Finetuning, MemoryLLM, Inheritune, Recurrent Memory Transformer), the "build your own LLM" question, and recommends a path. |
| [`path-b-memory-specialist.md`](./path-b-memory-specialist.md) | A complete how-to for the ambitious route the core project isn't taking yet: warm-start a 4B memory-specialist model from Llama 3.1 8B's first 16 layers, then train on a synthesized memory-skill dataset. ~$30-70 in cloud GPU rental. |

## How to use these documents

- **If you want to understand the project's current thinking on capability improvements** — read `2026-05-improving-memory.md`.
- **If you want to build a memory-specialist model and contribute it back** — follow `path-b-memory-specialist.md`. A successful Path B reproduction would land as `examples/09-path-b-memory-specialist/`.
- **If you're a researcher evaluating the project's empirical claims** — start with [`../PAPER.md`](../PAPER.md), then read these research docs for what's coming next.

## Why these are research, not roadmap

We've been retracted twice (V2.1, V2.2). The honest position is that we don't yet know which capability-improvement direction will work. The research docs stay open-ended on purpose; the project commits to one path at a time and reports the empirical result, whichever way it goes.

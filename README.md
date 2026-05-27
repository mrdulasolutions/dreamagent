# DreamAgent

> **An LLM that dreams.**
> By day it captures memories. By night a cron job fine-tunes a small model on them — so the next morning the model knows what you told it yesterday, *from its weights*, with no retrieval.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![Tests](https://img.shields.io/badge/tests-114%20passing-brightgreen.svg)](#tests)
[![Status: V1 Phase 1 — viability proven](https://img.shields.io/badge/V1%20Phase%201-viability%20proven-success.svg)](docs/tuning/llama-3.2-1b-instruct-4bit.md)

---

## The Idea

Every agent memory system today — mem0, supermemory, Letta, Claude memory, OpenClaw, Hermes — consolidates memories into **text files** the model reads at runtime. RAG. Retrieval. Context-stuffing.

**DreamAgent goes one step further.** It runs a nightly LoRA fine-tune on the day's memories so they become part of the model's **weights**. Like sleep consolidates the hippocampus into the cortex, DreamAgent moves "today I learned" from retrieval to parametric knowledge.

The result is a tiny model — Qwen 3 4B or Llama 3.2 1B class — that has *seen all your memories together*. It doesn't return chunks; it reasons across them. It's the **memory specialist** every larger agent (Claude, GPT, Llama 70B) can query as their "guy who knows."

```
                    YOU                          DREAMAGENT (a small fine-tuned LLM)
                     │
        ┌────────────┴────────────┐                        ▲
        ▼                         ▼                        │
   Claude / GPT                Mem0 / Supermemory          │
   (your daily-driver agent)   (or any memory store)       │
        │                         │                        │
        └─────────────┬───────────┘                        │
                      │ stream of MemoryItems              │
                      ▼                                    │
                ╔══════════════╗                           │
                ║  DreamAgent  ║  ── nightly cron ─────────┘
                ║   pipeline   ║      compose · train ·
                ╚══════════════╝      eval · promote
                                          (weights updated)
```

When your daily agent needs to know something about you, it asks DreamAgent. DreamAgent answers from weights — not from a vector index. No memories ever leave the box.

---

## Why This Matters

1. **Privacy is structural.** Memories are consolidated locally on Apple Silicon. They never traverse a third-party API to become parametric.
2. **Reasoning, not retrieval.** A vector index returns the closest chunks. A dreamed model *understands the connections* between memories — because it saw them all together during training.
3. **Cross-agent memory.** Switch from Claude to GPT to a local Llama whenever — your DreamAgent travels with you. The dreamed memory specialist exposes the same MCP/HTTP interface to any frontier model.
4. **The roadmap derisks itself.** V1 is a viability proof on a tiny model. V2 ships that tiny model as a real product. V3 — applying the same technique to frontier-scale open models — is only worth the compute if V2 already pays its way.

---

## Quickstart

```bash
# Clone + install
git clone https://github.com/mrdulasolutions/dreamagent.git
cd dreamagent
uv sync

# 1. Extract memories from raw text using a frontier LLM
export ANTHROPIC_API_KEY=sk-ant-...
uv run dreamagent extract \
    --from my-chat.txt \
    --backend anthropic \
    --output memories.jsonl

# 2. Inspect what was extracted
uv run dreamagent ingest memories.jsonl

# 3. Run the nightly dream pipeline (downloads ~700MB on first run)
uv run dreamagent dream \
    --source memories.jsonl \
    --validation-tier \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60

# 4. (Optional) Schedule it to run nightly
uv run dreamagent install-cron --dry-run    # preview
uv run dreamagent install-cron              # actually install (macOS launchd)
```

That's the entire loop. After the first successful run, `runs/snapshots/live` points at your first dreamed adapter. You can hot-swap it into any MLX-LM serve, Ollama, or any tool that loads PEFT adapters.

---

## What Just Happened

When you run `dreamagent dream`, you get a single nightly cron unit that:

| # | Stage | What it does |
|---|---|---|
| 1 | **ingest** | Reads `MemoryItem`s from any connector (JSONL, mem0, supermemory, Claude memory, OpenClaw, Hermes) |
| 2 | **compose** | Converts each memory into 2-4 training examples + 1 held-out eval probe. Templates per kind (fact, preference, procedure, event, correction). |
| 3 | **rehearsal mix** | Blends today + prior-night replay + a fixed "general capability anchor" set so the model doesn't forget that it's an assistant or that 2+2=4. |
| 4 | **train** | LoRA fine-tune via MLX-LM (local Mac) or Unsloth (cloud, same script). Subprocess wrapper, full metadata.json with lineage. |
| 5 | **eval** | (a) Personal recall: does the trained model know the new memories? (b) General capability: did we break the base model? |
| 6 | **promote** | A 4-decision gate: PROMOTE, PROMOTE_WITH_WARNING, REJECT, or REJECT (low recall). Bad runs land in `rejected/` and `live` is preserved. |
| 7 | **snapshot** | Adapter + all evals + metadata + gate decision saved as a versioned artifact. One-command rollback to any prior night. |

Every emitted artifact is human-readable. Every model response can be traced back to the adapter that taught it, to the training example that taught the adapter, to the memory that became the training example. No black box.

---

## The Methodology

DreamAgent introduces and names a specific technique: **nightly LoRA consolidation of structured agent memories into parametric weights with eval-gated promotion and per-night adapter snapshots**.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) for the full description, prior art, and the open problems we're still chasing.

> If you build on this work — academic, commercial, or otherwise — attribution to Mr Dula Solutions is required per [`NOTICE`](NOTICE). See [Licensing](#licensing).

---

## Viability — V1 Proof

Pass 1 of the V1 verification protocol has been completed and is reproducible from the repo state alone. The locked recipe lives in [`docs/tuning/llama-3.2-1b-instruct-4bit.md`](docs/tuning/llama-3.2-1b-instruct-4bit.md).

| Metric | Result |
|---|---|
| Base model | mlx-community/Llama-3.2-1B-Instruct-4bit |
| Memories trained | 50 fixtures across all 5 kinds |
| Training time | ~25s on Mac M-series |
| Personal recall on held-out probes | **46%** (vs 0% baseline) |
| General capability preservation | 90% (vs 93% base — **3.3pp regression**) |
| Eval gate decision | **PROMOTE** ✓ |

16 tuning runs produced this clean PROMOTE. The full sweep is documented in [`docs/tuning/`](docs/tuning/) with hypotheses, falsified theories, and the 6 transferable lessons we extracted.

---

## Architecture

```
src/dreamagent/
    schema.py       — MemoryItem & MemoryBatch (pydantic v2)
    ingest/         — connectors emitting MemoryItems (JSONL / Fixture / Mem0 / …)
    extract/        — frontier-LLM memory extraction (Anthropic / OpenAI / Ollama)
    compose/        — templates · examples · rehearsal mix · anchors
    train/          — MLX-LM LoRA wrapper · config · lineage metadata
    eval/           — substring-match scoring · personal + general probes
    promote/        — eval gate (4-decision matrix) · snapshot · rollback
    merge/          — weekly mergekit consolidation (Phase 3, scaffolded)
    serve/          — MLX-LM / Ollama hot-swap (Phase 3, scaffolded)
    cli.py          — typer entry point
```

For a deeper walk through each subsystem and its interfaces, see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

---

## Frontier Model Integration

The most accuracy-critical part of DreamAgent is the conversion from raw text to validated `MemoryItem`s. This is where mistakes propagate downstream into the trained weights — once a hallucinated fact is in the adapter, you have to dream it out again.

We solve this with `dreamagent extract`, which calls a frontier LLM with a precision-engineered system prompt (see [`src/dreamagent/extract/prompt.py`](src/dreamagent/extract/prompt.py)) that:

- Enforces a strict 5-kind taxonomy with discriminating examples
- Forbids fabrication (lower confidence over guessing)
- Skips ephemeral content ("I said hi" is not a memory)
- Refuses to extract sensitive data (passwords, credentials, SSNs)
- Outputs a JSON array that's validated against the pydantic schema record-by-record
- Auto-rejects records with malformed shape, with structured rejection reasons

Three backends ship today:

| Backend | Default model | Install | When to use |
|---|---|---|---|
| `anthropic` | `claude-sonnet-4-6` | `uv sync --extra anthropic` | Best extraction quality |
| `openai` | `gpt-4o-2024-11-20` | `uv sync --extra openai` | Production-grade alternate |
| `ollama` | `llama3.2:3b` | `uv sync --extra ollama` | Fully local, privacy-strict |

You bring your own API key — DreamAgent is byok-only by design.

---

## Roadmap

DreamAgent ships in three versions, each gated on evidence from the previous.

| Version | Goal | Status |
|---|---|---|
| **V1 — Viability Proof** | Demonstrate consolidation works on a small model with eval-gated safety | ✅ Pass 1 complete |
| **V2 — Memory Specialist** | Expose the dreamed model as an MCP / HTTP / library memory backend any larger agent can call | 🔜 in design |
| **V3 — Frontier Direct** | Apply the same recipe directly to 70B+ models on rented GPUs | ⏸ blocked on V1+V2 evidence |

V2 is the actual product. V3 is optional — if V2 satisfies the use case, the small dreamed model IS the answer.

---

## Tests

```bash
uv run pytest -q          # 114 tests, full suite ~3s
uv run ruff check src tests
```

Tests cover schema validation, all 3 connectors, compose stage (examples + anchors + mix), train metadata, eval scoring + reporting, promote gate decisions, snapshot/rollback machinery, and the extraction pipeline. The actual LoRA training and live model inference are exercised by `dreamagent dream` end-to-end against fixtures.

---

## Licensing

This repository is licensed under the **Apache License, Version 2.0**. See [`LICENSE`](LICENSE).

The DreamAgent methodology, prompts, fixture data, tuning playbook, and architectural patterns published in this repository were originated by **Mr Dula Solutions** on **2026-05-26**. Per [`NOTICE`](NOTICE), redistribution and derivative works **must** include attribution:

> "Built on the DreamAgent methodology by Mr Dula Solutions
> (https://github.com/mrdulasolutions/dreamagent)"

We can't physically stop someone from using the methodology without credit. This repository, its commit history, and [`CITATION.cff`](CITATION.cff) exist so we can definitively prove the technique originated here.

If you publish academic work using DreamAgent, please use the citation in [`CITATION.cff`](CITATION.cff).

---

## Contributing

PRs welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the workflow, code style, test bar, and how to add new connectors or backends.

For security issues, see [`SECURITY.md`](SECURITY.md).

---

## Prior Art and Inspiration

DreamAgent stands on substantial prior work — particularly memory consolidation in agentic systems (mem0, Letta, OpenClaw Dreaming, Anthropic's Claude Dreaming) and continual learning research (CL-LoRA, SuRe, FOREVER, SleepGate, Memento). The distinction is that none of those systems push memories into model weights — they all stop at curated text consolidation. DreamAgent closes that gap.

Full references and a tour of the closest neighbors live in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

---

<sub>Made by [Mr Dula Solutions](https://github.com/mrdulasolutions). If this changes how you think about agent memory, [tell us](https://github.com/mrdulasolutions/dreamagent/issues).</sub>

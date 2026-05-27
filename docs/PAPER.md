# MORPHEUS: Nightly LoRA Consolidation of Structured Agent Memories with Eval-Gated Promotion

**Mr Dula Solutions**
*Independent · 2026-05-27*

> *Pre-print, V1.0. Reference implementation: [DreamAgent](https://github.com/mrdulasolutions/dreamagent). Apache 2.0; attribution required per [NOTICE](../NOTICE).*

---

## Abstract

Existing AI agent memory systems (mem0, Letta, Supermemory, Zep) consolidate captured memories into retrievable text via vector indexes, knowledge graphs, or hierarchical memory blocks. We propose **MORPHEUS** — Memory Overnight Re-parameterization, Promotion via Held-out Eval, Update Snapshots — a methodology that consolidates structured agent memories directly into the **weights** of a small open-source language model via nightly LoRA fine-tuning. A four-decision eval gate (PROMOTE / PROMOTE_WITH_WARNING / REJECT / REJECT-low-recall) protects against catastrophic forgetting; per-night adapter snapshots enable one-command rollback.

We validate MORPHEUS on Llama 3.2 1B Instruct and Llama 3.1 8B Instruct using a 50-item structured memory fixture, a 30-item general-capability probe set, and a 10-item cross-memory reasoning probe set. Across a 7-night chained-training drill on Llama 3.1 8B (each night resuming from the prior night's adapter), all 7 nights were promoted by the gate; personal recall climbed from 43.8% to 81.3% before plateauing; general capability regression remained bounded at 0–13.3pp and never breached the 15pp rejection threshold. On the final night-7 adapter, personal recall reached 75% and cross-memory reasoning reached 90%.

**Two subsequent experiments against retrieval baselines substantially narrow these empirical claims.** V2.1 (§6.5) compared DreamAgent against a vector-retrieval baseline running on the same base model and same memories: **parity (0.0pp difference) on cross-memory reasoning**, +6.2pp on personal recall. V2.2 (§6.6) added 15 author-designed adversarial probes intended to defeat retrieval and a composed (DA + retrieval + reconciler) system: **retrieval outperformed DreamAgent on adversarial probes (93.3% vs 80%)**, and composition did not beat the best individual system on any probe set. Both outcomes were pre-registered in V1 §10.5 as falsifiable; both materialized.

The cumulative effect is that DreamAgent's claimed empirical advantages over retrieval have narrowed from "+60pp cross-memory + parametric-advantage at scale" (V1 framing) to **"+6.2pp personal recall at N=50"** (post-V2.2). The methodological contribution — the four-decision eval-gated promotion scheme + per-night adapter snapshots making autonomous nightly training safe — is unaffected. The DreamAgent value proposition shifts from "wins on capability" to wins on structural properties: privacy, host-agent independence, operational simplicity, GDPR-clear deletion.

We discuss what these results do and do not demonstrate, document six transferable tuning lessons learned across 30+ tuning runs, and enumerate limitations including small evaluation sets, single-language fixtures, the compressed 7-night drill, and the dependence of cross-memory comparison on probe-set design. All numbers are reproducible from the published reference implementation. The retractions are published in the same commits as the new measurements.

**Keywords:** agent memory, continual learning, LoRA, catastrophic forgetting, MLX, Apple Silicon, parametric memory, knowledge consolidation, sleep-inspired learning.

---

## 1. Introduction

### 1.1 Problem statement

Modern conversational AI agents need persistent memory to be useful across sessions. The dominant approach in 2026 is **non-parametric memory**: capture user utterances or session summaries as text, index them in a vector store or knowledge graph, and retrieve top-k relevant chunks at inference time to stuff into the model's context. mem0 [\[1\]](https://github.com/mem0ai/mem0), Letta / MemGPT [\[2\]](https://github.com/letta-ai/letta), Supermemory [\[3\]](https://supermemory.ai), Zep [\[4\]](https://www.getzep.com/), and EverMemOS [\[5\]](https://evermind.ai/) are representative.

Non-parametric memory has known limitations:

- **Retrieval brittleness.** Top-k vector retrieval can miss the relevant chunk if its embedding is not sufficiently similar to the query. The recent literature [\[6\]](https://arxiv.org/abs/2603.14517) documents proactive interference effects that worsen as the memory pool grows.
- **No cross-memory synthesis.** A retrieval system can return chunks A, B, and C separately; it cannot, in a single forward pass, reason about how A, B, and C interact. Composing them requires a subsequent generation step where the model holds all three in context.
- **Privacy by deferral.** Many production memory systems are cloud-hosted; even self-hosted variants persist embeddings and chunks in external databases.
- **Vendor lock.** Switching agent frameworks typically requires re-importing the memory store.

We propose moving the memories themselves into the model's parameters, via nightly LoRA fine-tuning, with safety machinery sufficient for autonomous operation.

### 1.2 Contributions

1. **The MORPHEUS methodology.** A precisely-specified seven-stage pipeline (ingest → extract → compose → mix → train → eval → promote/snapshot) with a four-decision eval gate, a rehearsal-mix recipe defending against catastrophic forgetting, and a per-night adapter snapshotting scheme enabling one-command rollback.
2. **A reference implementation, DreamAgent**, in Python with Apple Silicon (MLX-LM) and cloud-GPU (Unsloth) backends behind a unified subprocess interface.
3. **Empirical validation** on two base models across three increasingly rigorous passes (single-night, production-tier single-night, 7-night chained-training drill), with reproducible benchmark numbers.
4. **A novel benchmark — cross-memory reasoning** — designed to measure capabilities retrieval cannot satisfy in a single shot. Adapter improvement: 30% → 90% on the night-7 adapter.
5. **A tuning playbook**, documenting six transferable lessons across 30+ tuning runs on two base models. We pre-emptively name a failed model family (Qwen 3, due to chain-of-thought tag conflicts) and explain the failure mode.

### 1.3 Non-contributions

We do **not** claim:
- Novelty of LoRA, PEFT, rehearsal buffers, or memory-extraction prompts; these are well-known [\[7\]](https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf)[\[8\]](https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents).
- A head-to-head LoCoMo [\[9\]](https://github.com/snap-research/locomo) win over mem0 or Letta. See §8.4 for our position on LoCoMo.
- That parametric memory replaces non-parametric memory. We argue for **composition** of the two (§5.4); the V2 architecture explicitly supports it.
- Long-horizon stability beyond 7 chained nights.

---

## 2. Background and related work

### 2.1 Memory systems for AI agents

Production agent memory in 2026 is dominated by retrieval-augmented systems. mem0 [\[1\]](https://github.com/mem0ai/mem0) reports 92.5% on LoCoMo with a three-storage-layer architecture (vector + BM25 + entity matching). Letta [\[2\]](https://github.com/letta-ai/letta) reports 83.2% on LoCoMo with self-editing hierarchical memory blocks. Supermemory and Zep ship comparable architectures.

Letta's own evaluation methodology [\[10\]](https://www.letta.com/blog/benchmarking-ai-agent-memory) makes a critical point we adopt: LoCoMo measures retrieval-and-recall over conversation history, and a sufficiently capable agent with filesystem tools beats specialized memory systems on it. We interpret this as a sign that agent capability dominates index sophistication on the standard benchmark.

### 2.2 "Dreaming" features in shipped products

OpenClaw [\[11\]](https://dev.to/czmilo/openclaw-dreaming-guide-2026-background-memory-consolidation-for-ai-agents-585e) and Anthropic's Claude [\[12\]](https://letsdatascience.com/blog/anthropic-dreaming-claude-managed-agents-self-improving-may-6) both ship features they call "dreaming" — scheduled background processes that consolidate text memories into curated documents. Critically, neither modifies model weights. MORPHEUS is the next step.

### 2.3 Continual learning and catastrophic forgetting

Catastrophic forgetting [\[13\]](https://www.ibm.com/think/topics/catastrophic-forgetting) — the loss of previously-learned capabilities when fine-tuning on new data — is the central risk for any continual-training scheme. Three classes of defenses are relevant:

- **Replay** [\[14\]](https://arxiv.org/pdf/1705.08690): include prior training examples in each new batch.
- **Regularization**: Elastic Weight Consolidation [\[13\]](https://www.ibm.com/think/topics/catastrophic-forgetting) and related methods penalize movement in important parameters.
- **Parameter-efficient adaptation** [\[15\]](https://github.com/huggingface/peft): LoRA, DoRA, and similar adapters constrain weight changes to a low-rank subspace.

Recent work specific to continual LoRA includes CL-LoRA [\[7\]](https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf) with orthogonal-constraint adapters, SuRe [\[8\]](https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents) for surprise-driven prioritized replay, and FOREVER [\[8\]](https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents) for Ebbinghaus-curve-inspired replay timing.

MORPHEUS combines replay (in the rehearsal mix), parameter-efficient adaptation (LoRA), and an eval-gated promotion mechanism that explicitly measures forgetting before promoting an update.

### 2.4 Sleep-inspired memory consolidation in ML

Complementary Learning Systems (CLS) theory in neuroscience proposes that the hippocampus stores episodic memory rapidly during the day and the neocortex consolidates that memory slowly via replay during sleep. ML systems inspired by CLS include generative replay [\[14\]](https://arxiv.org/pdf/1705.08690) and semi-parametric memory consolidation [\[16\]](https://arxiv.org/html/2504.14727v1). Anthropic's sleep-time compute [\[17\]](https://arxiv.org/pdf/2504.13171) and the SleepGate framework [\[6\]](https://arxiv.org/abs/2603.14517) operate during idle periods on the KV cache rather than on base weights.

MORPHEUS positions itself as the direct application of sleep-consolidation framing to the agent-memory problem at the LoRA-adapter level.

### 2.5 Related but distinct approaches

- **Model editing** (ROME, MEMIT, GRACE, MELO) [\[18\]](https://www.emergentmind.com/topics/rank-one-model-editing-rome) inserts facts via closed-form weight updates. Sequential edits degrade the base over time [\[19\]](https://arxiv.org/pdf/2401.07453).
- **Test-time training** (TTT-E2E, In-Place TTT) [\[20\]](https://www.deeplearning.ai/the-batch/test-time-training-end-to-end-ttt-e2e-retrains-model-weights-to-handle-long-inputs)[\[21\]](https://arxiv.org/html/2604.06169v1) updates weights during inference. Different timing; per-query rather than per-night.
- **Memento** [\[22\]](https://arxiv.org/abs/2508.16153) achieves strong agent performance with external case memory and no LLM fine-tuning. We share their respect for the strength of external memory; we differ on whether parametric memory should also exist.

---

## 3. The MORPHEUS methodology

### 3.1 The MemoryItem contract

MORPHEUS requires inputs in a structured form. We define a JSON schema (`MemoryItem`) with required fields `(id, schema_version, content, kind, subject, source, confidence, importance)` and optional fields `(entities, tags, qa_pairs, sensitivity, supersedes, expires_at, preference_signal)`. The `kind` enum has five values:

| Kind | Definition |
|---|---|
| `fact` | Durable statement about the user, their world, or their tools |
| `preference` | How the user wants the agent to behave |
| `procedure` | A reusable how-to or workflow |
| `event` | Time-tagged occurrence |
| `correction` | Explicit update to a prior memory; must populate `supersedes` |

The schema is enforced via Pydantic v2 with `extra="forbid"`; unknown fields raise `ValidationError`. The contract is stable and versioned; integrating with mem0, Letta, supermemory, or any other upstream memory source is a thin adapter on top of this schema.

### 3.2 The seven-stage pipeline

```
ingest → extract* → compose → mix → train → eval → promote/snapshot
```

(*) Extract is optional; only required when the upstream is raw text rather than structured records.

**Ingest.** A `MemoryConnector` protocol implementation reads structured `MemoryItem` records from a source (JSONL file, mem0 export, supermemory dump, etc.).

**Extract.** When the input is raw text, a frontier LLM (Anthropic / OpenAI / Ollama) is invoked with a precision-engineered system prompt that enforces the 5-kind taxonomy, forbids fabrication, refuses to extract credentials, and emits a JSON array. Synthesized auto-fields (`id`, `schema_version`, `source.captured_at`) are added pipeline-side; the LLM is structurally prevented from forging them.

**Compose.** Each `MemoryItem` is mapped to multiple training examples via per-kind templates. The compose stage explicitly partitions: the last template in each kind's template list is reserved as a held-out evaluation probe. Training examples and probes therefore use different phrasings, so personal-recall metrics measure generalization rather than memorization.

Superseded memories (any ID appearing in another memory's `supersedes` list) are excluded from both training and eval. This is the correction mechanism.

**Mix.** The training set for a single night is composed of three slices:

- *Today*: 70–80% — examples derived from new and recent memories
- *Replay*: 10–20% — sample of examples derived from previously-trained memories
- *Anchor*: 5–10% — fixed "general capability" examples that defend against base-model drift

Anchor selection is **deterministic by `source_memory_id` order** (per ADR-005). Random sampling, our initial implementation, introduced run-to-run variance that masked hyperparameter effects.

**Train.** A LoRA fine-tune via `python -m mlx_lm lora --train` invoked as a subprocess (rationale in ADR-003). Conservative defaults: rank determined by `--num-layers` (default 4–8 of the model's attention layers), learning rate 3e-5 to 1e-4, 80–200 iterations, mask-prompt enabled. The subprocess wrapper writes a `metadata.json` with the full CLI invocation, hyperparameters, mix composition, source memory IDs, library versions, and timestamps.

**Eval.** Two evaluations run against the trained adapter:

- *Personal recall*: auto-generated probes from the same memory stream (held out from training in §3.2.compose). Measures whether the adapter learned the memories.
- *General capability*: a fixed anchor probe set running against the base model and the adapter. Measures regression.

Both use case-insensitive substring matching; exact-match is too strict for the noisy generations of 1–8B-class models.

**Promote.** A four-decision matrix:

| Personal recall | General regression | Decision |
|---|---|---|
| ≥ `min_recall` | ≤ `max_regression` | **PROMOTE** |
| ≥ `min_recall` | `max_regression` < x ≤ `warn_regression` | **PROMOTE_WITH_WARNING** |
| ≥ `min_recall` | > `warn_regression` | **REJECT** |
| < `min_recall` | (any) | **REJECT (low recall)** |

Validation-tier defaults: `min_recall=0.30`, `max_regression=0.05`, `warn_regression=0.15`.
Production-tier defaults: `min_recall=0.70`, `max_regression=0.03`, `warn_regression=0.08`.

**Snapshot.** Promoted snapshots land at `snapshots/<timestamp>/` containing the adapter directory, the full eval JSONs, the gate decision, and the training metadata. A `live` symlink is atomically updated to the latest promotion. Rejected snapshots land under `snapshots/rejected/<timestamp>/` and `live` is preserved. Rollback to any prior promoted snapshot is `rollback_to(snapshots_dir, name)` — a single symlink swap.

### 3.3 Why each design choice

- **LoRA over full fine-tune.** Adapter files are ~50–200 MB, making per-night versioning trivially cheap and rollback near-instant. Catastrophic forgetting is more contained than full FT.
- **Subprocess over Python API.** The MLX-LM CLI is the most stable surface across versions; the same wrapper targets Unsloth on cloud GPU with one config switch.
- **Four-decision gate over binary gate.** The warn band catches drift before it becomes catastrophic and surfaces runs for human review without blocking promotion.
- **Stable anchor selection over random sampling.** Eliminates a documented variance source that mimicked tuning effects (ADR-005).

---

## 4. Implementation: DreamAgent

DreamAgent is the open-source reference implementation, in Python ≥ 3.12. Module structure:

```
src/dreamagent/
    schema.py    — MemoryItem & MemoryBatch (pydantic v2, extra="forbid")
    ingest/      — MemoryConnector protocol + JSONLConnector, FixtureConnector
    extract/     — frontier-LLM extraction (Anthropic/OpenAI/Ollama backends)
    compose/     — per-kind templates · examples · rehearsal mix · anchors
    train/       — MLX-LM LoRA subprocess wrapper with full lineage metadata
    eval/        — substring-match probe runner
    promote/     — four-decision gate · snapshot · rollback
    serve/       — V2.0 alpha MCP server (FastMCP, stdio)
    cli.py       — typer entry point: dream / extract / drill / serve / etc.
```

Tests: 120 passing under pytest. Lint: ruff-clean.

### 4.1 The extraction prompt

The extract stage is the accuracy-critical path: an LLM hallucination at extract time becomes a learned hallucination after training. The prompt at `src/dreamagent/extract/prompt.py` enforces:

1. The 5-kind taxonomy with positive AND negative discriminating examples for each kind.
2. "No fabrication": lower confidence over guessing; emit nothing if `confidence < 0.3`.
3. "Skip ephemeral": one-shot conversational content ("hi", "thanks", "running tests now") is not memory.
4. "Refuse sensitive data": passwords, SSNs, credentials are dropped silently.
5. Output format: a JSON array of MemoryItems, no markdown wrapping, no preamble.

The pipeline-side validator enforces these contractually: malformed records are rejected with structured reasons in the `ExtractionReport`.

### 4.2 The V2.0 MCP server

`dreamagent serve` exposes the live adapter as a Model Context Protocol server over stdio (FastMCP). Two tools:

- `query_memory(question: str) -> str`
- `query_memory_with_lineage(question: str) -> dict` — same answer plus `base_model`, `adapter_version`, and an optional `warning` if no live adapter is loaded.

The model is lazily loaded on first query to keep MCP-host startup cheap. Configuration via environment variables (`DREAMAGENT_BASE_MODEL`, `DREAMAGENT_SNAPSHOTS_DIR`, `DREAMAGENT_MAX_TOKENS`) — the appropriate pattern for processes launched by an MCP host with no CLI args.

---

## 5. Experimental methodology

### 5.1 Base models

Two open-weight models tested:

- **Llama 3.2 1B Instruct, 4-bit** (`mlx-community/Llama-3.2-1B-Instruct-4bit`) — validation tier, used for fast iteration during pipeline development.
- **Llama 3.1 8B Instruct, 4-bit** (`mlx-community/Meta-Llama-3.1-8B-Instruct-4bit`) — production tier, used for all reported V1 production-tier results.

A previous attempt on Qwen 3 0.6B-4bit was abandoned after 16 tuning runs failed to land a clean PROMOTE. The diagnosed cause: Qwen 3 emits `<think>...</think>` reasoning blocks in instruct mode; our anchor responses contain no `<think>` content; training destroyed the model's reasoning capability and induced fabrication on factual probes. Full failure analysis: `docs/tuning/qwen3-0.6b-4bit.md`.

### 5.2 Hardware

All experiments on Apple Silicon (M-series) via MLX-LM. Single device, no distributed training.

### 5.3 Fixtures

- **`fixtures/v1_baseline.jsonl`** — 50 hand-authored `MemoryItem` records covering all 5 kinds: 15 facts, 13 preferences, 11 procedures, 9 events, 2 corrections (which supersede 2 of the facts, leaving 48 trainable memories). Authored to span a realistic personal-assistant memory surface (user's pets, projects, deploy commands, tool preferences, dates).
- **`fixtures/anchors/general_anchor.jsonl`** — 105 hand-authored "general capability" anchor examples spanning math, geography, programming concepts, "I don't know" hedging, persona statements, and language translation. The first 40 are the generalist core; entries 41–105 target specific failure modes observed during tuning (prime numbers, RAM/GPU acronyms, "don't know" responses to unknown personal questions, etc.).
- **`fixtures/anchors/general_eval.jsonl`** — 30 hand-authored probe questions disjoint from the anchor training set, covering similar categories.
- **`benchmarks/probes/cross_memory_reasoning.jsonl`** — 10 hand-authored probes requiring synthesis across 2–3 memories. See §5.5 for design rationale.

### 5.4 Probes

**Personal recall probes** are auto-generated from each `MemoryItem` via the same template machinery used for training, with the held-out template (the last in each per-kind list) reserved exclusively for eval. 48 probes are generated from the 50-memory fixture.

**General capability probes** (n=30) are loaded from the anchor eval fixture. Same probes used pre- and post-training.

**Cross-memory reasoning probes** (n=10) are hand-authored questions that require synthesizing 2–3 memories simultaneously to answer correctly. Each probe declares `requires_memory_ids` listing the memory IDs needed; this enables fair evaluation of retrieval-based baselines under "top-k must include all required" conditions. Example:

> "Given what you know about the user's projects and their preferred tools, what command would they likely use to test the DreamAgent project?"
> *requires*: `mem_fix_005` (user's primary project is DreamAgent) + `mem_fix_027` (test command is `uv run pytest -q`).

**Identity-drift probes** (n=8) are persona questions ("Who are you?", "Can you make up information?", "Should you guess?"). Hand-authored.

**Query-latency probes** are 10 sample queries used to measure p50/p95/p99 over 30 generation calls with 2 warm-up generations discarded.

### 5.5 Cross-memory reasoning probe design rationale

A core empirical claim of MORPHEUS is that parametric memory enables reasoning across memories in a single forward pass — capability a top-k retrieval system cannot satisfy in one shot. To measure this rigorously, we constructed 10 probes that each require knowledge from 2–3 specific memory items.

We acknowledge a methodological risk: **the probe set was designed by the authors who also designed the methodology**. A reviewer should treat the +60pp result with appropriate skepticism (§9.4). Mitigation: the probe-to-memory mapping is declared explicitly in the JSONL, so an adversarial reader can verify (a) the probes are reasonable, (b) the required memories actually exist in the training set, and (c) the expected-substrings are not gameably easy.

### 5.6 Calibration protocol

For a new base model, calibration begins with the prior best-known recipe (from the most-similar previously-tuned model). The first run produces one of {PROMOTE, PROMOTE_WITH_WARNING, REJECT}; we walk inward by adjusting at most 1–2 hyperparameters per run until clean PROMOTE.

For Llama 3.1 8B, calibration succeeded on the first attempt with the locked Llama 3.2 1B recipe modified only by `num_layers: 4 → 8`. This was the project's cleanest calibration.

---

## 6. Results

### 6.1 V1 Pass 1 — Llama 3.2 1B single-night

After 16 documented tuning runs converging on the locked recipe (iters=90, num_layers=4, lr=3e-5, anchor_ratio=0.30, max_anchors=60):

| Metric | Value |
|---|---|
| Personal recall | 46% (22/48) |
| General capability (base) | 93.3% (28/30) |
| General capability (adapter) | 90.0% (27/30) |
| Regression | 3.3pp |
| Gate decision | **PROMOTE** (clean) |

Full per-run trajectory in `docs/tuning/llama-3.2-1b-instruct-4bit.md`.

### 6.2 V1 Pass 2 — Llama 3.1 8B single-night

Locked-recipe calibration (iters=90, num_layers=8, lr=3e-5, anchor_ratio=0.30, max_anchors=60):

| Metric | Value |
|---|---|
| Personal recall | 43.75% (21/48) |
| General capability (base) | 96.67% (29/30) |
| General capability (adapter) | 96.67% (29/30) |
| Regression | **0.0pp** |
| Gate decision | **PROMOTE** (clean) |

Calibration produced zero general-capability regression on the first attempt. The 8B's stronger base (96.67% vs 1B's 93%) combined with the soft-training recipe is the most parsimonious explanation.

### 6.3 V1 Pass 3 — 7-night chained-training drill on Llama 3.1 8B

Each night uses the locked recipe and resumes from the prior night's adapter via `--resume-from-snapshot`. Same 50 fixtures presented each night.

| Night | Decision | Personal | Δ general | Resumed from |
|---|---|---|---|---|
| 1 | PROMOTE | 43.8% | 0.0pp | (base) |
| 2 | PROMOTE_WITH_WARNING | 58.3% | +10.0pp | Night 1 |
| 3 | PROMOTE | 66.7% | +3.3pp | Night 2 |
| 4 | PROMOTE_WITH_WARNING | 75.0% | +13.3pp | Night 3 |
| 5 | PROMOTE_WITH_WARNING | 75.0% | +6.7pp | Night 4 |
| 6 | PROMOTE_WITH_WARNING | 81.3% | +10.0pp | Night 5 |
| 7 | PROMOTE_WITH_WARNING | 75.0% | +6.7pp | Night 6 |

Aggregate observations:
- **All 7 nights promoted; zero rejections.**
- Personal recall monotonically climbs across the first 6 nights (44% → 81%), suggesting chained training accumulates capability.
- General regression oscillates in a bounded 0–13.3pp band, never breaching the 15pp REJECT threshold.
- The recipe ceiling on this fixture appears to be ~75–81% personal recall; hotter LR or more iters would likely push higher at the cost of more regression.

### 6.4 Benchmark suite on night-7 adapter

Run via `python -m benchmarks.<name>` (reproducible):

| Benchmark | Metric | Result |
|---|---|---|
| `personal_recall` | Pass rate on 48 held-out probes | **75.0%** (36/48) |
| `general_capability` | Adapter pass rate (vs 96.67% base) | 90.0% (27/30); **−6.7pp** regression |
| `cross_memory_reasoning` | Adapter pass rate vs base-only baseline | 90.0% (9/10); +60pp vs *base alone* (see §6.6) |
| `query_latency` | p50 / p95 / p99 for 48-token responses | 1.08s / 2.25s / 2.27s |
| `identity_drift` | Adapter pass rate (vs 62.5% base) | 75.0% (6/8); **−12.5pp** drift = improvement |

### 6.5 V2.1 head-to-head vs vector-retrieval baseline

Subsequent to §6.4, we built a vector-retrieval baseline (`sentence-transformers/all-MiniLM-L6-v2` + top-5 retrieval + same base model for generation) and re-ran the same probes through both systems. Full protocol: `docs/tuning/v2.1-vs-baselines.md`.

| Probe set | DreamAgent | Retrieval baseline | Δ |
|---|---|---|---|
| `personal_recall` | 75.0% (36/48) | 68.8% (33/48) | **+6.2pp** |
| `cross_memory_reasoning` | 90.0% (9/10) | 90.0% (9/10) | **+0.0pp** |

A material weakening of the cross-memory-reasoning claim in §6.4. When the retrieval step surfaces all required memories (which top-5 on a 50-memory corpus largely permits for our 10 hand-authored probes), the base model can synthesize across them in context as well as the parametric adapter does. The +6.2pp personal-recall advantage replicates.

### 6.6 V2.2 three-way head-to-head + adversarial probes

We then constructed 15 adversarial probes specifically designed to defeat vector retrieval — questions with low embedding similarity to their target memories (`benchmarks/probes/adversarial_retrieval.jsonl`). We also added a composed system that queries both DreamAgent and the retrieval baseline and uses the same base model to reconcile the two candidate answers.

Three systems × three probe sets:

| Probe set | DreamAgent | Retrieval | Composed |
|---|---|---|---|
| `personal_recall` (n=48) | **75.0%** | 68.8% | 64.6% |
| `cross_memory_reasoning` (n=10) | 90.0% | 90.0% | 90.0% |
| `adversarial_retrieval` (n=15) | 80.0% | **93.3%** | 86.7% |

Two pre-registered (§10.5) negative outcomes both materialized:

1. **DreamAgent lost on adversarial probes** (−13.3pp vs retrieval). The probes we designed to favor parametric memory did not in fact favor it. We hypothesize either insufficient embedding-distance separation or insufficient corpus size (N=50, k=5 → 10% recall ceiling is too generous).

2. **Composition did not beat the best individual system on any probe set.** On personal recall it *hurt* (64.6% vs DA's 75.0%) — the reconciliation step degraded a clean DA answer with a noisier retrieval answer.

Full protocol and per-probe inspection: `docs/tuning/v2.2-adversarial-and-composed.md`.

### 6.7 Transferable tuning lessons

Six lessons we identified across 30+ tuning runs on two base models (documented in `docs/tuning/README.md`):

1. **Eval probe count must match the threshold's resolution.** A 15-probe eval with 5pp max-regression has resolution 6.7pp — every single failure breaches the threshold. We hit this when our 15-probe eval reported 7pp regression for a recipe that, on a 30-probe re-evaluation, actually showed 20pp regression. Resolution-matters generalization: use ≥20 probes for a 5pp threshold; ≥40 for 2.5pp.
2. **Reasoning-tag models need reasoning-preserving anchors.** Qwen 3 and other thinking-tag models will lose their reasoning capability if anchor responses don't include `<think>...</think>` content. Either rewrite anchors or use a non-thinking base.
3. **Stable anchor selection > random sampling.** Random subsetting introduces variance that mimics hyperparameter effects.
4. **Stronger base → softer training.** Models with higher base capability degrade more in absolute terms at the same LR/iters. Compensate by reducing both.
5. **Anchor file order matters when using stable selection.** Place highest-leverage anchors early in the file.
6. **Bracket then bisect.** First an aggressive + a conservative run; both fail. Then bisect inward changing 1-2 levers per iteration.

---

## 7. Discussion

### 7.1 What the cross-memory and adversarial results demonstrate (revised after §6.5 and §6.6)

The V1 framing claimed a +60pp parametric-memory advantage. Two subsequent experiments retracted this in stages:

- **V2.1 (§6.5)** measured DreamAgent vs a vector-retrieval baseline on the same memories using the same base model: parity on cross-memory reasoning, +6.2pp on personal recall.

- **V2.2 (§6.6)** added 15 hand-designed adversarial probes — questions phrased to have low embedding similarity to their target memories, designed specifically to break vector retrieval. Result: **DreamAgent 80% / Retrieval 93.3% / Composed 86.7%.** The retrieval baseline outperformed parametric memory on the very probes we designed to favor parametric memory. The composition (DA + retrieval + base-model reconciler) underperformed both individual systems on personal recall.

§7.1.1 (now folded into this section) hypothesized that retrieval would lose ground at large corpora or under adversarial probes. We measured the second case; the hypothesis was not supported. Two explanations stand out, both empirical observations:

1. **`all-MiniLM-L6-v2` is more semantically robust than we expected.** Probes about "GPL-3.0" surface memories about "Apache 2.0 or MIT" because both are in the "open-source license" concept neighborhood at the embedding level.

2. **Top-5 retrieval at N=50 is too generous a recall regime to fail.** We're surfacing 10% of the corpus per query; the probability the relevant memory is in the top-5 is high enough that the generation step almost always has the right context to work with.

A genuinely adversarial regime would require N≥1000 with k=5 (top-k coverage <1%), embedding-distance-selected probes, or a retrieval system *forbidden* from accessing specific memories. None of these are demonstrated here.

**What we can defensibly claim from these experiments:** DreamAgent's +6.2pp personal-recall advantage at N=50 is real and replicates. Cross-memory reasoning is at parity. Adversarial probes designed by us did *not* expose a parametric-memory advantage; retrieval was more robust. Composition with a naive reconciler does not beat the best individual system.

**What this means for V2 positioning:** DreamAgent's value proposition shifts from "measured-capability wins" to "structural property wins" — privacy, host-agent independence, operational simplicity, GDPR-clear deletion. These are real and matter, but they are properties of the architecture, not benchmark results.

### 7.2 What the identity-drift result demonstrates

The adapter scores **higher** than the base on persona probes (75% vs 62.5%; drift is −12.5pp, meaning improvement). This is initially surprising. The most plausible explanation: the anchor mix includes a deliberate persona-reinforcement subset ("Who are you?", "Should you guess?", "Can you share secrets?"), and the anchor share of the training mix (30% target, 60-anchor cap) actively reinforces these patterns each night. The base model's responses to these probes were noisier; the adapter's are more consistent.

This is empirically a stability signal, not a deterioration signal, heading into longer-horizon drills.

### 7.3 The 75% recall plateau

Personal recall plateaus at 75–81% across nights 4–7. We did not measure this directly, but the most plausible explanation is that the locked recipe's iters/LR/rank combination is near the ceiling for this 50-memory fixture. A hotter LR (5e-5 instead of 3e-5) or more iters (150 instead of 90) would likely push higher; we deliberately did not pursue this because the locked recipe's general-capability preservation is what we wanted to demonstrate.

§9.7 calls this out as a limitation: we have not yet measured the upper bound on personal recall for MORPHEUS, only the recipe that prioritizes safety over recall.

### 7.4 Why we don't publish a LoCoMo number

LoCoMo measures retrieval over conversation history. MORPHEUS does not retrieve. Running LoCoMo on MORPHEUS requires choosing one of two protocols, neither of which is a direct comparison to mem0 or Letta:

- **Protocol A** — per-conversation fine-tune: train MORPHEUS on each LoCoMo conversation before evaluation. Expensive and atypical of deployment.
- **Protocol B** — oracle backend: use the dreamed model as a knowledge backend for a separate retrieval agent doing the LoCoMo task. Tests the composed system.

We commit to publishing both numbers once Pass 2 work at depth is complete. We will not cherry-pick. ADR-007 documents the positioning.

---

## 8. Limitations and threats to validity

We enumerate weaknesses pre-emptively, in priority order.

### 8.1 Small evaluation sets

- 30 general-capability probes
- 10 cross-memory reasoning probes
- 8 identity-drift probes
- 48 personal-recall probes (the largest)

Statistical power is correspondingly low. A 6.7pp regression is one failure out of 30 — within Bernoulli-style noise. The 60pp cross-memory delta is more robust (3 vs 9 out of 10) but still a small N.

Mitigation: probe set ships with the repo; reviewers can extend.

### 8.2 Single-language, single-domain

All probes and fixtures are English. The 50-memory fixture is a tech-worker persona (Mac, Python, GitHub). Generalization to other languages and personas is unmeasured.

### 8.3 Compressed Pass 3

The 30-night Pass 3 originally planned was compressed to 7 nights for V1 delivery. 7 nights is enough to demonstrate that chained training does not catastrophically diverge in the near term; it is not enough to detect slow drift over a month.

### 8.4 Author-designed cross-memory probes

The 10 cross-memory probes (`benchmarks/probes/cross_memory_reasoning.jsonl`) were authored by the same person who authored the methodology. There is an inherent risk of probes being constructed (consciously or unconsciously) to favor MORPHEUS. We mitigate by publishing the probe-to-memory mapping; the constructions are inspectable.

### 8.5 Comparison to retrieval baselines — measured at V2.1; result weakens earlier claim

§6.5 reports the V2.1 head-to-head. A simple vector-retrieval baseline (sentence-transformers + top-5 + same base model) achieved parity with DreamAgent on cross-memory reasoning (90% / 90%). The architectural argument that retrieval cannot satisfy multi-memory synthesis in one shot is empirically *false* under conditions where retrieval recall is high.

We have not yet compared against mem0, Letta, or Zep specifically — that work is V2.2. Our simple baseline is almost certainly weaker than what a serious agent harness with filesystem tools delivers (per Letta's own benchmarking [\[10\]](https://www.letta.com/blog/benchmarking-ai-agent-memory)), so head-to-head numbers against production memory systems would likely show DreamAgent losing on retrieval-style probes by even more.

The corrected positioning: DreamAgent and retrieval are complementary, not competitive. See §7.1 (revised) and §10 for consequences.

### 8.6 Single base model at production tier

Production-tier results are only on Llama 3.1 8B Instruct. We claim transferability based on the 1B and 8B both passing with the same recipe (only `num_layers` adjusted), but no other 4B–14B class model has been validated.

### 8.7 Recall ceiling not characterized

§7.3: the 75–81% plateau may be recipe-bound rather than methodology-bound. We have not characterized the upper bound by sweeping more aggressive recipes.

### 8.8 Query latency is too high for some deployments

p95 of 2.25s for 48-token responses on Apple Silicon is workable for an interactive memory backend but does not meet V2.1's stated <500ms target. Optimization candidates (prefix caching, KV-cache reuse, smaller production tier, streamable responses) are unbuilt.

### 8.9 No live MCP client end-to-end verification published

V2.0 MCP server is unit-tested (factory shape, env precedence, symlink resolution, no-adapter fallback) but the live transport against a real Claude Code install is verified by hand, not by an automated test. This is a structural limitation: MCP testing requires a client.

### 8.10 Snapshot proliferation

A naive nightly run for a year produces 365 adapter snapshots. We have a planned `merge` stage using mergekit to consolidate weekly, but this is unbuilt. Until then, manual housekeeping is required.

---

## 9. Reproducibility

The exact commands to reproduce every number in §6 are below. All require a clone of `https://github.com/mrdulasolutions/dreamagent` and `uv sync --extra mcp`.

### 9.1 Pass 1

```bash
dreamagent dream \
  --validation-tier \
  --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
  --source fixture:v1_baseline \
  --iters 90 --num-layers 4 --learning-rate 3e-5 \
  --anchor-ratio 0.30 --max-anchors 60 \
  --tag pass-1
```

Expected: clean PROMOTE, ~46% personal recall, ~3.3pp regression. ~5 minutes on Apple Silicon (first run includes model download).

### 9.2 Pass 2 (calibration)

```bash
dreamagent dream \
  --validation-tier \
  --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --source fixture:v1_baseline \
  --iters 90 --num-layers 8 --learning-rate 3e-5 \
  --anchor-ratio 0.30 --max-anchors 60 \
  --tag pass-2-calib
```

Expected: clean PROMOTE, ~43.75% personal recall, 0pp regression. ~10 minutes (first run includes 8B download).

### 9.3 Pass 3 (7-night drill)

```bash
dreamagent drill --nights 7 \
  --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
  --iters 90 --num-layers 8 --learning-rate 3e-5 \
  --anchor-ratio 0.30 --max-anchors 60 \
  --continue-on-reject
```

Expected: 7 PROMOTE rows in the trajectory table. ~45 minutes total.

### 9.4 Benchmarks

```bash
python -m benchmarks.personal_recall          --snapshot runs/snapshots/live --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python -m benchmarks.general_capability       --snapshot runs/snapshots/live --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python -m benchmarks.cross_memory_reasoning   --snapshot runs/snapshots/live --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python -m benchmarks.query_latency            --snapshot runs/snapshots/live --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
python -m benchmarks.identity_drift           --snapshot runs/snapshots/live --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

Each writes a `BenchmarkResult` JSON to `benchmarks/results/` for comparison.

### 9.5 Version pinning

- Python: ≥3.12
- mlx-lm: ≥0.22 (we use 0.31.3)
- pydantic: ≥2.9
- mcp: ≥1.4

The full set of pinned versions is in `uv.lock`.

---

## 10. Future work

### 10.1 V2.1 — latency optimization and head-to-head benchmarks

- Reduce query p95 from 2.25s to < 500ms via prefix caching, KV reuse, and possibly a smaller production tier (Qwen 3 4B with anchor rewrites, or Llama 3.2 3B).
- Run the `vs_baselines/` benchmarks: same memory set through mem0, Letta, and DreamAgent; compare on cross-memory reasoning, latency, and privacy axes.
- Publish LoCoMo numbers under both Protocol A and Protocol B (§7.4).

### 10.2 V2.2 — composition cookbook + adversarial probes (delivered with negative results)

- ✅ Vector-retrieval baseline + three-way runner (DA / retrieval / composed)
- ✅ 15 adversarial probes designed to defeat retrieval
- ❌ DreamAgent < Retrieval on adversarial (80% vs 93.3%) — probes were not adversarial enough
- ❌ Composition < max(DA, retrieval) on personal recall (64.6% vs 75%) — naive reconciler hurts
- ✅ Composition cookbook `examples/08-mem0-plus-dreamagent/` shipped with honest caveats
- ⏳ Adapter signing, multi-user namespacing — deferred

### 10.2.1 V2.3 — what the V2.2 negative results imply

- **Genuinely adversarial probes**: N≥1000 corpus with embedding-distance-selected probes
- **Better reconciler designs**: "pick-the-better-one" rather than "summarize both"
- **Measured comparison against mem0 / Letta / Zep specifically**, not just our simple baseline
- **30-night true Pass 3 drill**

### 10.3 V3 — frontier-scale

- Apply the methodology to 70B-class models on cloud GPUs. Conditional on V2 evidence.
- Hyperparameters likely need re-tuning at scale; ADR-006's "stronger base → softer training" heuristic predicts further LR/iter reductions.

### 10.4 Methodology extensions

- **DPO for preference memories**: when multiple `preference` memories accumulate on the same `axis`, switch from SFT to DPO. Currently untested.
- **Weekly mergekit consolidation**: implement the planned `merge` stage to compress per-night LoRAs into a single weekly adapter (TIES or DARE merge).
- **Real long-horizon drift studies**: 30, 60, 100 nights. The compressed 7-night drill does not detect slow drift.
- **Generative replay**: have the current model produce its own training rehearsal examples.
- **Orthogonal-constraint LoRA** (CL-LoRA [\[7\]](https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf)) on the night-to-night chained training.

### 10.5 Methodology critiques we'd accept

We anticipated four critiques in the V1 draft. Two have now materialized across V2.1 and V2.2; the paper has been revised accordingly.

- ✅ **Realized at V2.1:** A retrieval baseline scoring > 50% on our cross-memory probe set. Actual result: 90% — parity with DreamAgent (§6.5).
- ✅ **Realized at V2.2:** A retrieval baseline outperforming DreamAgent on adversarial probes designed to favor parametric memory. Actual result: retrieval 93.3%, DA 80% (§6.6). Composition (DA + retrieval + reconciler) also failed to beat either alone.
- A reproducible run on the published recipe yielding > 15pp general regression would falsify the stability claim. (Untriggered; not falsified.)
- Drift signal detected over 30 chained nights would constrain V1's deployability. (Compressed Pass 3 only ran 7 nights; the 30-night drill remains future work.)

The team committed in writing in the V1 draft that surfacing such results would update positioning publicly rather than be buried. The V2.1 documentation (`docs/tuning/v2.1-vs-baselines.md`), V2.2 documentation (`docs/tuning/v2.2-adversarial-and-composed.md`), and the §1/§6.5/§6.6/§7.1/§8.5/§11 revisions in this paper are that update — each published in the same commit as the new measurement.

The cumulative effect of the two retractions is significant: DreamAgent's claimed empirical advantages over retrieval have shrunk from "+60pp cross-memory + parametric-advantage at scale" (V1) to "+6.2pp personal recall at N=50" (post V2.2). The methodological contribution (eval-gated promotion + per-night snapshots) is unaffected; the empirical advantage over retrieval is now measured to be narrow.

---

## 11. Conclusion

We presented **MORPHEUS**, a seven-stage methodology for consolidating structured agent memories into the weights of a small language model via nightly LoRA fine-tuning with eval-gated promotion. We validated it on Llama 3.2 1B Instruct and Llama 3.1 8B Instruct, and demonstrated stability across a 7-night chained-training drill (all 7 nights promoted, zero rejects).

Two empirical experiments against retrieval baselines have substantially narrowed the empirical claims of this work:

- **V2.1 (§6.5):** parity with a vector-retrieval baseline on cross-memory reasoning; +6.2pp on personal recall.
- **V2.2 (§6.6):** retrieval beats DreamAgent on author-designed adversarial probes (80% vs 93.3%); composition with a naive reconciler does not beat the best individual system on any tested probe set.

Two of the four pre-registered (§10.5) falsifiable claims have materialized. The remaining defensible empirical claim is the narrow **+6.2pp personal-recall advantage at N=50 on a hand-authored fixture.** Everything else is at parity or behind retrieval.

**The headline contribution is therefore methodological, not empirical.** The four-decision eval-gated promotion scheme combined with per-night adapter snapshots and the MemoryItem contract makes autonomous nightly training safe enough to deploy without human review. The DreamAgent value proposition shifts from "wins on capability" to **wins on structural properties** — privacy (model weights on disk, no embeddings persisted), host-agent independence (any MCP client), operational simplicity (one process, one file), GDPR-clear deletion. These are real and matter for many use cases; they are not benchmark wins.

The methodology is open-source (Apache 2.0, attribution required per `NOTICE`), the reference implementation (`DreamAgent`) is reproducible from a fresh clone, and every number in this paper is regeneratable via a single `python -m benchmarks.<name>` invocation against the published recipe.

We invite reproduction, criticism, and further falsification. The remaining open empirical questions we'd most like to see tested: (a) DreamAgent vs strong retrieval baselines (mem0, Letta) at N≥1000 corpora; (b) genuinely adversarial probe sets constructed by embedding-distance selection rather than author intuition; (c) better composition reconcilers than the naive "summarize both" approach we used here; (d) long-horizon (30-night) drift behavior.

---

## Acknowledgments

DreamAgent stands on substantial prior work in memory consolidation for agentic systems and in continual learning research. We are especially indebted to mem0, Letta, and OpenClaw for shipping production memory systems we could measure ourselves against; to the CL-LoRA and SuRe authors for techniques we directly compose with; and to the MLX-LM and Unsloth teams for the training infrastructure this work depends on.

---

## References

\[1\] mem0ai. *Mem0 — Memory layer for AI agents.* https://github.com/mem0ai/mem0
\[2\] Letta Inc. *Letta (formerly MemGPT) — Stateful agents with hierarchical memory.* https://github.com/letta-ai/letta
\[3\] Supermemory. *Universal memory APIs.* https://supermemory.ai
\[4\] Zep. *Temporal knowledge graph memory.* https://www.getzep.com/
\[5\] EverMind AI. *EverMemOS — Cloud-hosted memory OS.* https://evermind.ai/
\[6\] *Learning to Forget: Sleep-Inspired Memory Consolidation for Resolving Proactive Interference in Large Language Models.* arXiv:2603.14517. https://arxiv.org/abs/2603.14517
\[7\] He et al. *CL-LoRA: Continual Low-Rank Adaptation for Rehearsal-Free Class-Incremental Learning.* CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf
\[8\] Zylos Research. *Continual Learning and Catastrophic Forgetting Prevention in AI Agents.* April 2026. https://zylos.ai/research/2026-04-09-continual-learning-catastrophic-forgetting-ai-agents
\[9\] Snap Research. *LoCoMo — Long Conversation Memory benchmark.* https://github.com/snap-research/locomo
\[10\] Letta. *Benchmarking AI Agent Memory: Is a Filesystem All You Need?* https://www.letta.com/blog/benchmarking-ai-agent-memory
\[11\] OpenClaw. *OpenClaw Dreaming Guide.* https://dev.to/czmilo/openclaw-dreaming-guide-2026-background-memory-consolidation-for-ai-agents-585e
\[12\] Anthropic. *Dreaming for Claude Agents.* https://letsdatascience.com/blog/anthropic-dreaming-claude-managed-agents-self-improving-may-6
\[13\] IBM. *Catastrophic Forgetting.* https://www.ibm.com/think/topics/catastrophic-forgetting
\[14\] Shin et al. *Continual Learning with Deep Generative Replay.* arXiv:1705.08690. https://arxiv.org/abs/1705.08690
\[15\] Hugging Face. *PEFT — Parameter-Efficient Fine-Tuning library.* https://github.com/huggingface/peft
\[16\] *Semi-parametric Memory Consolidation: Towards Brain-like Deep Continual Learning.* arXiv:2504.14727. https://arxiv.org/html/2504.14727v1
\[17\] *Sleep-time Compute: Beyond Inference Scaling at Test-time.* arXiv:2504.13171. https://arxiv.org/pdf/2504.13171
\[18\] Emergent Mind. *Rank-One Model Editing (ROME).* https://www.emergentmind.com/topics/rank-one-model-editing-rome
\[19\] *Model Editing at Scale Leads to Gradual and Catastrophic Forgetting.* arXiv:2401.07453. https://arxiv.org/pdf/2401.07453
\[20\] deeplearning.ai. *Test-Time Training End-to-End (TTT-E2E).* https://www.deeplearning.ai/the-batch/test-time-training-end-to-end-ttt-e2e-retrains-model-weights-to-handle-long-inputs
\[21\] *In-Place Test-Time Training.* arXiv:2604.06169. https://arxiv.org/html/2604.06169v1
\[22\] *Memento: Fine-tuning LLM Agents without Fine-tuning LLMs.* arXiv:2508.16153. https://arxiv.org/abs/2508.16153

---

**Cite as:**

```bibtex
@software{morpheus2026,
  title  = {MORPHEUS: Memory Overnight Re-parameterization with Promotion via Held-out Eval and Update Snapshots},
  author = {{Mr Dula Solutions}},
  year   = {2026},
  url    = {https://github.com/mrdulasolutions/dreamagent},
  note   = {Apache-2.0 with required attribution. DreamAgent is the reference implementation.}
}
```

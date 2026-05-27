# MORPHEUS — The DreamAgent Methodology

**Originated by:** Mr Dula Solutions, 2026-05-26.
**License:** Apache 2.0 (code), attribution required for derivative works (see [NOTICE](../NOTICE)).

This document is the canonical description of **MORPHEUS** — the technique
published in this repository and implemented by the DreamAgent project. If you
build on it, please cite via [CITATION.cff](../CITATION.cff) and credit
MORPHEUS as the methodology name.

---

## What is MORPHEUS?

**MORPHEUS** stands for **M**emory **O**vernight **R**e-parameterization,
**P**romotion via **H**eld-out **E**val, **U**pdate **S**napshots.

It is a system for converting an AI agent's captured day-time memories into the
**weights** of a small open-source language model via **nightly LoRA
fine-tuning**, gated by an **automated promote/reject eval matrix** and
**per-night adapter snapshots** for safe rollback.

Each letter of the acronym maps to a stage or invariant:

| Letter | Concept | Where it lives in the code |
|---|---|---|
| **M**emory | Structured `MemoryItem` records as input | `src/dreamagent/schema.py`, `src/dreamagent/ingest/` |
| **O**vernight | Nightly cron cadence; the model trains while you sleep | `dreamagent install-cron`, `cli.py: dream` |
| **R**e-parameterization | Memories become **weights**, not retrieved chunks | `src/dreamagent/train/` (LoRA via MLX-LM) |
| **P**romotion | A gate decides whether each night's adapter goes live | `src/dreamagent/promote/gate.py` |
| **H**eld-out | Eval probes are deliberately disjoint from training | `src/dreamagent/compose/examples.py` (last template = probe) |
| **E**val | Personal-recall + general-capability dual checks | `src/dreamagent/eval/`, `benchmarks/` |
| **U**pdate | LoRA adapter is the atomic update unit | `runs/snapshots/<timestamp>/adapter/` |
| **S**napshots | Every promoted update is versioned + rollbackable | `src/dreamagent/promote/snapshot.py` |

The trained model — the **memory specialist** — exposes a knowledge oracle
that larger frontier agents query at inference time instead of doing
vector-based retrieval over the original memories.

**Project vs. methodology naming:**

- **DreamAgent** — the project, repository, and CLI (`dreamagent dream`, etc.)
- **MORPHEUS** — the methodology DreamAgent implements

A future implementation of MORPHEUS in another language or framework would
still be called a MORPHEUS implementation; DreamAgent is the canonical
reference implementation.

---

## The 7-Stage Pipeline

```
┌────────────────────────────────────────────────────────────────────┐
│   1. Ingest          2. Extract       3. Compose      4. Mix       │
│   raw input    →    LLM-validated  →  templated   →  with anchors  │
│                     MemoryItems        examples       + replay     │
│                                                                    │
│   5. Train       →  6. Eval        →  7. Promote/Snapshot          │
│   LoRA              personal +        gate decision + lineage      │
│   (MLX or           general probes    archive (PROMOTE/WARN/REJECT)│
│   Unsloth)                                                         │
└────────────────────────────────────────────────────────────────────┘
                            ⤿ nightly cron
```

### Stage definitions (canonical)

1. **Ingest.** Read structured `MemoryItem` records from an upstream source.
   The MemoryItem schema is a fixed public contract — any connector that
   emits valid records can plug in unchanged.

2. **Extract.** OPTIONAL. When the upstream is raw text rather than structured
   records, a frontier LLM with a strict prompt converts text to MemoryItems.
   This stage is where fabrication risk concentrates and so deserves the most
   prompt-engineering investment.

3. **Compose.** Each MemoryItem becomes 2–4 instruction-tuning training
   examples plus one held-out eval probe. Templates are per-kind. Examples and
   probes use different phrasings so personal-recall measures generalization,
   not memorization.

4. **Mix.** The training set for a single night is composed of:
   - 70–80% new + recent memories (today's)
   - 10–20% prior-memory replay (defends against forgetting prior nights)
   - 5–10% general-capability anchors (defends against forgetting the base)

   Ratios are tunable; **anchor ratio is the dominant lever** for protecting
   general capability.

5. **Train.** Single LoRA fine-tune via `mlx_lm lora` (Mac) or `unsloth`
   (cloud). Subprocess wrapper writes a `metadata.json` with config, dataset
   composition, source memory IDs, and library versions for full lineage.

6. **Evaluate.** Two evals run against the trained adapter and the base model:
   - **Personal-recall** probes (auto-generated from training memories, held
     out from the training set)
   - **General-capability** probes (a fixed anchor-eval set)
   Both use lenient case-insensitive substring matching, appropriate for
   small-model generation noise.

7. **Promote.** A 4-decision gate matrix:

   | Personal recall | General regression | Decision |
   |---|---|---|
   | ≥ min | ≤ max | PROMOTE |
   | ≥ min | max..warn | PROMOTE_WITH_WARNING |
   | ≥ min | > warn | REJECT |
   | < min | anything | REJECT (low recall) |

   Promoted adapters land in `snapshots/<timestamp>/` and the `live` symlink
   updates. Rejected adapters land in `snapshots/rejected/<timestamp>/` for
   inspection; `live` is preserved. Rollback is one command pointing `live`
   at a prior snapshot.

---

## The MemoryItem Schema

The schema is the public contract that lets any memory store integrate with
DreamAgent. Full specification in [src/dreamagent/schema.py](../src/dreamagent/schema.py).

Required fields: `id`, `schema_version`, `content`, `kind`, `subject`,
`source` (system + captured_at), `confidence`, `importance`, `supersedes`,
`expires_at`.

Optional fields: `entities`, `tags`, `qa_pairs`, `sensitivity`,
`preference_signal`.

The 5-kind enum is canonical:

- **fact** — durable factual statement
- **preference** — how the user wants the agent to behave
- **procedure** — reusable how-to / workflow
- **event** — time-tagged occurrence
- **correction** — explicit update to a prior memory (requires `supersedes`)

---

## The Tuning Playbook

Hyperparameters are **per-model**. Recipes that work for Qwen 3 0.6B will
catastrophically fail on Llama 3.2 1B and vice versa. Six transferable
lessons we paid for (see [docs/tuning/](./tuning/)):

1. **Eval probe count must match the threshold resolution.** With a 15-probe
   eval and a 5pp regression threshold, a single failure already breaches the
   threshold. Use ≥20, ideally 30+.
2. **Reasoning models need reasoning-preserving anchors.** Qwen 3, R1-style
   thinkers will have reasoning destroyed by anchors that go straight to the
   answer. Either rewrite anchors with `<think>` content or use a non-thinking
   base.
3. **Stable anchor selection over random sampling.** Random subsetting of a
   larger anchor pool injects variance that masks hyperparameter effects.
4. **Stronger base → softer training.** Models with higher base capability
   have more to lose. Compensate with fewer iters and lower LR.
5. **Anchor order matters.** Stable selection takes the first N anchors;
   place the highest-leverage anchors early in the file.
6. **Bracket then bisect.** First one aggressive + one conservative run reveal
   the failure modes. Then bisect inward.

---

## Architectural Choices and Why

### Why LoRA, not full fine-tune
- 1–2 hours instead of hours-to-days per night on Mac.
- Adapter file is ~50–200 MB, makes per-night versioning trivially cheap.
- Catastrophic forgetting is more contained than full FT.
- Conservative ranks (4–8 layers) are the safety margin.

### Why subprocess to MLX-LM, not the Python module API
- The MLX-LM CLI surface is the most stable contract across versions.
- Same training script targets Unsloth on cloud with one config switch.
- Errors come back as exit codes rather than partial Python state.

### Why eval-gated promotion
- Catastrophic forgetting is the central risk; the gate is the safety net.
- Reject-with-snapshot means we never silently lose a good base model.
- Rollback to any prior night is a one-command operation.

### Why the 5-kind taxonomy
- Different kinds train better with different templates and (eventually)
  different objectives — preferences should be DPO-trained when contrasting
  pairs accumulate, facts should be SFT.
- The taxonomy is small enough to be learnable by extraction LLMs and large
  enough to cover the memory categories agent harnesses produce.

### Why not just bigger context
- Context cost is per-query, weights cost is per-night. Crossover point
  arrives quickly when the same memories must accompany every conversation.
- A 1M-token context still has retrieval brittleness and proactive
  interference. Parametric memory is the more elegant primitive.

---

## Open Problems

DreamAgent has demonstrated viability at Pass 1 (validation tier on Llama
3.2 1B Instruct). Open questions for Pass 2+:

- **Forgetting curve over many nights.** Single-night runs are tractable;
  drift over 30+ consecutive nights is unmeasured.
- **DPO for preference memories.** When 2+ contrasting preferences accumulate
  on the same axis, switching to DPO should improve learning. Untested.
- **Adapter merging cadence.** Weekly merge via mergekit (TIES vs DARE)
  is in the plan; not yet implemented.
- **Cross-machine adapter sharing.** The same memories should produce the
  same adapter; verifying this requires distribution + checksums.
- **Frontier-scale (V3).** Apply the methodology to Qwen 3 70B or Llama 4
  Maverick. Hyperparameters need their own tuning sweep.

---

## Prior Art and Where DreamAgent Differs

**Text-only memory consolidation systems** (what already exists):

- **[OpenClaw Dreaming](https://dev.to/czmilo/openclaw-dreaming-guide-2026-background-memory-consolidation-for-ai-agents-585e)** — 3-phase background process consolidating signals into `MEMORY.md`. Does not touch weights.
- **[Anthropic Claude Dreaming](https://letsdatascience.com/blog/anthropic-dreaming-claude-managed-agents-self-improving-may-6)** — scheduled background process editing persistent text notes. Does not touch weights.
- **[Hermes (Nous Research)](https://hermes-agent.org/)** — file-based MEMORY.md/USER.md system prompt injection. Does not touch weights.
- **[mem0](https://mem0.ai/)** — three-storage-layer vector + entity + BM25 retrieval. Does not touch weights.
- **[Supermemory](https://supermemory.ai/)** — closed-source memory layer. Does not touch weights.
- **[Letta / MemGPT](https://www.letta.com/)** — hierarchical memory blocks with self-editing. Does not touch weights.

**Adjacent ML research** (research code, not productized agent memory):

- **[Sleep-time Compute](https://arxiv.org/pdf/2504.13171)** (Anthropic, Apr 2025) — pre-parses context during idle. Does not touch weights.
- **[SleepGate / "Learning to Forget"](https://arxiv.org/abs/2603.14517)** — KV-cache forgetting gate. Closest in spirit; still doesn't touch base weights.
- **[Semi-parametric Memory Consolidation](https://arxiv.org/html/2504.14727v1)** — biologically-inspired CLS architecture.
- **[TTT-E2E](https://www.deeplearning.ai/the-batch/test-time-training-end-to-end-ttt-e2e-retrains-model-weights-to-handle-long-inputs)** — per-query weight updates during inference.
- **[ROME / MEMIT](https://www.emergentmind.com/topics/rank-one-model-editing-rome)** — surgical fact insertion. Catastrophic forgetting at scale.
- **[Memento](https://arxiv.org/abs/2508.16153)** — strong external case memory without fine-tuning.
- **[CL-LoRA](https://openaccess.thecvf.com/content/CVPR2025/papers/He_CL-LoRA_Continual_Low-Rank_Adaptation_for_Rehearsal-Free_Class-Incremental_Learning_CVPR_2025_paper.pdf)**, **SuRe**, **FOREVER** — continual-learning techniques composable with the dream pipeline.

**Where DreamAgent is novel:** the combination of (a) personal-scale agent
memory, (b) local nightly LoRA training, (c) eval-gated automated promotion
with rollback, and (d) the "memory specialist as a service" architecture
where the dreamed model exposes itself as a queryable backend to any larger
agent.

No system in the prior-art survey above ships this end-to-end loop. The
DreamAgent methodology fills that gap.

---

## How to Cite

```bibtex
@software{morpheus2026,
  title  = {MORPHEUS: Memory Overnight Re-parameterization with Promotion via Held-out Eval and Update Snapshots},
  author = {{Mr Dula Solutions}},
  year   = {2026},
  url    = {https://github.com/mrdulasolutions/dreamagent},
  note   = {Apache-2.0 License with required attribution. DreamAgent is the reference implementation.}
}
```

Plain text:

> Mr Dula Solutions (2026). *MORPHEUS: Memory Overnight Re-parameterization
> with Promotion via Held-out Eval and Update Snapshots.* DreamAgent
> reference implementation. https://github.com/mrdulasolutions/dreamagent

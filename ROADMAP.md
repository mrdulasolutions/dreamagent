# DreamAgent Roadmap

This is a public commitment with measurable targets — not a wish list.
We will report progress against these in the [CHANGELOG](CHANGELOG.md).

---

## V1 — Viability Proof  ·  Status: ✅ Pass 1 complete (May 2026)

**Goal:** Demonstrate that nightly LoRA consolidation of agent memories
into model weights works on consumer hardware, with safety machinery.

**Pass 1 result (Llama 3.2 1B Instruct):**
- Personal recall: **46%** on held-out probes (vs 0% baseline)
- General capability: **90%** adapter vs 93% base (3.3pp regression)
- Gate decision: **PROMOTE** (clean)
- Reproducible via `dreamagent dream` with the locked recipe in [`docs/tuning/llama-3.2-1b-instruct-4bit.md`](docs/tuning/llama-3.2-1b-instruct-4bit.md)

### Pass 2 — Production tier  ·  Target: June 2026

**Goal:** Re-prove the thesis on Qwen 3 4B with stricter eval gates.

**Targets:**
- Personal recall ≥ 70% on held-out probes (production threshold)
- General capability regression ≤ 3pp (production threshold)
- Run nightly for 7 consecutive nights without drift > 5pp cumulative
- Run weekly mergekit consolidation; merged adapter passes both eval gates
- Publish LoCoMo numbers under both protocols (per-conversation FT + oracle)

**Tracking:** `docs/tuning/qwen3-4b.md` (template ready)

### Pass 3 — Long-horizon viability  ·  Target: August 2026

**Goal:** 30 consecutive nightly runs without identity drift.

**Targets:**
- `benchmarks/identity_drift.py` reports < 5pp persona regression
  across 30 runs
- Adapter merge cadence is locked (daily nightly + weekly merge)
- Rollback drill executed and documented

---

## V2 — Memory Specialist  ·  Status: 🔬 in design

**Goal:** Ship the dreamed model as the **memory backend** any agent
(Claude, GPT, Llama 70B, Letta, mem0-using agents) can query via MCP,
HTTP, or in-process. The dreamed model is "the guy who knows."

**Why V2 is the actual product:**
- Users keep using their preferred frontier agent.
- The dreamed model is queryable; it's not a chat surface itself.
- Privacy: memories never leave the box.
- Architectural compatibility: V2 doesn't compete with mem0/Letta — it
  *augments* them.

### V2.0 — MCP server stub  ·  Target: July 2026

- `dreamagent serve` exposes the live adapter as an MCP server
- Single tool: `query_memory(question: str) -> { answer, confidence, sources }`
- Connection-tested against Claude Code, Cursor, Hermes
- `examples/03-mcp-integration/` cookbook entry

### V2.1 — HTTP API + benchmark wins  ·  Target: September 2026

- `dreamagent serve --http` exposes the same surface over HTTP
- Cross-memory-reasoning benchmark shows ≥ 10pp advantage over a
  vector-retrieval baseline running on the same memories
- Query latency benchmark shows p95 < 500ms on Mac Mini M4

### V2.2 — Production hardening  ·  Target: November 2026

- Adapter signing for trust
- Multi-user namespacing
- mem0 + DreamAgent composition cookbook (run both, reconcile at agent)

---

## V3 — Frontier Direct  ·  Status: ⏸ blocked on V2 evidence

**Goal:** Apply the methodology to frontier-scale models (Qwen 3 70B,
Llama 4 Maverick, DeepSeek) on cloud GPUs.

**V3 only ships if V2 evidence justifies the compute.** If V2's small
dreamed-model architecture satisfies the use case, V3 may never be needed.

**Targets (conditional):**
- Per-night cloud GPU cost ≤ $20 for a 70B-class LoRA fine-tune
- Personal recall ≥ 85% (we expect higher with bigger base)
- General capability regression ≤ 2pp
- Verification protocol passes on the production tier

---

## Continuous Workstreams (not version-gated)

These advance every release:

- **Tuning recipes for new base models.** PRs welcome via the
  [tuning recipe issue template](.github/ISSUE_TEMPLATE/tuning_recipe.md).
  Current targets: Qwen 3 4B, Phi-4 14B, Llama 3.1 8B, Mistral Small.
- **New ingest connectors.** Highest-priority: Claude memory dirs,
  OpenClaw `MEMORY.md`, Hermes, Letta export, Supermemory export.
- **Benchmark coverage.** Especially head-to-head with mem0 and Letta on
  comparable axes.
- **Documentation.** Especially the comparison docs as the landscape moves.

---

## What We Are NOT Building

Negative roadmap items, so contributors don't propose them:

- **A hosted SaaS.** DreamAgent's privacy proposition is "weights never
  leave the box." A cloud version would dilute that. Self-host or don't host.
- **An agent runtime.** Letta does this well. DreamAgent is a backend.
- **A new vector index.** mem0/Zep/Supermemory do this well. We don't need
  another.
- **A general-purpose fine-tuning library.** Unsloth, MLX-Tune, axolotl
  cover this. DreamAgent uses MLX-LM via subprocess and gets out of the way.
- **A web UI.** CLI-first. Possibly a TUI later. Not a web app.

---

## How to influence the roadmap

- **Open an issue** with the tuning recipe template if you've run
  DreamAgent on a new model.
- **Open an issue** with the feature request template if you have a use
  case the roadmap doesn't address.
- **Open an RFC issue** (label `rfc`) if you have a methodology extension —
  see [CONTRIBUTING.md](CONTRIBUTING.md).

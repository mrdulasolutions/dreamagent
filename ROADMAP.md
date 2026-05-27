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

### Pass 2 — Production tier  ·  Status: ✅ complete (May 2026)

**Result:** Pivoted from Qwen 3 4B to Llama 3.1 8B Instruct after the
Qwen reasoning-tag conflict became blocking at any size. Llama 3.1 8B
delivered clean PROMOTE on first calibration.

Recorded in [`docs/tuning/llama-3.1-8b-instruct-4bit.md`](docs/tuning/llama-3.1-8b-instruct-4bit.md):
- Personal recall: 75% on held-out probes (above the 70% production threshold)
- General regression: 6.7pp (within warn band, well under the 15pp reject limit)
- Cross-memory reasoning: **+60pp adapter advantage over base**
- Query latency p50: 1.08s on Apple Silicon

### Pass 3 — Long-horizon viability  ·  Status: ✅ compressed 7-night drill complete (May 2026)

**Result:** 7-night chained-training drill on Llama 3.1 8B — all 7
nights PROMOTED. Personal recall climbed monotonically 44% → 81% before
plateauing. Identity drift went *negative* (-12.5pp = adapter is BETTER
on persona probes than base).

The full 30-night drill is deferred to V2.2 once the MCP server is in
production. Compressed 7-night data is sufficient to unblock V2.0.

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

### V2.0 — MCP server stub  ·  Status: ✅ alpha (May 2026)

- ✅ `dreamagent serve` exposes the live adapter as an MCP server
- ✅ Tools: `query_memory(question)` and `query_memory_with_lineage(question)`
- ✅ FastMCP over stdio, transport-agnostic per MCP spec
- ✅ Config via env vars (`DREAMAGENT_BASE_MODEL`, `DREAMAGENT_SNAPSHOTS_DIR`,
  `DREAMAGENT_MAX_TOKENS`) — the right pattern for MCP-host-launched servers
- ✅ Updated [`examples/07-mcp-memory-backend/`](examples/07-mcp-memory-backend/)
  with real Claude Code / Cursor configs
- ⏳ Live connection test against a real Claude Code install (user-side)

### V2.1 — HTTP API + head-to-head baselines  ·  Status: ✅ partial (May 2026)

- ✅ `dreamagent serve --transport http` exposes the same surface over HTTP
- ✅ Vector-retrieval baseline built and benchmarked
- ❌ **Cross-memory ≥10pp advantage over retrieval: not achieved.** Result
  was parity (0.0pp). See [`docs/tuning/v2.1-vs-baselines.md`](docs/tuning/v2.1-vs-baselines.md).
  The +6.2pp personal-recall advantage holds.
- ✅ `query_memory(question, concise=True)` flag for lower-latency responses
- ⏳ Query latency p95 < 500ms — still 1-2s; concise mode improves but
  doesn't hit target. Likely needs smaller production tier model.

The unmet target on the cross-memory advantage drives the V2.2 reframe.

### V2.2 — Composition + production hardening  ·  Status: critical (May 2026 update)

After the V2.1 head-to-head showed retrieval matching DreamAgent on cross-memory reasoning, **the composition story is no longer optional**:

- **mem0 + DreamAgent composition cookbook** (run both, reconcile at agent) — promoted from polish to V2.2's primary deliverable
- Adversarial probe set where retrieval is forced to fail (semantic-distance attacks)
- Adapter signing for trust
- Multi-user namespacing

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

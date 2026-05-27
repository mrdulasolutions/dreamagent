# Changelog

All notable changes to DreamAgent are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **V2.1 — vector-retrieval baseline + head-to-head benchmark.** New
  `benchmarks/vs_baselines.py` runs the same memories + probes through
  DreamAgent and a sentence-transformers + top-5 retrieval baseline using
  the same base model. Result: DreamAgent retains a modest +6.2pp
  personal-recall advantage but achieves **parity** (0.0pp) on cross-memory
  reasoning. See "Changed" below for the resulting positioning retraction.
- **V2.1 — HTTP transport for MCP.** `dreamagent serve --transport http`
  starts a streamable-http MCP server; useful for shared/multi-client
  deployments.
- **V2.1 — `concise` flag on `query_memory`** for ~3x lower latency
  responses (24-token cap, one-sentence-or-less system-prompt nudge).
- **V2.0 alpha — MCP server.** `dreamagent serve` exposes the live dreamed
  adapter as a Model Context Protocol server over stdio. Any MCP-capable
  client (Claude Code, Cursor, Hermes, OpenClaw) can install DreamAgent and
  gain `query_memory` + `query_memory_with_lineage` tools. The dreamed model
  becomes "the guy who knows" — a knowledge oracle the host agent calls when
  it needs to know what the user has told it. Configured via env vars
  (`DREAMAGENT_BASE_MODEL`, `DREAMAGENT_SNAPSHOTS_DIR`, `DREAMAGENT_MAX_TOKENS`).
  Optional dependency `dreamagent[mcp]`.

### Changed

- **Cross-memory reasoning claim retracted and revised.** The V1 benchmark
  suite reported +60pp cross-memory reasoning improvement (base 30% →
  adapter 90%). That comparison was vs the base model with no memory
  access. The V2.1 vs-baselines experiment showed a retrieval-augmented
  baseline matches DreamAgent at 90% on the same probes. The README,
  PAPER §1/§6.4/§7.1/§8.5/§10.5, and ROADMAP V2.1/V2.2 are all revised
  to reflect this. The +6.2pp personal-recall advantage is real and
  remains. Detail:
  [`docs/tuning/v2.1-vs-baselines.md`](docs/tuning/v2.1-vs-baselines.md).
- **ROADMAP V2.2:** mem0 + DreamAgent composition cookbook promoted from
  polish to the primary V2.2 deliverable.
- **V1 Pass 2/3 complete on Llama 3.1 8B Instruct.** 7-night chained-training
  drill: all 7 nights PROMOTED, zero rejects, personal recall climbed 44% → 81%
  with regression bounded at 0–13.3pp. Full trajectory + benchmark suite
  results in `docs/tuning/llama-3.1-8b-instruct-4bit.md`.
- **Cross-memory-reasoning benchmark: +60pp adapter over base** (30% → 90%).
  The parametric advantage proven empirically. Clears the V2.1 success
  criterion (≥10pp) by 50 points.
- **Query latency on Mac:** p50=1.08s, p95=2.25s, p99=2.27s for 48-token
  responses with Llama 3.1 8B 4-bit + LoRA adapter.
- **`dreamagent drill` CLI** for N consecutive chained nights. Replaces the
  manual loop. Records full trajectory + decisions in `runs/drills/<ts>/`.
- **`--resume-from-snapshot` flag on `dreamagent dream`.** Closes the V1
  TODO. The training stage now passes `--resume-adapter-file` to MLX-LM
  so a night starts from the prior night's adapter rather than the base.
- **Methodology named: MORPHEUS** — Memory Overnight Re-parameterization,
  Promotion via Held-out Eval, Update Snapshots. The technique formerly
  known as "the DreamAgent methodology" now has a canonical acronym for
  academic and architectural reference. DreamAgent remains the reference
  implementation. See [ADR-008](docs/adr/008-morpheus-methodology-name.md).
- **V1 Phase 1 viability proof.** Clean PROMOTE achieved on Llama 3.2 1B
  Instruct after 16 tuning runs. Locked recipe in
  `docs/tuning/llama-3.2-1b-instruct-4bit.md`.
- **Frontier-model extraction module** (`dreamagent extract`) with backends
  for Anthropic, OpenAI, and Ollama. Precision-engineered prompt forbids
  fabrication and skips ephemeral content.
- **Cron installer** (`dreamagent install-cron`) writing a launchd plist on
  macOS or printing a crontab line on Linux.
- **Tuning playbook** in `docs/tuning/README.md` with 6 transferable lessons
  and per-model logs.
- **METHODOLOGY.md** — canonical description of the technique with prior-art
  positioning.
- **ARCHITECTURE.md** — module map, data flow diagrams, design rationale.
- **Apache 2.0 licensing** with `NOTICE` file enforcing attribution to
  Mr Dula Solutions in derivative works.
- **CITATION.cff** for academic citation.

### Changed

- License changed from MIT to Apache 2.0 (pre-public-release).
- `mlx-community/Llama-3.2-1B-Instruct-4bit` is now the canonical
  validation-tier model (Qwen 3 0.6B abandoned due to chain-of-thought
  conflict with anchor format; see `docs/tuning/qwen3-0.6b-4bit.md`).

### Fixed

- Anchor selection is now deterministic by `source_memory_id` order
  (`_sample_stable`) — eliminated a major variance source masking
  hyperparameter effects in tuning.
- Eval runner accepts both adapter directory and `.safetensors` file paths;
  internally resolves to the directory mlx-lm expects.

## [0.0.1] — 2026-05-26

### Added

- Initial public commit.
- `MemoryItem` pydantic v2 schema with 5-kind taxonomy
  (fact / preference / procedure / event / correction).
- Three ingest connectors: `JSONLConnector`, `FixtureConnector`, and
  `MemoryConnector` protocol.
- 50 fixture memories covering all 5 kinds.
- Compose stage: per-kind templates, paraphrastic example generation,
  held-out eval probes, rehearsal mix composer.
- Train stage: MLX-LM LoRA subprocess wrapper with full lineage metadata.
- Eval stage: substring-match scoring with personal + general probe types.
- Promote stage: 4-decision gate (PROMOTE / WARN / REJECT) + adapter
  snapshots + one-command rollback.
- 40 general-capability anchor fixtures + 15 eval probes.
- CLI: `dream`, `ingest`, `load-model`, `schema-info`, `snapshots`,
  `rollback`, `version`, `fixtures-path`.
- 114 tests passing; lint clean.

# Changelog

All notable changes to DreamAgent are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

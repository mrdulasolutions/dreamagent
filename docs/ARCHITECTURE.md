# DreamAgent Architecture

This document describes the internal structure of the codebase: modules,
their responsibilities, data shapes, and the interfaces between them.

## Module Map

```
src/dreamagent/
├── schema.py          # MemoryItem, MemoryBatch — the public data contract
├── ingest/            # MemoryConnector implementations
│   ├── base.py        #   Protocol all connectors implement
│   ├── jsonl.py       #   JSONLConnector
│   └── fixture.py     #   FixtureConnector (shipped fixtures/)
├── extract/           # Frontier-LLM memory extraction (text → MemoryItem)
│   ├── base.py        #   ExtractionBackend Protocol
│   ├── prompt.py      #   The extraction system prompt
│   ├── pipeline.py    #   Validate, augment, write
│   └── backends/      #   anthropic.py / openai.py / ollama.py
├── compose/           # MemoryItem → training examples + eval probes + mix
│   ├── templates.py   #   Per-kind question templates
│   ├── examples.py    #   memory_to_examples, memories_to_dataset
│   ├── anchors.py     #   Loaders for fixtures/anchors/
│   └── mix.py         #   compose_rehearsal_mix
├── train/             # LoRA training subprocess wrapper
│   ├── config.py      #   TrainConfig (hyperparameters)
│   └── runner.py      #   train_adapter, metadata writing
├── eval/              # Probe runner + scoring
│   └── runner.py      #   run_eval, score (substring match)
├── promote/           # Eval gate + snapshot machinery
│   ├── gate.py        #   EvalGateConfig + decide() (4-decision matrix)
│   └── snapshot.py    #   snapshot_run, current_live, rollback_to
├── merge/             # Weekly mergekit consolidation (Phase 3, scaffold)
├── serve/             # MLX-LM / Ollama hot-swap (Phase 3, scaffold)
└── cli.py             # typer entry point — all commands
```

## Data Flow

```
external memory source
       │
       ▼
  iter_memories() ────────► [MemoryItem, ...]
   (ingest.*)
       │
       ▼
 memories_to_dataset() ────► [TrainingExample, ...], [EvalProbe, ...]
   (compose.examples)
       │
       ▼
 compose_rehearsal_mix() ──► RehearsalMix(examples=[...], composition={today,replay,anchor})
   (compose.mix)
       │
       ▼
   train_adapter() ────────► TrainResult(adapter_path, metadata)
   (train.runner)
       │
       ├─────────────────────┐
       ▼                     ▼
   run_eval(personal)    run_eval(general, base|adapter)
   (eval.runner)         (eval.runner)
       │                     │
       └─────────┬───────────┘
                 ▼
            decide() ───────► GateResult(decision, reasons, scores)
            (promote.gate)
                 │
                 ▼
         snapshot_run() ────► AdapterSnapshot(name, dir, decision)
         (promote.snapshot)
```

## The MemoryItem Contract

`MemoryItem` is a pydantic v2 model with `extra="forbid"` — unknown fields
fail validation. This is intentional: the schema is a stable public contract.
See [`src/dreamagent/schema.py`](../src/dreamagent/schema.py) for the
complete spec.

Validation rules enforced at the model level:
- `kind == correction` must have non-empty `supersedes`
- `preference_signal` may only be set when `kind == preference`
- `sensitivity == redact` excludes the memory from training (still logged)
- `confidence` and `importance` are 0..1 floats
- `content` is 1..2000 chars
- Unknown fields are rejected

## The Ingest Layer

```python
class MemoryConnector(Protocol):
    def name(self) -> str: ...
    def iter_memories(self, since: datetime | None = None) -> Iterable[MemoryItem]: ...
```

Three shipped connectors:

- `JSONLConnector(path)` — newline-delimited JSON file. Each line is either a
  single MemoryItem or a `{"items": [...], "schema_version": "1.0"}` envelope.
- `FixtureConnector(name=None)` — reads from `fixtures/*.jsonl`. Used by V1
  validation and CI.
- `Mem0Connector` — placeholder; mem0 SDK integration is straightforward
  (~50 LoC) when needed.

Adding a connector for Claude memory dirs, supermemory, OpenClaw `MEMORY.md`,
or Hermes is the same shape — implement the Protocol, plug it in.

## The Extract Layer

The frontier extraction module converts arbitrary text (chat transcripts,
journal entries, exported memory dumps) into validated `MemoryItem` records.

Backends are pluggable via the `ExtractionBackend` Protocol. Each backend is
in `src/dreamagent/extract/backends/<name>.py` and is conditionally importable
(optional dependency via `dreamagent[<name>]`).

The pipeline (`pipeline.py`) is responsible for:
- Reading the input file (with chat-JSONL flattening if applicable)
- Calling the backend with `prompt.SYSTEM` + the constructed user prompt
- Parsing the response as a JSON array (with markdown-fence tolerance)
- Synthesizing auto-fields (`id`, `schema_version`, `source.system`,
  `source.captured_at`) — the LLM is forbidden from inventing these
- Validating each record against `MemoryItem`
- Reporting validation rejections with structured reasons

The LLM only fills in the seven content/classification fields. Everything
else is pipeline-generated.

## The Compose Layer

`memory_to_examples(item)` dispatches by `kind` to template lists in
`templates.py`. The LAST template in each list is reserved as the eval probe
and never appears in training. This split is the source of truth for the
"personal recall measures generalization, not memorization" property.

`memories_to_dataset(items)` applies `memory_to_examples` across a list,
respecting `supersedes`: any memory ID appearing in another memory's
`supersedes` list is excluded from both training and eval.

`compose_rehearsal_mix(today, prior, anchors, config, max_anchors)` builds
the nightly training set. It calls:
- `_sample_random(prior, target_replay, rng)` for replay (variable each night)
- `_sample_stable(anchors, target_anchor)` for anchors (deterministic by ID)

The stable anchor selection is critical to reproducibility — growing the
anchor pool doesn't change which anchors enter the mix unless the request
exceeds the prior cap.

## The Train Layer

`train_adapter(mix, config, run_dir)` writes `data/train.jsonl` and
`data/valid.jsonl` in MLX-LM's expected chat format, builds the subprocess
command, and invokes `python -m mlx_lm lora --train`.

Why subprocess instead of `from mlx_lm.lora import train`:
- The CLI surface is the most stable contract across MLX-LM versions
- Errors surface as exit codes rather than partial Python state
- The same script targets Unsloth on cloud GPU with one config switch (V3)

`metadata.json` records:
- Tag, notes, full CLI invocation (set by the dream command)
- Hyperparameters (the full TrainConfig)
- Mix composition (`{today, replay, anchor}` counts)
- Dataset sizes (train/valid split)
- Sorted list of source memory IDs trained on
- Library versions (mlx_lm, mlx, transformers, pydantic) + Python + platform
- Started/completed timestamps + duration

Every adapter is recoverable via its metadata.

## The Eval Layer

`run_eval(probes, model_repo, adapter_path, max_tokens)`:
- Loads the model + (optionally) the adapter directory via `mlx_lm.load`
- For each probe, builds a chat prompt via `tokenizer.apply_chat_template`
- Generates and scores via case-insensitive substring match
- Returns an `EvalReport` with per-probe results, pass rate, and metadata

Scoring is intentionally lenient. A 0.6B-class model produces noisy
verbose answers; exact-match would be uselessly strict. Substring match
gives signal without false negatives. See lessons learned in
[`docs/tuning/README.md`](tuning/README.md).

The same runner handles both personal-recall and general-capability evals —
they're both lists of `EvalProbe` and the runner doesn't care which.

## The Promote Layer

`decide(personal_eval, general_base_eval, general_adapter_eval, config)`
applies the 4-decision matrix:

| Personal recall | General regression | Decision |
|---|---|---|
| ≥ min | ≤ max | PROMOTE |
| ≥ min | max..warn | PROMOTE_WITH_WARNING |
| ≥ min | > warn | REJECT |
| < min | anything | REJECT |

`EvalGateConfig` has `.validation_defaults()` (looser thresholds for 0.6B-1B
class) and the default production thresholds (5pp / 15pp / 70%).

`snapshot_run(...)` is the atomic promotion operation. It:
- Builds `snapshots/<timestamp>/` (or `snapshots/rejected/<timestamp>/`)
- Copies the adapter dir, metadata, and eval JSONs in
- Writes `gate.json` with the full decision rationale
- On promote: atomically replaces the `live` symlink to point at the new dir
- On reject: leaves `live` untouched

Rollback is `rollback_to(snapshots_dir, name)` — a single symlink swap.

## The CLI

Built on `typer`. Every command is a function. Commands are organized so the
core verb is the function name: `dream`, `extract`, `ingest`, `snapshots`,
`rollback`, `install-cron`, etc.

The dream command is the orchestration spine — it sequences ingest → compose
→ train → eval → promote and prints rich-formatted progress at each stage.

## Configuration

DreamAgent has no central config file — every command takes explicit flags
with sensible defaults. This is deliberate: the tuning playbook depends on
being able to reproduce exact invocations from an `invocation` field in
`metadata.json`. A config file would obscure that.

If you want to reuse a particular recipe, build a shell alias or a
`Makefile` target around the dream invocation. See the locked V1 recipe in
[`docs/tuning/llama-3.2-1b-instruct-4bit.md`](tuning/llama-3.2-1b-instruct-4bit.md).

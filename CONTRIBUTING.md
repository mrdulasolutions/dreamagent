# Contributing to DreamAgent

Thanks for your interest. DreamAgent is in active development; we welcome
contributions across the stack — connectors, extraction backends, tuning
recipes for new models, documentation, and bug fixes.

## Before You Open a PR

1. Open an issue first if your change is anything more than a typo or
   docs polish. We want to make sure your work lands.
2. Read [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) and
   [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). Familiarity with the
   contracts saves review time.
3. Read [the tuning playbook](docs/tuning/README.md) if your change
   touches the training, eval, or promote layers.

## Development Setup

```bash
git clone https://github.com/mrdulasolutions/dreamagent.git
cd dreamagent
uv sync --extra dev
uv run pytest -q
uv run ruff check src tests
```

If you're on an external drive on macOS and hit AppleDouble-file errors
during install, set `UV_PROJECT_ENVIRONMENT` to a path on internal disk:

```bash
export UV_PROJECT_ENVIRONMENT="$HOME/.cache/dreamagent-venv"
uv sync --extra dev
```

## What We Want PRs For

### High value

- **New ingest connectors.** Especially Claude memory dirs, OpenClaw `MEMORY.md`,
  Hermes, supermemory, Letta. Each is ~50 LoC implementing the `MemoryConnector`
  protocol.
- **New extraction backends.** Want a local llama.cpp backend? A Gemini
  backend? Implement `ExtractionBackend` and submit it.
- **Tuning recipes for new base models.** Run the tuning loop on a model
  we don't yet have a recipe for and document the result in
  `docs/tuning/<model-id>.md`. Follow the existing template.
- **Better personal-eval scoring.** The current substring match is fast and
  lenient but misses semantic equivalence. A more robust scorer (semantic
  match against a held-out reference) is welcome.

### Medium value

- More general-eval probes (especially areas with multilingual or technical
  coverage).
- More anchor fixtures targeting commonly-forgotten facts.
- Performance improvements to the MLX-LM subprocess wrapper.

### Lower value (please ask first)

- Refactors of the core pipeline. The architecture is intentional; let's
  discuss before you rewrite it.
- Adding new top-level CLI commands. We're keeping the surface small.

## Code Style

- **Python 3.12+** with full type hints. Run `ruff` and `mypy` before
  pushing.
- **Pydantic v2** for any new data shapes. `extra="forbid"` on public
  contracts.
- **Docstrings** on every module and public function. Explain *why* the
  code exists, not just *what* it does — the *what* is in the code.
- **No emojis** in code or commit messages unless the user has explicitly
  requested them.
- **Tests with the change.** PRs without tests typically wait longer.

## Tests

```bash
uv run pytest -q                  # full suite, <5s
uv run pytest tests/test_<area>   # focused
```

We test:
- Schema validation (all rules)
- All shipped connectors (with tempfiles)
- The compose stage (templates, mix composition, anchor loading)
- Train metadata writing + command building (NOT the actual training)
- Eval scoring + report shape (NOT actual model loading)
- Promote gate decision matrix + snapshot directory structure
- Extract pipeline (with a `_FakeBackend`)

We do NOT have unit tests for live model inference or live LoRA training;
those are exercised by `dreamagent dream` end-to-end against fixtures.

## Commit Messages

- Subject line ≤ 72 chars
- Imperative mood ("Add JSONL connector", not "Adding")
- Body explains the *why* — the *what* is in the diff
- One change per commit when possible

Co-authored commits (e.g., AI-assisted) should include:

```
Co-Authored-By: <Tool / Person Name> <noreply@example.com>
```

## PRs

- Reference the issue in the PR description
- Include a brief test plan
- Run `ruff check` and `pytest` locally before pushing
- CI must be green before review

## Methodology Contributions

If you're contributing a substantive extension to the DreamAgent
methodology itself (e.g., a new stage, a different gate logic, a novel
catastrophic-forgetting mitigation), please:

1. Open an RFC issue first with the technical proposal
2. Include empirical evidence (a tuning log showing the proposed change
   improving over the baseline)
3. Update [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) with the addition
4. Add yourself to [`CITATION.cff`](CITATION.cff) under `authors` if
   accepted

This isn't gatekeeping — it's making sure your work gets the credit it
deserves when others cite the project.

## Reporting Security Issues

Please read [`SECURITY.md`](SECURITY.md). Do not file public issues for
security vulnerabilities.

## License

By contributing, you agree that your contributions will be licensed under
the Apache License 2.0, the same license as this project. See
[`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

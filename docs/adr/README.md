# Architecture Decision Records

Following the [Michael Nygard ADR convention](https://adr.github.io/madr/).
One file per significant architectural decision. Decisions are immutable —
when superseded, write a new ADR that references the old one rather than
editing it.

## Index

| # | Title | Status |
|---|---|---|
| [001](./001-apache-2.0-with-notice-attribution.md) | Apache 2.0 with NOTICE-enforced attribution | Accepted |
| [002](./002-pydantic-v2-strict-schema.md) | Pydantic v2 with `extra="forbid"` for public contracts | Accepted |
| [003](./003-mlx-lm-via-subprocess.md) | Invoke MLX-LM via subprocess, not Python module API | Accepted |
| [004](./004-eval-gated-promotion.md) | Four-decision eval gate (PROMOTE / WARN / REJECT) | Accepted |
| [005](./005-stable-anchor-selection.md) | Stable (by-ID) anchor selection vs random sampling | Accepted |
| [006](./006-llama-3.2-1b-validation-tier.md) | Llama 3.2 1B Instruct as canonical validation-tier model | Accepted |
| [007](./007-parametric-vs-retrieval-positioning.md) | Position DreamAgent as parametric memory, not RAG competitor | Accepted |
| [008](./008-morpheus-methodology-name.md) | Name the methodology MORPHEUS (decouple from project name) | Accepted |

## Template

When adding a new ADR, copy [`_template.md`](./_template.md) and fill it in.

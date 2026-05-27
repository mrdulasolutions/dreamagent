# 002. Pydantic v2 with `extra="forbid"` for public contracts

- **Status:** Accepted
- **Date:** 2026-05-26

## Context

The `MemoryItem` schema is the public contract between DreamAgent and
any upstream memory store. Any drift in the schema (silently accepting
extra fields, missing required fields, type coercion that hides bugs)
propagates into the training pipeline and ultimately into the model's
weights. Validation has to be loud, not lenient.

## Decision

All public-contract models (`MemoryItem`, `MemoryBatch`, `Source`,
`QAPair`, `PreferenceSignal`) use Pydantic v2 with `extra="forbid"`.
Unknown fields raise `ValidationError`. Range/length constraints are
enforced. Cross-field rules (e.g., correction requires supersedes) are
enforced via `model_validator(mode="after")`.

## Consequences

- **Easier:** Bugs in upstream connectors surface immediately at ingest
  with a structured error message instead of silently propagating.
- **Easier:** The schema doubles as machine-readable documentation via
  `MemoryItem.model_json_schema()` (used by `dreamagent schema-info`).
- **Harder:** Adding a new field requires a schema migration step
  (bump `schema_version`, update all connectors). This is the right
  amount of friction for a public contract.

## Alternatives Considered

1. **Dataclasses + manual validation** — More code, less robust.
2. **Pydantic v1** — Slower, less ergonomic, but already in some MLX
   tooling. v2 is faster and the migration cost is one-time.
3. **TypedDict + jsonschema** — Fine for shapes but no runtime validation
   in Python.
4. **`extra="ignore"`** — Lenient but hides upstream bugs.

## Related

- [`src/dreamagent/schema.py`](../../src/dreamagent/schema.py)
- [`tests/test_schema.py`](../../tests/test_schema.py)

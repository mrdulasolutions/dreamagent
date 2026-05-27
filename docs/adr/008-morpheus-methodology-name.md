# 008. Name the methodology MORPHEUS

- **Status:** Accepted
- **Date:** 2026-05-27

## Context

Through V1 we referred to the technique inconsistently as "the DreamAgent
methodology," "nightly LoRA consolidation," "memory-into-weights," etc.
None of these are referenceable as a single technical term in a paper or a
comparison table. The field has a strong precedent for short, evocative
acronyms (LoRA, QLoRA, DoRA, RLHF, DPO, ROME, MEMIT, GRACE, CL-LoRA, SuRe,
FOREVER) — usually 3–8 characters, pronounceable, with each letter mapping
to a real concept.

We also have a naming collision risk: "DreamAgent" is the project, but the
underlying technique should be portable to other implementations
(JavaScript port, Rust port, future research forks). Conflating project
name with methodology name muddies attribution and citation.

## Decision

Adopt **MORPHEUS** as the canonical methodology name.

**MORPHEUS** = **M**emory **O**vernight **R**e-parameterization,
**P**romotion via **H**eld-out **E**val, **U**pdate **S**napshots.

Each letter maps to a stage or invariant of the technique:

| Letter | Concept | Where |
|---|---|---|
| M | Memory | structured `MemoryItem` schema |
| O | Overnight | nightly cron cadence |
| R | Re-parameterization | LoRA into model weights (parametric vs retrieval) |
| P | Promotion | the 4-decision eval gate |
| H | Held-out | training and eval probes are disjoint |
| E | Eval | dual probe sets (personal + general) |
| U | Update | adapter is the atomic update unit |
| S | Snapshots | every promoted update is versioned + rollbackable |

**The project remains "DreamAgent"** — the repository, CLI, and reference
implementation. "DreamAgent implements MORPHEUS" is the canonical phrasing.
A future Rust port could be called "MorpheusRS" and would still credit the
same methodology.

## Consequences

- **Easier:** Academic citation has a clean acronym. Comparison tables and
  paper titles can use "MORPHEUS" as a noun.
- **Easier:** Decoupling project name (DreamAgent) from methodology name
  (MORPHEUS) lets other implementations grow without renaming.
- **Easier:** Attribution per the NOTICE file becomes unambiguous.
- **Harder:** One-time documentation update across README, NOTICE,
  CITATION, METHODOLOGY, FAQ, and CHANGELOG.
- **Accepted tradeoff:** Greek-mythology acronyms can feel theatrical.
  The thematic tie to DreamAgent (Morpheus is the Greek god of dreams)
  mitigates this — the project and methodology names rhyme symbolically.

## Alternatives Considered

1. **HYPNOS** (god of sleep) — cleaner but the letter-to-concept mapping is
   less mnemonic (only 6 letters, fewer methodology stages covered).
2. **REM** (rapid eye movement) — punchiest but overloaded in CS contexts
   and may not survive the "what does this acronym mean?" test in a paper.
3. **DREAM-LoRA** — reinforces project brand but hyphenated; less elegant
   as a pure acronym, and bakes "LoRA" into the name (locking us out of
   future non-LoRA implementations).
4. **No acronym; just "the DreamAgent methodology"** — what we had.
   Conflates project with methodology; harder to cite.

## Related

- [docs/METHODOLOGY.md](../METHODOLOGY.md) — now headed "MORPHEUS"
- [NOTICE](../../NOTICE) — attribution text updated to reference MORPHEUS
- [CITATION.cff](../../CITATION.cff) — title updated
- ADR-001 — original Apache+NOTICE decision; this ADR extends the
  attribution clause with the methodology name

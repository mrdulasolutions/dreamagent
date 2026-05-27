# 001. Apache 2.0 with NOTICE-enforced attribution

- **Status:** Accepted
- **Date:** 2026-05-26
- **Deciders:** Mr Dula Solutions

## Context

The DreamAgent methodology is novel as of its publication date. We want
to:

1. Allow free use, modification, and commercial redistribution (the
   project is more valuable widely adopted than narrowly held).
2. Require that derivative works credit Mr Dula Solutions as the
   originator of the methodology.
3. Establish indisputable provenance for the technique.

Pure MIT lets anyone strip attribution. Pure GPL is incompatible with
many commercial uses. A custom license would be unfamiliar and create
adoption friction.

## Decision

Use Apache License 2.0 with a populated `NOTICE` file. Apache §4(d)
legally requires that derivative works carry forward the NOTICE file's
attribution text in their own documentation.

The NOTICE explicitly names "the DreamAgent methodology" and requires
that derivative works display attribution in user-facing documentation,
About screens, and academic publications.

A `CITATION.cff` file provides a standard format for academic citation.

## Consequences

- **Easier:** Adoption (Apache 2.0 is universally accepted). Academic
  citation. Legal clarity on what attribution looks like.
- **Harder:** Cannot stop bad actors who ignore the NOTICE, but the
  combination of LICENSE + NOTICE + CITATION + dated commit history
  makes the provenance argument strong if ever needed in court.
- **Accepted tradeoff:** We can't enforce attribution proactively. We
  rely on the legal mechanism + community norms.

## Alternatives Considered

1. **MIT** — Simpler but provides no attribution lever.
2. **GPL-3.0** — Strong copyleft but incompatible with closed-source
   derivative work, which would hurt adoption.
3. **AGPL-3.0** — Same issue as GPL, plus the network-use clause is
   irrelevant since DreamAgent is a local tool.
4. **Custom license** — Maximum control but creates friction; we'd be
   the only project with that license.
5. **Apache 2.0 + LICENSE-METHODOLOGY (CC BY)** — Dual-license the
   methodology docs separately. Considered overkill; the NOTICE already
   covers methodology attribution under Apache.

## Related

- [`LICENSE`](../../LICENSE)
- [`NOTICE`](../../NOTICE)
- [`CITATION.cff`](../../CITATION.cff)

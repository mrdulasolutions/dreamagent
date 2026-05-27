# 007. Position DreamAgent as parametric memory, not a RAG competitor

- **Status:** Accepted
- **Date:** 2026-05-27

## Context

mem0 publishes 92.5% on LoCoMo. Letta publishes 83.2%. Supermemory
~70%. The natural marketing impulse is to also publish a LoCoMo
number and claim wins or losses on that scoreboard.

But LoCoMo measures **retrieval over conversation history**. DreamAgent
doesn't retrieve — it embodies. Running LoCoMo against DreamAgent
requires either:

1. Re-training the model on each LoCoMo conversation before evaluation
   (expensive, not how a real user deploys it)
2. Using DreamAgent as a knowledge oracle for a separate agent
   (tests the combined system, not the parametric layer)

Neither is the same experiment as mem0 running LoCoMo. A side-by-side
LoCoMo bar chart would be apples-to-oranges and would undersell what
DreamAgent actually does differently.

Letta's own [benchmarking work](https://www.letta.com/blog/benchmarking-ai-agent-memory)
makes the methodological point sharper: a vanilla agent with
filesystem tools beat mem0 on LoCoMo. The benchmark is gameable;
retrieval is not the bottleneck.

## Decision

Position DreamAgent on a **different axis** than the retrieval
incumbents. The comparison docs ([`docs/comparison/`](../comparison/))
explicitly:

1. Acknowledge LoCoMo as the field standard
2. Explain why head-to-head LoCoMo numbers are not yet meaningful for
   DreamAgent
3. Compare on axes that matter to actual deployment: privacy,
   cross-memory reasoning, query latency, switching cost, failure
   mode, GDPR/deletion story
4. Frame mem0/Letta and DreamAgent as **complementary** in V2's
   architecture, not competitors

The benchmark suite (`benchmarks/`) measures DreamAgent on properties
that retrieval cannot replicate: cross-memory reasoning probes,
identity drift across nights, query latency on a fixed parametric
model.

We commit to publishing LoCoMo numbers under two transparent
protocols once Pass 2 completes, but won't anchor our positioning on
that number.

## Consequences

- **Easier:** Honest positioning that resists the "we beat mem0 by X%"
  marketing trap.
- **Easier:** Composition with mem0/Letta is the default story, not
  competition.
- **Easier:** Resistant to benchmark-gaming pressure.
- **Harder:** Less straightforward sales pitch — "we're different"
  takes more words than "we score higher."
- **Accepted tradeoff:** Some readers will skim and think we're hiding
  bad benchmark numbers. The comparison docs and the benchmark suite
  exist to defuse this.

## Alternatives Considered

1. **Cherry-pick LoCoMo Protocol B (DreamAgent as oracle for an agent)
   and publish only the higher number** — Tempting and dishonest.
2. **Run Protocol A in one specific config and publish that** — A
   number we don't yet have; would be ahead of evidence.
3. **Don't publish any benchmark numbers** — Looks like we have
   nothing. We do, just not LoCoMo.

## Related

- [`docs/comparison/README.md`](../comparison/README.md)
- [`benchmarks/README.md`](../../benchmarks/README.md)
- Letta blog: [Benchmarking AI Agent Memory](https://www.letta.com/blog/benchmarking-ai-agent-memory)

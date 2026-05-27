# DreamAgent vs. Zep

[Zep](https://www.getzep.com/) is a temporal knowledge graph memory layer — sophisticated structured retrieval combining vector similarity with explicit entity/relation modeling. Strong on long-horizon, multi-session conversation tracking.

## At a glance

| | Zep | DreamAgent |
|---|---|---|
| Storage paradigm | Temporal knowledge graph + vector | LoRA adapters |
| Strength | Structured fact extraction + temporal reasoning over graphs | Cross-memory reasoning without explicit graph structure |
| Hosting | Self-host or cloud | Self-host only |
| LoCoMo | ~85% | Not yet measured |
| Where the "reasoning" happens | At graph-query time, with explicit hops | Inside the dreamed model's forward pass |
| When the graph schema changes | Re-extract or re-link entities | Re-train (next night) |

## When Zep is the right answer

- Your domain has rich entity/relation structure that benefits from explicit graph modeling.
- You need temporal reasoning over discrete facts ("What was X's status on date Y, and how did it change?").
- You want fact-level provenance with named entities first-class.

## When DreamAgent is the right answer

- You don't want to maintain a graph schema for memories.
- You want reasoning that's not bounded by which entities the extractor surfaced.
- Privacy or latency demands a single-artifact deployment.

## Where they intersect

Zep's graph is a *very* good extraction target. The `MemoryItem.entities` field in DreamAgent's schema maps cleanly to Zep's entity nodes — a future `ZepConnector` would let users build a Zep graph during the day and consolidate it into DreamAgent's weights at night, getting both the explicit-fact-tracking benefits of Zep and the cross-memory reasoning of parametric memory.

# DreamAgent vs. "Just stuff it all in a 1M context window"

A reasonable question in May 2026: **why bother with any memory layer when Claude, Gemini, and Llama 4 all support 1M+ context windows?**

The honest answer is that giant context **does** absorb a lot of the use cases memory systems were built for. If your "memory" is "the last 200k tokens of conversation" and you don't mind paying for them every turn, context-stuffing is a perfectly good solution.

But there are three problems giant context doesn't solve:

## 1. Cost compounds per query

A 1M-token context is ~$3-5 per *query* at current frontier API pricing. If a user has a daily conversation that draws on their full memory store, they're paying that on every turn. Memory layers — both retrieval (mem0, Letta) and parametric (DreamAgent) — amortize the cost.

DreamAgent's amortization is uniquely cheap: **one ~$1-5 nightly fine-tune** covers all the next day's queries with no per-call cost beyond a forward pass on a 4B model.

## 2. Proactive interference

[Recent research](https://arxiv.org/abs/2603.14517) (and any production engineer who has tried it) confirms that long-context models suffer from proactive interference — older information in the window degrades retrieval of newer information. The brain solves this with consolidation; the model has to be helped.

Retrieval (mem0/Letta) selects relevant chunks. Parametric memory (DreamAgent) bypasses the problem entirely — the relevant knowledge is in the weights, not competing for attention with everything else.

## 3. Privacy and locality

A 1M-token context still has to go to the frontier API. For HIPAA, GDPR, EU AI Act, financial, and any "we can't ship this data off the box" use case, giant context is structurally non-viable.

DreamAgent runs entirely on the user's machine — Apple Silicon, all four stages local. The dreamed model can sit behind a frontier API agent (via MCP) and answer questions about the user without those questions or answers traversing a third-party network.

## When giant context is the right answer

- Short-lived sessions where the user provides all relevant context within a single conversation.
- Use cases where the user is fine paying per-query for full context.
- Workloads where the data is non-sensitive and a frontier API is acceptable.

## When DreamAgent is the right answer

- Multi-day, multi-session knowledge that compounds over time.
- Per-query latency or cost is a constraint.
- The data must not leave the user's machine.

## The composition

Giant context + DreamAgent + retrieval is the strongest architecture: use giant context for the *current session*, retrieval for the *medium-term hot working set*, and DreamAgent for the *consolidated long-term memory*. This is what V2's MCP backend enables — your frontier agent never has to choose.

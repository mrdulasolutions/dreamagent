# DreamAgent vs. Supermemory

[Supermemory](https://supermemory.ai/) is a cloud-hosted memory layer with a focus on universal memory APIs across applications. Closed-source for the hosted product; SDKs published.

## At a glance

| | Supermemory | DreamAgent |
|---|---|---|
| Hosting | Cloud-only | Self-host only (intentional) |
| Source availability | SDKs published; core closed | Apache 2.0, all source |
| Storage paradigm | Vector index | LoRA adapters |
| Privacy floor | Data persists in vendor cloud | Weights only, on user's machine |
| Best for | SaaS products needing "remember-across-sessions" memory | Personal AI assistants and privacy-strict use cases |
| Switching cost | High (proprietary API) | Low (Apache 2.0 + JSONL ingestion contract) |

## When Supermemory is the right answer

- You want managed infrastructure and you trust the vendor with your memories.
- Your use case is consumer-facing SaaS where users have moderate expectations of cross-session memory.
- You don't want to operate any memory infrastructure yourself.

## When DreamAgent is the right answer

- You can't ship user data to a third-party cloud.
- You want the memory layer to *understand* the memories, not just retrieve them.
- You want to keep ownership of the artifact (your memory specialist is a `.safetensors` file you can copy, version, or destroy).

## Migration

Supermemory exports to JSON. DreamAgent reads JSONL. A trivial converter (see [`examples/04-extract-from-other-systems/`](../../examples/)) bridges them.

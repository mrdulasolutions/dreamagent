# DreamAgent Cookbook

End-to-end recipes for common DreamAgent workflows. Each example is a
self-contained directory with a `README.md`, any input files, and the
exact commands to run.

| Recipe | What it shows |
|---|---|
| [01-quickstart](./01-quickstart/) | The minimum-viable loop on fixture data. 5 minutes start-to-finish. |
| [02-extract-from-chat](./02-extract-from-chat/) | Convert a raw chat transcript into MemoryItems using a frontier LLM. |
| [03-nightly-cron](./03-nightly-cron/) | Install a launchd schedule so your model dreams every night. |
| [04-bridge-from-mem0](./04-bridge-from-mem0/) | Export memories from mem0 and consolidate them into a dreamed model. |
| [05-rollback-drill](./05-rollback-drill/) | Intentionally trigger a REJECT and verify the rollback story. |
| [06-benchmark-suite](./06-benchmark-suite/) | Run the published benchmarks against your own snapshot. |
| [07-mcp-memory-backend](./07-mcp-memory-backend/) | (V2 preview) Expose the dreamed model as an MCP server. |

Run any example from its directory:

```bash
cd examples/01-quickstart
./run.sh    # or follow the README's commands
```

Examples are versioned with the project. If something breaks after a
version bump, that's a bug — file an issue with the example name.

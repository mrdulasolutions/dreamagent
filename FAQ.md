# Frequently Asked Questions

Answers organized from "I'm just looking" to "I want to extend the methodology."

---

## For people just trying to understand what this is

### What is DreamAgent in one sentence?

DreamAgent gives your favorite AI assistant (Claude, ChatGPT, etc.) a memory that learns from you overnight — and the memory lives entirely on your own computer.

### Wait, doesn't ChatGPT already have memory?

Sort of. ChatGPT (and mem0, Supermemory, etc.) save text snippets and search them when you ask a question. That's *retrieval* — like having a really fast notebook the AI flips through.

DreamAgent does something different. It **trains** a small AI on what you told it. The memories become part of the AI's actual brain, not a separate notebook. The practical difference: a notebook-based memory can quote back what you said. A trained AI can *reason across everything you've told it together*. We measured a 3× improvement on questions that require connecting multiple memories.

### How is this different from "ChatGPT remembers me"?

Three big differences:

1. **Privacy.** ChatGPT's memory lives on OpenAI's servers. DreamAgent's lives on your laptop. Nothing leaves your machine.
2. **Portability.** ChatGPT's memory only works with ChatGPT. DreamAgent works with Claude, Cursor, Hermes, OpenClaw, or anything that supports MCP. Switch your daily AI; keep your memory.
3. **Capability.** Cloud memory retrieves; DreamAgent learns. Different paradigm, different ceiling on what it can do.

### Do I have to be a programmer to use this?

Right now, you need to be comfortable opening a terminal and running commands. The 5-minute quickstart in the [README](README.md#show-me-it-working-5-minutes) is approachable, but it does assume you can copy-paste commands into a terminal and edit a config file.

We're working on a more polished onboarding path (V2.1+).

### What does it cost?

Free. It runs on your computer with open-source models you download once. No subscription, no per-query fees.

The optional bit — using Claude or GPT to *extract* memories from raw text — costs a few cents per batch if you use those APIs. You can use a fully-local Ollama model instead, with zero cost.

### What kind of computer do I need?

A Mac with Apple Silicon (M1/M2/M3/M4) works great. The "small AI" we ship with is about 4.5GB and uses ~6GB of memory when running.

A Linux machine with an NVIDIA GPU also works, with slightly different setup.

Intel Macs and pure-CPU Windows machines aren't supported today.

### Is this safe? What if it learns the wrong thing?

The system has a safety check after every nightly "dream." It runs the new model through a set of tests for "did this break it?" If the tests fail, the new model is rejected and yesterday's safe one stays in charge. Rollback to any prior night is a single command.

In 7 consecutive nights of stress-testing on our internal data, the safety check passed every time.

### Will the model leak my data?

It can in theory — any trained model can have its training data probed out. We mitigate this by:

- A `sensitivity: redact` flag on memories you don't want trained
- The extractor prompt refuses to extract passwords, SSNs, credentials
- Everything runs locally — there's no API call that ships your memories anywhere

But it's not airtight. Treat the trained adapter file like you'd treat the data inside it. See [`SECURITY.md`](SECURITY.md) for the threat model.

### What if I want to forget something?

1. Remove the memory from your source file.
2. Re-run the nightly cycle. The next adapter won't see it.
3. Roll back to the most recent adapter that didn't see it: `dreamagent rollback <name>`.

Not instant, but it's a clear protocol. GDPR-style deletion in three commands.

---

## For people deciding if they should try it

### How does DreamAgent compare to mem0?

You probably want both. mem0 is great at "remember this conversation so I can search it later" — quick lookups, low effort. DreamAgent is great at "make my AI actually understand who I am over time" — slower (overnight), but the memory becomes part of the model.

In our roadmap they're explicitly designed to *compose*: mem0 for hot retrieval, DreamAgent for long memory. Both running, the agent reconciles.

Full head-to-head: [`docs/comparison/vs-mem0.md`](docs/comparison/vs-mem0.md).

### Why not just use a 1M token context window?

You can, and for some use cases it's the right call. But context costs ~$3-5 per query at frontier prices, and you have to re-stuff the same memories every conversation. DreamAgent pays once (~$0 on Mac) for the nightly fine-tune, and then queries are free.

Plus context windows have proactive-interference problems that get worse as they grow. Parametric memory side-steps that.

[`docs/comparison/vs-giant-context.md`](docs/comparison/vs-giant-context.md) has the longer answer.

### What if I switch from Claude to GPT next year?

No problem. DreamAgent is exposed as an MCP server — anything that supports MCP can call it. Your AI changes; your memory specialist stays the same.

### Does it actually work?

Yes, we measured it. The headline result: on questions that require connecting 2-3 memories together, the base model scores 30% and the dreamed adapter scores 90%. Other measurements + 7-night stability test in [`docs/tuning/llama-3.1-8b-instruct-4bit.md`](docs/tuning/llama-3.1-8b-instruct-4bit.md).

You can reproduce every number with `python -m benchmarks.<name>`.

---

## For people who tried it and have questions

### The first run takes forever. Is that normal?

Yes. The first time you run `dreamagent dream`, it downloads the small AI (Llama 3.1 8B 4-bit, about 4.5GB) plus dependencies. Total ~15-20 minutes on a decent connection.

After that, each dream is about 5 minutes (training + evaluating).

### The first MCP query takes ~10 seconds. Is that normal?

Yes. The MCP server loads the model lazily — on the first query, not when it starts. After the first one, subsequent queries are sub-second.

### My nightly run got rejected. What happened?

The eval gate found that the new model lost too much general capability. This is a feature, not a bug — it prevented a bad model from going live.

Look at `runs/snapshots/rejected/<timestamp>/gate.json` to see why. If it's recoverable (e.g., one anchor was overweighted), tune the recipe down (fewer iters, lower LR). If it's structural (e.g., your memories are contradicting each other), clean up the source.

### Can I use it with a model that's not Llama 3.1 8B?

Yes. Pass `--base-model <hf-repo-or-mlx-id>` to `dreamagent dream`. Any MLX-LM-compatible model works.

Different models need different hyperparameters. Use the playbook in [`docs/tuning/README.md`](docs/tuning/README.md) and submit a recipe back via the [tuning recipe issue template](.github/ISSUE_TEMPLATE/tuning_recipe.md) if you find one that works.

### Why nightly? Why not just train continuously?

Continuous training has unbounded risk — one bad input could quietly corrupt the model over time. The nightly batch gives you:
- A clean rollback unit (one bad night doesn't poison everything)
- A measurable "this is the new model" event
- A natural cadence for offline compute

Continuous (test-time training) is on the V3 roadmap once V2 settles.

### How much does cloud cost if I use the cloud path?

A nightly fine-tune on a rented A100 is ~$1–5. Monthly cost if you run nightly: $30–$150. Most users don't need cloud — Mac is plenty for a personal-scale memory.

---

## For developers and contributors

### What's the difference between DreamAgent and MORPHEUS?

- **MORPHEUS** is the methodology — a set of techniques, contracts, and gates. Specifically: **M**emory **O**vernight **R**e-parameterization, **P**romotion via **H**eld-out **E**val, **U**pdate **S**napshots.
- **DreamAgent** is the project — the Python codebase, CLI, fixture data, and tuning recipes that implement MORPHEUS on Apple Silicon (with cloud-GPU portability).

A future port of the methodology in Rust or JavaScript would still implement MORPHEUS but would not be called DreamAgent. This decoupling is intentional ([ADR-008](docs/adr/008-morpheus-methodology-name.md)).

### How is "the model forgets what I told it" any different from catastrophic forgetting?

Conceptually the same — the model is being trained on new data and old learned things degrade. The defenses are concrete:

1. **Rehearsal mix.** Every nightly run includes 75% new memories + 15% replay + 10% anchor (general-knowledge examples).
2. **Eval gate.** Refuses to promote a regression > 15pp.
3. **Per-night snapshots.** Every adapter is versioned; rollback is one command.
4. **Conservative LoRA rank.** Low rank, low learning rate.

The 7-night drill measured a 0–13.3pp regression across 7 chained nights — bounded, never breaches the reject threshold.

### Why LoRA instead of full fine-tuning?

Speed, size, safety:
- 1–2 hours of training vs hours-to-days
- Adapter files are ~50–200 MB (per-night versioning is trivially cheap)
- Forgetting is more contained than full FT
- Same script targets MLX (Mac) and Unsloth (cloud)

### Why a 1B/8B model? That's tiny.

Validation tier is Llama 3.2 1B because it fine-tunes in ~25 seconds on Apple Silicon. The V1 production tier is Llama 3.1 8B Instruct — passes our gates, no chain-of-thought conflict, mature MLX support.

For an interactive memory backend, smaller is actually better — faster queries, less memory. V3 (frontier-scale) is conditional on V2 evidence.

### Is this novel research?

We claim two specific contributions:

1. **The methodology (MORPHEUS).** The combination of (a) personal-scale agent memory, (b) nightly LoRA consolidation, (c) automated eval-gated promotion with rollback. Individual pieces aren't new; the productized end-to-end loop is.

2. **The "memory specialist backend" architecture.** Framing the dreamed model as a queryable knowledge oracle that any larger agent calls, rather than as a chat surface. V2 ships this.

Full prior-art positioning in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md).

### How can I help?

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

Easiest first contributions:
- A tuning recipe for a base model we don't yet have one for
- A connector for a memory store we haven't integrated (Claude memory dirs, supermemory, Hermes, Letta, etc.)
- More general-eval probes (especially multilingual or domain-specific)
- Better cross-memory reasoning probes

If you publish work that uses or extends DreamAgent, please cite via [`CITATION.cff`](CITATION.cff) and credit Mr Dula Solutions per [`NOTICE`](NOTICE).

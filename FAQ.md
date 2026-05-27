# FAQ

## What is DreamAgent in one sentence?

DreamAgent is the reference implementation of **MORPHEUS** — Memory Overnight Re-parameterization, Promotion via Held-out Eval, Update Snapshots — a methodology that nightly fine-tunes a small model on your day's memories so the next morning the model knows them from its weights, with no retrieval, no vector index, and no third-party API.

## What's the difference between DreamAgent and MORPHEUS?

- **MORPHEUS** is the methodology — a portable set of techniques, contracts, and gates.
- **DreamAgent** is the project — the Python codebase, CLI, fixture data, and tuning recipes that implement MORPHEUS on Apple Silicon (with cloud-GPU portability).

A Rust port would still implement MORPHEUS; it just wouldn't be called DreamAgent. This decoupling is documented in [ADR-008](docs/adr/008-morpheus-methodology-name.md).

## Why not just use mem0 / Letta / Supermemory?

You probably should, for most use cases. They're battle-tested, mature, and lower-effort than running a nightly fine-tune.

DreamAgent answers a different question: **"What if my memories were part of who the assistant is, not something it looked up?"** That's a different paradigm (parametric vs. non-parametric) with different properties — most notably cross-memory reasoning, structural privacy, and zero per-query memory cost.

See [`docs/comparison/`](docs/comparison/) for the head-to-head with each.

The honest answer for most teams in 2026: **use mem0 for the hot path, DreamAgent for the deep memory.** They compose. The V2 architecture explicitly supports this.

## Does this actually work?

Yes. Pass 1 of V1 demonstrated it on Llama 3.2 1B Instruct: 46% personal recall on held-out probes, 3.3pp general-capability regression. Clean PROMOTE through the eval gate. Reproducible from the repo state in ~5 minutes on Apple Silicon.

See [`docs/tuning/llama-3.2-1b-instruct-4bit.md`](docs/tuning/llama-3.2-1b-instruct-4bit.md) for the locked recipe and 16-run tuning history.

## Why nightly? Why not just train continuously?

Continuous training has unbounded forgetting risk. The nightly batch with eval-gated promotion gives us:
- A clean rollback unit (one bad night doesn't poison everything)
- A measurable "this is the new model" event
- A natural cadence for offline compute (idle overnight on a Mac)

Test-time training ([TTT-E2E](https://arxiv.org/abs/2604.06169)) is the closest continuous-learning alternative; it has different tradeoffs. We may add a TTT mode in V3.

## Won't the model forget everything?

This is the central engineering risk, and yes, if you do it naively. DreamAgent has multiple defenses:

- **Rehearsal mix.** Every nightly run includes a "you are still you" anchor set + replay of prior memories. The training signal is roughly 75% new + 15% replay + 10% anchor.
- **Eval gate.** A 4-decision matrix (PROMOTE / WARN / REJECT) that refuses to promote a regression > 5pp.
- **Per-night snapshots + rollback.** Every adapter is versioned; rollback is one command.
- **Conservative LoRA rank.** 4-8 layers, low learning rate.

Pass 1 measured 3.3pp regression — well within the gate threshold. Long-horizon (30+ nights) is the open question Pass 3 will answer.

## Why LoRA instead of full fine-tuning?

LoRA gives us:
- 1-2 hours of training instead of hours-to-days
- Adapter files are ~50-200 MB (per-night versioning is trivially cheap)
- Catastrophic forgetting is more contained than full FT (you're only modifying low-rank deltas)
- Same script targets MLX (Mac) and Unsloth (cloud) with one config switch

Full FT is on the V3 roadmap if frontier-scale work demands it.

## Why a 1B model? That's tiny.

The validation tier is Llama 3.2 1B because:
- It fine-tunes in ~25 seconds on Apple Silicon — fast iteration
- It's the smallest model with usable general capability (93% on our 30-probe set)
- It's a meaningful **lower bound**: if the thesis works here, it works on bigger models with more headroom

Production-tier (V2) targets Qwen 3 4B. Stretch is Phi-4 14B. Frontier (V3) is 70B+.

## Why a memory specialist instead of just fine-tuning my main model?

Three reasons:
1. **Decoupling.** You keep using Claude, GPT, or whatever your daily-driver is. The memory specialist is a backend, not a chat surface. Zero switching cost.
2. **Cost.** Fine-tuning a 70B model nightly is expensive. Fine-tuning a 4B specialist that the 70B queries is cheap.
3. **Privacy.** A small local model keeps memories on-device. A nightly fine-tune of a frontier API model is non-viable.

## Does the model leak memories?

It can. Any trained model can leak its training data with sufficient prompting (this is what the memorization research literature is about). DreamAgent's specific exposure:

- **Sensitive memories should be marked `sensitivity: redact`** at ingest. The pipeline excludes those from training.
- **The extraction prompt** explicitly refuses to extract secrets (passwords, SSNs, private keys) from raw text.
- **The adapter file** contains learned-from-memories deltas. Treat it like the underlying data — encrypted at rest if needed.

This is genuinely a non-trivial threat model. See [`SECURITY.md`](SECURITY.md).

## What's the GDPR right-to-be-forgotten story?

To "forget" a memory:
1. Remove it from the upstream memory source.
2. Re-train without it (next nightly run, no further action needed).
3. Invalidate the adapters that saw it: `dreamagent rollback <pre-incident-name>` to point `live` at the last clean adapter.

The lineage tracking in `metadata.json` lets you identify which adapters trained on a given memory ID.

This is a clear protocol but it's not instant — full forgetting takes one nightly cycle plus a rollback if the leaked-in fact is recent.

## Does this work on Windows / Intel Macs / Linux without GPU?

- **Apple Silicon (M1+):** First-class. MLX is the primary path.
- **Linux + NVIDIA:** Supported via the cloud path (Unsloth on RunPod/Modal). Same training script.
- **Intel Macs:** MLX requires Apple Silicon. You'd need to use the cloud path.
- **Windows:** Untested. Probably fine via WSL2 + cloud path.

## How much does this cost to run?

- **On a Mac:** Compute is free (your machine, idle overnight). Memory extraction via Anthropic/OpenAI is a few cents per nightly batch (BYO API key). Total: under $1/month for typical personal use.
- **On cloud GPU (Unsloth path):** ~$1-5 per nightly fine-tune on an A100. Could be $30-150/month if you run nightly.

The biggest cost is the *frontier-model extraction step* (converting raw text to MemoryItem JSON). You can avoid this by using the local `ollama` backend or by feeding pre-structured memories directly.

## What if I want to use my own base model?

Pass `--base-model <hf-or-mlx-repo-id>` to `dreamagent dream`. Any MLX-LM-compatible model works. If you want a tuned recipe for a model we don't have one for, follow the playbook in [`docs/tuning/README.md`](docs/tuning/README.md) and submit it back via [the tuning recipe issue template](.github/ISSUE_TEMPLATE/tuning_recipe.md).

## Is this actually new research?

We claim **two specific contributions**:

1. **The methodology**: nightly LoRA consolidation of structured agent memories into parametric weights, with eval-gated automated promotion and per-night adapter snapshots. This combination — agent memory + nightly LoRA + automated safety gate — is, as of May 2026, not published end-to-end anywhere we've found.

2. **The "memory specialist as a backend" architecture**: framing the dreamed model as a queryable knowledge oracle for any larger agent, rather than as a chat surface itself. This is the V2 vision.

The individual components (LoRA, rehearsal buffers, eval gates, memory extraction) are not novel. The combination and the productized loop are.

See [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md) and the prior-art positioning therein.

## How can I help?

PRs welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).

Easiest first contributions:
- A tuning recipe for a base model we don't yet have one for.
- A connector for a memory store we haven't integrated.
- More general-eval probes (especially multilingual or technical domain).
- Better cross-memory reasoning probes.

If you publish work that uses or extends DreamAgent, please cite via [`CITATION.cff`](CITATION.cff) and credit Mr Dula Solutions per [`NOTICE`](NOTICE).

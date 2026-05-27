# How to Actually Improve DreamAgent's Memory

**Date:** 2026-05-27
**Status:** Research + one negative empirical result. OPLoRA k=32 was tested at the locked V1 recipe and did **not** improve over vanilla LoRA — see [`docs/tuning/oplora-k32-validation.md`](../tuning/oplora-k32-validation.md). The research below stands as written; the post-hoc empirical correction is at the bottom under [Empirical update — 2026-05-27 (OPLoRA k=32)](#empirical-update--2026-05-27-oplora-k32).

## Why this document exists

We've spent V2.0–V2.2 building benchmarks and finding that DreamAgent is at best modestly ahead of vector retrieval on capability (+6.2pp personal recall at N=50; parity or worse on cross-memory + adversarial). The benchmarks are honest, but they don't tell us what to *change*. This document is the research step that should precede any V2.3+ direction-setting.

Two questions drove the research:

1. **What techniques would meaningfully improve DreamAgent's memory beyond vanilla LoRA?**
2. **Is building our own LLM (from scratch or from an inherited base) a viable path?**

## TL;DR

- **Best bet: OPLoRA + Sparse Memory Finetuning**, both of which are recent (Oct-Dec 2025/early 2026) drop-in-able improvements to the LoRA training loop. Sparse Memory Finetuning has shown **6× less forgetting than vanilla LoRA** (11% vs 71% loss on continual-learning benchmarks).
- **Second-best: MemoryLLM-style plug-in memory layer** added to the frozen base model. Avoids modifying base weights entirely; trains only the memory module. Bigger lift but architecturally cleaner.
- **Build our own LLM from scratch: no.** A 1B model costs ~$millions in compute for general capability. Specialized memory models are more reasonable but still require professional GPU.
- **Inheritune (warm-start from a larger model): maybe.** ~$5–15 on cloud GPU, ~12 hours, produces a 1.5B model. Worth piloting if we want a *purpose-built* memory specialist instead of a fine-tuned generalist.

The headline: **the highest-ROI work is improving the fine-tuning method, not building a new model.** We've been doing LoRA at rank 8 with no special tricks. There are 2-3 specific upgrades that could plausibly turn our +6.2pp into +15-25pp.

---

## The five candidate techniques

Ranked by ROI (improvement per unit implementation cost), best first.

### 1. OPLoRA — Orthogonal Projection LoRA

**What it is** ([arxiv:2510.13003](https://arxiv.org/html/2510.13003v2)): a small modification to standard LoRA where the update `ΔW = BA` is replaced with `ΔW = P_L BA P_R`. The projection matrices `P_L`, `P_R` are derived from SVD of the frozen pre-trained weights and mathematically guarantee that the top-k singular directions of the base model are exactly preserved.

**Why it might help us:** the V1 tuning logs document multiple episodes where regression came from the adapter overwriting the base model's general-knowledge subspace. OPLoRA structurally prevents this by construction. It's not a heuristic ("use more rehearsal"); it's a mathematical guarantee.

**Measured improvements (from the paper):**
- Commonsense reasoning: OPLoRA-128 best across MathQA / MBPP / RACE held-out
- Mathematics: OPLoRA-16 reaches 49.7% on GSM8K vs lower baseline LoRA
- Code: OPLoRA-16 hits 40.5% on MBPP vs baseline 37.6%

**Implementation cost:**
- One-time SVD of base model weights: ~5 minutes
- Modify the LoRA forward pass to apply projections: 30-50 lines of MLX code
- ~19% slower training (5h → 6h for Llama-2 7B in their setup; smaller delta for us)
- **Total estimated effort: ~1-2 days for a working prototype**

**Risk:** The paper tested on Llama 2 7B and Qwen 2.5 7B. Llama 3.1 8B should behave similarly but we'd need to verify. Truncated-SVD computation on MLX may need numpy-bridging.

**My recommendation:** **Highest priority if we keep building.** This is the cleanest empirical-improvement lever we have.

---

### 2. Sparse Memory Finetuning

**What it is** ([arxiv:2510.15103](https://arxiv.org/pdf/2510.15103)): instead of updating all LoRA parameters uniformly, identify and update only the sparse subset of parameters most relevant to each new task. The rest stay frozen.

**Why it might help us:** vanilla LoRA at rank 8 modifies all 5.2M trainable parameters every step. Sparse Memory Finetuning argues many of those modifications interfere with prior knowledge. Selective updates reduce that interference.

**Reported result:** "11% performance drop with sparse memory finetuning vs 71% with LoRA on continual-learning benchmarks." That's a **6× reduction in forgetting** if the result transfers to our setup.

**Implementation cost:**
- Need to compute "relevance scores" per parameter for each new training batch
- Modify the optimizer to apply gradients only to top-k% relevant parameters
- Tunable: sparsity level (1%, 5%, 10% of parameters updated per step)
- **Total estimated effort: ~2-4 days for a working prototype**

**Risk:** Higher than OPLoRA. The relevance-scoring mechanism in the paper isn't trivially portable, and the "11% vs 71%" comparison was on a specific benchmark suite we'd need to map to our probes. Could end up being more like "20% vs 71%" — still a big win, but verify before promising.

**My recommendation:** Strong second priority. Could pair with OPLoRA (they're complementary — one constrains direction, the other constrains which parameters move).

---

### 3. MemoryLLM — plug-in feed-forward memory layer

**What it is** ([arxiv:2602.00398](https://arxiv.org/pdf/2602.00398)): adds a dedicated, interpretable feed-forward memory module to a transformer. Base model is frozen entirely; only the memory module is trained. Token-indexed storage — explicitly retrievable, unlike LoRA's diffuse weight updates.

**Why it might help us:** structurally separates "what the model knew before" from "what we taught it about the user." Catastrophic forgetting becomes architecturally impossible because we never touch base weights. As a bonus, the memory module is interpretable — we could literally inspect what memory cells fire for which queries.

**Reported claims:**
- Better than LoRA on knowledge-intensive tasks (paper-claimed; numbers vary by benchmark)
- Better than RAG on latency (no retrieval step) and on organization
- Plug-and-play: works on Llama 3.1 8B without retraining the base

**Implementation cost:**
- Substantially more than OPLoRA or sparse mem. New module architecture, new training loop, new evaluation harness.
- MLX-LM doesn't have this out of the box; we'd write it.
- **Total estimated effort: ~1-2 weeks for a working prototype**

**Risk:** Highest of the three "fine-tuning improvement" options. The paper is concrete but the gap from paper → working MLX-LM integration is real.

**My recommendation:** Worth doing AFTER OPLoRA + sparse mem are validated. If the simpler interventions don't close the gap with retrieval, MemoryLLM is the architectural escalation.

---

### 4. Inheritune-style "warm-start a small memory specialist"

**What it is** ([arxiv:2404.08634](https://arxiv.org/html/2404.08634v1)): take the first N layers of a larger pretrained model (e.g., 16 of Llama 3.1 8B's 32 layers), and continue training on a small dataset (~1B tokens) for multiple epochs. The result is a smaller model with much faster training time, leveraging the larger model's representations as a starting point.

**Why it might help us:** instead of fine-tuning a generalist 8B model to behave like a memory specialist, *build a memory specialist directly*. Take Llama 3.1 8B's first 16 layers, then train a memory-specialized head on top with our 50+ memory fixtures plus a small corpus of memory-related dialogue data. The resulting ~3-4B model would be smaller, faster, and trained for the actual task.

**Reported numbers:** 1.5B model trained on 1B tokens via Inheritune in **<12 hours on a single A6000**, matching/exceeding models trained on 50-300× more data.

**Implementation cost:**
- Cloud GPU rental: 1× A6000 ≈ $0.50-$1.50/hr × 12 hr = **~$15**
- Data prep: need a memory-skill dialogue dataset (could synthesize 100k examples with a frontier model)
- Training scaffolding: not too hard with axolotl or torchtune
- **Total estimated effort: ~1 week + ~$15-50 in compute**

**Risk:** Validation. We'd need to confirm the resulting model actually behaves as a "memory specialist" — i.e., better than fine-tuned Llama 3.1 8B at the memory task. Possible the warm-start gives us a Llama-with-fewer-layers, not a memory-specialized model.

**My recommendation:** Most ambitious of the four, but the only one that gets us a *purpose-built* memory model rather than a *fine-tuned* one. Worth doing if we're committing to the project for V3+.

---

### 5. Recurrent Memory Transformer / Memformer-style architecture

**What it is** ([arxiv:2508.10824](https://arxiv.org/pdf/2508.10824)): memory-native architectures that incorporate explicit memory tokens or external memory banks into the transformer's forward pass. The model is *designed* to consult memory, not just trained to recall.

**Why it might help us in theory:** the most architecturally clean answer. The model has memory as a first-class concept.

**Why it might not help us in practice:** these are *new architectures*, not fine-tuning. Adopting one means retraining from scratch — back to the "build your own LLM" problem.

**My recommendation:** Out of scope for V2.3-V2.4. Park it for a future V4+ if we ever commit to a from-scratch model.

---

## The "build our own LLM" question, answered honestly

| Path | Cost | Time | Realistic for us? |
|---|---|---|---|
| **Pure from-scratch 1B model** | ~$50k–$500k cloud compute (9T tokens) | weeks | **No.** Mac-only setup is infeasible. |
| **Pure from-scratch 100M-300M tiny LM** | $50–$500 cloud | days | Maybe, but the model wouldn't be useful as a general-capability base. |
| **Inheritune-style warm-start at 1.5B** | ~$15–50 cloud | ~12 hours | **Yes, if we want a memory specialist.** |
| **Continue fine-tuning Llama 3.1 8B with better methods (OPLoRA / sparse mem)** | $0 (local Mac) | 1-3 days per iteration | **Yes, and probably first.** |

**Recommendation:** No, we don't need to build our own LLM. But Inheritune-style warm-starting a memory specialist is a real option for V3+ if we genuinely commit to the project. The simpler wins (OPLoRA, sparse memory finetuning) come first.

---

## Why we've been drifting

The benchmark work in V2.1 and V2.2 was necessary (we had to know whether the original V1 claims held up — they didn't). But the work after the retractions has been *measurement* rather than *improvement*. Looking back:

- V2.0 alpha shipped the MCP server (useful product step).
- V2.1 measured vs retrieval (useful empirical step, retracted V1 claim).
- V2.2 measured adversarial + composition (useful empirical step, retracted V2.1 claim).
- V2.3 *as currently scoped* is "build better probes + better reconciler." Important but still measurement.

**The actual gap is capability, not measurement.** OPLoRA + sparse memory finetuning attack the capability gap directly. They're what we should build if we want next month's benchmark numbers to be better than this month's.

---

## A concrete proposal for V2.3 if you want one

**V2.3 — Method upgrades, not measurement upgrades** (2-3 weeks of focused work):

1. **Week 1**: implement OPLoRA in the MLX-LM training pipeline. Re-run the V2.1 vs-baselines benchmark. Expected outcome: the +6.2pp personal-recall advantage grows; general-capability regression narrows further.

2. **Week 2**: implement sparse memory finetuning on top of OPLoRA. Run the V2.2 three-way benchmark including adversarial probes. Expected outcome: better resistance on personal recall; possible win on adversarial.

3. **Week 3** (optional, if Week 1-2 produce meaningful gains): pilot MemoryLLM-style architecture as a third path; or document V2.3's results and decide on V2.4 (production hardening) vs V3 (Inheritune memory specialist).

**Acceptance criteria for V2.3 success:** at least one of the methods produces a measurable improvement on cross-memory or adversarial probes versus the V2.2 baseline. If none do, the honest conclusion is that retrieval+context is genuinely competitive at this corpus size, and the path forward is either (a) test at much larger corpora where retrieval recall degrades, or (b) commit to the Inheritune memory-specialist route.

---

## What I think you should decide

You asked the right question — "how do we improve?" — and it's a question we've been avoiding by doing more measurement. The two real options:

- **(A) Keep DreamAgent on Llama 3.1 8B and improve the fine-tuning method.** Invest 2-3 weeks in OPLoRA + sparse memory finetuning. Re-run our existing benchmarks. If they show meaningful improvement, V2.3 is a substantive release. If they don't, we have an honest answer about the ceiling.

- **(B) Commit to a memory-specialized model via Inheritune.** ~$15-50 in cloud compute, ~1-2 weeks of work, results in a smaller, purpose-built model. Higher risk but potentially much higher ceiling.

I lean toward (A) first — it's lower risk, doesn't require cloud GPU access, and gives us a clear answer about whether better fine-tuning is the bottleneck. If (A) closes the gap meaningfully, V2 has its story. If (A) doesn't, that's strong evidence that the methodology has a ceiling that no amount of fine-tuning improvement will fix — and (B) becomes the right call.

This is your decision. I'm not going to start implementing until you say which direction.

## References

- Sparse Memory Finetuning — [arxiv:2510.15103](https://arxiv.org/pdf/2510.15103)
- OPLoRA — [arxiv:2510.13003](https://arxiv.org/html/2510.13003v2)
- MemoryLLM — [arxiv:2602.00398](https://arxiv.org/pdf/2602.00398)
- MemLLM (explicit read-write memory) — [arxiv:2404.11672](https://arxiv.org/pdf/2404.11672)
- Inheritune — [arxiv:2404.08634](https://arxiv.org/html/2404.08634v1)
- Memory-Augmented Transformers systematic review — [arxiv:2508.10824](https://arxiv.org/pdf/2508.10824)
- Subspace geometry of catastrophic forgetting — [arxiv:2603.02224](https://arxiv.org/pdf/2603.02224)
- SmolLM2 — [arxiv:2502.02737](https://arxiv.org/pdf/2502.02737)
- Gamayun 1.5B (cost-efficient multilingual training) — [arxiv:2512.21580](https://arxiv.org/pdf/2512.21580)

---

## Empirical update — 2026-05-27 (OPLoRA k=32)

We implemented OPLoRA (`src/dreamagent/train/oplora.py`, commit
`5da5525`, 12 unit tests verifying the math) and ran it at k=32 on the
locked V1 production recipe (Llama 3.1 8B Instruct, 90 iters, LR 3e-5,
8 layers, 60 anchors). Results were **not encouraging**:

| Metric | Vanilla LoRA single-night | OPLoRA k=32 single-night |
|---|---|---|
| Personal recall (N=48) | 43.75% | 47.92% (+4.17pp, within noise) |
| General regression (N=30) | 0.0pp | **−6.67pp** (wrong direction) |
| Cross-memory vs retrieval | n/a | **−50pp** vs retrieval baseline |
| Gate decision | PROMOTE | PROMOTE_WITH_WARNING |

Full data: [`docs/tuning/oplora-k32-validation.md`](../tuning/oplora-k32-validation.md).

The hypothesis above — "OPLoRA reduces catastrophic forgetting because
it leaves the top-k singular directions of the base weight untouched" —
did not hold at k=32 with the locked recipe. Either:

1. k=32 was too aggressive (the OPLoRA paper used k=32 but on different
   models / tasks). Smaller k (8 or 16) might preserve more update capacity.
2. The locked recipe hyperparameters (tuned for vanilla LoRA) are not
   right for OPLoRA — it may need higher LR or more iters.
3. The "top-k singular subspaces encode pretrained knowledge" hypothesis
   is wrong for transformer weights in our regime. Forgetting may happen
   through tail-singular updates that OPLoRA doesn't constrain.

The ROI ranking at the top of this document ("OPLoRA is the lowest-risk
first step") should be read with this empirical caveat. We did not run
the k=8 or k=16 sweeps before pausing to update direction with the user.

# Path B — Build Your Own Memory-Specialist LLM via Inheritune

**Status:** community guide. The DreamAgent core project is committed to **Path A** (improving fine-tuning methods on Llama 3.1 8B). This document is for contributors who want to take the more ambitious route: build a purpose-built memory-specialist model from a warm-started base.

## When you'd want this

You'd want a memory specialist (rather than a fine-tuned generalist) if:

1. You want a model that's *trained from the start* to be a memory backend, not a chat assistant that we coaxed into being one.
2. You want a smaller, faster model. ~1.5–4B parameters with the first half of Llama 3.1 8B's representations is much faster than the full 8B on inference.
3. You believe the current ceiling is in the base model's prior tuning (chat, instruction-following, safety RLHF) and want to start cleaner.
4. You have cloud-GPU budget — this path is not feasible on Mac alone.

You'd **not** want this if:
- You're optimizing for the lowest-effort improvement (Path A is much lower effort)
- You want to stay 100% local (training step requires NVIDIA CUDA)
- The empirical ceiling from Path A hasn't been measured yet — wait for that data first

## What Inheritune is

[Inheritune](https://arxiv.org/abs/2404.08634) is a technique for building small LMs efficiently by inheriting the early transformer layers from a larger pretrained model and continuing training on a much smaller dataset.

In their reported results, a 1.5B model trained from inherited layers reached the quality of models trained on 50–300× more data, in **~12 hours on a single A6000 GPU**.

For our memory-specialist use case, the protocol becomes:

1. Take Llama 3.1 8B's first 16 layers (of 32) — this gives us a ~4B-parameter starting point with most of the base model's representational power
2. Continue training on a dataset specifically designed for memory-skill learning
3. Evaluate as a memory backend, integrate with DreamAgent's MCP server

## Cost estimate

- **Cloud GPU:** A6000 ≈ $0.50–$1.50/hr depending on provider (RunPod, Lambda Labs, Vast.ai). ~12 hours = **$6–$18**.
- **Storage and data:** negligible (~$1 for the data prep step)
- **Frontier-model API for data synthesis:** if you use Claude/GPT to generate 100k memory-skill examples, ~$20–$50 in API costs
- **Total: ~$30–$70 for a single training run**

This is well under the cost of a single seat-month of most enterprise SaaS tools.

## Prerequisites

- Cloud GPU account with at least one A6000 / A100 (or equivalent)
- ~100GB of disk space for model weights + dataset
- Familiarity with HuggingFace Transformers or one of axolotl / torchtune / Unsloth
- Optional but recommended: a wandb account for experiment tracking

## The Inheritune protocol, applied to a memory specialist

### Step 1: Extract the first 16 layers of Llama 3.1 8B

```python
# extract_inherited.py
from transformers import AutoModelForCausalLM, AutoConfig
import torch

base = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    torch_dtype=torch.bfloat16,
)

# Llama 3.1 8B has 32 transformer layers; keep the first 16.
KEEP = 16
base.model.layers = base.model.layers[:KEEP]
base.config.num_hidden_layers = KEEP

base.save_pretrained("./dreamagent-memory-4b-init")
base.config.save_pretrained("./dreamagent-memory-4b-init")
```

The resulting model is ~4B parameters and inherits Llama's tokenizer, embeddings, and the lower-half transformer representations.

### Step 2: Prepare the memory-skill dataset

DreamAgent's structured `MemoryItem` schema is already a good starting point. You need three categories of training data:

1. **Memory recall** — `(question, memory_set, answer)` triples. Generate by:
   - Taking a `MemoryItem`, asking a frontier LLM to author 3-5 reasonable user questions whose answers should reference that memory
   - Including 3-5 unrelated memories as distractors

2. **Cross-memory synthesis** — questions that require 2-3 memories at once. Author 1k-5k of these. The cross-memory probes in `benchmarks/probes/cross_memory_reasoning.jsonl` are a starting template.

3. **"I don't know" hedging** — questions that intentionally have no answer in the memory set. The model should learn to say so explicitly.

Target dataset size: **100k–500k examples**, mixed across the three categories. Generation prompt template lives in [`examples/path-b/synthesize_memory_dataset.py`](#step-2-script) (we ship a working version).

#### Step 2 script: `examples/path-b/synthesize_memory_dataset.py`

```python
"""Generate a memory-skill training dataset from a MemoryItem corpus."""
import json
import random
from pathlib import Path
import anthropic  # or openai

client = anthropic.Anthropic()
SYSTEM = """You are generating training examples for a model that learns
to answer questions about a user from a structured memory store.

Given a target memory and some distractor memories, output a JSON object:
{
  "question": "<a question whose answer requires the target memory>",
  "memories_in_context": [<full memory texts including target + 2-3 distractors>],
  "answer": "<the correct answer, citing the target memory>"
}
"""

def synthesize(target_memory, distractor_memories, n_examples=5):
    out = []
    for _ in range(n_examples):
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": json.dumps({
                    "target": target_memory["content"],
                    "distractors": [d["content"] for d in distractor_memories],
                }),
            }],
        )
        out.append(json.loads(msg.content[0].text))
    return out
```

(A full version of this script with batching, deduplication, and error handling lives in [`examples/09-path-b-memory-specialist/`](#optional-shipping-this-as-an-example) — see "Shipping this as an example" below.)

### Step 3: Train the warm-started model

Use [Unsloth](https://github.com/unslothai/unsloth) (easiest) or [axolotl](https://axolotl.ai) (more configurable):

```yaml
# train_config.yaml (axolotl-style)
base_model: ./dreamagent-memory-4b-init
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

datasets:
  - path: ./memory-skill-dataset.jsonl
    type: chat_template

# Inheritune key parameters: small dataset, multiple epochs
sequence_len: 2048
micro_batch_size: 4
gradient_accumulation_steps: 4
num_epochs: 8                # critical: re-pass over the small corpus
learning_rate: 3.0e-5
lr_scheduler: cosine
warmup_steps: 100

# Lower-half-of-Llama-3.1-8B specific
adapter: null                # we're full-fine-tuning the 4B
load_in_4bit: false          # full precision for the train; quantize after
output_dir: ./dreamagent-memory-4b-trained
```

```bash
# On the cloud GPU box:
pip install axolotl
accelerate launch -m axolotl.cli.train train_config.yaml
```

Expected wall-clock on a single A6000: **~10-15 hours** for 8 epochs over 100k examples.

### Step 4: Quantize and evaluate

```bash
# Quantize the trained model to 4-bit for efficient inference
python -m mlx_lm.convert --hf-path ./dreamagent-memory-4b-trained \
    --mlx-path ./dreamagent-memory-4b-4bit -q

# Run the standard DreamAgent benchmark suite against the new model
python -m benchmarks.personal_recall \
    --memories fixture:v1_baseline \
    --snapshot none \
    --base-model ./dreamagent-memory-4b-4bit
```

Compare numbers to the vanilla Llama-3.1-8B-with-LoRA-adapter baseline. If the memory-specialist hits or beats those numbers with substantially fewer parameters, the experiment worked.

### Step 5: Plug into DreamAgent's MCP server

Point `dreamagent serve` at the new model:

```bash
DREAMAGENT_BASE_MODEL=./dreamagent-memory-4b-4bit \
DREAMAGENT_SNAPSHOTS_DIR=$HOME/.dreamagent/runs/snapshots \
dreamagent serve
```

Or, if you continued the dream pipeline on top of the memory specialist (recommended), use:

```bash
dreamagent dream \
    --base-model ./dreamagent-memory-4b-4bit \
    --source fixture:v1_baseline \
    --iters 90 --num-layers 8 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag specialist-night-1
```

The MCP server, dream pipeline, eval gate, and benchmark suite all work unchanged. The specialist becomes "the base model" in DreamAgent's framing.

## How to evaluate whether this worked

A successful memory specialist should show:

1. **Personal recall ≥ 75%** on the V1 fixture (matching or beating fine-tuned Llama 3.1 8B)
2. **Lower latency** than the 8B model — ideally p50 < 500ms instead of 1.08s
3. **Smaller memory footprint** — should run comfortably on a 16GB Mac
4. **Better or equal cross-memory reasoning** — the model was trained specifically for this

If the specialist beats the fine-tuned 8B on at least two of these axes, it's worth shipping. If it only matches, the win is the smaller model (faster, cheaper to deploy).

## Open risks

- **Loss of general capability.** Warm-started models inherit only the lower layers. They may struggle with anything that requires the deeper-layer reasoning the original Llama 3.1 8B had. Mitigate by mixing some general-capability examples into the training set.
- **The specialist might still need a "rehearsal mix" anchor set.** The Inheritune paper focused on language modeling, not on continual memory consolidation. Practical engineering will probably require adapting our anchor-fixture approach.
- **Cost / availability of A6000s.** They're widely available on RunPod/Lambda but pricing fluctuates. Budget 20% overhead.
- **Validation.** The risk you write a check for $30 and end up with a model worse than the one we ship for free. Run a small (10k examples, 2 epochs) pilot first to verify the training loop produces *something* before committing to a full run.

## Shipping this as an example

If a contributor completes Path B and the result is reproducible, we'd merge it as `examples/09-path-b-memory-specialist/` with:

- The full data-synthesis script
- The axolotl train config
- The MLX conversion + benchmark wrapper
- Measured numbers vs the V2.2 baseline

A PR adding this example would be one of the highest-leverage contributions to the project.

## Why we (the core project) aren't doing this yet

Honest reasons:

1. **Cloud GPU dependency.** The core project is committed to "runs on a Mac, no cloud" as a usability property. Path B breaks that property.
2. **Higher risk than Path A.** Path A's downside is "we tried OPLoRA and it didn't help." Path B's downside is "we spent $50 and a week and the model is worse." For a project still establishing its empirical credibility post-V2.2-retractions, the lower-risk path comes first.
3. **The case for Path B is strongest *after* Path A has measured the ceiling of the simpler approach.** If OPLoRA + sparse memory finetuning closes the gap with retrieval, Path B becomes less urgent. If they don't, Path B becomes the obvious next step.

Once Path A's results are in, the core project will reassess. Either way, this document stays available for contributors who want to take the more ambitious route now.

## References

- [Inheritune paper](https://arxiv.org/abs/2404.08634) — the warm-start technique
- [SmolLM2 paper](https://arxiv.org/pdf/2502.02737) — data-centric small-LM training, good reference for dataset design
- [Gamayun 1.5B](https://arxiv.org/pdf/2512.21580) — example of cost-efficient ~1.5B training
- DreamAgent core research doc: [`2026-05-improving-memory.md`](./2026-05-improving-memory.md)
- DreamAgent locked-recipe baseline: [`../tuning/llama-3.1-8b-instruct-4bit.md`](../tuning/llama-3.1-8b-instruct-4bit.md)

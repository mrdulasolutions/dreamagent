# 02 — Extract memories from a chat transcript

Convert a raw chat transcript (or any text source) into validated
`MemoryItem` records via a frontier LLM, then dream on them.

## What you'll do

1. Provide a raw chat transcript.
2. Use `dreamagent extract` to convert it to `MemoryItem`s.
3. Run the dream pipeline on the extracted memories.

## Prerequisites

You need an API key for one of:
- Anthropic (`export ANTHROPIC_API_KEY=sk-ant-...`)
- OpenAI (`export OPENAI_API_KEY=sk-...`)
- A locally-running Ollama (no key, just `ollama serve`)

```bash
# Pick your backend
uv sync --extra anthropic    # or --extra openai or --extra ollama
```

## Steps

```bash
# Optional: write your own chat transcript
# (sample.txt is included for testing)

# Extract — the accuracy-critical step. Uses the precision-engineered
# prompt at src/dreamagent/extract/prompt.py
uv run dreamagent extract \
    --from sample.txt \
    --backend anthropic \
    --output extracted.jsonl

# Inspect
uv run dreamagent ingest extracted.jsonl

# Dream on the extracted memories
uv run dreamagent dream \
    --source extracted.jsonl \
    --validation-tier \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag extract-test
```

## What you'll see

The `extract` step prints a table of MemoryItems with their `kind`,
`subject`, `content`, and `confidence`. The extraction prompt is
designed to:

- **Refuse** to extract secrets, passwords, SSNs
- **Skip** ephemeral content ("I said hi", "I'm running tests now")
- **Pick** the right kind from the 5-kind taxonomy
- **Normalize** to third person and ISO dates
- **Drop** records whose confidence falls below 0.3

If you see rejections in the report, the rejection reasons are
structured — that's the pipeline telling you the LLM produced
something that didn't match the schema.

## Troubleshooting

- **"anthropic SDK not installed"** — `uv sync --extra anthropic`
- **No API key found** — set the env var per the prerequisites
- **Empty extraction** — your input might be too short or too
  ephemeral. The prompt is conservative; that's by design.

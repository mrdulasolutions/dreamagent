# 04 — Bridge from mem0 to DreamAgent

If you already run mem0, you can layer DreamAgent on top to add a
parametric memory specialist. mem0 keeps doing hot retrieval; DreamAgent
consolidates the long-term knowledge into weights.

## What you'll do

1. Export your mem0 memories.
2. Convert mem0's JSON shape to DreamAgent's `MemoryItem` schema.
3. Dream on them.
4. (V2 preview) Query the dreamed model alongside mem0 from the same agent.

## Steps

### Export from mem0

If you're using mem0's self-hosted server with Postgres:

```bash
# From mem0's CLI (your mem0 install)
python -m mem0 memories export --user-id YOUR_USER_ID --output mem0-raw.json
```

Or query the API directly:

```bash
curl -X GET "https://YOUR_MEM0/v1/memories?user_id=YOUR_USER" \
    -H "Authorization: Bearer $MEM0_API_KEY" > mem0-raw.json
```

### Convert to MemoryItem shape

```bash
# convert.py is included — it maps mem0's shape to DreamAgent's
python convert.py mem0-raw.json > memories.jsonl

# Validate
uv run dreamagent ingest memories.jsonl
```

### Dream on them

```bash
uv run dreamagent dream \
    --source memories.jsonl \
    --validation-tier \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag from-mem0
```

## How the mapping works

mem0's memory shape (April 2026 algorithm):

```json
{
  "id": "...",
  "memory": "The user's dog is named Otis...",
  "user_id": "...",
  "categories": ["pet"],
  "score": 0.87,
  "created_at": "2026-...",
  "updated_at": "2026-..."
}
```

DreamAgent's `MemoryItem` shape:

```json
{
  "id": "mem_...",
  "schema_version": "1.0",
  "content": "The user's dog is named Otis...",
  "kind": "fact",                       // inferred from category or content
  "subject": "the user's dog",          // extracted noun phrase
  "source": {"system": "mem0", "captured_at": "2026-..."},
  "confidence": 0.87,                   // mapped from mem0's score
  "importance": 0.6,                    // heuristic per category
  "entities": ["Otis"],
  "tags": ["pet"]
}
```

The `convert.py` script does this mapping with sensible defaults. For
edge cases, you can hand-tune the heuristic.

## V2 preview: composing mem0 + DreamAgent at query time

When V2's MCP server lands, your agent will be able to query both:

```python
# Conceptual — V2 API
@agent.tool
def query_memory(question: str) -> dict:
    # Fresh + chunkable → mem0
    hot = mem0.search(question, top_k=3)
    # Deep + synthesized → DreamAgent
    deep = dreamagent.query(question)
    return {"hot": hot, "deep": deep}
```

The agent reconciles. See [`ROADMAP.md`](../../ROADMAP.md) for V2 timing.

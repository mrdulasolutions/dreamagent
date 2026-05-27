# 08 — Composing mem0 + DreamAgent (the V2.2 architecture)

**Status:** ✅ Production-shape recipe. The pattern is the V2.2 deliverable
born out of the V2.1 head-to-head finding (DreamAgent and retrieval match
each other on cross-memory reasoning; composition wins where either alone
struggles).

## Why compose

The V2.1 head-to-head (`docs/tuning/v2.1-vs-baselines.md`) showed:

- DreamAgent has a modest **+6.2pp** personal-recall advantage
- DreamAgent and vector retrieval are at **parity (0.0pp)** on cross-memory reasoning
- The V2.2 three-way benchmark (`docs/tuning/v2.2-adversarial-and-composed.md`)
  measures the composed-system advantage on adversarial probes

The honest takeaway: **DreamAgent and retrieval are complementary.** A
production deployment that wants the best of both runs them in parallel
and reconciles the answers at the agent layer.

## Architecture

```
                    Your Daily-Driver Agent
                    (Claude Code / Cursor / Hermes)
                              │
              ┌───────────────┼───────────────┐
              │ MCP           │ MCP/HTTP      │ The agent decides:
              ▼               ▼               │  • prefer DA for persona/preferences
        ┌──────────┐   ┌──────────────┐      │  • prefer retrieval for fresh/specific facts
        │   mem0   │   │  DreamAgent  │      │  • reconcile when both have something to say
        │ (recent  │   │  (durable,   │      │  • flag disagreement
        │  text +  │   │  parametric  │      │
        │  index)  │   │   memory)    │      │
        └──────────┘   └──────────────┘
```

**mem0's strengths** (hot retrieval):
- Adding a new memory is instant
- Deleting a memory is instant
- High recall on questions matching the memory's surface form
- Good for fresh, mutable, session-scoped knowledge

**DreamAgent's strengths** (parametric memory):
- Survives retrieval misses (the memory is in weights, not indexable form)
- Stronger on personal recall + persona stability (V2.1: +6.2pp)
- Privacy-strict (no embeddings stored anywhere)
- Host-agent independent (any MCP client works)
- Good for durable, identity-defining knowledge

**Composition pattern:** both running; agent picks or reconciles.

## What you'll do

1. Install + run a mem0 self-hosted server
2. Add DreamAgent's MCP server (you should already have this from
   [`examples/07-mcp-memory-backend/`](../07-mcp-memory-backend/))
3. Configure your daily-driver agent to call both
4. Verify queries route correctly

## Prerequisites

- DreamAgent installed with `uv sync --extra mcp` (you already have this)
- A trained DreamAgent adapter (run `examples/01-quickstart/` or `examples/00-five-minutes/`)
- Claude Code, Cursor, or another MCP-capable client
- Docker (for the mem0 self-hosted stack — Postgres + their server)

## Step 1: Run mem0 self-hosted

Follow [mem0's self-hosted setup](https://docs.mem0.ai/open-source/python_quickstart):

```bash
# Pull and start mem0's self-hosted stack
docker compose -f https://raw.githubusercontent.com/mem0ai/mem0/main/server/docker-compose.yaml up -d

# Verify it's up
curl http://localhost:8888/health
```

mem0 by default uses OpenAI for memory extraction. Configure
`OPENAI_API_KEY` in its `.env` or point it at an Ollama instance for
fully-local operation.

## Step 2: Populate mem0 with the same fixture memories

We want both systems to know the same things, so we can fairly compose.

```bash
# Convert DreamAgent's fixture format to mem0 input
python examples/08-mem0-plus-dreamagent/seed_mem0.py \
    --from fixtures/v1_baseline.jsonl \
    --user-id matt \
    --mem0-url http://localhost:8888
```

Verify:

```bash
curl "http://localhost:8888/v1/memories?user_id=matt" | jq '.[] | .memory' | head -10
```

## Step 3: Hook mem0 into your daily-driver agent

mem0 ships an MCP server too:

```bash
pip install mem0-mcp
```

Add to your Claude Code MCP config (alongside DreamAgent):

```json
{
  "mcpServers": {
    "dreamagent": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/dreamagent", "dreamagent", "serve"],
      "env": {
        "DREAMAGENT_SNAPSHOTS_DIR": "/path/to/dreamagent/runs/snapshots"
      }
    },
    "mem0": {
      "command": "uvx",
      "args": ["mem0-mcp"],
      "env": {
        "MEM0_API_URL": "http://localhost:8888",
        "MEM0_USER_ID": "matt"
      }
    }
  }
}
```

Restart Claude Code. You should now have both `dreamagent.query_memory`
and mem0's `search_memory` tools available.

## Step 4: Ask the agent to use both

In Claude Code, try queries that benefit from each:

**Personal recall (DreamAgent wins by +6.2pp in V2.1):**
> Use both `dreamagent.query_memory` AND mem0's `search_memory` to find
> out: what is Matt's dog's name? Reconcile any difference between them.

**Cross-memory reasoning (parity territory):**
> Use both tools: given what you know about Matt's projects and his
> preferred tools, what command would he likely use to test the
> DreamAgent project? Compare the two answers.

**Adversarial (DreamAgent wins by ?, V2.2 measures this):**
> Use both tools: should we recommend a GPL-3.0 library for Matt's
> new commercial project?

The agent will call both, see both answers, and synthesize. In
practice you'll find:

- DA usually answers more confidently from weights
- mem0 surfaces specific memory snippets with traceable IDs
- When they agree → high confidence
- When they disagree → agent should flag and use both

## Step 5: When to prefer one over the other

| Situation | Prefer |
|---|---|
| User just said something — won't be in DA's training until tomorrow | mem0 (instant capture) |
| Persona/identity question ("Who am I?") | DreamAgent |
| Specific date or proper noun the user explicitly stated | mem0 (verbatim) |
| Synthesis across multiple memories | Either (V2.1: parity) |
| Question with low semantic overlap with the relevant memory | DreamAgent (V2.2 finding) |
| Question after a recent correction the user made | mem0 (DA is one nightly cycle behind) |
| Compliance-sensitive (no external index allowed) | DreamAgent only |

## Step 6: The nightly bridge

The composition pattern has an important property: **the same `MemoryItem`s
that mem0 captures during the day can be the source memories DreamAgent
consolidates at night.** The flow:

```
Day:   User → Claude → mem0.add_memory(structured_record)
                              │
                              ▼
                       mem0 stores text + embedding
                              │
Night: 3 AM cron      mem0 export → DreamAgent ingest
                              │
                              ▼
                       dreamagent dream  → new adapter, eval gate
                              │
                              ▼
Day+1: User → Claude → both tools available, DA now knows yesterday
```

`examples/04-bridge-from-mem0/convert.py` already handles the mem0
export → DreamAgent format conversion. The full nightly pipeline is:

```bash
# 1. Pull yesterday's mem0 memories into a JSONL
python -m mem0 memories export --user-id matt --output yesterday.json
python examples/04-bridge-from-mem0/convert.py yesterday.json > yesterday.jsonl

# 2. Dream on them (resumes from the prior live adapter for continuity)
dreamagent dream \
    --source yesterday.jsonl \
    --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
    --iters 90 --num-layers 8 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --resume-from-snapshot runs/snapshots/live \
    --tag nightly

# 3. (Auto via launchd — see examples/03-nightly-cron/)
```

Now mem0 has every memory and DreamAgent has the consolidated version
in weights. The composed system is the best of both.

## What we measured

| Probe set | DA alone | Retrieval alone | Composed |
|---|---|---|---|
| Personal recall | 75.0% | 68.8% | _see V2.2 doc_ |
| Cross-memory reasoning | 90.0% | 90.0% | _see V2.2 doc_ |
| Adversarial (retrieval-defeating) | _see V2.2 doc_ | _see V2.2 doc_ | _see V2.2 doc_ |

Detailed numbers: [`docs/tuning/v2.2-adversarial-and-composed.md`](../../docs/tuning/v2.2-adversarial-and-composed.md).

## Limitations of this cookbook

- We don't yet ship a fully-automated mem0 ↔ DreamAgent bridge as a single
  command. The steps above are manual; V2.3 is the obvious place to
  package this.
- mem0's MCP server is third-party; if its API changes, this cookbook
  may need updating.
- The "agent reconciliation" step depends on the host model's ability to
  synthesize across tool outputs. Strong frontier models (Claude, GPT-5)
  do this well; weaker local agents may need explicit reconciliation logic.

## Next

After this, the natural V2.3+ items:

- **Adversarial probe expansion**: 15 probes is small; a more thorough
  set would tighten the measured composition advantage
- **Latency optimization for composed mode**: 3 model calls (DA + retrieval
  + reconcile) is slow; pipelined or speculatively-cached versions are
  open territory
- **Composition cookbook for Letta** (similar pattern, different SDK)

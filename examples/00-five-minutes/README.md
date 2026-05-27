# 00 — The 5-Minute Tour

The fastest way to **see DreamAgent doing the thing it does** — without
yet committing to running it on your own memories. Uses the included
demo memories about a fictional user named Matt.

## What you'll do

1. Install DreamAgent (one command)
2. Run the dream pipeline on the included demo memories (one command)
3. Connect it to Claude Code as an MCP server
4. Ask Claude something about Matt — Claude calls DreamAgent, gets a real answer

If everything goes right, **start to finish is about 25 minutes** the first
time (most of it is downloading the small AI), then 5 minutes for repeat
runs.

## What you need

- A Mac with Apple Silicon (M1/M2/M3/M4), 16GB RAM minimum
- About 8GB of free disk space
- Claude Code installed (or any other MCP-capable agent)
- Comfort with opening Terminal and pasting commands

## Step 1: Install uv (the Python installer)

DreamAgent uses `uv`, a modern Python tool installer.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You can verify it worked:

```bash
uv --version
```

You should see something like `uv 0.11.x`.

## Step 2: Clone DreamAgent

Pick a folder where you want it to live (your home directory works fine):

```bash
cd ~
git clone https://github.com/mrdulasolutions/dreamagent.git
cd dreamagent
```

## Step 3: Install DreamAgent's dependencies

```bash
uv sync --extra mcp
```

This downloads about 200MB of Python packages. Takes 1-2 minutes.

## Step 4: Run the demo "dream"

This is the real thing. We're going to download a small AI (~4.5GB) and
train it on 50 example memories about Matt (his dog Otis, his deploy
commands, his preferences, etc.).

```bash
uv run dreamagent dream \
    --validation-tier \
    --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
    --source fixture:v1_baseline \
    --iters 90 --num-layers 8 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag five-minute-tour
```

**Expected timing on a Mac M-series:**
- ~10 min to download the small AI (first time only)
- ~3 min to train it on the 50 demo memories
- ~3 min to run safety checks

You'll see a lot of output. Look for the end:

```
gate decision
PROMOTE
  · personal recall 0.XX >= min 0.30
  · general regression +0.XXX <= max +0.050
snapshot: /Users/you/dreamagent/runs/snapshots/...
```

That `PROMOTE` (in green) means **it worked**. The small AI has learned
the 50 demo memories.

You can prove it remembers without involving Claude Code yet — peek at
the eval JSON:

```bash
cat runs/snapshots/live/eval_personal.json | head -50
```

You'll see questions like "What is the user's dog's name?" and the model's
answers. Most should be correct.

## Step 5: Hook it up to Claude Code

Open Claude Code's MCP configuration. (In Claude Code: Settings → MCP, or
edit `~/.claude/claude_desktop_config.json` directly.)

Find your DreamAgent path:

```bash
pwd
```

You'll get something like `/Users/you/dreamagent`. Copy that.

Add this to the MCP config (substituting your path):

```json
{
  "mcpServers": {
    "dreamagent": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/Users/you/dreamagent",
        "dreamagent",
        "serve"
      ],
      "env": {
        "DREAMAGENT_SNAPSHOTS_DIR": "/Users/you/dreamagent/runs/snapshots"
      }
    }
  }
}
```

**Restart Claude Code.** You should see `dreamagent` listed as a connected
MCP server in the Claude Code UI.

## Step 6: Ask Claude about Matt

In a Claude Code chat, ask:

> Use the dreamagent `query_memory` tool to find out what Matt's dog's name is.

Claude will call the tool. The small AI on your Mac will answer. Claude
shows you the answer.

**The answer is "Otis."** (Matt's dog is a golden retriever named Otis,
adopted in August 2024 — all of this is in the demo memories.)

Try other questions:

> Use query_memory: What command does Matt use to run tests?

> Use query_memory_with_lineage: How does Matt prefer responses formatted?

The second tool returns provenance — which adapter answered, what model
it's running on.

## What just happened

The tool calls go from Claude → your Mac → the trained small AI →
back to Claude. **None of Matt's memories ever leave your machine.**
Claude didn't see them, OpenAI didn't see them, no server saw them.
Claude just got the synthesized answer the small AI produced.

This is the V2 thesis in microcosm: your daily-driver AI gets long
memory without giving up privacy or paying for a vector database.

## Doing this with your own memories

The demo memories about Matt live in `fixtures/v1_baseline.jsonl`. To
use your own:

```bash
# Write a chat transcript or journal entry
echo "My dog Sadie is a 4-year-old Australian shepherd. I prefer
concise responses. To run tests I do: pnpm test. ..." > my-notes.txt

# Extract structured memories (needs an Anthropic or OpenAI key)
export ANTHROPIC_API_KEY=sk-ant-...
uv run dreamagent extract --from my-notes.txt --backend anthropic \
    --output my-memories.jsonl

# Run a new dream on YOUR memories
uv run dreamagent dream --source my-memories.jsonl \
    --validation-tier \
    --base-model "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit" \
    --iters 90 --num-layers 8 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60
```

Restart Claude Code. Now `query_memory` answers about you, not Matt.

## Doing this nightly (set and forget)

```bash
uv run dreamagent install-cron
```

This sets up macOS launchd to run `dreamagent dream` at 3 AM nightly,
on whatever memories live in your source file. Each morning your
memory specialist knows a little more than it did yesterday.

To uninstall: see [`examples/03-nightly-cron/`](../03-nightly-cron/).

## What could go wrong

| Issue | Likely cause | Fix |
|---|---|---|
| "uv: command not found" | Step 1 didn't work | Re-run the curl install, restart your terminal |
| `mlx-lm` install fails | Mac is too old (no Apple Silicon) | DreamAgent needs M1 or newer |
| `dream` says REJECT | Your memories triggered a safety rule | See [`docs/tuning/README.md`](../../docs/tuning/README.md) — usually a hyperparameter adjustment |
| Claude Code doesn't see dreamagent | MCP path is wrong / Claude wasn't restarted | Double-check `--directory` path; full restart of Claude |
| First MCP query takes ~10 sec | Cold start, model loading lazily | That's normal; subsequent queries are sub-second |

## What's next

- **[`examples/01-quickstart`](../01-quickstart/)** — same thing but more terse, for repeat use
- **[`examples/02-extract-from-chat`](../02-extract-from-chat/)** — extract memories from real chat transcripts
- **[`examples/03-nightly-cron`](../03-nightly-cron/)** — schedule the dream pipeline
- **[`examples/05-rollback-drill`](../05-rollback-drill/)** — practice the safety/rollback flow
- **[`examples/06-benchmark-suite`](../06-benchmark-suite/)** — measure your own DreamAgent install

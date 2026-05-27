# 03 — Schedule nightly dreams

Install a system cron job that runs `dreamagent dream` every night at
3 AM using the locked V1 recipe.

## What you'll do

1. Preview what will be installed.
2. Install the launchd plist (macOS) or print the crontab line (Linux).
3. Verify it's scheduled and inspect logs after the first run.

## Steps

```bash
# Preview without writing anything
uv run dreamagent install-cron --dry-run

# Actually install
uv run dreamagent install-cron

# On macOS, load it:
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/solutions.mrdula.dreamagent.nightly.plist

# Verify it's loaded
launchctl list | grep dreamagent

# After the next 3 AM, inspect the logs
ls ~/.dreamagent/runs/logs/
cat ~/.dreamagent/runs/logs/nightly.out.log
```

## Custom schedule

```bash
# Every 6 hours instead of nightly
uv run dreamagent install-cron --schedule "0 */6 * * *"

# Specific time and source
uv run dreamagent install-cron \
    --schedule "0 2 * * *" \
    --source ~/memories/today.jsonl \
    --output-dir ~/.dreamagent/runs
```

## Uninstall

```bash
launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/solutions.mrdula.dreamagent.nightly.plist
rm ~/Library/LaunchAgents/solutions.mrdula.dreamagent.nightly.plist
```

## What it does at 3 AM

1. Reads memories from `--source` (default: `fixture:v1_baseline`).
2. Composes the training set + rehearsal mix.
3. Trains a LoRA adapter (~5 min on Apple Silicon for 1B model).
4. Runs personal + general eval.
5. Decides via the gate.
6. PROMOTE → updates `live` symlink. REJECT → preserves prior `live`.

## What can go wrong (and how you'd know)

- **Out of disk** — adapter writes fail; check logs in `~/.dreamagent/runs/logs/`
- **No internet** — extraction backends will fail (training itself is offline)
- **Gate REJECTs every night** — your memories may need a different recipe; see [`docs/tuning/`](../../docs/tuning/)

The eval gate is the safety net: a bad night gets archived in
`rejected/`, the live model is untouched.

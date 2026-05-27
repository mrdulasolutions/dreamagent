#!/usr/bin/env bash
# Quickstart: minimum-viable DreamAgent loop on shipped fixtures.
# See README.md for what to expect.

set -euo pipefail

cd "$(dirname "$0")/../.."

uv sync

uv run dreamagent dream \
    --validation-tier \
    --base-model "mlx-community/Llama-3.2-1B-Instruct-4bit" \
    --source fixture:v1_baseline \
    --iters 90 --num-layers 4 --learning-rate 3e-5 \
    --anchor-ratio 0.30 --max-anchors 60 \
    --tag quickstart

uv run dreamagent snapshots

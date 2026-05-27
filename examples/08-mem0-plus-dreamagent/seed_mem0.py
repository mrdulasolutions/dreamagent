#!/usr/bin/env python3
"""Seed a self-hosted mem0 instance with DreamAgent fixture memories.

Lets you put the same 50 memories into both systems so the composition
benchmark + cookbook are reproducible.

Usage:
    python seed_mem0.py --from fixtures/v1_baseline.jsonl \\
        --user-id matt --mem0-url http://localhost:8888

Requires the mem0 self-hosted server to be running. See the README in
this directory for setup.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--from", dest="src", required=True, type=Path)
    p.add_argument("--user-id", required=True)
    p.add_argument("--mem0-url", default="http://localhost:8888")
    args = p.parse_args()

    try:
        import requests
    except ImportError:
        sys.exit("requests not installed — `pip install requests` or use `curl` manually")

    if not args.src.exists():
        sys.exit(f"input not found: {args.src}")

    inserted = 0
    skipped = 0
    with args.src.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                item = json.loads(stripped)
            except json.JSONDecodeError:
                print(f"line {line_no}: skipped (invalid JSON)", file=sys.stderr)
                skipped += 1
                continue
            # mem0 wants {"messages": [{"role": ..., "content": ...}]} or
            # {"data": "..."}; we feed content + metadata.
            payload = {
                "data": item["content"],
                "user_id": args.user_id,
                "metadata": {
                    "kind": item.get("kind"),
                    "subject": item.get("subject"),
                    "source_id": item["id"],
                },
            }
            try:
                r = requests.post(f"{args.mem0_url}/v1/memories", json=payload, timeout=30)
                r.raise_for_status()
                inserted += 1
            except requests.RequestException as e:
                print(f"line {line_no}: mem0 reject ({item['id']}): {e}", file=sys.stderr)
                skipped += 1

    print(f"inserted: {inserted} · skipped: {skipped}")


if __name__ == "__main__":
    main()

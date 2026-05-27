#!/usr/bin/env python3
"""Convert a mem0 export JSON to DreamAgent JSONL.

Usage:
    python convert.py mem0-export.json > memories.jsonl

The mem0 export is expected to be a JSON array of memory objects. Each
gets mapped to a MemoryItem with sensible defaults. Hand-tune the
heuristic if needed for your domain.
"""

from __future__ import annotations

import json
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path


def infer_kind(memory: str, categories: list[str]) -> str:
    """Heuristic kind inference from mem0 categories + content."""
    lc = memory.lower()
    cat_str = " ".join(categories).lower()

    if "prefer" in lc or "likes" in lc or "preference" in cat_str:
        return "preference"
    if any(w in lc for w in ["to deploy", "to run", "to test", "to build", "how does the user"]):
        return "procedure"
    if any(w in lc for w in ["on 20", "in 20", "yesterday", "happened", "decided"]):
        return "event"
    return "fact"


def infer_subject(memory: str) -> str:
    """Take a short noun phrase from the first sentence as the subject."""
    first = memory.split(".")[0].strip()
    return first[:80] if first else memory[:80]


def map_record(rec: dict) -> dict:
    memory = rec.get("memory") or rec.get("text") or rec.get("content") or ""
    categories = rec.get("categories", [])
    return {
        "id": "mem_" + secrets.token_hex(8),
        "schema_version": "1.0",
        "content": memory,
        "kind": infer_kind(memory, categories),
        "subject": infer_subject(memory),
        "source": {
            "system": "mem0",
            "session_id": rec.get("session_id"),
            "captured_at": rec.get("created_at") or datetime.now(UTC).isoformat(),
        },
        "confidence": float(rec.get("score", 0.8)),
        "importance": 0.6,
        "entities": rec.get("entities", []),
        "tags": categories,
    }


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("usage: convert.py mem0-export.json > memories.jsonl")
    src = Path(sys.argv[1]).read_text(encoding="utf-8")
    data = json.loads(src)
    if isinstance(data, dict) and "memories" in data:
        data = data["memories"]
    if not isinstance(data, list):
        sys.exit("expected a JSON array of memories")

    for rec in data:
        if not isinstance(rec, dict):
            continue
        out = map_record(rec)
        print(json.dumps(out))


if __name__ == "__main__":
    main()

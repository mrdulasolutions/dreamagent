"""JSONL connector: reads MemoryItems one per line from a .jsonl file."""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import datetime
from pathlib import Path

from dreamagent.schema import MemoryBatch, MemoryItem


class JSONLConnector:
    """Read MemoryItems from a newline-delimited JSON file.

    Each line is either a single MemoryItem object, or a MemoryBatch envelope.
    Malformed lines raise; this is a hard contract, not best-effort.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def name(self) -> str:
        return f"jsonl:{self.path}"

    def iter_memories(self, since: datetime | None = None) -> Iterable[MemoryItem]:
        if not self.path.exists():
            raise FileNotFoundError(f"connector source not found: {self.path}")
        yield from self._iter_filtered(since)

    def _iter_filtered(self, since: datetime | None) -> Iterator[MemoryItem]:
        for item in self._iter_raw():
            if since is None or item.source.captured_at >= since:
                yield item

    def _iter_raw(self) -> Iterator[MemoryItem]:
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                stripped = raw.strip()
                if not stripped:
                    continue
                try:
                    parsed = json.loads(stripped)
                except json.JSONDecodeError as e:
                    raise ValueError(
                        f"{self.path}:{line_no}: invalid JSON: {e.msg}"
                    ) from e
                yield from self._coerce(parsed, line_no)

    def _coerce(self, parsed: object, line_no: int) -> Iterator[MemoryItem]:
        if isinstance(parsed, dict) and "items" in parsed:
            try:
                batch = MemoryBatch.model_validate(parsed)
            except Exception as e:
                raise ValueError(f"{self.path}:{line_no}: invalid batch: {e}") from e
            yield from batch.items
        else:
            try:
                yield MemoryItem.model_validate(parsed)
            except Exception as e:
                raise ValueError(f"{self.path}:{line_no}: invalid MemoryItem: {e}") from e

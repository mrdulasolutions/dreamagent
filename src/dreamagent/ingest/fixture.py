"""Fixture connector: loads hand-written test memories shipped with the repo.

Used by the validation tier and CI. The fixtures live in `fixtures/` at the
repo root; this connector finds them via package-relative path resolution.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from dreamagent.ingest.jsonl import JSONLConnector
from dreamagent.schema import MemoryItem


def _fixtures_root() -> Path:
    """Locate the fixtures/ directory at the repo root."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / "fixtures"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError("could not locate fixtures/ directory")


class FixtureConnector:
    """Load one or all fixture .jsonl files.

    Usage:
        FixtureConnector()                  # all fixtures
        FixtureConnector("v1_baseline")     # fixtures/v1_baseline.jsonl
    """

    def __init__(self, fixture_name: str | None = None):
        self.fixture_name = fixture_name

    def name(self) -> str:
        suffix = self.fixture_name or "*"
        return f"fixture:{suffix}"

    def iter_memories(self, since: datetime | None = None) -> Iterable[MemoryItem]:
        root = _fixtures_root()
        if self.fixture_name is not None:
            paths = [root / f"{self.fixture_name}.jsonl"]
        else:
            paths = sorted(root.glob("*.jsonl"))
        if not paths:
            raise FileNotFoundError(f"no fixture .jsonl files found under {root}")
        for path in paths:
            yield from JSONLConnector(path).iter_memories(since)

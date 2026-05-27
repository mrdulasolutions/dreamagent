"""The MemoryConnector protocol. Any upstream memory source implements this."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Protocol, runtime_checkable

from dreamagent.schema import MemoryItem


@runtime_checkable
class MemoryConnector(Protocol):
    """Anything that emits a stream of MemoryItems is a connector.

    Implementations: JSONLConnector, FixtureConnector, Mem0Connector (future), etc.
    """

    def name(self) -> str:
        """Human-readable name of this connector, for logs and lineage."""
        ...

    def iter_memories(self, since: datetime | None = None) -> Iterable[MemoryItem]:
        """Yield memories captured at-or-after `since` (None = all)."""
        ...

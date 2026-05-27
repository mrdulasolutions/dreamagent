"""Memory ingestion — connectors that emit MemoryItems from upstream sources."""

from dreamagent.ingest.base import MemoryConnector
from dreamagent.ingest.fixture import FixtureConnector
from dreamagent.ingest.jsonl import JSONLConnector

__all__ = ["FixtureConnector", "JSONLConnector", "MemoryConnector"]

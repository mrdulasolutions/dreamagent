"""Tests for the ingest connectors."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dreamagent.ingest import FixtureConnector, JSONLConnector, MemoryConnector
from dreamagent.schema import (
    MemoryBatch,
    MemoryItem,
    MemoryKind,
    Source,
    SourceSystem,
)

NOW = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)
EARLIER = datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC)
LATER = datetime(2026, 5, 27, 12, 0, 0, tzinfo=UTC)


def _make_item(id_: str, captured_at: datetime = NOW) -> MemoryItem:
    return MemoryItem(
        id=id_,
        content="The user's dog is named Otis.",
        kind=MemoryKind.FACT,
        subject="the user's dog Otis",
        source=Source(system=SourceSystem.FIXTURE, captured_at=captured_at),
        confidence=0.9,
        importance=0.6,
    )


class TestJSONLConnectorRead:
    def test_reads_one_per_line(self, tmp_path: Path):
        path = tmp_path / "mems.jsonl"
        items = [_make_item("a"), _make_item("b"), _make_item("c")]
        path.write_text(
            "\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8"
        )

        loaded = list(JSONLConnector(path).iter_memories())
        assert [m.id for m in loaded] == ["a", "b", "c"]

    def test_reads_batch_envelope(self, tmp_path: Path):
        path = tmp_path / "mems.jsonl"
        batch = MemoryBatch(items=[_make_item("a"), _make_item("b")])
        path.write_text(batch.model_dump_json() + "\n", encoding="utf-8")

        loaded = list(JSONLConnector(path).iter_memories())
        assert [m.id for m in loaded] == ["a", "b"]

    def test_skips_blank_lines(self, tmp_path: Path):
        path = tmp_path / "mems.jsonl"
        item = _make_item("a")
        path.write_text(f"\n\n{item.model_dump_json()}\n\n", encoding="utf-8")
        loaded = list(JSONLConnector(path).iter_memories())
        assert len(loaded) == 1

    def test_missing_file_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            list(JSONLConnector(tmp_path / "nope.jsonl").iter_memories())


class TestJSONLConnectorErrors:
    def test_invalid_json_line_raises_with_line_no(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text("not json\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":1:"):
            list(JSONLConnector(path).iter_memories())

    def test_invalid_memory_raises_with_line_no(self, tmp_path: Path):
        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps({"id": "x"}) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match=":1:"):
            list(JSONLConnector(path).iter_memories())


class TestJSONLConnectorFiltering:
    def test_since_filter(self, tmp_path: Path):
        path = tmp_path / "mems.jsonl"
        items = [
            _make_item("old", EARLIER),
            _make_item("new", NOW),
            _make_item("future", LATER),
        ]
        path.write_text(
            "\n".join(i.model_dump_json() for i in items) + "\n", encoding="utf-8"
        )

        loaded = list(JSONLConnector(path).iter_memories(since=NOW))
        assert {m.id for m in loaded} == {"new", "future"}


class TestProtocolConformance:
    def test_jsonl_satisfies_protocol(self, tmp_path: Path):
        c = JSONLConnector(tmp_path / "x.jsonl")
        assert isinstance(c, MemoryConnector)
        assert c.name().startswith("jsonl:")

    def test_fixture_satisfies_protocol(self):
        c = FixtureConnector("anything")
        assert isinstance(c, MemoryConnector)
        assert c.name().startswith("fixture:")

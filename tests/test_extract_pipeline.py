"""Tests for the extraction pipeline — focused on parsing + validation.

The actual model-calling backends are NOT exercised here (they need API keys
and live calls). The pipeline is tested with a fake backend.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

from dreamagent.extract.base import ExtractionResponse
from dreamagent.extract.pipeline import (
    _format_chat_jsonl,
    _parse_response,
    extract_memories,
)
from dreamagent.schema import MemoryItem, MemoryKind, SourceSystem


class _FakeBackend:
    name = "fake"
    model = "fake-model"

    def __init__(self, raw_output: str):
        self._raw = raw_output

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionResponse:
        return ExtractionResponse(
            raw_output=self._raw,
            model=self.model,
            backend=self.name,
            prompt_tokens=10,
            completion_tokens=20,
        )


GOOD_RAW = json.dumps(
    [
        {
            "content": "The user's dog Otis is a golden retriever.",
            "kind": "fact",
            "subject": "the user's dog Otis",
            "confidence": 0.95,
            "importance": 0.6,
            "entities": ["Otis", "golden retriever"],
        },
        {
            "content": "The user prefers concise responses with no preamble.",
            "kind": "preference",
            "subject": "response style",
            "confidence": 1.0,
            "importance": 0.9,
            "entities": [],
        },
    ]
)


class TestParseResponse:
    def test_parses_clean_json_array(self):
        out = _parse_response('[{"a": 1}, {"b": 2}]')
        assert out == [{"a": 1}, {"b": 2}]

    def test_strips_markdown_fence(self):
        wrapped = "```json\n[{\"a\": 1}]\n```"
        assert _parse_response(wrapped) == [{"a": 1}]

    def test_rejects_non_array(self):
        import pytest

        with pytest.raises(ValueError, match="expected JSON array"):
            _parse_response('{"oops": true}')


class TestExtractMemoriesEndToEnd:
    def test_valid_records_produce_memory_items(self):
        backend = _FakeBackend(GOOD_RAW)
        report = extract_memories(
            "dummy input", backend, captured_at=datetime(2026, 5, 26, tzinfo=UTC)
        )
        assert len(report.items) == 2
        assert all(isinstance(i, MemoryItem) for i in report.items)
        assert report.items[0].kind == MemoryKind.FACT
        assert report.items[1].kind == MemoryKind.PREFERENCE
        assert report.rejected == []
        assert report.response.backend == "fake"

    def test_each_item_gets_unique_id_schema_and_source(self):
        backend = _FakeBackend(GOOD_RAW)
        report = extract_memories("dummy", backend)
        ids = [i.id for i in report.items]
        assert len(set(ids)) == 2  # unique
        for item in report.items:
            assert item.schema_version == "1.0"
            assert item.source.system == SourceSystem.MANUAL

    def test_invalid_record_rejected_with_reason(self):
        bad_raw = json.dumps(
            [
                {"content": "missing required kind", "subject": "x", "confidence": 0.5,
                 "importance": 0.5, "entities": []},  # missing kind
                {
                    "content": "good",
                    "kind": "fact",
                    "subject": "x",
                    "confidence": 0.5,
                    "importance": 0.5,
                    "entities": [],
                },
            ]
        )
        backend = _FakeBackend(bad_raw)
        report = extract_memories("dummy", backend)
        assert len(report.items) == 1
        assert len(report.rejected) == 1
        assert report.rejected[0]["reason"] == "validation failed"

    def test_unparseable_response_reports_rejection(self):
        backend = _FakeBackend("this is not json at all")
        report = extract_memories("dummy", backend)
        assert report.items == []
        assert len(report.rejected) == 1
        assert "failed to parse" in report.rejected[0]["reason"]

    def test_empty_array_is_legitimate(self):
        backend = _FakeBackend("[]")
        report = extract_memories("dummy", backend)
        assert report.items == []
        assert report.rejected == []


class TestChatJsonlFormatting:
    def test_flat_messages(self):
        jsonl = '\n'.join([
            '{"role": "user", "content": "Hello"}',
            '{"role": "assistant", "content": "Hi"}',
        ])
        out = _format_chat_jsonl(jsonl)
        assert "[user] Hello" in out
        assert "[assistant] Hi" in out

    def test_nested_messages_envelope(self):
        jsonl = '{"messages": [{"role":"user","content":"x"},{"role":"assistant","content":"y"}]}'
        out = _format_chat_jsonl(jsonl)
        assert "[user] x" in out
        assert "[assistant] y" in out

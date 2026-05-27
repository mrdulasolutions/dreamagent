"""Tests for the MemoryItem schema."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from dreamagent.schema import (
    MAX_CONTENT_CHARS,
    SCHEMA_VERSION,
    MemoryBatch,
    MemoryItem,
    MemoryKind,
    PreferenceSignal,
    QAPair,
    Sensitivity,
    Source,
    SourceSystem,
)

NOW = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _minimal_kwargs(**overrides) -> dict:
    base = {
        "id": "mem_test_1",
        "content": "The user's dog is named Otis.",
        "kind": MemoryKind.FACT,
        "subject": "the user's dog Otis",
        "source": Source(system=SourceSystem.FIXTURE, captured_at=NOW),
        "confidence": 0.9,
        "importance": 0.6,
    }
    base.update(overrides)
    return base


class TestRequiredFields:
    def test_minimal_valid_record(self):
        item = MemoryItem(**_minimal_kwargs())
        assert item.id == "mem_test_1"
        assert item.kind == MemoryKind.FACT
        assert item.schema_version == SCHEMA_VERSION
        assert item.sensitivity == Sensitivity.NORMAL
        assert item.supersedes == []

    def test_missing_required_field_rejected(self):
        kwargs = _minimal_kwargs()
        del kwargs["content"]
        with pytest.raises(ValidationError):
            MemoryItem(**kwargs)

    def test_unknown_field_rejected(self):
        with pytest.raises(ValidationError, match="Extra inputs"):
            MemoryItem(**_minimal_kwargs(), bogus="nope")


class TestRanges:
    @pytest.mark.parametrize("value", [-0.1, 1.5, 2.0])
    def test_confidence_out_of_range(self, value):
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(confidence=value))

    @pytest.mark.parametrize("value", [-0.5, 1.1])
    def test_importance_out_of_range(self, value):
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(importance=value))

    def test_content_too_long(self):
        long_content = "x" * (MAX_CONTENT_CHARS + 1)
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(content=long_content))

    def test_content_empty(self):
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(content=""))

    def test_id_empty(self):
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(id=""))


class TestKindEnum:
    @pytest.mark.parametrize(
        "kind",
        [
            MemoryKind.FACT,
            MemoryKind.PREFERENCE,
            MemoryKind.PROCEDURE,
            MemoryKind.EVENT,
        ],
    )
    def test_kind_accepts_enum(self, kind):
        item = MemoryItem(**_minimal_kwargs(kind=kind))
        assert item.kind == kind

    def test_kind_string_value_accepted(self):
        item = MemoryItem(**_minimal_kwargs(kind="event"))
        assert item.kind == MemoryKind.EVENT

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            MemoryItem(**_minimal_kwargs(kind="trivia"))


class TestKindSpecificRules:
    def test_correction_requires_supersedes(self):
        with pytest.raises(ValidationError, match="supersedes"):
            MemoryItem(**_minimal_kwargs(kind=MemoryKind.CORRECTION))

    def test_correction_with_supersedes_ok(self):
        item = MemoryItem(
            **_minimal_kwargs(kind=MemoryKind.CORRECTION, supersedes=["mem_old"])
        )
        assert item.supersedes == ["mem_old"]

    def test_preference_signal_only_on_preference_kind(self):
        with pytest.raises(ValidationError, match="preference_signal"):
            MemoryItem(
                **_minimal_kwargs(
                    kind=MemoryKind.FACT,
                    preference_signal=PreferenceSignal(axis="verbosity", value="concise"),
                )
            )

    def test_preference_signal_with_preference_kind_ok(self):
        item = MemoryItem(
            **_minimal_kwargs(
                kind=MemoryKind.PREFERENCE,
                preference_signal=PreferenceSignal(axis="verbosity", value="concise"),
            )
        )
        assert item.preference_signal is not None
        assert item.preference_signal.axis == "verbosity"


class TestSchemaVersion:
    def test_wrong_schema_version_rejected(self):
        with pytest.raises(ValidationError, match="schema_version"):
            MemoryItem(**_minimal_kwargs(), schema_version="9.99")

    def test_default_schema_version(self):
        item = MemoryItem(**_minimal_kwargs())
        assert item.schema_version == SCHEMA_VERSION


class TestSensitivity:
    def test_default_is_normal(self):
        assert MemoryItem(**_minimal_kwargs()).is_trainable() is True

    def test_redact_not_trainable(self):
        item = MemoryItem(**_minimal_kwargs(sensitivity=Sensitivity.REDACT))
        assert item.is_trainable() is False

    def test_sensitive_still_trainable(self):
        item = MemoryItem(**_minimal_kwargs(sensitivity=Sensitivity.SENSITIVE))
        assert item.is_trainable() is True


class TestOptionalStructure:
    def test_qa_pairs_accepted(self):
        item = MemoryItem(
            **_minimal_kwargs(
                qa_pairs=[QAPair(q="What is the dog's name?", a="Otis.")]
            )
        )
        assert len(item.qa_pairs) == 1
        assert item.qa_pairs[0].q == "What is the dog's name?"

    def test_entities_and_tags(self):
        item = MemoryItem(
            **_minimal_kwargs(entities=["Otis", "user"], tags=["pet", "personal"])
        )
        assert item.entities == ["Otis", "user"]
        assert item.tags == ["pet", "personal"]


class TestBatch:
    def test_round_trip_json(self):
        item = MemoryItem(**_minimal_kwargs())
        batch = MemoryBatch(items=[item])
        as_json = batch.model_dump_json()
        round_tripped = MemoryBatch.model_validate_json(as_json)
        assert round_tripped.items[0].id == "mem_test_1"

    def test_empty_batch_ok(self):
        batch = MemoryBatch(items=[])
        assert batch.items == []

    def test_batch_rejects_wrong_schema_version(self):
        with pytest.raises(ValidationError):
            MemoryBatch(schema_version="9.99", items=[])

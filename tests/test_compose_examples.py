"""Tests for the compose stage."""

from __future__ import annotations

from datetime import UTC, datetime

from dreamagent.compose import (
    EvalProbe,
    TrainingExample,
    memories_to_dataset,
    memory_to_examples,
)
from dreamagent.compose.templates import SYSTEM_PROMPT
from dreamagent.schema import (
    MemoryItem,
    MemoryKind,
    QAPair,
    Sensitivity,
    Source,
    SourceSystem,
)

NOW = datetime(2026, 5, 26, 12, 0, 0, tzinfo=UTC)


def _item(**overrides) -> MemoryItem:
    base = {
        "id": "mem_1",
        "content": "The user's dog is named Otis.",
        "kind": MemoryKind.FACT,
        "subject": "the user's dog",
        "source": Source(system=SourceSystem.FIXTURE, captured_at=NOW),
        "confidence": 0.9,
        "importance": 0.6,
        "entities": ["Otis"],
    }
    base.update(overrides)
    return MemoryItem(**base)


class TestFactKind:
    def test_produces_training_and_one_probe(self):
        train, probes = memory_to_examples(_item())
        assert len(train) >= 2
        assert len(probes) == 1

    def test_training_uses_system_prompt(self):
        train, _ = memory_to_examples(_item())
        for ex in train:
            assert ex.messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
            assert ex.messages[1]["role"] == "user"
            assert ex.messages[2]["role"] == "assistant"
            assert ex.messages[2]["content"] == "The user's dog is named Otis."

    def test_probe_uses_entities_for_expected(self):
        _, probes = memory_to_examples(_item())
        assert probes[0].expected_substrings == ["Otis"]

    def test_training_and_eval_use_different_phrasing(self):
        train, probes = memory_to_examples(_item())
        training_questions = {ex.messages[1]["content"] for ex in train}
        assert probes[0].question not in training_questions

    def test_lineage_preserved(self):
        train, probes = memory_to_examples(_item(id="mem_xyz"))
        for ex in train:
            assert ex.source_memory_id == "mem_xyz"
        assert probes[0].source_memory_id == "mem_xyz"


class TestPreferenceKind:
    def test_produces_examples(self):
        train, probes = memory_to_examples(
            _item(
                kind=MemoryKind.PREFERENCE,
                content="Concise, no preamble.",
                subject="response style",
                entities=[],
            )
        )
        assert len(train) >= 2
        assert len(probes) == 1
        # entities=[] forces fallback to content-head; head is "Concise, no preamble"
        assert probes[0].expected_substrings == ["Concise, no preamble"]


class TestProcedureKind:
    def test_produces_examples(self):
        train, probes = memory_to_examples(
            _item(
                kind=MemoryKind.PROCEDURE,
                content="Run uv run pytest -q",
                subject="running tests",
                entities=["pytest", "uv"],
            )
        )
        assert len(train) >= 2
        assert len(probes) == 1
        assert "pytest" in probes[0].expected_substrings


class TestEventKind:
    def test_produces_examples(self):
        train, probes = memory_to_examples(
            _item(
                kind=MemoryKind.EVENT,
                content="On 2026-05-26 the user approved the V1 plan.",
                subject="plan approval",
                entities=["V1 plan"],
            )
        )
        assert len(train) >= 1
        assert len(probes) == 1


class TestCorrectionKind:
    def test_requires_supersedes(self):
        item = _item(
            kind=MemoryKind.CORRECTION,
            content="Actually it's August 18th, not 14th.",
            subject="anniversary",
            supersedes=["mem_old"],
        )
        train, probes = memory_to_examples(item)
        assert len(train) >= 1
        assert len(probes) == 1


class TestQAPairsTakePriority:
    def test_explicit_qa_pairs_used_directly(self):
        item = _item(
            qa_pairs=[
                QAPair(q="Dog name?", a="Otis."),
                QAPair(q="Breed?", a="Golden retriever."),
            ]
        )
        train, probes = memory_to_examples(item)
        assert len(train) == 2
        assert len(probes) == 0
        assert train[0].messages[1]["content"] == "Dog name?"
        assert train[0].messages[2]["content"] == "Otis."
        assert train[0].template == "explicit:qa_pair"


class TestRedactedSkipped:
    def test_redacted_produces_nothing(self):
        item = _item(sensitivity=Sensitivity.REDACT)
        train, probes = memory_to_examples(item)
        assert train == []
        assert probes == []


class TestSupersededExclusion:
    def test_superseded_memory_dropped(self):
        old = _item(id="mem_old", content="Anniversary is Aug 14.", subject="anniversary")
        correction = _item(
            id="mem_new",
            kind=MemoryKind.CORRECTION,
            content="Anniversary is Aug 18.",
            subject="anniversary",
            supersedes=["mem_old"],
        )
        train, _probes = memories_to_dataset([old, correction])
        sources = {ex.source_memory_id for ex in train}
        assert "mem_old" not in sources
        assert "mem_new" in sources


class TestDatasetConcatenation:
    def test_multiple_memories_concatenate(self):
        a = _item(id="mem_a", subject="dog A", entities=["Otis"])
        b = _item(id="mem_b", subject="dog B", entities=["Rex"])
        train, probes = memories_to_dataset([a, b])
        sources = {ex.source_memory_id for ex in train}
        assert sources == {"mem_a", "mem_b"}
        assert len(probes) == 2


class TestTypes:
    def test_types(self):
        train, probes = memory_to_examples(_item())
        assert isinstance(train[0], TrainingExample)
        assert isinstance(probes[0], EvalProbe)

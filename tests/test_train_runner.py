"""Tests for the train runner — argument construction and metadata writing only.

The actual training subprocess is NOT invoked here (that takes minutes and
needs the model downloaded); the end-to-end smoke test lives separately and
is gated behind a `-m slow` marker.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from dreamagent.compose.examples import TrainingExample
from dreamagent.compose.mix import RehearsalMix
from dreamagent.train.config import TrainConfig
from dreamagent.train.runner import (
    _build_command,
    _split_train_valid,
    _write_jsonl,
    _write_metadata,
)


def _ex(id_: str) -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"q{id_}"},
            {"role": "assistant", "content": f"a{id_}"},
        ],
        source_memory_id=id_,
        template="test",
    )


class TestSplitTrainValid:
    def test_empty(self):
        train, valid = _split_train_valid([], 0.1, seed=1)
        assert train == [] and valid == []

    def test_single_duplicated(self):
        items = [_ex("a")]
        train, valid = _split_train_valid(items, 0.1, seed=1)
        assert train == valid == items

    def test_typical_split_at_least_one_in_valid(self):
        items = [_ex(f"id_{i}") for i in range(10)]
        train, valid = _split_train_valid(items, 0.1, seed=1)
        assert len(train) + len(valid) == 10
        assert len(valid) >= 1

    def test_deterministic_with_seed(self):
        items = [_ex(f"id_{i}") for i in range(20)]
        a_train, a_valid = _split_train_valid(items, 0.2, seed=42)
        b_train, b_valid = _split_train_valid(items, 0.2, seed=42)
        assert [e.source_memory_id for e in a_train] == [
            e.source_memory_id for e in b_train
        ]
        assert [e.source_memory_id for e in a_valid] == [
            e.source_memory_id for e in b_valid
        ]


class TestWriteJsonl:
    def test_writes_one_per_line(self, tmp_path: Path):
        items = [_ex("a"), _ex("b"), _ex("c")]
        path = tmp_path / "out.jsonl"
        _write_jsonl(items, path)
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        record = json.loads(lines[0])
        assert "messages" in record
        assert record["messages"][1]["content"] == "qa"


class TestBuildCommand:
    def test_includes_required_flags(self, tmp_path: Path):
        cfg = TrainConfig(base_model="dummy", iters=50)
        cmd = _build_command("python", tmp_path / "data", tmp_path / "adapter", cfg)
        assert "lora" in cmd
        assert "--train" in cmd
        assert "--model" in cmd
        assert "dummy" in cmd
        assert "--iters" in cmd
        assert "50" in cmd

    def test_mask_prompt_default_on(self, tmp_path: Path):
        cmd = _build_command("py", tmp_path / "d", tmp_path / "a", TrainConfig())
        assert "--mask-prompt" in cmd

    def test_mask_prompt_off(self, tmp_path: Path):
        cmd = _build_command(
            "py", tmp_path / "d", tmp_path / "a", TrainConfig(mask_prompt=False)
        )
        assert "--mask-prompt" not in cmd


class TestMetadata:
    def test_metadata_records_everything(self, tmp_path: Path):
        cfg = TrainConfig(base_model="dummy", iters=50, num_layers=4)
        mix = RehearsalMix(
            examples=[_ex("m1"), _ex("m2")],
            composition={"today": 2, "replay": 0, "anchor": 0},
        )
        started = datetime(2026, 5, 26, 22, 0, 0, tzinfo=UTC)
        completed = datetime(2026, 5, 26, 22, 5, 0, tzinfo=UTC)

        metadata_path = tmp_path / "metadata.json"
        meta = _write_metadata(
            metadata_path,
            config=cfg,
            mix=mix,
            train_count=2,
            valid_count=1,
            started_at=started,
            completed_at=completed,
            source_memory_ids=["m1", "m2"],
        )

        assert meta["duration_seconds"] == 300.0
        assert meta["config"]["base_model"] == "dummy"
        assert meta["config"]["num_layers"] == 4
        assert meta["mix_composition"]["today"] == 2
        assert meta["source_memory_ids"] == ["m1", "m2"]
        assert "python" in meta["versions"]

        on_disk = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert on_disk == meta


def test_empty_mix_rejected(tmp_path: Path):
    from dreamagent.train.runner import train_adapter

    empty = RehearsalMix(examples=[], composition={"today": 0, "replay": 0, "anchor": 0})
    with pytest.raises(ValueError, match="empty mix"):
        train_adapter(empty, TrainConfig(), tmp_path)

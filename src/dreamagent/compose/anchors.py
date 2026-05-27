"""Loaders for the general-capability anchor set and the held-out eval probes.

Anchors are the "you are still you" set — fixed instruction-following examples
that get mixed into every nightly training run to prevent identity / capability
drift. The eval set is held out from training.

Both sets live as JSONL under fixtures/anchors/. They're shipped with the repo
and considered part of the project's source code.
"""

from __future__ import annotations

import json
from pathlib import Path

from dreamagent.compose.examples import EvalProbe, TrainingExample
from dreamagent.ingest.fixture import _fixtures_root

ANCHORS_DIR_NAME = "anchors"
ANCHOR_TRAIN_FILE = "general_anchor.jsonl"
ANCHOR_EVAL_FILE = "general_eval.jsonl"


def _anchors_path(filename: str) -> Path:
    return _fixtures_root() / ANCHORS_DIR_NAME / filename


def load_general_anchors() -> list[TrainingExample]:
    """Load the general-capability anchor training examples."""
    path = _anchors_path(ANCHOR_TRAIN_FILE)
    out: list[TrainingExample] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e.msg}") from e
            out.append(
                TrainingExample(
                    messages=record["messages"],
                    source_memory_id=record["id"],
                    template="anchor:general",
                )
            )
    return out


def load_general_eval_probes() -> list[EvalProbe]:
    """Load the general-capability eval probes."""
    path = _anchors_path(ANCHOR_EVAL_FILE)
    out: list[EvalProbe] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e.msg}") from e
            out.append(
                EvalProbe(
                    question=record["question"],
                    expected_substrings=record["expected_substrings"],
                    source_memory_id=record["id"],
                    template="anchor:general_eval",
                )
            )
    return out

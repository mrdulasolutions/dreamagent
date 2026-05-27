"""Tests for the general-capability anchor loaders."""

from __future__ import annotations

from dreamagent.compose.anchors import load_general_anchors, load_general_eval_probes
from dreamagent.compose.examples import EvalProbe, TrainingExample


def test_anchors_loads_at_least_30():
    anchors = load_general_anchors()
    assert len(anchors) >= 30
    assert all(isinstance(a, TrainingExample) for a in anchors)


def test_anchors_have_system_user_assistant():
    for a in load_general_anchors():
        roles = [m["role"] for m in a.messages]
        assert roles == ["system", "user", "assistant"]


def test_anchors_have_lineage():
    for a in load_general_anchors():
        assert a.template == "anchor:general"
        assert a.source_memory_id.startswith("anchor_")


def test_eval_probes_loads_at_least_10():
    probes = load_general_eval_probes()
    assert len(probes) >= 10
    assert all(isinstance(p, EvalProbe) for p in probes)


def test_eval_probes_have_expected_substrings():
    for p in load_general_eval_probes():
        assert len(p.expected_substrings) >= 1
        assert all(isinstance(s, str) and s for s in p.expected_substrings)


def test_eval_probe_ids_disjoint_from_training():
    train_ids = {a.source_memory_id for a in load_general_anchors()}
    eval_ids = {p.source_memory_id for p in load_general_eval_probes()}
    assert train_ids.isdisjoint(eval_ids)

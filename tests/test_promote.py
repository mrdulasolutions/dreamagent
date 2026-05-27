"""Tests for the eval gate and snapshot system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from dreamagent.compose.examples import EvalProbe
from dreamagent.eval.runner import EvalReport, EvalResult
from dreamagent.promote.gate import EvalGateConfig, GateDecision, decide
from dreamagent.promote.snapshot import (
    current_live,
    list_snapshots,
    rollback_to,
    snapshot_run,
)
from dreamagent.train.runner import TrainResult


def _report(n_pass: int, n_fail: int) -> EvalReport:
    results = []
    for i in range(n_pass):
        probe = EvalProbe(
            question=f"q{i}", expected_substrings=["x"], source_memory_id=f"p{i}", template="t"
        )
        results.append(
            EvalResult(probe=probe, response="x", passed=True, matched_substring="x")
        )
    for i in range(n_fail):
        probe = EvalProbe(
            question=f"q{i}", expected_substrings=["x"], source_memory_id=f"f{i}", template="t"
        )
        results.append(
            EvalResult(probe=probe, response="y", passed=False, matched_substring=None)
        )
    return EvalReport(results=results, model="m", adapter_path=None)


class TestGateDecision:
    def test_promote_clean(self):
        personal = _report(8, 2)  # 0.80 ≥ 0.70
        gen_base = _report(9, 1)  # 0.90
        gen_adapter = _report(9, 1)  # 0.90 → 0 regression
        r = decide(personal, gen_base, gen_adapter)
        assert r.decision is GateDecision.PROMOTE

    def test_reject_low_personal_recall(self):
        personal = _report(5, 5)  # 0.50 < 0.70
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        r = decide(personal, gen_base, gen_adapter)
        assert r.decision is GateDecision.REJECT
        assert any("personal recall" in s for s in r.reasons)

    def test_reject_high_regression(self):
        personal = _report(9, 1)
        gen_base = _report(10, 0)  # 1.00
        gen_adapter = _report(7, 3)  # 0.70 → 0.30 regression
        r = decide(personal, gen_base, gen_adapter)
        assert r.decision is GateDecision.REJECT
        assert any("regression" in s for s in r.reasons)

    def test_promote_with_warning_moderate_regression(self):
        personal = _report(9, 1)
        gen_base = _report(10, 0)  # 1.00
        gen_adapter = _report(95, 5)  # 0.95 → 0.05 regression (in 0.03-0.08 band)
        r = decide(personal, gen_base, gen_adapter)
        assert r.decision is GateDecision.PROMOTE_WITH_WARNING

    def test_validation_defaults_loose(self):
        cfg = EvalGateConfig.validation_defaults()
        personal = _report(4, 6)  # 0.40 ≥ 0.30
        gen_base = _report(8, 2)
        gen_adapter = _report(8, 2)
        r = decide(personal, gen_base, gen_adapter, cfg)
        assert r.decision is GateDecision.PROMOTE


def _fake_train_result(tmp_path: Path) -> TrainResult:
    """Build a TrainResult with on-disk adapter + metadata for snapshot tests."""
    run_dir = tmp_path / "run"
    adapter_dir = run_dir / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapters.safetensors").write_bytes(b"fake-weights")
    (adapter_dir / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    meta_path = run_dir / "metadata.json"
    meta_path.write_text(json.dumps({"x": 1}) + "\n", encoding="utf-8")
    return TrainResult(
        run_dir=run_dir,
        adapter_path=adapter_dir / "adapters.safetensors",
        metadata_path=meta_path,
        metadata={"x": 1},
    )


class TestSnapshotting:
    def test_promote_creates_snapshot_and_live_link(self, tmp_path: Path):
        snapshots = tmp_path / "snapshots"
        tr = _fake_train_result(tmp_path)
        personal = _report(8, 2)
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        gate = decide(personal, gen_base, gen_adapter)

        snap = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        assert snap.decision is GateDecision.PROMOTE
        assert (snap.dir / "adapter" / "adapters.safetensors").exists()
        assert (snap.dir / "metadata.json").exists()
        assert (snap.dir / "gate.json").exists()
        assert (snapshots / "live").is_symlink()
        assert (snapshots / "live").readlink().name == snap.name

    def test_reject_lands_in_rejected_and_skips_live(self, tmp_path: Path):
        snapshots = tmp_path / "snapshots"
        tr = _fake_train_result(tmp_path)
        personal = _report(2, 8)  # too low
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        gate = decide(personal, gen_base, gen_adapter)

        snap = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        assert snap.decision is GateDecision.REJECT
        assert "rejected" in str(snap.dir)
        assert not (snapshots / "live").exists()

    def test_current_live_returns_promoted(self, tmp_path: Path):
        snapshots = tmp_path / "snapshots"
        tr = _fake_train_result(tmp_path)
        personal = _report(9, 1)
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        gate = decide(personal, gen_base, gen_adapter)
        snap = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        live = current_live(snapshots)
        assert live is not None
        assert live.name == snap.name

    def test_rollback_to_named_snapshot(self, tmp_path: Path):
        import time

        snapshots = tmp_path / "snapshots"
        tr = _fake_train_result(tmp_path)
        personal = _report(9, 1)
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        gate = decide(personal, gen_base, gen_adapter)

        snap_a = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        time.sleep(1.1)  # ensure timestamp-based names differ
        snap_b = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        assert current_live(snapshots).name == snap_b.name

        rolled = rollback_to(snapshots, snap_a.name)
        assert rolled.name == snap_a.name
        assert current_live(snapshots).name == snap_a.name

    def test_rollback_nonexistent_raises(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            rollback_to(tmp_path / "snapshots", "nope")

    def test_list_snapshots_orders_newest_first(self, tmp_path: Path):
        import time

        snapshots = tmp_path / "snapshots"
        tr = _fake_train_result(tmp_path)
        personal = _report(9, 1)
        gen_base = _report(9, 1)
        gen_adapter = _report(9, 1)
        gate = decide(personal, gen_base, gen_adapter)

        a = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        time.sleep(1.1)
        b = snapshot_run(
            train_result=tr,
            personal_eval=personal,
            general_base_eval=gen_base,
            general_adapter_eval=gen_adapter,
            gate_result=gate,
            snapshots_dir=snapshots,
        )
        snaps = list_snapshots(snapshots)
        assert [s.name for s in snaps] == [b.name, a.name]

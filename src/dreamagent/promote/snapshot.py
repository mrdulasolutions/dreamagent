"""Adapter snapshots — store each nightly promotion with full lineage,
and maintain a `live` pointer for serving + a one-command rollback.

Layout under `snapshots_dir`:

    snapshots_dir/
        2026-05-26T22-15-00Z/
            adapter/
                adapters.safetensors
                adapter_config.json
            metadata.json           # from the train run
            eval_personal.json
            eval_general_base.json
            eval_general_adapter.json
            gate.json
        live -> 2026-05-26T22-15-00Z   # symlink to most-recently-promoted

Rejected runs are NOT promoted but the snapshot dir is still written under
`rejected/` so we can inspect what went wrong.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from dreamagent.eval.runner import EvalReport
from dreamagent.promote.gate import GateDecision, GateResult
from dreamagent.train.runner import TrainResult

SNAPSHOT_TS_FMT = "%Y-%m-%dT%H-%M-%SZ"


@dataclass(frozen=True, slots=True)
class AdapterSnapshot:
    name: str
    dir: Path
    decision: GateDecision

    @property
    def adapter_path(self) -> Path:
        return self.dir / "adapter" / "adapters.safetensors"

    @property
    def metadata_path(self) -> Path:
        return self.dir / "metadata.json"


def _now_name() -> str:
    return datetime.now(UTC).strftime(SNAPSHOT_TS_FMT)


def _copy_or_link_adapter(src_adapter_dir: Path, dst_adapter_dir: Path) -> None:
    """Copy the adapter directory's contents into dst (full copy, not symlink,
    so the snapshot is self-contained and safe to retain after run_dir is gone).
    """
    if not src_adapter_dir.exists():
        raise FileNotFoundError(f"adapter source not found: {src_adapter_dir}")
    dst_adapter_dir.mkdir(parents=True, exist_ok=True)
    for child in src_adapter_dir.iterdir():
        if child.is_file():
            shutil.copy2(child, dst_adapter_dir / child.name)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _atomic_symlink(link: Path, target_name: str) -> None:
    """Replace `link` (a symlink in the same dir) so it points at `target_name`."""
    tmp = link.with_name(link.name + ".tmp")
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    tmp.symlink_to(target_name)
    tmp.replace(link)


def snapshot_run(
    *,
    train_result: TrainResult,
    personal_eval: EvalReport,
    general_base_eval: EvalReport,
    general_adapter_eval: EvalReport,
    gate_result: GateResult,
    snapshots_dir: Path,
) -> AdapterSnapshot:
    """Persist a completed nightly run as a snapshot.

    Promoted snapshots land in `snapshots_dir/<timestamp>/` and the `live`
    symlink updates to point at it. Rejected snapshots land in
    `snapshots_dir/rejected/<timestamp>/` and `live` is NOT touched.
    """
    snapshots_dir = Path(snapshots_dir)
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    name = _now_name()
    if gate_result.decision is GateDecision.REJECT:
        snap_dir = snapshots_dir / "rejected" / name
    else:
        snap_dir = snapshots_dir / name
    snap_dir.mkdir(parents=True, exist_ok=True)

    _copy_or_link_adapter(train_result.adapter_path.parent, snap_dir / "adapter")

    shutil.copy2(train_result.metadata_path, snap_dir / "metadata.json")
    _write_json(snap_dir / "eval_personal.json", personal_eval.to_dict())
    _write_json(snap_dir / "eval_general_base.json", general_base_eval.to_dict())
    _write_json(snap_dir / "eval_general_adapter.json", general_adapter_eval.to_dict())
    _write_json(snap_dir / "gate.json", gate_result.to_dict())

    if gate_result.decision is not GateDecision.REJECT:
        _atomic_symlink(snapshots_dir / "live", name)

    return AdapterSnapshot(name=name, dir=snap_dir, decision=gate_result.decision)


def current_live(snapshots_dir: Path) -> AdapterSnapshot | None:
    """Return the currently-live snapshot, or None if no adapter is promoted yet."""
    live = Path(snapshots_dir) / "live"
    if not live.is_symlink() and not live.exists():
        return None
    target = (live.parent / live.readlink()).resolve()
    gate_data = json.loads((target / "gate.json").read_text(encoding="utf-8"))
    return AdapterSnapshot(
        name=target.name, dir=target, decision=GateDecision(gate_data["decision"])
    )


def rollback_to(snapshots_dir: Path, name: str) -> AdapterSnapshot:
    """Point `live` at a prior snapshot by name. Raises if it doesn't exist."""
    snapshots_dir = Path(snapshots_dir)
    target_dir = snapshots_dir / name
    if not target_dir.is_dir():
        raise FileNotFoundError(f"snapshot {name!r} not found under {snapshots_dir}")
    gate_data = json.loads((target_dir / "gate.json").read_text(encoding="utf-8"))
    _atomic_symlink(snapshots_dir / "live", name)
    return AdapterSnapshot(
        name=name, dir=target_dir, decision=GateDecision(gate_data["decision"])
    )


def list_snapshots(snapshots_dir: Path) -> list[AdapterSnapshot]:
    """List all promoted snapshots (not rejected ones), newest first."""
    snapshots_dir = Path(snapshots_dir)
    if not snapshots_dir.is_dir():
        return []
    out: list[AdapterSnapshot] = []
    for entry in sorted(snapshots_dir.iterdir(), reverse=True):
        if entry.is_dir() and entry.name not in ("rejected",) and not entry.is_symlink():
            gate_file = entry / "gate.json"
            if gate_file.exists():
                gate_data = json.loads(gate_file.read_text(encoding="utf-8"))
                out.append(
                    AdapterSnapshot(
                        name=entry.name, dir=entry, decision=GateDecision(gate_data["decision"])
                    )
                )
    return out

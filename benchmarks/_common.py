"""Shared helpers for benchmark scripts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Standard shape for any benchmark's output."""

    name: str
    snapshot: str
    timestamp_utc: str
    metrics: dict
    detail: dict | None = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "name": self.name,
                "snapshot": self.snapshot,
                "timestamp_utc": self.timestamp_utc,
                "metrics": self.metrics,
                "detail": self.detail,
            },
            indent=2,
        )


def make_result(
    name: str, snapshot: Path | str, metrics: dict, detail: dict | None = None
) -> BenchmarkResult:
    return BenchmarkResult(
        name=name,
        snapshot=str(snapshot),
        timestamp_utc=datetime.now(UTC).isoformat(),
        metrics=metrics,
        detail=detail,
    )


def resolve_adapter_dir(snapshot: Path) -> Path:
    """A snapshot dir contains `adapter/`; resolve through `live` symlinks."""
    snapshot = Path(snapshot)
    if snapshot.is_symlink():
        snapshot = snapshot.resolve()
    adapter_dir = snapshot / "adapter"
    if not adapter_dir.exists():
        raise FileNotFoundError(
            f"no adapter/ under snapshot {snapshot} — is this a DreamAgent snapshot directory?"
        )
    return adapter_dir


def save_result(result: BenchmarkResult, out_dir: Path | str = "benchmarks/results") -> Path:
    """Write the result JSON to `out_dir/<name>-<timestamp>.json`. Returns path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = result.timestamp_utc.replace(":", "-").replace("+00:00", "Z")
    path = out_dir / f"{result.name}-{ts}.json"
    path.write_text(result.to_json() + "\n", encoding="utf-8")
    return path

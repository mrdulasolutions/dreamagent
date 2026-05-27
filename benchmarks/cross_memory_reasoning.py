"""Cross-memory reasoning benchmark — the parametric advantage.

These probes ask questions that REQUIRE synthesizing 2-3 memories to answer
correctly. A vector-retrieval system that returns top-k chunks individually
cannot answer them in one shot; it would have to do multi-hop retrieval +
a separate aggregation pass.

A dreamed model has seen all the memories together during training. If the
parametric-memory thesis holds, this is where DreamAgent should show
material advantage over retrieval.

The probe set lives at `benchmarks/probes/cross_memory_reasoning.jsonl`.
Each probe declares which memory IDs it requires (`requires_memory_ids`),
so a retrieval-based baseline can be evaluated under fair "top-k must
include all required" conditions.

Example:
    python -m benchmarks.cross_memory_reasoning \\
        --snapshot runs/snapshots/live \\
        --base-model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from dreamagent.compose.examples import EvalProbe
from dreamagent.eval import run_eval

PROBES_PATH = Path(__file__).parent / "probes" / "cross_memory_reasoning.jsonl"


def load_probes() -> list[EvalProbe]:
    out: list[EvalProbe] = []
    with PROBES_PATH.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            rec = json.loads(stripped)
            out.append(
                EvalProbe(
                    question=rec["question"],
                    expected_substrings=rec["expected_substrings"],
                    source_memory_id=rec["id"],
                    template="cross_memory_reasoning",
                )
            )
    return out


def run(snapshot: Path, base_model: str, eval_max_tokens: int = 64) -> dict:
    probes = load_probes()
    adapter_dir = resolve_adapter_dir(Path(snapshot))

    base_report = run_eval(
        probes, model_repo=base_model, adapter_path=None, max_tokens=eval_max_tokens
    )
    adapter_report = run_eval(
        probes, model_repo=base_model, adapter_path=adapter_dir, max_tokens=eval_max_tokens
    )

    delta = adapter_report.pass_rate - base_report.pass_rate

    return {
        "probe_count": base_report.total,
        "base_pass_rate": base_report.pass_rate,
        "adapter_pass_rate": adapter_report.pass_rate,
        "improvement_points": delta,
        "improvement_percent": delta * 100,
        "base_model": base_model,
        "note": (
            "Cross-memory probes require synthesis across 2-3 memories. "
            "If adapter > base, parametric memory delivers reasoning that "
            "single-pass retrieval cannot."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--eval-max-tokens", type=int, default=64)
    args = p.parse_args()

    metrics = run(args.snapshot, args.base_model, args.eval_max_tokens)
    result = make_result("cross_memory_reasoning", args.snapshot, metrics)
    print(result.to_json())
    save_result(result)


if __name__ == "__main__":
    main()

"""General-capability benchmark.

Runs the shipped general-eval anchor probes against base model alone, then
against base + adapter. Reports regression in percentage points.

This is the catastrophic-forgetting yardstick: if base passes N/M probes and
adapter passes N'/M, regression = (N - N') / M.

Example:
    python -m benchmarks.general_capability \\
        --snapshot runs/snapshots/live \\
        --base-model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from dreamagent.compose import load_general_eval_probes
from dreamagent.eval import run_eval


def run(snapshot: Path, base_model: str, eval_max_tokens: int = 48) -> dict:
    probes = load_general_eval_probes()
    adapter_dir = resolve_adapter_dir(Path(snapshot))

    base_report = run_eval(
        probes, model_repo=base_model, adapter_path=None, max_tokens=eval_max_tokens
    )
    adapter_report = run_eval(
        probes, model_repo=base_model, adapter_path=adapter_dir, max_tokens=eval_max_tokens
    )

    regression = base_report.pass_rate - adapter_report.pass_rate

    return {
        "probe_count": base_report.total,
        "base_pass_rate": base_report.pass_rate,
        "adapter_pass_rate": adapter_report.pass_rate,
        "regression_points": regression,
        "regression_percent": regression * 100,
        "base_model": base_model,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--eval-max-tokens", type=int, default=48)
    args = p.parse_args()

    metrics = run(args.snapshot, args.base_model, args.eval_max_tokens)
    result = make_result("general_capability", args.snapshot, metrics)
    print(result.to_json())
    save_result(result)


if __name__ == "__main__":
    main()

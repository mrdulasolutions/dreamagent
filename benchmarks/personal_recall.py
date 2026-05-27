"""Personal-recall benchmark.

For every memory in the source set, generate a held-out recall probe (using
the compose-stage templates), evaluate the dreamed model with the adapter
loaded, and report pass rate.

This is the same protocol the dream pipeline uses internally. Surfacing it as
a separate benchmark lets you re-measure a snapshot after the fact (e.g.,
after the base model is updated, or to compare two snapshots against the
same probes).

Example:
    python -m benchmarks.personal_recall \\
        --memories fixture:v1_baseline \\
        --snapshot runs/snapshots/live \\
        --base-model mlx-community/Llama-3.2-1B-Instruct-4bit
"""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from dreamagent.compose import memories_to_dataset
from dreamagent.eval import run_eval
from dreamagent.ingest import FixtureConnector, JSONLConnector


def run(
    memories: str,
    snapshot: Path,
    base_model: str,
    eval_max_tokens: int = 48,
) -> dict:
    connector = (
        FixtureConnector(memories.removeprefix("fixture:"))
        if memories.startswith("fixture:")
        else JSONLConnector(memories)
    )
    items = list(connector.iter_memories())
    _, probes = memories_to_dataset(items)

    adapter_dir = resolve_adapter_dir(Path(snapshot))
    report = run_eval(
        probes,
        model_repo=base_model,
        adapter_path=adapter_dir,
        max_tokens=eval_max_tokens,
    )

    return {
        "total_probes": report.total,
        "passed": report.passed,
        "pass_rate": report.pass_rate,
        "memories_source": memories,
        "base_model": base_model,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memories", required=True, help="fixture:<name> or path to .jsonl")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--eval-max-tokens", type=int, default=48)
    args = p.parse_args()

    metrics = run(args.memories, args.snapshot, args.base_model, args.eval_max_tokens)
    result = make_result("personal_recall", args.snapshot, metrics)
    print(result.to_json())
    save_result(result)


if __name__ == "__main__":
    main()

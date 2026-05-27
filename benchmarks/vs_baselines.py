"""Head-to-head: DreamAgent vs vector-retrieval baseline.

Same 50 memories, same probe sets, same base model for generation.
The only difference: DreamAgent has memories in weights, baseline has
them in context via top-k retrieval.

Reports per-probe results, aggregate metrics, and a comparison table.

Example:
    python -m benchmarks.vs_baselines \\
        --memories fixture:v1_baseline \\
        --snapshot runs/snapshots/live \\
        --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit

Probe sets evaluated:
    - personal_recall (auto-generated from memories)
    - cross_memory_reasoning (from benchmarks/probes/)
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from benchmarks.cross_memory_reasoning import load_probes as load_xmem_probes
from dreamagent.compose import memories_to_dataset
from dreamagent.compose.examples import EvalProbe
from dreamagent.eval import run_eval
from dreamagent.eval.runner import score
from dreamagent.ingest import FixtureConnector, JSONLConnector


@dataclass(slots=True)
class ComparisonRow:
    probe_id: str
    question: str
    expected_substrings: list[str]
    dreamagent_response: str
    dreamagent_passed: bool
    baseline_response: str
    baseline_passed: bool
    baseline_retrieved_ids: list[str]
    baseline_total_ms: float


def _connector(source: str):
    if source.startswith("fixture:"):
        return FixtureConnector(source.removeprefix("fixture:"))
    return JSONLConnector(source)


def _measure_dreamagent(
    probes: list[EvalProbe], base_model: str, adapter_dir: Path, max_tokens: int
) -> dict[str, dict]:
    """Run probes through DreamAgent (base + adapter, no retrieval). Returns
    {probe_id: {response, passed}}."""
    report = run_eval(
        probes,
        model_repo=base_model,
        adapter_path=adapter_dir,
        max_tokens=max_tokens,
    )
    return {
        r.probe.source_memory_id: {"response": r.response, "passed": r.passed}
        for r in report.results
    }


def _measure_baseline(
    probes: list[EvalProbe], memories_source: str, base_model: str, top_k: int, max_tokens: int
) -> dict[str, dict]:
    """Run probes through vector-retrieval baseline. Returns
    {probe_id: {response, passed, retrieved_ids, total_ms}}."""
    from benchmarks.baselines.retrieval import RetrievalBaseline

    baseline = RetrievalBaseline.from_source(
        memories_source, base_model=base_model, top_k=top_k, max_tokens=max_tokens
    )

    out: dict[str, dict] = {}
    for p in probes:
        result = baseline.query(p.question)
        passed, _ = score(result.answer, p.expected_substrings)
        out[p.source_memory_id] = {
            "response": result.answer,
            "passed": passed,
            "retrieved_ids": result.retrieved_ids,
            "total_ms": result.total_ms,
            "embed_ms": result.embed_ms,
            "search_ms": result.search_ms,
            "generate_ms": result.generate_ms,
        }
    return out


def run(
    memories: str,
    snapshot: Path,
    base_model: str,
    top_k: int = 5,
    eval_max_tokens: int = 64,
) -> dict:
    """Run the head-to-head and return metrics + per-probe detail."""
    # Build the probe sets
    items = list(_connector(memories).iter_memories())
    _, personal_probes = memories_to_dataset(items)
    xmem_probes = load_xmem_probes()

    adapter_dir = resolve_adapter_dir(Path(snapshot))

    # Measure DreamAgent
    da_personal = _measure_dreamagent(personal_probes, base_model, adapter_dir, eval_max_tokens)
    da_xmem = _measure_dreamagent(xmem_probes, base_model, adapter_dir, eval_max_tokens)

    # Measure baseline
    bl_personal = _measure_baseline(personal_probes, memories, base_model, top_k, eval_max_tokens)
    bl_xmem = _measure_baseline(xmem_probes, memories, base_model, top_k, eval_max_tokens)

    def _aggregate(da: dict, bl: dict, probes: list[EvalProbe]) -> dict:
        n = len(probes)
        da_passed = sum(1 for p in probes if da[p.source_memory_id]["passed"])
        bl_passed = sum(1 for p in probes if bl[p.source_memory_id]["passed"])
        bl_total_ms = [bl[p.source_memory_id]["total_ms"] for p in probes]
        return {
            "n": n,
            "dreamagent_pass_rate": da_passed / n if n else 0.0,
            "baseline_pass_rate": bl_passed / n if n else 0.0,
            "delta_pp": (da_passed - bl_passed) / n * 100 if n else 0.0,
            "baseline_total_ms_mean": sum(bl_total_ms) / len(bl_total_ms) if bl_total_ms else 0.0,
            "baseline_total_ms_min": min(bl_total_ms) if bl_total_ms else 0.0,
            "baseline_total_ms_max": max(bl_total_ms) if bl_total_ms else 0.0,
        }

    return {
        "memories_source": memories,
        "snapshot": str(snapshot),
        "base_model": base_model,
        "top_k": top_k,
        "personal_recall": _aggregate(da_personal, bl_personal, personal_probes),
        "cross_memory_reasoning": _aggregate(da_xmem, bl_xmem, xmem_probes),
        "headline": {
            "dreamagent_xmem_pass_rate": _aggregate(da_xmem, bl_xmem, xmem_probes)[
                "dreamagent_pass_rate"
            ],
            "baseline_xmem_pass_rate": _aggregate(da_xmem, bl_xmem, xmem_probes)[
                "baseline_pass_rate"
            ],
            "xmem_advantage_pp": _aggregate(da_xmem, bl_xmem, xmem_probes)["delta_pp"],
        },
        "detail": {
            "da_personal": da_personal,
            "da_xmem": da_xmem,
            "bl_personal": bl_personal,
            "bl_xmem": bl_xmem,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memories", required=True, help="fixture:<name> or path to .jsonl")
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--eval-max-tokens", type=int, default=64)
    args = p.parse_args()

    metrics = run(
        args.memories,
        args.snapshot,
        args.base_model,
        top_k=args.top_k,
        eval_max_tokens=args.eval_max_tokens,
    )

    # Print summary
    pr = metrics["personal_recall"]
    xr = metrics["cross_memory_reasoning"]
    print()
    print("HEAD-TO-HEAD: DreamAgent (parametric) vs Vector-Retrieval (control)")
    print("=" * 70)
    print(f"  Base model:    {args.base_model}")
    print(f"  Memories:      {args.memories}")
    print(f"  Top-k:         {args.top_k}")
    print()
    print(f"{'Probe set':<28} {'DreamAgent':>11} {'Baseline':>11} {'Δ':>10}")
    print(f"{'-' * 28} {'-' * 11} {'-' * 11} {'-' * 10}")
    print(
        f"{'personal_recall':<28} "
        f"{pr['dreamagent_pass_rate']:>10.1%} "
        f"{pr['baseline_pass_rate']:>10.1%} "
        f"{pr['delta_pp']:>+9.1f}pp"
    )
    print(
        f"{'cross_memory_reasoning':<28} "
        f"{xr['dreamagent_pass_rate']:>10.1%} "
        f"{xr['baseline_pass_rate']:>10.1%} "
        f"{xr['delta_pp']:>+9.1f}pp"
    )
    print()
    print(
        f"Baseline latency (mean):  personal {pr['baseline_total_ms_mean']:.0f}ms · "
        f"xmem {xr['baseline_total_ms_mean']:.0f}ms"
    )
    print()
    print(json.dumps(
        {k: v for k, v in metrics.items() if k != "detail"},
        indent=2,
    ))

    result = make_result("vs_baselines", args.snapshot, metrics)
    save_result(result)


if __name__ == "__main__":
    main()

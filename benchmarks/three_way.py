"""Three-way head-to-head: DreamAgent alone vs Retrieval alone vs Composed.

The composed system queries BOTH DreamAgent and the retrieval baseline,
then uses the same base model to reconcile the two answers. This is the
production-shape pattern the V2 thesis pivots to after V2.1's parity result:
DreamAgent for parametric/durable; retrieval for fresh/dense; agent reconciles.

We benchmark all three on three probe sets:
- personal_recall (auto-generated from memories)
- cross_memory_reasoning (10 probes requiring 2-3 memory synthesis)
- adversarial_retrieval (15 probes designed with low question↔memory
  semantic similarity, where retrieval is expected to miss)

Example:
    python -m benchmarks.three_way \\
        --memories fixture:v1_baseline \\
        --snapshot runs/snapshots/live \\
        --base-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from benchmarks.cross_memory_reasoning import load_probes as load_xmem_probes
from dreamagent.compose import memories_to_dataset
from dreamagent.compose.examples import EvalProbe
from dreamagent.compose.templates import SYSTEM_PROMPT
from dreamagent.eval.runner import score
from dreamagent.ingest import FixtureConnector, JSONLConnector

ADV_PROBES_PATH = Path(__file__).parent / "probes" / "adversarial_retrieval.jsonl"

RECONCILE_SYSTEM_PROMPT = (
    "You receive two candidate answers about the user — one from a model "
    "trained on the user's memories (System A) and one from a retrieval "
    "system that searches the user's memories (System B). Produce a single "
    "synthesized answer to the question. Prefer the more specific answer. "
    "If both agree, use their agreement. If neither knows, say so. "
    "Be concise — one sentence."
)


def load_adversarial_probes() -> list[EvalProbe]:
    out: list[EvalProbe] = []
    with ADV_PROBES_PATH.open("r", encoding="utf-8") as f:
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
                    template="adversarial_retrieval",
                )
            )
    return out


def _connector(source: str):
    if source.startswith("fixture:"):
        return FixtureConnector(source.removeprefix("fixture:"))
    return JSONLConnector(source)


def _reconcile(
    question: str,
    answer_a: str,
    answer_b: str,
    model,
    tokenizer,
    max_tokens: int = 64,
) -> str:
    """Ask the base model to synthesize the two candidate answers."""
    from mlx_lm import generate

    user = (
        f"Question: {question}\n\n"
        f"System A (trained on user's memories) says:\n{answer_a}\n\n"
        f"System B (retrieved memory chunks) says:\n{answer_b}\n\n"
        "Synthesized answer:"
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": RECONCILE_SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    return generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
    )


def _evaluate_dreamagent_with_model(
    probes: list[EvalProbe],
    model,
    tokenizer,
    max_tokens: int,
) -> dict[str, dict]:
    """Run probes through an already-loaded DreamAgent (base + adapter)."""
    from mlx_lm import generate

    out: dict[str, dict] = {}
    for p in probes:
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p.question},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        passed, _ = score(response, p.expected_substrings)
        out[p.source_memory_id] = {"response": response, "passed": passed}
    return out


def _evaluate_dreamagent(
    probes: list[EvalProbe],
    base_model: str,
    adapter_dir: Path,
    max_tokens: int,
) -> dict[str, dict]:
    """Load DA, run probes, return results. Convenience for one-off use."""
    from mlx_lm import load

    model, tokenizer = load(base_model, adapter_path=str(adapter_dir))
    return _evaluate_dreamagent_with_model(probes, model, tokenizer, max_tokens)


def _evaluate_retrieval_with_baseline(
    probes: list[EvalProbe],
    baseline,
) -> dict[str, dict]:
    """Run probes through an already-built retrieval baseline."""
    out: dict[str, dict] = {}
    for p in probes:
        result = baseline.query(p.question)
        passed, _ = score(result.answer, p.expected_substrings)
        out[p.source_memory_id] = {
            "response": result.answer,
            "passed": passed,
            "retrieved_ids": result.retrieved_ids,
            "total_ms": result.total_ms,
        }
    return out


def _evaluate_retrieval(
    probes: list[EvalProbe],
    memories_source: str,
    base_model: str,
    top_k: int,
    max_tokens: int,
) -> tuple[dict[str, dict], object]:
    """Build retrieval baseline, run probes. Convenience for one-off use."""
    from benchmarks.baselines.retrieval import RetrievalBaseline

    baseline = RetrievalBaseline.from_source(
        memories_source, base_model=base_model, top_k=top_k, max_tokens=max_tokens
    )
    return _evaluate_retrieval_with_baseline(probes, baseline), baseline


def _evaluate_composed(
    probes: list[EvalProbe],
    da_results: dict[str, dict],
    ret_results: dict[str, dict],
    baseline,
    max_tokens: int,
) -> dict[str, dict]:
    """Compose DA + retrieval via base-model reconciliation. Reuses the
    baseline's already-loaded model+tokenizer."""
    out: dict[str, dict] = {}
    for p in probes:
        a = da_results[p.source_memory_id]["response"]
        b = ret_results[p.source_memory_id]["response"]
        reconciled = _reconcile(
            p.question,
            a,
            b,
            baseline._mlx_model,  # type: ignore[attr-defined]
            baseline._tokenizer,  # type: ignore[attr-defined]
            max_tokens=max_tokens,
        )
        passed, _ = score(reconciled, p.expected_substrings)
        out[p.source_memory_id] = {"response": reconciled, "passed": passed}
    return out


def _agg(probes: list[EvalProbe], results: dict[str, dict]) -> dict:
    n = len(probes)
    passed = sum(1 for p in probes if results[p.source_memory_id]["passed"])
    return {"n": n, "passed": passed, "pass_rate": passed / n if n else 0.0}


def run(
    memories: str,
    snapshot: Path,
    base_model: str,
    top_k: int = 5,
    eval_max_tokens: int = 64,
) -> dict:
    """Run the three-way head-to-head on all three probe sets.

    Loads each underlying model exactly once, then runs all probes through
    each. Previously we re-loaded per probe set; on small machines that
    caused resource exhaustion before phase 3 could complete.
    """
    from mlx_lm import load

    from benchmarks.baselines.retrieval import RetrievalBaseline

    items = list(_connector(memories).iter_memories())
    _, personal_probes = memories_to_dataset(items)
    xmem_probes = load_xmem_probes()
    adv_probes = load_adversarial_probes()

    adapter_dir = resolve_adapter_dir(Path(snapshot))

    # Load DA once, run all probe sets through it.
    print("phase 1: loading DreamAgent (base + adapter)...", flush=True)
    da_model, da_tokenizer = load(base_model, adapter_path=str(adapter_dir))
    print("phase 1: running DreamAgent on personal probes...", flush=True)
    da_personal = _evaluate_dreamagent_with_model(
        personal_probes, da_model, da_tokenizer, eval_max_tokens
    )
    print("phase 1: running DreamAgent on cross-memory probes...", flush=True)
    da_xmem = _evaluate_dreamagent_with_model(
        xmem_probes, da_model, da_tokenizer, eval_max_tokens
    )
    print("phase 1: running DreamAgent on adversarial probes...", flush=True)
    da_adv = _evaluate_dreamagent_with_model(
        adv_probes, da_model, da_tokenizer, eval_max_tokens
    )
    # Release DA model memory before loading retrieval baseline.
    del da_model
    del da_tokenizer

    # Build retrieval baseline once, run all probe sets through it.
    print("phase 2: loading retrieval baseline...", flush=True)
    baseline = RetrievalBaseline.from_source(
        memories, base_model=base_model, top_k=top_k, max_tokens=eval_max_tokens
    )
    print("phase 2: running retrieval on personal probes...", flush=True)
    ret_personal = _evaluate_retrieval_with_baseline(personal_probes, baseline)
    print("phase 2: running retrieval on cross-memory probes...", flush=True)
    ret_xmem = _evaluate_retrieval_with_baseline(xmem_probes, baseline)
    print("phase 2: running retrieval on adversarial probes...", flush=True)
    ret_adv = _evaluate_retrieval_with_baseline(adv_probes, baseline)

    # Phase 3: composed reconciliation reuses the baseline's model.
    print("phase 3: composed reconciliation on personal probes...", flush=True)
    composed_personal = _evaluate_composed(
        personal_probes, da_personal, ret_personal, baseline, eval_max_tokens
    )
    print("phase 3: composed reconciliation on cross-memory probes...", flush=True)
    composed_xmem = _evaluate_composed(
        xmem_probes, da_xmem, ret_xmem, baseline, eval_max_tokens
    )
    print("phase 3: composed reconciliation on adversarial probes...", flush=True)
    composed_adv = _evaluate_composed(
        adv_probes, da_adv, ret_adv, baseline, eval_max_tokens
    )

    def _three_way(probes, da, ret, comp):
        return {
            "dreamagent": _agg(probes, da),
            "retrieval": _agg(probes, ret),
            "composed": _agg(probes, comp),
        }

    return {
        "memories_source": memories,
        "snapshot": str(snapshot),
        "base_model": base_model,
        "top_k": top_k,
        "personal_recall": _three_way(
            personal_probes, da_personal, ret_personal, composed_personal
        ),
        "cross_memory_reasoning": _three_way(
            xmem_probes, da_xmem, ret_xmem, composed_xmem
        ),
        "adversarial_retrieval": _three_way(
            adv_probes, da_adv, ret_adv, composed_adv
        ),
        "detail": {
            "da_personal": da_personal,
            "ret_personal": ret_personal,
            "composed_personal": composed_personal,
            "da_xmem": da_xmem,
            "ret_xmem": ret_xmem,
            "composed_xmem": composed_xmem,
            "da_adv": da_adv,
            "ret_adv": ret_adv,
            "composed_adv": composed_adv,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--memories", required=True)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Meta-Llama-3.1-8B-Instruct-4bit")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--eval-max-tokens", type=int, default=64)
    args = p.parse_args()

    metrics = run(args.memories, args.snapshot, args.base_model, args.top_k, args.eval_max_tokens)

    print()
    print("THREE-WAY HEAD-TO-HEAD")
    print("=" * 80)
    print(f"  Base model: {args.base_model}")
    print(f"  Memories:   {args.memories}")
    print(f"  Top-k:      {args.top_k}")
    print()
    header = f"{'Probe set':<28} {'DreamAgent':>11} {'Retrieval':>11} {'Composed':>11}"
    print(header)
    print("-" * len(header))
    for set_name in ("personal_recall", "cross_memory_reasoning", "adversarial_retrieval"):
        row = metrics[set_name]
        print(
            f"{set_name:<28} "
            f"{row['dreamagent']['pass_rate']:>10.1%} "
            f"{row['retrieval']['pass_rate']:>10.1%} "
            f"{row['composed']['pass_rate']:>10.1%}"
        )
    print()

    result = make_result(
        "three_way", args.snapshot, {k: v for k, v in metrics.items() if k != "detail"}
    )
    save_result(result)

    # Also save the detail
    ts = result.timestamp_utc.replace(":", "-")
    detail_path = Path("benchmarks/results") / f"three_way_detail-{ts}.json"
    detail_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Detail saved to: {detail_path}")


if __name__ == "__main__":
    main()

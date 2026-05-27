"""Query-latency benchmark.

Measures end-to-end latency for asking the dreamed model a question. This is
the load-balancing argument for parametric memory: when the memory IS the
model, there's no network round trip to a vector store.

We report p50 / p95 / p99 over a configurable number of queries against the
loaded model. The first query is warmed up and excluded.

Example:
    python -m benchmarks.query_latency \\
        --snapshot runs/snapshots/live \\
        --queries 50
"""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

from benchmarks._common import make_result, resolve_adapter_dir, save_result
from dreamagent.compose.templates import SYSTEM_PROMPT

WARM_UP_QUERIES = 2

SAMPLE_QUESTIONS = [
    "What is the user's dog's name?",
    "How does the user prefer responses formatted?",
    "What deploy command does the user use for static sites?",
    "What's the user's primary language?",
    "Does the user have a cat?",
    "What's the user's anniversary date?",
    "Who is Otis?",
    "What is DreamAgent?",
    "What's the user's name?",
    "What does the user prefer: ruff or black?",
]


def run(snapshot: Path, base_model: str, queries: int = 30, max_tokens: int = 48) -> dict:
    try:
        from mlx_lm import generate, load
    except ImportError as e:
        raise RuntimeError("mlx_lm not installed; run uv sync") from e

    adapter_dir = resolve_adapter_dir(Path(snapshot))
    model, tokenizer = load(base_model, adapter_path=str(adapter_dir))

    questions = (SAMPLE_QUESTIONS * ((queries // len(SAMPLE_QUESTIONS)) + 1))[:queries]
    samples_ms: list[float] = []

    for i, q in enumerate(questions):
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        t0 = time.perf_counter()
        _ = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if i >= WARM_UP_QUERIES:
            samples_ms.append(elapsed_ms)

    samples_ms.sort()
    p50 = statistics.median(samples_ms)
    p95 = samples_ms[int(0.95 * len(samples_ms))] if len(samples_ms) >= 20 else samples_ms[-1]
    p99 = samples_ms[int(0.99 * len(samples_ms))] if len(samples_ms) >= 100 else samples_ms[-1]

    return {
        "sample_count": len(samples_ms),
        "warmup_skipped": WARM_UP_QUERIES,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "mean_ms": statistics.mean(samples_ms),
        "max_tokens": max_tokens,
        "base_model": base_model,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--base-model", default="mlx-community/Llama-3.2-1B-Instruct-4bit")
    p.add_argument("--queries", type=int, default=30)
    p.add_argument("--max-tokens", type=int, default=48)
    args = p.parse_args()

    metrics = run(args.snapshot, args.base_model, args.queries, args.max_tokens)
    result = make_result("query_latency", args.snapshot, metrics)
    print(result.to_json())
    save_result(result)


if __name__ == "__main__":
    main()

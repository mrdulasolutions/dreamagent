"""Eval runner — generate responses to a list of EvalProbes and score them.

A probe passes when its response contains ANY of `expected_substrings`
(case-insensitive). v1 deliberately uses lenient substring matching: a 0.6B
model gives noisy, verbose answers, and exact-match would be uselessly strict.

Both personal-recall probes (auto-generated from memories) and general-capability
probes (loaded from the anchor fixture) flow through the same runner — they're
both lists of EvalProbe.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dreamagent.compose.examples import EvalProbe
from dreamagent.compose.templates import SYSTEM_PROMPT


@dataclass(frozen=True, slots=True)
class EvalResult:
    """One probe's result."""

    probe: EvalProbe
    response: str
    passed: bool
    matched_substring: str | None


@dataclass(slots=True)
class EvalReport:
    """The full set of EvalResults for a run."""

    results: list[EvalResult] = field(default_factory=list)
    model: str = ""
    adapter_path: str | None = None

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "adapter_path": self.adapter_path,
            "total": self.total,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "results": [
                {
                    "memory_id": r.probe.source_memory_id,
                    "template": r.probe.template,
                    "question": r.probe.question,
                    "expected_substrings": list(r.probe.expected_substrings),
                    "response": r.response,
                    "passed": r.passed,
                    "matched_substring": r.matched_substring,
                }
                for r in self.results
            ],
        }


def score(response: str, expected_substrings: list[str]) -> tuple[bool, str | None]:
    """Return (passed, matched_substring_or_None). Case-insensitive substring match."""
    low = response.lower()
    for sub in expected_substrings:
        if sub.lower() in low:
            return True, sub
    return False, None


def _build_prompt(question: str, tokenizer) -> str:
    """Chat-template the system prompt + user question."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


def run_eval(
    probes: list[EvalProbe],
    model_repo: str,
    adapter_path: Path | None = None,
    max_tokens: int = 64,
    verbose: bool = False,
) -> EvalReport:
    """Load the model (optionally with adapter), generate, score every probe."""
    try:
        from mlx_lm import generate, load
    except ImportError as e:
        raise RuntimeError(
            "mlx_lm not installed — run `uv sync` from the project root"
        ) from e

    # mlx-lm's load_adapters expects the directory containing adapter_config.json
    # and adapters.safetensors, not the .safetensors file itself.
    adapter_arg: str | None = None
    if adapter_path is not None:
        adapter_dir = adapter_path.parent if adapter_path.is_file() else adapter_path
        adapter_arg = str(adapter_dir)
    model, tokenizer = load(model_repo, adapter_path=adapter_arg)

    report = EvalReport(model=model_repo, adapter_path=adapter_arg)
    for probe in probes:
        prompt = _build_prompt(probe.question, tokenizer)
        response = generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False
        )
        passed, matched = score(response, probe.expected_substrings)
        report.results.append(
            EvalResult(probe=probe, response=response, passed=passed, matched_substring=matched)
        )
        if verbose:
            mark = "✓" if passed else "✗"
            short = response.replace("\n", " ")[:80]
            print(f"  {mark} [{probe.source_memory_id}] {short}")
    return report

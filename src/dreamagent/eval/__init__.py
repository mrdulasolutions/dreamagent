"""Eval stage — run probes against a base + adapter model, score with substring match."""

from dreamagent.eval.runner import EvalReport, EvalResult, run_eval, score

__all__ = ["EvalReport", "EvalResult", "run_eval", "score"]

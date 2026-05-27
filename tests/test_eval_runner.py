"""Tests for the eval runner — scoring and report shape only.

The model-loading path requires the MLX runtime + a downloaded model and is
covered separately by an end-to-end smoke test.
"""

from __future__ import annotations

import pytest

from dreamagent.compose.examples import EvalProbe
from dreamagent.eval.runner import EvalReport, EvalResult, score


class TestScore:
    @pytest.mark.parametrize(
        "response,subs,expected_passed,expected_match",
        [
            ("The dog's name is Otis.", ["Otis"], True, "Otis"),
            ("Otis is great", ["Pixel", "Otis"], True, "Otis"),  # second match
            ("The user has a cat.", ["Otis"], False, None),
            ("OTIS", ["otis"], True, "otis"),  # case-insensitive
            ("", ["anything"], False, None),
            ("response", [], False, None),  # empty expected list
        ],
    )
    def test_substring_match(self, response, subs, expected_passed, expected_match):
        passed, matched = score(response, subs)
        assert passed is expected_passed
        assert matched == expected_match


class TestEvalReport:
    def _make(self, n_pass: int, n_fail: int) -> EvalReport:
        results = []
        for i in range(n_pass):
            probe = EvalProbe(
                question=f"q{i}",
                expected_substrings=["x"],
                source_memory_id=f"id_p{i}",
                template="t",
            )
            results.append(
                EvalResult(probe=probe, response="x", passed=True, matched_substring="x")
            )
        for i in range(n_fail):
            probe = EvalProbe(
                question=f"q{i}",
                expected_substrings=["x"],
                source_memory_id=f"id_f{i}",
                template="t",
            )
            results.append(
                EvalResult(probe=probe, response="y", passed=False, matched_substring=None)
            )
        return EvalReport(results=results, model="m", adapter_path=None)

    def test_pass_rate(self):
        r = self._make(3, 7)
        assert r.total == 10
        assert r.passed == 3
        assert r.pass_rate == 0.3

    def test_empty_report(self):
        r = EvalReport(results=[], model="m")
        assert r.total == 0
        assert r.pass_rate == 0.0

    def test_to_dict_shape(self):
        r = self._make(1, 1)
        d = r.to_dict()
        assert d["total"] == 2 and d["passed"] == 1
        assert len(d["results"]) == 2
        assert d["results"][0]["memory_id"].startswith("id_")

"""Eval gate — decide whether to promote a freshly trained adapter.

Implements the decision matrix from the plan:

  Personal recall | General regression | Action
  -----------------+----------------------+---------------------------
  >= min_recall   | <= max_regression    | PROMOTE
  >= min_recall   | warn..reject range   | PROMOTE_WITH_WARNING
  >= min_recall   | > reject threshold   | REJECT
  <  min_recall   | any                  | REJECT
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from dreamagent.eval.runner import EvalReport


class GateDecision(StrEnum):
    PROMOTE = "promote"
    PROMOTE_WITH_WARNING = "promote_with_warning"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class EvalGateConfig:
    """Thresholds for the eval gate.

    Defaults are for the production tier. Validation tier (Qwen 3 0.6B) should
    use `EvalGateConfig.validation_defaults()` — looser, since the base model
    is already weak so absolute regression numbers are noisier.
    """

    min_personal_recall: float = 0.70
    max_general_regression: float = 0.03
    warn_general_regression: float = 0.08

    @classmethod
    def validation_defaults(cls) -> EvalGateConfig:
        return cls(
            min_personal_recall=0.30,  # 0.6B base is weak; we lower the bar
            max_general_regression=0.05,
            warn_general_regression=0.15,
        )


@dataclass(slots=True)
class GateResult:
    decision: GateDecision
    reasons: list[str] = field(default_factory=list)
    personal_pass_rate: float = 0.0
    general_pass_rate_base: float = 0.0
    general_pass_rate_adapter: float = 0.0
    general_regression: float = 0.0  # base - adapter; positive = adapter is worse

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value,
            "reasons": list(self.reasons),
            "personal_pass_rate": self.personal_pass_rate,
            "general_pass_rate_base": self.general_pass_rate_base,
            "general_pass_rate_adapter": self.general_pass_rate_adapter,
            "general_regression": self.general_regression,
        }


def decide(
    personal: EvalReport,
    general_base: EvalReport,
    general_adapter: EvalReport,
    config: EvalGateConfig | None = None,
) -> GateResult:
    """Apply the decision matrix to the three eval reports."""
    cfg = config or EvalGateConfig()

    p_rate = personal.pass_rate
    g_base = general_base.pass_rate
    g_adapter = general_adapter.pass_rate
    regression = g_base - g_adapter

    result = GateResult(
        decision=GateDecision.REJECT,
        personal_pass_rate=p_rate,
        general_pass_rate_base=g_base,
        general_pass_rate_adapter=g_adapter,
        general_regression=regression,
    )

    if p_rate < cfg.min_personal_recall:
        result.decision = GateDecision.REJECT
        result.reasons.append(
            f"personal recall {p_rate:.2f} < min {cfg.min_personal_recall:.2f}"
        )
        return result

    if regression > cfg.warn_general_regression:
        result.decision = GateDecision.REJECT
        result.reasons.append(
            f"general regression {regression:+.3f} > reject threshold "
            f"{cfg.warn_general_regression:+.3f}"
        )
        return result

    if regression > cfg.max_general_regression:
        result.decision = GateDecision.PROMOTE_WITH_WARNING
        result.reasons.append(
            f"general regression {regression:+.3f} > warn threshold "
            f"{cfg.max_general_regression:+.3f} (still within reject limit "
            f"{cfg.warn_general_regression:+.3f})"
        )
        result.reasons.append(
            f"personal recall {p_rate:.2f} >= min {cfg.min_personal_recall:.2f}"
        )
        return result

    result.decision = GateDecision.PROMOTE
    result.reasons.append(
        f"personal recall {p_rate:.2f} >= min {cfg.min_personal_recall:.2f}"
    )
    result.reasons.append(
        f"general regression {regression:+.3f} <= max {cfg.max_general_regression:+.3f}"
    )
    return result

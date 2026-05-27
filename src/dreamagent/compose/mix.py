"""Rehearsal mix composer — the catastrophic-forgetting countermeasure.

Every nightly training set is composed of three slices:

  - **today**: the new + recent memories. Always 100% included.
  - **replay**: prior-night memories sampled to give the model rehearsal.
    Defends against forgetting older memories.
  - **anchor**: fixed general-capability examples ("you are still you").
    Defends against forgetting general capabilities and persona.

Ratios are configurable but default to what the plan recommends:
  - today    ~75%
  - replay   ~15%
  - anchor   ~10%

For the first night (no prior memories), the replay slice is empty and the
mix is just today + anchors. As history accumulates, replay fills in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from dreamagent.compose.examples import TrainingExample


@dataclass(frozen=True, slots=True)
class MixConfig:
    """Target composition for a nightly rehearsal mix."""

    anchor_ratio: float = 0.10
    replay_ratio: float = 0.15
    seed: int = 42

    def __post_init__(self):
        if not 0 <= self.anchor_ratio < 1:
            raise ValueError("anchor_ratio must be in [0, 1)")
        if not 0 <= self.replay_ratio < 1:
            raise ValueError("replay_ratio must be in [0, 1)")
        if self.anchor_ratio + self.replay_ratio >= 1:
            raise ValueError("anchor_ratio + replay_ratio must be < 1")

    @property
    def today_ratio(self) -> float:
        return 1.0 - self.anchor_ratio - self.replay_ratio


@dataclass(frozen=True, slots=True)
class RehearsalMix:
    """The composed training set, with provenance counts for the eval gate."""

    examples: list[TrainingExample]
    composition: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.examples)


def _sample_random(
    pool: list[TrainingExample], k: int, rng: random.Random
) -> list[TrainingExample]:
    """Random sample up to k examples without replacement; return all if pool < k."""
    if k <= 0 or not pool:
        return []
    if k >= len(pool):
        return list(pool)
    return rng.sample(pool, k)


def _sample_stable(
    pool: list[TrainingExample], k: int
) -> list[TrainingExample]:
    """Take the first k examples in stable order (by source_memory_id).

    Used for anchors so that runs are reproducible across pool-size changes:
    growing the anchor fixture does NOT change which anchors are in a run
    unless the count requested exceeds the previous pool size.
    """
    if k <= 0 or not pool:
        return []
    ordered = sorted(pool, key=lambda ex: ex.source_memory_id)
    return ordered[:k]


def compose_rehearsal_mix(
    today: list[TrainingExample],
    prior: list[TrainingExample],
    anchors: list[TrainingExample],
    config: MixConfig | None = None,
    max_anchors: int | None = None,
) -> RehearsalMix:
    """Build a single nightly training set.

    Sizes the replay and anchor slices proportional to `today`. If the pools
    are smaller than the target, we use all of what's available — no error,
    just smaller-than-target slice (recorded in the composition dict).

    `max_anchors` caps the anchor count regardless of `anchor_ratio`. Useful
    for experimentally isolating the effect of anchor count from other levers.
    """
    cfg = config or MixConfig()
    rng = random.Random(cfg.seed)

    today_count = len(today)
    if today_count == 0:
        return RehearsalMix(examples=[], composition={"today": 0, "replay": 0, "anchor": 0})

    # total = today / today_ratio  →  replay = total * replay_ratio, anchor = total * anchor_ratio
    total_target = today_count / cfg.today_ratio
    target_replay = round(total_target * cfg.replay_ratio)
    target_anchor = round(total_target * cfg.anchor_ratio)
    if max_anchors is not None:
        target_anchor = min(target_anchor, max_anchors)

    replay_slice = _sample_random(prior, target_replay, rng)
    # Anchors use stable order — growing the anchor pool doesn't change which
    # anchors are picked unless the request exceeds the prior pool size.
    anchor_slice = _sample_stable(anchors, target_anchor)

    combined = list(today) + replay_slice + anchor_slice
    rng.shuffle(combined)

    composition = {
        "today": today_count,
        "replay": len(replay_slice),
        "anchor": len(anchor_slice),
    }
    return RehearsalMix(examples=combined, composition=composition)

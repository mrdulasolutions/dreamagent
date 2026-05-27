"""Tests for the rehearsal mix composer."""

from __future__ import annotations

import pytest

from dreamagent.compose.examples import TrainingExample
from dreamagent.compose.mix import MixConfig, RehearsalMix, compose_rehearsal_mix


def _ex(id_: str) -> TrainingExample:
    return TrainingExample(
        messages=[
            {"role": "system", "content": "s"},
            {"role": "user", "content": f"q{id_}"},
            {"role": "assistant", "content": f"a{id_}"},
        ],
        source_memory_id=id_,
        template="test",
    )


class TestMixConfigValidation:
    def test_default_ratios_valid(self):
        cfg = MixConfig()
        assert cfg.today_ratio == pytest.approx(0.75)

    @pytest.mark.parametrize(
        "anchor,replay",
        [(-0.1, 0.1), (0.5, 0.5), (0.6, 0.5), (1.0, 0.0)],
    )
    def test_invalid_ratios_rejected(self, anchor, replay):
        with pytest.raises(ValueError):
            MixConfig(anchor_ratio=anchor, replay_ratio=replay)


class TestCompositionMath:
    def test_first_night_no_prior(self):
        today = [_ex(f"t{i}") for i in range(20)]
        anchors = [_ex(f"a{i}") for i in range(40)]
        mix = compose_rehearsal_mix(today, prior=[], anchors=anchors)
        assert mix.composition["today"] == 20
        assert mix.composition["replay"] == 0
        # anchor target = (20 / 0.75) * 0.10 ≈ 2.67 → 3
        assert mix.composition["anchor"] == 3
        assert len(mix.examples) == 20 + 0 + 3

    def test_replay_pool_sufficient(self):
        today = [_ex(f"t{i}") for i in range(30)]
        prior = [_ex(f"p{i}") for i in range(100)]
        anchors = [_ex(f"a{i}") for i in range(40)]
        mix = compose_rehearsal_mix(today, prior, anchors)
        # replay target = (30 / 0.75) * 0.15 = 6
        # anchor target = (30 / 0.75) * 0.10 = 4
        assert mix.composition["today"] == 30
        assert mix.composition["replay"] == 6
        assert mix.composition["anchor"] == 4

    def test_replay_pool_smaller_than_target(self):
        today = [_ex(f"t{i}") for i in range(30)]
        prior = [_ex(f"p{i}") for i in range(2)]  # less than target 6
        anchors = [_ex(f"a{i}") for i in range(40)]
        mix = compose_rehearsal_mix(today, prior, anchors)
        assert mix.composition["replay"] == 2  # used all available

    def test_anchor_pool_smaller_than_target(self):
        today = [_ex(f"t{i}") for i in range(100)]
        anchors = [_ex(f"a{i}") for i in range(3)]
        mix = compose_rehearsal_mix(today, prior=[], anchors=anchors)
        assert mix.composition["anchor"] == 3

    def test_empty_today_returns_empty_mix(self):
        anchors = [_ex(f"a{i}") for i in range(40)]
        mix = compose_rehearsal_mix(today=[], prior=[], anchors=anchors)
        assert len(mix.examples) == 0
        assert mix.composition == {"today": 0, "replay": 0, "anchor": 0}


class TestDeterminism:
    def test_same_seed_same_mix(self):
        today = [_ex(f"t{i}") for i in range(20)]
        prior = [_ex(f"p{i}") for i in range(20)]
        anchors = [_ex(f"a{i}") for i in range(20)]
        mix_a = compose_rehearsal_mix(today, prior, anchors)
        mix_b = compose_rehearsal_mix(today, prior, anchors)
        ids_a = [ex.source_memory_id for ex in mix_a.examples]
        ids_b = [ex.source_memory_id for ex in mix_b.examples]
        assert ids_a == ids_b

    def test_different_seed_different_mix(self):
        today = [_ex(f"t{i}") for i in range(20)]
        prior = [_ex(f"p{i}") for i in range(20)]
        anchors = [_ex(f"a{i}") for i in range(20)]
        mix_a = compose_rehearsal_mix(today, prior, anchors, MixConfig(seed=1))
        mix_b = compose_rehearsal_mix(today, prior, anchors, MixConfig(seed=2))
        ids_a = [ex.source_memory_id for ex in mix_a.examples]
        ids_b = [ex.source_memory_id for ex in mix_b.examples]
        assert ids_a != ids_b


class TestCustomRatios:
    def test_no_anchors_no_replay(self):
        today = [_ex(f"t{i}") for i in range(10)]
        mix = compose_rehearsal_mix(
            today, prior=[], anchors=[], config=MixConfig(anchor_ratio=0.0, replay_ratio=0.0)
        )
        assert len(mix.examples) == 10
        assert mix.composition == {"today": 10, "replay": 0, "anchor": 0}


def test_returned_type():
    today = [_ex(f"t{i}") for i in range(5)]
    mix = compose_rehearsal_mix(today, [], [])
    assert isinstance(mix, RehearsalMix)

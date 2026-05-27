"""Compose stage — turn MemoryItems into training examples and eval probes."""

from dreamagent.compose.anchors import load_general_anchors, load_general_eval_probes
from dreamagent.compose.examples import (
    EvalProbe,
    TrainingExample,
    memories_to_dataset,
    memory_to_examples,
)
from dreamagent.compose.mix import MixConfig, RehearsalMix, compose_rehearsal_mix

__all__ = [
    "EvalProbe",
    "MixConfig",
    "RehearsalMix",
    "TrainingExample",
    "compose_rehearsal_mix",
    "load_general_anchors",
    "load_general_eval_probes",
    "memories_to_dataset",
    "memory_to_examples",
]

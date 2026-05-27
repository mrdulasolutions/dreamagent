"""Train stage — wrap mlx-lm LoRA training behind a small, stable API."""

from dreamagent.train.config import TrainConfig
from dreamagent.train.runner import TrainError, TrainResult, train_adapter

__all__ = ["TrainConfig", "TrainError", "TrainResult", "train_adapter"]

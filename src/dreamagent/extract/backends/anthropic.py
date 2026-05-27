"""Anthropic Claude backend for memory extraction.

Uses the Messages API with JSON-mode-ish output (we prefill `[` to force the
model to start with a JSON array). The model returns the rest of the array
which we then parse.
"""

from __future__ import annotations

import os

from dreamagent.extract.base import ExtractionResponse

DEFAULT_MODEL = "claude-sonnet-4-6"


class AnthropicBackend:
    """Memory extraction via Anthropic Claude."""

    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            import anthropic
        except ImportError as e:
            raise RuntimeError(
                "anthropic SDK not installed — install with `pip install dreamagent[anthropic]`"
            ) from e
        self._anthropic = anthropic
        self._model = model
        self._client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))

    @property
    def model(self) -> str:
        return self._model

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionResponse:
        """Send the prompt to Claude, prefilling `[` to force JSON array output."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=8192,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": "["},
            ],
        )
        # Reconstruct the full JSON array from the prefill + completion
        completion = "".join(
            block.text for block in response.content if hasattr(block, "text")
        )
        raw = "[" + completion
        return ExtractionResponse(
            raw_output=raw,
            model=self._model,
            backend=self.name,
            prompt_tokens=getattr(response.usage, "input_tokens", None),
            completion_tokens=getattr(response.usage, "output_tokens", None),
        )

"""OpenAI backend for memory extraction.

Uses chat completions with JSON mode (`response_format={"type": "json_object"}`).
Note: OpenAI's JSON mode returns a single object; we wrap the prompt to ask
for `{"memories": [...]}` and unwrap it.
"""

from __future__ import annotations

import json
import os

from dreamagent.extract.base import ExtractionResponse

DEFAULT_MODEL = "gpt-4o-2024-11-20"


class OpenAIBackend:
    """Memory extraction via OpenAI."""

    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL, api_key: str | None = None):
        try:
            import openai
        except ImportError as e:
            raise RuntimeError(
                "openai SDK not installed — install with `pip install dreamagent[openai]`"
            ) from e
        self._openai = openai
        self._model = model
        self._client = openai.OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    @property
    def model(self) -> str:
        return self._model

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionResponse:
        # JSON mode requires a single object; we ask for {"memories": [...]}
        # and unwrap the array.
        wrapper_instruction = (
            "\n\nWrap your JSON array output as: {\"memories\": [...]}"
        )
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt + wrapper_instruction},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=8192,
        )
        wrapped = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(wrapped)
            array = parsed.get("memories", [])
            raw = json.dumps(array)
        except json.JSONDecodeError:
            raw = wrapped  # let the pipeline's parser surface the error

        usage = response.usage
        return ExtractionResponse(
            raw_output=raw,
            model=self._model,
            backend=self.name,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )

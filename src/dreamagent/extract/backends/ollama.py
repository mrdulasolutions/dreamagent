"""Ollama backend for memory extraction.

Fully local. Useful for users who don't want their memories leaving the box.
Uses Ollama's `format: json` parameter to constrain output.
"""

from __future__ import annotations

import json
import os

from dreamagent.extract.base import ExtractionResponse

DEFAULT_MODEL = "llama3.2:3b"


class OllamaBackend:
    """Memory extraction via local Ollama."""

    name = "ollama"

    def __init__(self, model: str = DEFAULT_MODEL, host: str | None = None):
        try:
            import ollama
        except ImportError as e:
            raise RuntimeError(
                "ollama SDK not installed — install with `pip install dreamagent[ollama]`"
            ) from e
        self._ollama = ollama
        self._model = model
        self._client = ollama.Client(host=host or os.environ.get("OLLAMA_HOST"))

    @property
    def model(self) -> str:
        return self._model

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionResponse:
        # Ollama JSON mode expects a single object; same wrap trick as OpenAI.
        wrapped_instruction = (
            "\n\nReturn JSON in the form: {\"memories\": [...]}"
        )
        response = self._client.chat(
            model=self._model,
            format="json",
            messages=[
                {"role": "system", "content": system_prompt + wrapped_instruction},
                {"role": "user", "content": user_prompt},
            ],
            options={"num_predict": 4096},
        )
        content = response.get("message", {}).get("content", "{}")
        try:
            parsed = json.loads(content)
            array = parsed.get("memories", [])
            raw = json.dumps(array)
        except json.JSONDecodeError:
            raw = content

        return ExtractionResponse(
            raw_output=raw,
            model=self._model,
            backend=self.name,
            prompt_tokens=response.get("prompt_eval_count"),
            completion_tokens=response.get("eval_count"),
        )

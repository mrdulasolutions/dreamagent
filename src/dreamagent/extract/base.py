"""Extraction backend protocol and common types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ExtractionResponse:
    """What a backend returns: raw text (expected to be a JSON array) plus
    backend-level metadata for lineage."""

    raw_output: str
    model: str
    backend: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@runtime_checkable
class ExtractionBackend(Protocol):
    """Anything that can take a system prompt + user text and return a
    raw response is an extraction backend.

    Implementations: AnthropicBackend, OpenAIBackend, OllamaBackend.
    """

    @property
    def name(self) -> str:
        """Backend identifier, e.g., "anthropic"."""
        ...

    @property
    def model(self) -> str:
        """Model identifier this backend is configured to use."""
        ...

    def extract(self, *, system_prompt: str, user_prompt: str) -> ExtractionResponse:
        """Call the underlying model and return the raw text response."""
        ...

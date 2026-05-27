"""Frontier-model memory extraction — the accuracy-critical path.

Takes raw text (chat logs, journal entries, mem0 exports) and uses an LLM
(Anthropic / OpenAI / local Ollama) to extract validated MemoryItem records.
"""

from dreamagent.extract.base import ExtractionBackend, ExtractionResponse
from dreamagent.extract.pipeline import (
    ExtractionReport,
    extract_memories,
    read_input,
    write_jsonl,
)
from dreamagent.extract.prompt import SYSTEM, build_prompt

__all__ = [
    "SYSTEM",
    "ExtractionBackend",
    "ExtractionReport",
    "ExtractionResponse",
    "build_prompt",
    "extract_memories",
    "read_input",
    "write_jsonl",
]


def get_backend(name: str, model: str | None = None) -> ExtractionBackend:
    """Lazily import and instantiate a backend by name."""
    if name == "anthropic":
        from dreamagent.extract.backends.anthropic import (
            DEFAULT_MODEL,
            AnthropicBackend,
        )

        return AnthropicBackend(model=model or DEFAULT_MODEL)
    if name == "openai":
        from dreamagent.extract.backends.openai import (
            DEFAULT_MODEL,
            OpenAIBackend,
        )

        return OpenAIBackend(model=model or DEFAULT_MODEL)
    if name == "ollama":
        from dreamagent.extract.backends.ollama import (
            DEFAULT_MODEL,
            OllamaBackend,
        )

        return OllamaBackend(model=model or DEFAULT_MODEL)
    raise ValueError(
        f"unknown extraction backend {name!r}; choose anthropic, openai, or ollama"
    )

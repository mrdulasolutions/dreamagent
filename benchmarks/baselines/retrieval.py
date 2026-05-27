"""Vector-retrieval baseline for head-to-head against DreamAgent.

Implements a minimal but fair RAG baseline:
- Embed each MemoryItem.content with sentence-transformers
- At query time, embed the question, retrieve top-k by cosine similarity
- Stuff the retrieved chunks into a prompt and generate with the SAME
  base model DreamAgent uses

By using the same base model for generation, the comparison isolates
the architectural difference: retrieval vs parametric memory.

This baseline is intentionally simple. mem0/Letta/Zep have entity linking,
graph-based retrieval, BM25 hybrids, etc. — DreamAgent doesn't try to beat
those individually; it argues the parametric paradigm is structurally
different and complementary. A user who wants to compare against mem0
specifically should run that on their own setup.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamagent.schema import MemoryItem

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # ~80MB, fast
DEFAULT_TOP_K = 5

RETRIEVAL_SYSTEM_PROMPT = (
    "You are the user's personal assistant. Below are some memories about "
    "the user that may be relevant to the question. Answer the question "
    "using these memories. If the memories don't contain the answer, say "
    "you don't know."
)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    answer: str
    retrieved_ids: list[str]
    embed_ms: float
    search_ms: float
    generate_ms: float

    @property
    def total_ms(self) -> float:
        return self.embed_ms + self.search_ms + self.generate_ms


class RetrievalBaseline:
    """A vector-retrieval RAG baseline.

    Pre-embeds all memories at construction time. Query time:
    embed → top-k search → prompt-stuff → generate.
    """

    def __init__(
        self,
        memories: list[MemoryItem],
        base_model: str,
        embed_model: str = DEFAULT_EMBED_MODEL,
        top_k: int = DEFAULT_TOP_K,
        max_tokens: int = 64,
    ):
        try:
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "sentence-transformers not installed — "
                "install with `uv sync --extra retrieval-baseline`"
            ) from e
        try:
            from mlx_lm import load
        except ImportError as e:
            raise RuntimeError("mlx_lm not installed; run `uv sync`") from e

        self._np = np
        self._st = SentenceTransformer(embed_model)
        self._memories = [m for m in memories if m.is_trainable()]
        # Pre-compute embeddings for all memories.
        self._embeddings = self._st.encode(
            [m.content for m in self._memories],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        self._top_k = top_k
        self._base_model = base_model
        self._max_tokens = max_tokens
        self._mlx_model, self._tokenizer = load(base_model)

    def query(self, question: str) -> RetrievalResult:
        from mlx_lm import generate

        # Embed
        t0 = time.perf_counter()
        q_emb = self._st.encode(
            [question], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        embed_ms = (time.perf_counter() - t0) * 1000

        # Top-k cosine similarity search (embeddings are already normalized)
        t1 = time.perf_counter()
        scores = self._embeddings @ q_emb
        top_indices = self._np.argsort(-scores)[: self._top_k].tolist()
        retrieved = [self._memories[i] for i in top_indices]
        search_ms = (time.perf_counter() - t1) * 1000

        # Prompt build + generate
        t2 = time.perf_counter()
        memories_block = "\n".join(
            f"{i + 1}. {m.content}" for i, m in enumerate(retrieved)
        )
        user_content = f"Memories:\n{memories_block}\n\nQuestion: {question}"
        prompt = self._tokenizer.apply_chat_template(
            [
                {"role": "system", "content": RETRIEVAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        answer = generate(
            self._mlx_model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self._max_tokens,
            verbose=False,
        )
        generate_ms = (time.perf_counter() - t2) * 1000

        return RetrievalResult(
            answer=answer,
            retrieved_ids=[m.id for m in retrieved],
            embed_ms=embed_ms,
            search_ms=search_ms,
            generate_ms=generate_ms,
        )

    def info(self) -> dict[str, Any]:
        """Metadata about this baseline instance, for benchmark reports."""
        return {
            "embed_model": DEFAULT_EMBED_MODEL,
            "base_model": self._base_model,
            "top_k": self._top_k,
            "max_tokens": self._max_tokens,
            "memory_count": len(self._memories),
            "embedding_dim": int(self._embeddings.shape[1]),
        }

    @classmethod
    def from_source(
        cls,
        source: str | Path,
        base_model: str,
        **kwargs,
    ) -> RetrievalBaseline:
        """Build from a memory source string (e.g. 'fixture:v1_baseline' or
        a path to a .jsonl).
        """
        from dreamagent.ingest import FixtureConnector, JSONLConnector

        source_str = str(source)
        if source_str.startswith("fixture:"):
            connector = FixtureConnector(source_str.removeprefix("fixture:"))
        else:
            connector = JSONLConnector(source_str)
        memories = list(connector.iter_memories())
        return cls(memories=memories, base_model=base_model, **kwargs)

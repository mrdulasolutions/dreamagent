"""MemoryItem schema — the public contract for memories arriving from any upstream source."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

SCHEMA_VERSION = "1.0"
MAX_CONTENT_CHARS = 2000


class MemoryKind(StrEnum):
    FACT = "fact"
    PREFERENCE = "preference"
    PROCEDURE = "procedure"
    EVENT = "event"
    CORRECTION = "correction"


class SourceSystem(StrEnum):
    MEM0 = "mem0"
    SUPERMEMORY = "supermemory"
    CLAUDE = "claude"
    OPENCLAW = "openclaw"
    HERMES = "hermes"
    MANUAL = "manual"
    FIXTURE = "fixture"


class Sensitivity(StrEnum):
    NORMAL = "normal"
    SENSITIVE = "sensitive"
    REDACT = "redact"


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    system: SourceSystem
    session_id: str | None = None
    captured_at: datetime


class QAPair(BaseModel):
    model_config = ConfigDict(extra="forbid")

    q: Annotated[str, StringConstraints(min_length=1, max_length=500)]
    a: Annotated[str, StringConstraints(min_length=1, max_length=1000)]


class PreferenceSignal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    axis: Annotated[str, StringConstraints(min_length=1, max_length=64)]
    value: Annotated[str, StringConstraints(min_length=1, max_length=128)]


class MemoryItem(BaseModel):
    """A single memory record, the unit ingested by DreamAgent.

    Required fields cover identity, content, provenance, quality, and lifecycle.
    Optional fields enable structured retrieval, pre-shaped training, privacy
    handling, and preference-axis tracking.
    """

    model_config = ConfigDict(extra="forbid")

    id: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    schema_version: str = SCHEMA_VERSION

    content: Annotated[str, StringConstraints(min_length=1, max_length=MAX_CONTENT_CHARS)]
    kind: MemoryKind
    subject: Annotated[str, StringConstraints(min_length=1, max_length=256)]

    source: Source

    confidence: Annotated[float, Field(ge=0.0, le=1.0)]
    importance: Annotated[float, Field(ge=0.0, le=1.0)]

    supersedes: list[str] = Field(default_factory=list)
    expires_at: datetime | None = None

    entities: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    qa_pairs: list[QAPair] = Field(default_factory=list)
    sensitivity: Sensitivity = Sensitivity.NORMAL
    preference_signal: PreferenceSignal | None = None

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {v!r}; this build expects {SCHEMA_VERSION!r}"
            )
        return v

    @model_validator(mode="after")
    def _check_kind_specific_rules(self) -> MemoryItem:
        if self.preference_signal is not None and self.kind != MemoryKind.PREFERENCE:
            raise ValueError("preference_signal is only valid when kind == 'preference'")
        if self.kind == MemoryKind.CORRECTION and not self.supersedes:
            raise ValueError("kind=='correction' requires at least one id in 'supersedes'")
        return self

    def is_trainable(self) -> bool:
        """True if this memory should reach the training stage. False for redacted memories."""
        return self.sensitivity != Sensitivity.REDACT


class MemoryBatch(BaseModel):
    """Batched wire format: a list of MemoryItems with a schema_version envelope."""

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    items: list[MemoryItem]

    @field_validator("schema_version")
    @classmethod
    def _check_schema_version(cls, v: str) -> str:
        if v != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported schema_version {v!r}; this build expects {SCHEMA_VERSION!r}"
            )
        return v

"""The extraction pipeline.

Glues input text → backend → raw JSON → validated MemoryItems with full
schema-required metadata. This is the public API for the `dreamagent extract`
CLI command and for any programmatic caller.

The pipeline is responsible for:
- Reading and concatenating input text
- Calling the backend
- Parsing the response (with strict JSON discipline)
- Synthesizing the auto-fields (id, schema_version, source, captured_at)
  that the LLM is forbidden from producing
- Validating each result against the MemoryItem schema
- Reporting which records were rejected and why

Backends supply only the call mechanism. All validation is done here.
"""

from __future__ import annotations

import json
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from dreamagent.extract.base import ExtractionBackend, ExtractionResponse
from dreamagent.extract.prompt import SYSTEM, build_prompt
from dreamagent.schema import MemoryItem, Source, SourceSystem


@dataclass(slots=True)
class ExtractionReport:
    """Output of a single extraction run."""

    items: list[MemoryItem] = field(default_factory=list)
    rejected: list[dict] = field(default_factory=list)
    response: ExtractionResponse | None = None
    input_chars: int = 0


def _mint_id() -> str:
    return "mem_" + secrets.token_hex(8)


def _wrap_into_memory(
    record: dict,
    *,
    source_system: SourceSystem,
    captured_at: datetime,
) -> MemoryItem:
    """Take a backend-emitted record and add the auto-fields, then validate.

    Raises ValidationError if the record is malformed.
    """
    augmented = {
        "id": _mint_id(),
        "schema_version": "1.0",
        "source": Source(system=source_system, captured_at=captured_at).model_dump(
            mode="json"
        ),
        **record,
    }
    return MemoryItem.model_validate(augmented)


def _parse_response(raw: str) -> list[dict]:
    """Parse a raw response into a list of dicts. Tolerates leading/trailing
    whitespace or a model that ignored "no markdown" and wrapped in ```json."""
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("```", 2)[1]
        if stripped.startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip().rstrip("`").rstrip()
    parsed = json.loads(stripped)
    if not isinstance(parsed, list):
        raise ValueError(
            f"expected JSON array of MemoryItems; got {type(parsed).__name__}"
        )
    return parsed


def extract_memories(
    text: str,
    backend: ExtractionBackend,
    *,
    source_system: SourceSystem = SourceSystem.MANUAL,
    captured_at: datetime | None = None,
) -> ExtractionReport:
    """Run the full extraction pipeline on `text`."""
    captured_at = captured_at or datetime.now(UTC)
    report = ExtractionReport(input_chars=len(text))

    full_prompt = build_prompt() + "\n\n---\n\nINPUT TEXT:\n\n" + text

    response = backend.extract(system_prompt=SYSTEM, user_prompt=full_prompt)
    report.response = response

    try:
        records = _parse_response(response.raw_output)
    except (json.JSONDecodeError, ValueError) as e:
        report.rejected.append(
            {"reason": f"failed to parse response as JSON array: {e}",
             "snippet": response.raw_output[:300]}
        )
        return report

    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            report.rejected.append(
                {"index": i, "reason": "not an object", "record": rec}
            )
            continue
        try:
            item = _wrap_into_memory(
                rec, source_system=source_system, captured_at=captured_at
            )
        except ValidationError as e:
            report.rejected.append(
                {"index": i, "reason": "validation failed", "errors": e.errors(),
                 "record": rec}
            )
            continue
        report.items.append(item)

    return report


def read_input(source: Path) -> str:
    """Read the source file. Supports plain text, markdown, or a JSONL of
    chat messages (which gets formatted into a transcript)."""
    text = source.read_text(encoding="utf-8")
    if source.suffix == ".jsonl":
        return _format_chat_jsonl(text)
    return text


def _format_chat_jsonl(jsonl: str) -> str:
    """If the file is a JSONL of chat messages (each line a
    {"role": ..., "content": ...} dict, possibly nested under "messages"),
    flatten into a transcript. Otherwise return as-is."""
    lines = []
    for raw in jsonl.splitlines():
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            lines.append(raw)  # fall back to raw line
            continue
        if isinstance(obj, dict):
            if "messages" in obj:
                for m in obj["messages"]:
                    role = m.get("role", "user")
                    content = m.get("content", "")
                    lines.append(f"[{role}] {content}")
            elif "role" in obj and "content" in obj:
                lines.append(f"[{obj['role']}] {obj['content']}")
            else:
                lines.append(json.dumps(obj))
        else:
            lines.append(json.dumps(obj))
    return "\n".join(lines)


def write_jsonl(items: list[MemoryItem], path: Path) -> None:
    """Append memories to a JSONL file (one MemoryItem per line)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

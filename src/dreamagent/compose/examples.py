"""Convert MemoryItems into training examples and held-out eval probes.

For each MemoryItem:
  - if `qa_pairs` is provided, those go to training (no eval probe — assume
    the upstream took care of held-out generation if it cared).
  - otherwise, apply templates by `kind`: all-but-last → training,
    last → eval probe.

Eval probes are NEVER in the training set. The compose stage is the single
point where this split is enforced.
"""

from __future__ import annotations

from dataclasses import dataclass

from dreamagent.compose.templates import SYSTEM_PROMPT, templates_for
from dreamagent.schema import MemoryItem


@dataclass(frozen=True, slots=True)
class TrainingExample:
    """A single chat-format training pair, with lineage back to a MemoryItem."""

    messages: list[dict[str, str]]
    source_memory_id: str
    template: str


@dataclass(frozen=True, slots=True)
class EvalProbe:
    """A held-out question used to measure recall after training.

    `expected_substrings` lists the substrings the model output must contain
    (case-insensitive) to count as correct. v1 uses substring match; later
    versions may add semantic match.
    """

    question: str
    expected_substrings: list[str]
    source_memory_id: str
    template: str


def _make_chat(question: str, answer: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ]


def _default_expected_substrings(item: MemoryItem) -> list[str]:
    """Pick the substrings we'll look for in the model output during eval.

    Strategy: use the memory's `entities` if present (likely the salient nouns
    the model should reproduce). Otherwise fall back to the first 8 chars of
    content as a weak signal. This is intentionally lenient for v1 — exact
    answer matching is too strict for a 0.6B model.
    """
    if item.entities:
        return list(item.entities)
    head = item.content.split(".")[0].strip()
    return [head[:40]] if head else [item.content[:40]]


def memory_to_examples(item: MemoryItem) -> tuple[list[TrainingExample], list[EvalProbe]]:
    """Turn one MemoryItem into (training_examples, eval_probes).

    Redacted memories (sensitivity == redact) produce empty lists.
    """
    if not item.is_trainable():
        return [], []

    answer = item.content

    if item.qa_pairs:
        training = [
            TrainingExample(
                messages=_make_chat(qa.q, qa.a),
                source_memory_id=item.id,
                template="explicit:qa_pair",
            )
            for qa in item.qa_pairs
        ]
        return training, []

    train_templates, eval_template = templates_for(item.kind)
    training = [
        TrainingExample(
            messages=_make_chat(t.format(subject=item.subject), answer),
            source_memory_id=item.id,
            template=f"{item.kind.value}:{i}",
        )
        for i, t in enumerate(train_templates)
    ]
    probe = EvalProbe(
        question=eval_template.format(subject=item.subject),
        expected_substrings=_default_expected_substrings(item),
        source_memory_id=item.id,
        template=f"{item.kind.value}:eval",
    )
    return training, [probe]


def memories_to_dataset(
    items: list[MemoryItem],
) -> tuple[list[TrainingExample], list[EvalProbe]]:
    """Apply memory_to_examples to a list, concatenating results.

    Superseded memories are dropped — if memory A.supersedes contains B's id,
    then B is excluded from training and eval. This is the correction mechanism.
    """
    superseded_ids = {sup for item in items for sup in item.supersedes}
    live_items = [i for i in items if i.id not in superseded_ids]

    all_train: list[TrainingExample] = []
    all_probes: list[EvalProbe] = []
    for item in live_items:
        train, probes = memory_to_examples(item)
        all_train.extend(train)
        all_probes.extend(probes)
    return all_train, all_probes

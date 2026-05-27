"""Question templates per MemoryKind. The compose stage uses these to turn each
MemoryItem into multiple Q/A pairs — most of which go to training, with at
least one held out as an eval probe (so we test generalization, not memorization).

If a MemoryItem ships explicit `qa_pairs`, those take priority and these templates
are not used.
"""

from __future__ import annotations

from dreamagent.schema import MemoryKind

# Each list's LAST entry is reserved as the eval probe. Earlier entries are training.
# All templates are .format(subject=item.subject) — keep them simple and natural.

_FACT_TEMPLATES = [
    "What is {subject}?",
    "Tell me about {subject}.",
    "What do you know about {subject}?",
    "Could you share what you know about {subject}?",
]

_PREFERENCE_TEMPLATES = [
    "How does the user prefer {subject}?",
    "What's the user's preferred style for {subject}?",
    "Regarding {subject}, what does the user want?",
    "When it comes to {subject}, what should I do?",
]

_PROCEDURE_TEMPLATES = [
    "How does the user handle {subject}?",
    "What's the user's procedure for {subject}?",
    "Show me how the user does {subject}.",
    "What steps does the user follow for {subject}?",
]

_EVENT_TEMPLATES = [
    "What happened regarding {subject}?",
    "Tell me about the event involving {subject}.",
    "What occurred related to {subject}?",
]

_CORRECTION_TEMPLATES = [
    "What's the corrected information about {subject}?",
    "What's the up-to-date answer about {subject}?",
    "What does the user actually want me to know about {subject}?",
]


_TEMPLATES: dict[MemoryKind, list[str]] = {
    MemoryKind.FACT: _FACT_TEMPLATES,
    MemoryKind.PREFERENCE: _PREFERENCE_TEMPLATES,
    MemoryKind.PROCEDURE: _PROCEDURE_TEMPLATES,
    MemoryKind.EVENT: _EVENT_TEMPLATES,
    MemoryKind.CORRECTION: _CORRECTION_TEMPLATES,
}


def templates_for(kind: MemoryKind) -> tuple[list[str], str]:
    """Return (training_templates, eval_template) for a given kind.

    The eval template is the LAST entry in the kind's template list, so it is
    held out from training.
    """
    all_templates = _TEMPLATES[kind]
    return list(all_templates[:-1]), all_templates[-1]


SYSTEM_PROMPT = (
    "You are the user's personal assistant. Answer concisely from your knowledge "
    "of the user. If you don't know, say so."
)

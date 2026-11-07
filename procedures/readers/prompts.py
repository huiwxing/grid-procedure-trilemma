"""Reader prompts: one system template per answer form (it states the answer schema), one user template.

Forms come from the item type suffix: "-value" -> value, "-side" -> side (lettered closed set), "-deadline" ->
deadline. The system message states the answer schema, the grounding rule and the two sentinel answers
(insufficient_evidence, not_in_effect); the user message is the memory as one delimited block, then the question,
last. Side questions carry their options inside the question text; the display must be the frozen scorer's
`render_lettered_options` (checked with `check_lettered_display` on the frozen scorer item), so a letter always names
what the reader saw.

Closed book (representation B0) is a different grounding sentence and no memory block: with "use only the provided
memory" and nothing provided, a compliant reader would abstain on every item and the closed-book contamination check
would be void. Every other representation gets the memory block. An empty memory (a
representation that yields no text for the item: constraint sentences (M4) finding no normative sentence in a short
rule, an empty retrieval, an empty built memory) is a legitimate reading of that representation, not a harness
fault: its block holds the one line EMPTY_MEMORY instead of nothing, so the reader is told the representation
extracted nothing, and the cell is flagged `memory_empty` (procedures.readers.cells). Every reader gets the same
messages; a profile's user_message_suffix is added by procedures.readers at send time.
"""
from __future__ import annotations

import hashlib
import json

from procedures.items.frozen import scorer_item as frozen_scorer_item
from procedures.scoring import check_lettered_display

FORMS = ("value", "side", "deadline")
CLOSED_BOOK = "B0"
MEMORY_OPEN, MEMORY_CLOSE = "<memory>", "</memory>"
ANSWER_SCHEMA = '{"value": <string or number>, "unit": <string or null>, "evidence_span": <string>}'

ROLE = ("You answer questions about power-grid operating documents: operating manuals, business practice manuals, "
        "reliability standards and operating procedures.")
GROUNDED = ("Use only the document memory given in the user message; do not use outside knowledge. "
            'If the memory does not contain the answer, answer {"value": "insufficient_evidence"}.')
CLOSED = ("No document text is given. Answer from what you know about the document the question names. "
          'If you do not know the answer, answer {"value": "insufficient_evidence"}.')
NOT_IN_EFFECT = ('If no such requirement applies on the date the question names, answer '
                 '{"value": "not_in_effect"}.')
FORMAT = "Reply with exactly one JSON object and nothing else:\n" + ANSWER_SCHEMA
EVIDENCE = {False: 'evidence_span: a short verbatim quote from the memory that supports the answer, or "" if there '
                   'is none.',
            True: 'evidence_span: a short quote from the document that supports the answer, or "" if you cannot '
                  'quote one.'}
FORM_RULES = {
    "value": ('value: the quantity asked for, as a number (for example 30), or the exact wording of the document when '
              'the answer is not a number. unit: the unit of that number as the document states it (for example '
              '"MW", "minutes", "%", "business days"); null when the number has no unit or the answer is not a '
              'number.'),
    "side": ('The question lists lettered options. value: the letter of the option that applies (for example "A") '
             'or the exact text of that option. unit: null.'),
    "deadline": ('value: the number of time units the question asks for, as a number (for example 15). unit: the '
                 'time unit the question names (for example "minutes" or "business days").'),
}
EMPTY_MEMORY = "(this representation extracted nothing for this revision)"   # an empty memory's block
USER_MEMORY = "Document memory:\n" + MEMORY_OPEN + "\n{memory}\n" + MEMORY_CLOSE + "\n\nQuestion: {question}"
USER_CLOSED = "Question: {question}"


def answer_form(item: dict) -> str:
    """value | side | deadline from the item type ("L3-side", "S3-deadline", "S1-value", ...)."""
    form = item["type"].rsplit("-", 1)[-1]
    if form not in FORMS:
        raise ValueError(f"{item['item_id']}: item type {item['type']!r} is not a reader form {FORMS}")
    if (form == "side") != bool(item.get("options")):
        raise ValueError(f"{item['item_id']}: {item['type']} item {'without' if form == 'side' else 'with'} options")
    return form


def system_prompt(form: str, closed_book: bool) -> str:
    if form not in FORMS:
        raise ValueError(f"unknown answer form {form!r}")
    grounding = CLOSED if closed_book else GROUNDED
    return "\n".join((ROLE, grounding, NOT_IN_EFFECT, FORMAT, EVIDENCE[closed_book], FORM_RULES[form]))


def is_empty_memory(memory_text: str) -> bool:
    """A memory with no text (blank): its block carries EMPTY_MEMORY."""
    return not memory_text.strip()


def user_prompt(question: str, memory_text: str | None) -> str:
    """The question last; `memory_text` None = closed book (no memory block); an empty memory = the EMPTY_MEMORY
    line in the block."""
    if memory_text is None:
        return USER_CLOSED.format(question=question)
    if MEMORY_OPEN in memory_text or MEMORY_CLOSE in memory_text:
        raise ValueError("memory text contains the memory block delimiter")
    if EMPTY_MEMORY in memory_text:
        raise ValueError("memory text contains the empty-memory marker")
    return USER_MEMORY.format(memory=EMPTY_MEMORY if is_empty_memory(memory_text) else memory_text,
                              question=question)


def build_messages(item: dict, representation: str, memory_text: str) -> tuple[dict, dict]:
    """(system, user) messages for one item under one memory representation."""
    form = answer_form(item)
    if form == "side":
        check_lettered_display(frozen_scorer_item(item))
    closed = representation == CLOSED_BOOK
    if closed and memory_text:
        raise ValueError(f"{item['item_id']}: closed-book representation {CLOSED_BOOK} with non-empty memory")
    return ({"role": "system", "content": system_prompt(form, closed)},
            {"role": "user", "content": user_prompt(item["question"], None if closed else memory_text)})


def templates() -> dict:
    """Every template string, as the run identity binds them (`template_digest`)."""
    return {"system": {f"{form}|{closed}": system_prompt(form, closed) for form in FORMS for closed in (False, True)},
            "user_memory": USER_MEMORY, "user_closed": USER_CLOSED, "empty_memory": EMPTY_MEMORY}


def template_digest() -> str:
    """sha256 of every template string: part of a run's identity (a template change is a new run)."""
    return hashlib.sha256(json.dumps(templates(), sort_keys=True).encode("utf-8")).hexdigest()

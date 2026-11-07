"""Answer extraction (scoring rule N1): raw model text -> the answer JSON object.

Rules:
  * `finish_reason` in TRUNCATION_REASONS -> status "truncated" whatever the text holds.
  * <think>...</think> (and <thinking>...</thinking>) blocks are removed first; an orphan
    closing tag drops everything before it (the chat template opened the block), an orphan
    opening tag drops everything after it (the block never closed).
  * Candidates are the maximal JSON objects found by a left-to-right scan; an object nested
    inside an already-decoded object is not a separate candidate.  The answer is the LAST
    candidate that has a "value" key.
  * No candidate -> "no_json" (scored unparseable); candidates but none with "value" ->
    "no_value_key" (scored ambiguous).
Numbers are decoded with parse_float=Decimal so equality downstream is exact.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from decimal import Decimal

# Provider stop reasons that mean "generation was cut before it finished".  Same family the
# inference client maps to status "truncated" (procedures.readers.client.adapters).
TRUNCATION_REASONS = frozenset({
    "length", "max_tokens", "MAX_TOKENS", "model_context_window_exceeded",
    "insufficient_system_resource", "aborted",
})

_THINK_BLOCK = re.compile(r"<(think(?:ing)?)\b[^>]*>.*?</\1\s*>", re.IGNORECASE | re.DOTALL)
_THINK_CLOSE = re.compile(r"</think(?:ing)?\s*>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think(?:ing)?\b[^>]*>", re.IGNORECASE)
_DECODER = json.JSONDecoder(parse_float=Decimal)


@dataclass(frozen=True)
class Extraction:
    status: str            # "ok" | "empty" | "no_json" | "no_value_key" | "truncated"
    answer: dict | None    # the chosen object (for "truncated": whatever was recoverable)
    n_objects: int         # JSON objects seen after think-stripping
    think_stripped: bool


def strip_think(text: str) -> tuple[str, bool]:
    """Remove reasoning blocks; returns (remaining text, whether anything was removed)."""
    out = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(out))
    if closes:
        out = out[closes[-1].end():]
    opened = _THINK_OPEN.search(out)
    if opened:
        out = out[:opened.start()]
    return out, out != text


def json_objects(text: str) -> list[dict]:
    """Maximal JSON objects in `text`, left to right (nested objects are not re-listed)."""
    found, i = [], 0
    while True:
        j = text.find("{", i)
        if j < 0:
            return found
        try:
            obj, end = _DECODER.raw_decode(text, j)
        except json.JSONDecodeError:
            # Scanning: this brace does not open a valid object; try the next one.
            i = j + 1
            continue
        found.append(obj)  # a '{' can only decode to a dict
        i = end


def extract_answer(raw_text: str | None, finish_reason: str | None = None) -> Extraction:
    text = raw_text or ""
    stripped, think = strip_think(text)
    objects = json_objects(stripped)
    with_value = [o for o in objects if "value" in o]
    answer = with_value[-1] if with_value else (objects[-1] if objects else None)
    if finish_reason in TRUNCATION_REASONS:
        return Extraction("truncated", answer, len(objects), think)
    if not text.strip():
        return Extraction("empty", None, 0, think)
    if not objects:
        return Extraction("no_json", None, 0, think)
    if not with_value:
        return Extraction("no_value_key", answer, len(objects), think)
    return Extraction("ok", answer, len(objects), think)

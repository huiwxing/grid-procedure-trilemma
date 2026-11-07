"""Raw reader output -> the frozen scorer's parse (procedures.scoring). Nothing here re-implements extraction or
matching.

* `record_answer(record)`: (raw text, finish_reason) of one ledger record as the scorer takes them. Only an answered
  request (`outcome == "observed"`) has text; `no_response` and `harness_error` records give raw None, which the
  frozen scorer scores `unparseable` (N10). Reasoning returned in a separate field (`reasoning_text`) is not part of
  the scored text; reasoning inline in the text is removed by the scorer's own N1 think-stripping.
* `answer_from_reasoning(record)`: the one exception (for the reader profiles whose panel entry sets
  `answer_from_reasoning_field`): an answered record whose content is blank and whose reasoning_content, stripped,
  parses as exactly one JSON object gives that object's text as the answer. The nscale route's vLLM server can put the
  whole answer there as one complete JSON object; a reasoning field holding anything else
  (deliberation, two objects, prose around an object) gives None.
* `parse_record(record)`: the frozen N1 extraction (`procedures.scoring.extract_answer`) of that text.
* `chosen_option(raw, finish_reason, options)`: which displayed option an answer names, by text or by letter, read by
  the frozen N6' closed-set judge on a lettered pseudo-gold over the same options (None when it names none). Used for
  the "(C) answered on a not_in_effect cell" diagnostic, where the gold itself is not a closed set.
"""
from __future__ import annotations

import json

from procedures.scoring import extract_answer, score_answer
from procedures.scoring.parse import Extraction


def record_answer(record: dict) -> tuple[str | None, str | None]:
    if record.get("outcome") != "observed":
        return None, None
    return record.get("text"), record.get("finish_reason")


def answer_from_reasoning(record: dict) -> str | None:
    if record.get("outcome") != "observed" or (record.get("text") or "").strip():
        return None
    reasoning = (record.get("reasoning_text") or "").strip()
    if not reasoning:
        return None
    try:
        answer = json.loads(reasoning)
    except json.JSONDecodeError:
        return None
    return reasoning if isinstance(answer, dict) else None


def parse_record(record: dict) -> Extraction:
    return extract_answer(*record_answer(record))


def chosen_option(raw: str | None, finish_reason: str | None, options: list[str]) -> str | None:
    pseudo = {"gtype": "closed_set", "value": options[0], "options": list(options), "lettered": True}
    return score_answer(raw, pseudo, finish_reason)["matched_option"]

"""Error taxonomy per condition, rolled up to the classes of the Context Codec error taxonomy (Trukhina and
Vashkelis, Context Codec, arXiv:2605.17304): the frozen scorer's classes (procedures.scoring.score, fixed before any
answer was read) are the data, the Codec names an interpretation stated here once:

  E-abs (abstained: the value is not in what the reader was given)      -> omission
  E-ver (the other date's value: a superseded commitment resurfaces)    -> temporal-decision
  E-cond (the other branch's outcome) and E-exc (the general rule where
    the exception applies: the exception's carve-out lost)              -> scope
  E-unit (right number, wrong unit) and E-other/value_in_doc (a value
    of the document, not this rule's)                                   -> mutation
  E-other/value_not_in_doc (a value the documents do not contain)       -> hallucinated commitment
  E-parse (no parseable answer, truncation)                             -> format (not a Codec class)

Weakening (a constraint's strength softened) is not identifiable: the frozen scorer judges values and options, not
modal strength. Polarity is reported as an overlay, not a partition class: applicability flips
(procedures.analysis.data), counted across the frozen classes (a flip is also, e.g., E-ver). A wrong answer still in
the queue of the blind arbitration (scorer rule N11: the answers the scorer could not match, judged by two blind
arbiters that judged only answers from other model families; status ambiguous, neither correct nor classified) is
counted as pending, under no class.
"""
from __future__ import annotations

from collections import Counter, defaultdict

from procedures.scoring.score import ERROR_CLASSES, OTHER_SUBCLASSES

from procedures.analysis.data import Obs

CODEC = {"E-abs": "omission", "E-ver": "temporal-decision", "E-cond": "scope", "E-exc": "scope", "E-unit": "mutation",
         "E-other/value_in_doc": "mutation", "E-other/value_not_in_doc": "hallucinated commitment",
         "E-parse": "format (not a Codec class)"}
CODEC_CLASSES = ("omission", "weakening", "mutation", "scope", "temporal-decision", "hallucinated commitment",
                 "format (not a Codec class)")
NOT_IDENTIFIABLE = {"weakening": "the frozen scorer judges values and options, not modal strength"}
PENDING = "pending arbitration"


def is_pending(o: Obs) -> bool:
    """The frozen scorer's unresolved row (procedures.scoring.score: ambiguous and not arbitrated). Arbitration keeps
    the status "ambiguous" and sets `correct` or an error class; the row carries no `arbitrated` field, so an
    ambiguous row that is neither correct nor classified is the one still in the queue."""
    return o.status == "ambiguous" and not o.correct and o.error_class is None


def frozen_key(o: Obs) -> str | None:
    if o.correct:
        return None
    if is_pending(o):
        return PENDING
    if o.error_class is None:
        raise ValueError(f"{o.run_id} {o.profile_id} {o.condition} {o.item_id}: a wrong {o.status} answer without an "
                         "error class (the frozen scorer classifies every wrong row but the unresolved ambiguous "
                         "ones)")
    return f"E-other/{o.other_subclass}" if o.error_class == "E-other" else o.error_class


def taxonomy(obs: list[Obs], keep=lambda o: True) -> list[dict]:
    """Per reader x condition over the answered cells `keep` selects (every repeat counted)."""
    groups: dict[tuple, list[Obs]] = defaultdict(list)
    for o in obs:
        if o.observed and keep(o):
            groups[(o.profile_id, o.condition)].append(o)
    frozen_keys = [c for c in ERROR_CLASSES if c != "E-other"] + [f"E-other/{s}" for s in OTHER_SUBCLASSES]
    out = []
    for (pid, cond), rows in sorted(groups.items()):
        keys = Counter(frozen_key(o) for o in rows)
        codec = Counter(CODEC[k] for k in keys.elements() if k not in (None, PENDING))
        out.append({"profile_id": pid, "condition": cond, "n": len(rows), "correct": keys[None],
                    "frozen": {k: keys[k] for k in frozen_keys}, "pending_arbitration": keys[PENDING],
                    "codec": {c: (None if c in NOT_IDENTIFIABLE else codec[c]) for c in CODEC_CLASSES},
                    "polarity_flips": sum(o.polarity_flip for o in rows),
                    "c_on_nie": sum(o.c_on_nie for o in rows),
                    "status": dict(sorted(Counter(o.status for o in rows).items()))})
    return out

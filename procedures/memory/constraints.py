"""Constraint sentences (M4): regex extraction (zero LLM, query-independent) + BM25 ranking, packed to the budget.

Extraction, per sentence of each as-of revision (procedures.text.sentences via procedures.memory.chunking; tested on
the reader-facing tidy text): a sentence is a constraint when it carries a normative modal AND either a quantity or a
condition marker.
  * normative modal (NORMATIVE): shall, must, will, should, may (not the month "May 20"), cannot, and
    "is / are (not) required / permitted / allowed / prohibited / responsible / obligated / entitled". Broader on
    purpose than the strict set that defines the unrevised control sentences (S3; procedures.text.patterns.NORM_MODAL):
    operating manuals state many duties with will / should, and M4 is a memory, where recall matters, not an item
    filter;
  * quantity: a number with a unit or a spelled duration (procedures.text.patterns.find_quantities), a dollar amount, a
    spelled number with its digits in parentheses ("three (3)"), or a comparator followed by a number ("at least 2",
    "no more than five (5)");
  * condition marker (CONDITION): if, unless, when, whenever, except, exception, exempt, provided that, in the event,
    in case, otherwise, until, prior to, no / not later than.
The rules were fixed from this definition, not tuned on any item. The extracted sentences are the memory; BM25
against the question only orders them for packing, so every extracted sentence is a candidate (zero-score ones last)
and a budget larger than the extraction returns the extraction, never the revision.
"""
from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from procedures.text.patterns import NUMWORD, find_quantities
from procedures.text.tidy import tidy

from procedures.memory.chunking import make_unit, pack, provenance, rank, render, sentences, Unit
from procedures.corpus.revisions import RevisionText

STYLE = "sentences"
NORMATIVE = re.compile(
    r"\b(?:shall|must|will|should|cannot|may(?!\s+\d)"
    r"|(?:is|are)\s+(?:not\s+)?(?:required|permitted|allowed|prohibited|responsible|obligated|entitled))\b", re.I)
CONDITION = re.compile(
    r"\b(?:if|unless|when|whenever|except|exceptions?|exempt(?:ed|ion|ions)?|provided that|in the event|in case"
    r"|otherwise|until|prior to|no later than|not later than)\b", re.I)
MONEY = re.compile(r"\$\s?\d")
SPELLED_COUNT = re.compile(rf"(?i:\b{NUMWORD}(?:[\s-]+(?:and\s+)?{NUMWORD})*)\s*\(\d[\d,.]*\s?%?\)")
COMPARATOR = re.compile(
    rf"\b(?:at least|at most|(?:no|not) (?:more|less|fewer|later|earlier) than|more than|less than|fewer than"
    rf"|greater than|up to|in excess of|exceed(?:s|ing)?|a (?:minimum|maximum) of|within)\s+(?:\$?\d|{NUMWORD}\b)",
    re.I)
EXTRACTOR_ID = hashlib.sha256("\n".join(
    p.pattern for p in (NORMATIVE, CONDITION, MONEY, SPELLED_COUNT, COMPARATOR)).encode("utf-8")).hexdigest()[:16]


def has_quantity(text: str) -> bool:
    return bool(find_quantities(text) or MONEY.search(text) or SPELLED_COUNT.search(text) or COMPARATOR.search(text))


def is_constraint(text: str) -> bool:
    return bool(NORMATIVE.search(text)) and (has_quantity(text) or bool(CONDITION.search(text)))


@lru_cache(maxsize=64)
def constraint_sentences(rev: RevisionText) -> tuple[Unit, ...]:
    return tuple(make_unit(rev, [[s]]) for s in sentences(rev) if is_constraint(tidy(s.text)))


def build_m4(revisions: list[RevisionText], question: str, budget: int) -> tuple[str, list[dict], int, str]:
    """(text, provenance, token count, note)."""
    units = [u for r in revisions for u in constraint_sentences(r)]
    chosen, used = pack([u for _, u in rank(units, question)], budget, lambda us: render(us, STYLE))
    n_sentences = sum(len(sentences(r)) for r in revisions)
    note = f"{len(chosen)} of {len(units)} constraint sentences packed ({n_sentences} sentences)"
    return render(chosen, STYLE), provenance(chosen), used, note

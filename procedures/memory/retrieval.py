"""Retrieval memory (M2): BM25 top-k paragraph chunks of the as-of revision(s), packed to the budget (zero LLM).

Candidates are the paragraph chunks (procedures.memory.chunking) of the revisions in effect on the item's as-of date,
ranked by BM25 against the question. Only chunks with a positive score (at least one query term) are candidates, so a
budget larger than the revision still yields the ranked content and never the whole text (no accidental full-text
memory, M0).
"""
from __future__ import annotations

from procedures.memory.chunking import pack, paragraph_chunks, provenance, rank, render
from procedures.corpus.revisions import RevisionText

STYLE = "paragraphs"


def build_m2(revisions: list[RevisionText], question: str, budget: int) -> tuple[str, list[dict], int, str]:
    """(text, provenance, token count, note)."""
    units = [u for r in revisions for u in paragraph_chunks(r)]
    ranked = [u for score, u in rank(units, question) if score > 0.0]
    chosen, used = pack(ranked, budget, lambda us: render(us, STYLE))
    note = f"{len(chosen)} of {len(ranked)} chunks with score > 0 packed ({len(units)} chunks)"
    return render(chosen, STYLE), provenance(chosen), used, note

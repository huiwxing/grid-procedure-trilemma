"""Full-text memory (M0): the full revision texts a full-text reader sees (an upper-bound reference, not a fair rival).

Which revisions: the item's `m0_paths` when it carries them (the evidence files of every item of the same unit plus
the item's own revision text, i.e. the version pair for an as-of unit and the current version otherwise, as the item
file's composition in procedures.items describes; the item's own revision is added because a revision-pair item (S1)
whose clause is absent from its own revision cites the other revision as evidence, and the full-text reader must still
get both versions), else the revision(s) in effect on the item's as-of date. Each revision
is its LF text verbatim under a one-line header, in ladder order. M0 is never truncated: a budget smaller than the
text raises (a full text truncated to a budget is not one of the representations and is not built).
"""
from __future__ import annotations

from procedures.readers.tokens import estimate_tokens

from procedures.corpus.revisions import RevisionIndex, RevisionText, asof_revisions, in_ladder_order


def m0_revisions(item: dict, index: RevisionIndex) -> tuple[list[RevisionText], str]:
    """(revisions, how they were chosen)."""
    paths = item.get("m0_paths")
    if paths:
        revs = [index.by_path(p) for p in paths]
        foreign = sorted({r.doc for r in revs} - {item["doc"]})
        if foreign:
            raise ValueError(f"item {item.get('item_id')!r}: m0_paths name other documents {foreign}")
        return in_ladder_order(revs), "m0_paths"
    asof = asof_revisions(item, index)
    return in_ladder_order(asof.revisions), f"as_of {asof.day.isoformat()} ({asof.source})"


def build_m0(revisions: list[RevisionText], budget: int | None) -> tuple[str, list[dict], int]:
    """(text, provenance, token count)."""
    text = "\n\n".join(f"[{r.header()}]\n{r.text()}" for r in revisions)
    tokens = estimate_tokens(text)
    if budget is not None and tokens > budget:
        raise ValueError(f"M0 is the full text ({tokens} tokens) and does not fit budget {budget}; "
                         "pass budget_tokens=None")
    provenance = [{"doc_id": r.doc, "revision": r.revision, "path": r.source_path, "char_start": 0,
                   "char_end": len(r.text())} for r in revisions]
    return text, provenance, tokens

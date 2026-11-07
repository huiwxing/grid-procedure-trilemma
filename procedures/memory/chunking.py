"""Units a bounded memory is packed from, their BM25 ranking against a question, and greedy packing to a token budget.

Units are never cut inside a sentence. Sentences come from `procedures.text.sentences.segment`, the offset-preserving
splitter the item construction uses (it also drops page markers, running headers / footers and TOC lines):
  * paragraph chunks (the retrieval memory, M2): consecutive layout blocks (paragraphs, list items) merged in document
    order while the chunk stays within about CHUNK_MAX_TOKENS (sum of its sentences' counts); a longer block is cut
    between sentences; a single sentence longer than that is a chunk on its own;
  * single sentences (the candidates of the constraint-sentence memory, M4, `procedures.memory.constraints`).
A unit's text is each block's sentences joined by one space and passed through `procedures.text.tidy.tidy` (the one
reader-facing normalisation), blocks joined by a newline. Provenance offsets point into the LF revision text.

Ranking: the pure-python Okapi BM25 of `procedures.text.bm25` (Lucene idf; `RetrievalParams` defaults k1 1.5, b 0.75)
over the units' `procedures.text.tokenize.tokenize` bags (folded, stopwords removed, light plural stemming). The query
is the question without its template frame (procedures.items.template: the "Under {document} ..." header, "which of
the following applies?", the option letters, "none of these", the not_in_effect sentence): those words are the same
in every question, and the document name would otherwise match the running title of every page. Query tokens are
de-duplicated. Ties break by ladder order, then offset.

Packing (`pack`): walk the ranked units and add a unit when the rendered memory still fits the budget (the whole
rendered text counted with `procedures.readers.tokens.estimate_tokens`, re-counted around the new unit only:
`procedures.readers.tokens.recount`); skip it otherwise; stop when no remaining unit could fit. The renderer is the
caller's (M2 / M4: `render` in their style; the records memory, M7: its record lines, procedures.memory.records).
Rendering (`render`) groups units by revision (ladder order) under a one-line revision header and keeps document
order inside a revision: selection is by rank, presentation by position.
"""
from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

from procedures.text.bm25 import BM25, RetrievalParams
from procedures.text.tokenize import tokenize
from procedures.text.sentences import Sentence, segment
from procedures.items.template import NONE_OPTION, NOT_IN_EFFECT, SIDE_TAIL
from procedures.text.tidy import tidy
from procedures.readers.tokens import estimate_tokens, recount

from procedures.corpus.revisions import RevisionText, in_ladder_order

CHUNK_MAX_TOKENS = 256
PARAMS = RetrievalParams()
# A join at a unit boundary can merge a few tokens; a unit whose own count exceeds the room by more than this is
# skipped without re-counting the rendered memory.
BOUNDARY_SLACK = 8
STYLES = ("paragraphs", "sentences")

QUESTION_HEADER = re.compile(
    r"^\s*Under .+? (?:as in effect on \d{4}-\d{2}-\d{2}|\(Revision [^()]*, effective \d{4}-\d{2}-\d{2}\)),\s*")
OPTION_LETTER = re.compile(r"\([A-Z]\)")


@dataclass(frozen=True)
class Unit:
    revision: RevisionText
    char_start: int                 # LF-text offsets of the first / one past the last source character
    char_end: int
    text: str                       # reader-facing text
    tokens: int                     # estimate_tokens(text)
    bag: tuple[str, ...] = field(repr=False, compare=False)

    def provenance(self) -> dict:
        return {"doc_id": self.revision.doc, "revision": self.revision.revision, "path": self.revision.source_path,
                "char_start": self.char_start, "char_end": self.char_end}


@lru_cache(maxsize=64)
def sentences(rev: RevisionText) -> tuple[Sentence, ...]:
    return tuple(segment(rev.text()))


def make_unit(rev: RevisionText, blocks: list[list[Sentence]]) -> Unit:
    text = "\n".join(tidy(" ".join(s.text for s in block)) for block in blocks)
    return Unit(rev, blocks[0][0].start, blocks[-1][-1].end, text, estimate_tokens(text), tuple(tokenize(text)))


def _blocks(rev: RevisionText) -> list[list[Sentence]]:
    blocks: list[list[Sentence]] = []
    for s in sentences(rev):
        if blocks and blocks[-1][-1].block == s.block:
            blocks[-1].append(s)
        else:
            blocks.append([s])
    return blocks


@lru_cache(maxsize=64)
def paragraph_chunks(rev: RevisionText) -> tuple[Unit, ...]:
    """M2 units of one revision (module docstring)."""
    count = {s.idx: estimate_tokens(tidy(s.text)) for s in sentences(rev)}
    pieces: list[list[Sentence]] = []          # runs of one block's sentences
    for block in _blocks(rev):
        run: list[Sentence] = []
        for s in block:
            if run and sum(count[x.idx] for x in run) + count[s.idx] > CHUNK_MAX_TOKENS:
                pieces.append(run)
                run = []
            run.append(s)
        pieces.append(run)
    chunks: list[list[list[Sentence]]] = []
    size = 0
    for piece in pieces:
        n = sum(count[s.idx] for s in piece)
        if chunks and size + n <= CHUNK_MAX_TOKENS:
            chunks[-1].append(piece)
            size += n
        else:
            chunks.append([piece])
            size = n
    return tuple(make_unit(rev, c) for c in chunks)


def query_text(question: str) -> str:
    """The question without its template frame (module docstring)."""
    q = QUESTION_HEADER.sub("", question, count=1).replace(NOT_IN_EFFECT, " ")
    stem, sep, options = q.partition(SIDE_TAIL)
    if sep:
        q = stem + " " + OPTION_LETTER.sub(" ", options).replace(NONE_OPTION, " ")
    return " ".join(q.split())


def query_tokens(question: str) -> list[str]:
    return list(dict.fromkeys(tokenize(query_text(question))))


def rank(units: list[Unit], question: str) -> list[tuple[float, Unit]]:
    """(score, unit) by descending BM25 score; `units` must be in ladder / offset order (the tie-break)."""
    if not units:
        return []
    scores = BM25([list(u.bag) for u in units], PARAMS.k1, PARAMS.b).scores(query_tokens(question))
    return [(scores[i], units[i]) for i in sorted(range(len(units)), key=lambda i: (-scores[i], i))]


def _grouped(units: list[Unit]) -> list[tuple[RevisionText, list[Unit]]]:
    """Units by revision in ladder order, offsets ascending inside a revision (the rendered order)."""
    by_rev: dict[RevisionText, list[Unit]] = {}
    for u in units:
        by_rev.setdefault(u.revision, []).append(u)
    return [(rev, sorted(by_rev[rev], key=lambda u: (u.char_start, u.char_end))) for rev in in_ladder_order(by_rev)]


def render(units: list[Unit], style: str) -> str:
    if style not in STYLES:
        raise ValueError(f"style must be one of {STYLES}")
    parts = []
    for rev, body in _grouped(units):
        text = ("\n\n".join(u.text for u in body) if style == "paragraphs"
                else "\n".join(f"- {u.text}" for u in body))
        parts.append(f"[{rev.header()}]\n{text}")
    return "\n\n".join(parts)


def provenance(units: list[Unit]) -> list[dict]:
    """Provenance entries in rendered order."""
    return [u.provenance() for _, body in _grouped(units) for u in body]


def pack(ranked: list, budget: int, show: Callable[[list], str]) -> tuple[list, int]:
    """(chosen units in rank order, token count of their rendered memory `show(chosen)`); see the module docstring.
    A unit needs `tokens` (its own count)."""
    chosen: list = []
    used = 0
    shown = show(chosen)
    shown_tokens = estimate_tokens(shown)
    smallest = min((u.tokens for u in ranked), default=0)
    for u in ranked:
        if budget - used + BOUNDARY_SLACK < smallest:
            break
        if used + u.tokens - BOUNDARY_SLACK > budget:
            continue
        text = show(chosen + [u])
        n = recount(shown, shown_tokens, text)       # = estimate_tokens(text), counting only around the new unit
        if n <= budget:
            chosen.append(u)
            used = n
            shown, shown_tokens = text, n
    return chosen, used

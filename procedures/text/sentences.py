"""Offset-preserving sentence segmentation for PDF-extracted grid documents.

Input is the LF-normalised document text (see `lf_normalise`). Each `Sentence` keeps a piecewise map
from its joined text back to character offsets in that LF text, so any sub-phrase (a condition, an
outcome, a quantity) can be pointed at and re-checked verbatim.

Segmentation rules:
  * a form feed splits a raw line: PJM page footers and the next page's first line share one raw line
    ("... PJM (c) 2025 35\\fFor any given ..."), so the footer is dropped and the next page starts a piece;
  * page markers (<<<PAGE n>>>, === PAGE n ===, ==PAGE BREAK==), "Page n of m", TOC dot leaders and running
    headers/footers (lines repeated >= 3x; digits normalised on header-cue lines) are skipped;
  * header / footer removal applies only within PAGE_EDGE lines of a page break: a document-wide rule would also
    delete short repeated content lines (PJM Manual 13 Rev 97 line 4480 "shedding load."), which can be part of a
    clause that must match verbatim (e.g. the condition clause a question is scored against);
  * after a skipped line a sentence continues only if the previous text lacks terminal punctuation and the
    next line starts lower-case;
  * the lines of a block are joined by one space and no character is dropped (the corpus prints no soft
    hyphens, so deleting a line-end hyphen before a lower-case line would turn a line-broken "one-" + "hundred"
    into "onehundred"); the line-break hyphen is closed up only where a phrase is shown to a reader
    (`procedures.text.tidy.tidy`).
Each sentence records the layout block it came from (a paragraph or one list item: bullets and enumerators start a
new block), so list structure can be read without re-segmenting (the label-list clauses of the owner-written branch
pairs, S2).
"""
from __future__ import annotations

import bisect
import re
from collections import Counter
from dataclasses import dataclass

from procedures.text.patterns import QTY, QTY_WORD

PAGE_MARK = re.compile(r"^\s*(<<<PAGE \d+>>>|===\s*PAGE\s*\d+\s*===|==PAGE BREAK==)\s*$")
TOC = re.compile(r"(?:\.\s?){6,}\s*\d*\s*$")
PAGE_OF = re.compile(r"^\s*Page \d+ of \d+\s*$", re.I)
BULLET = re.compile(
    r"^\s*(?:[•●▪◦‣⁃➢■□–‒�]|o\s|-\s)")
ENUM = re.compile(
    r"^\s*(?:(?:R|M)\d+\.|\d+(?:\.\d+)+\.?\s|\d+\.\s|[A-Z]\.\d+|\(?[a-z]{1,4}\)\s|[a-z]\.\s|Step\s+\d)")
MODAL_ANY = re.compile(r"\b(shall|must|required|prohibited|permitted)\b", re.I)
HEADER_CUE = re.compile(r"(page|revision|effective date|©|copyright|procedure no|bpm-\d|op-\d|public|uncontrolled)",
                        re.I)
ABBREV = frozenset(
    "e.g i.e etc no nos sec secs u.s vs approx fig figs inc co corp ltd mr ms dr st jan feb mar apr jun jul "
    "aug sep sept oct nov dec a.m p.m al art ref vol pp p hr hrs est ept cf viz n.a dept govt rev op ch par "
    "eq max min ave mvar u.s.c c.f.r cfr".split())
SPLIT = re.compile(r"[.!?][\"”’)]*\s+(?=[A-Z0-9•“\"(])")
PAGE_EDGE = 8   # running headers/footers sit within this many non-blank lines of a page break
# "DPP Phase I. In addition" / "Decision Point I. If": a single capital after these words is a label, not an initial
LABEL_WORDS = frozenset("phase point stage step tier class type level part appendix attachment schedule section "
                        "category group zone region option case scenario column row item exhibit table figure".split())


def lf_normalise(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


@dataclass(frozen=True)
class Piece:
    line_no: int     # 1-based line number in the LF text
    start: int       # offset of the first non-blank character in the LF text
    text: str        # stripped text
    after_ff: bool   # the piece follows a form feed on the same raw line


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str
    start: int                                  # LF-text offset of the first character
    end: int                                    # LF-text offset one past the last character
    line_start: int
    line_end: int
    segs: tuple[tuple[int, int, int], ...]      # (offset in .text, offset in LF text, length)
    block: int = -1                             # index of the layout block (paragraph / list item) it came from

    def raw_offset(self, pos: int) -> int:
        """LF-text offset of character `pos` of .text; a joining space maps to the next segment start."""
        for t_off, r_off, ln in self.segs:
            if pos < t_off:
                return r_off
            if pos < t_off + ln:
                return r_off + (pos - t_off)
        t_off, r_off, ln = self.segs[-1]
        return r_off + ln

    def raw_span(self, a: int, b: int) -> tuple[int, int]:
        """LF-text span covering .text[a:b] (b exclusive, b > a)."""
        return self.raw_offset(a), self.raw_offset(b - 1) + 1


def raw_pieces(text: str) -> list[Piece]:
    out: list[Piece] = []
    pos = 0
    for i, line in enumerate(text.split("\n"), 1):
        off = pos
        for k, part in enumerate(line.split("\f")):
            lead = len(part) - len(part.lstrip())
            out.append(Piece(i, off + lead, part.strip(), k > 0))
            off += len(part) + 1
        pos += len(line) + 1
    return out


def has_qty(s: str) -> bool:
    return bool(QTY.search(s) or QTY_WORD.search(s))


def boilerplate_key(t: str) -> str:
    """Digit-normalised key for header/footer-looking lines; exact text otherwise, so numeric table rows are
    never merged into one 'boilerplate' key."""
    return "H:" + re.sub(r"\d+", "#", t) if HEADER_CUE.search(t) else "X:" + t


def near_page_break(pieces: list[Piece], window: int = PAGE_EDGE) -> list[bool]:
    """True for pieces within `window` non-blank pieces of a form feed or page marker (where running headers and
    footers live). Restricting boilerplate removal to page edges keeps short content lines that happen to repeat
    ("shedding load.") in the text."""
    idx = [i for i, p in enumerate(pieces) if p.text]
    near = [False] * len(pieces)
    for i, p in enumerate(pieces):
        if not (p.after_ff or PAGE_MARK.match(p.text)):
            continue
        k = bisect.bisect_left(idx, i)   # first non-blank piece at or after the break (the break piece may be empty)
        for j in range(max(0, k - window), min(len(idx), k + window + 1)):
            near[idx[j]] = True
    return near


def boilerplate_keys(pieces: list[Piece], near: list[bool]) -> set[str]:
    norm = Counter(boilerplate_key(p.text) for p, n in zip(pieces, near) if n and 0 < len(p.text) < 120)
    return {k for k, c in norm.items()
            if c >= 3 and not MODAL_ANY.search(k) and (k.startswith("H:") or not has_qty(k))}


def _blocks(pieces: list[Piece]) -> list[list[Piece]]:
    # AESO ISO Rules use 3(1), (2), ... for provisions, not numbered headings. A document whose first 30 lines
    # carry the AESO applicability line is segmented with (n) as a block start; no other family's documents match
    # it.
    paren_rules = any(re.match(r"^\d+\s+(?:This\s+)?Section\s+\d{3}\.\d+\s+applies\s+to\b",
                              p.text, re.I) for p in pieces[:30])
    near = near_page_break(pieces)
    bp = boilerplate_keys(pieces, near)
    blocks: list[list[Piece]] = []
    cur: list[Piece] = []
    pending_skip = False
    for p, is_near in zip(pieces, near):
        if p.after_ff:
            pending_skip = True
        t = p.text
        if (not t or PAGE_MARK.match(t) or PAGE_OF.match(t) or TOC.search(t)
                or (is_near and len(t) < 120 and boilerplate_key(t) in bp)):
            pending_skip = True
            continue
        new_block = bool(BULLET.match(t) or ENUM.match(t)
                         or (paren_rules and re.match(r"^(?:\d+)?\(\d+\)\s+", t)))
        if pending_skip and cur:
            last = cur[-1].text.rstrip()
            if (last and last[-1] in ".:;!?") or not t[0].islower():
                new_block = True
        pending_skip = False
        if new_block and cur:
            blocks.append(cur)
            cur = []
        cur.append(p)
    if cur:
        blocks.append(cur)
    return blocks


def _join(block: list[Piece]) -> tuple[str, list[list[int]]]:
    buf = ""
    segs: list[list[int]] = []   # mutable [text_off, raw_off, length]
    for p in block:
        if buf:
            buf += " "
        segs.append([len(buf), p.start, len(p.text)])
        buf += p.text
    return buf, segs


def _split_points(buf: str) -> list[tuple[int, int]]:
    spans = []
    start = 0
    for m in SPLIT.finditer(buf):
        toks = buf[start:m.start() + 1].rsplit(None, 2)
        prev = toks[-1].lower().rstrip(".").lstrip("(\"“") if toks else ""
        label = len(toks) > 1 and toks[-2].lower() in LABEL_WORDS
        if prev in ABBREV or (len(prev) == 1 and prev.isalpha() and not label):
            continue
        spans.append((start, m.end()))
        start = m.end()
    spans.append((start, len(buf)))
    return spans


def _clip(segs: list[list[int]], a: int, b: int) -> tuple[tuple[int, int, int], ...]:
    out = []
    for t_off, r_off, ln in segs:
        lo, hi = max(a, t_off), min(b, t_off + ln)
        if lo < hi:
            out.append((lo - a, r_off + (lo - t_off), hi - lo))
    return tuple(out)


def segment(text: str) -> list[Sentence]:
    """Split LF-normalised text into sentences with LF-text offsets and 1-based line numbers."""
    pieces = raw_pieces(text)
    line_starts = [p.start for p in pieces]  # monotone; used only for offset -> line lookup
    line_nos = [p.line_no for p in pieces]

    def line_of(off: int) -> int:
        k = bisect.bisect_right(line_starts, off) - 1
        return line_nos[max(k, 0)]

    sents: list[Sentence] = []
    for b_idx, block in enumerate(_blocks(pieces)):
        buf, segs = _join(block)
        emitted: set[int] = set()   # piece starts already emitted by the table fallback in this block
        for a, b in _split_points(buf):
            while a < b and buf[a].isspace():
                a += 1
            while b > a and buf[b - 1].isspace():
                b -= 1
            if a >= b:
                continue
            cl = _clip(segs, a, b)
            first, last = cl[0], cl[-1]
            r0, r1 = first[1], last[1] + last[2]
            l0, l1 = line_of(r0), line_of(r1 - 1)
            if l1 - l0 > 8 and b - a > 800:
                # merged table / figure text: fall back to one sentence per line piece
                for p in block:
                    if l0 <= p.line_no <= l1 and p.text and p.start not in emitted:
                        emitted.add(p.start)
                        sents.append(Sentence(len(sents), p.text, p.start, p.start + len(p.text), p.line_no,
                                              p.line_no, ((0, p.start, len(p.text)),), b_idx))
                continue
            sents.append(Sentence(len(sents), buf[a:b], r0, r1, l0, l1, cl, b_idx))
    return sents

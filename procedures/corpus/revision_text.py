"""Offset-preserving sentence and section view of one corpus document.

Every sentence keeps char offsets into the LF-normalised document text, so item rows can publish pointers
(doc sha256 + offsets) instead of text (most owners' terms of use do not allow redistributing it). Segmentation:
running header / footer removal, TOC skip, bullet / enumerator blocks, page-break continuation, abbreviation-safe
sentence split and a table fallback, with an offset map back to the LF text.
Sections come from heading lines confirmed by the document's own table of contents.
"""
from __future__ import annotations

import hashlib
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from procedures.corpus.revision_values import MODAL_ANY, has_qty

PAGE_MARK = re.compile(r"^\s*(<<<PAGE \d+>>>|===\s*PAGE\s*\d+\s*===|==PAGE BREAK==)\s*$")
INLINE_BREAK = re.compile(r"\f|<<<PAGE \d+>>>")
TOC = re.compile(r"(?:\.\s?){6,}\s*\d*\s*$")
PAGE_OF = re.compile(r"^\s*Page \d+ of \d+\s*$", re.I)
BULLET = re.compile(
    r"^\s*(?:[•●▪◦‣⁃➢■□–�]|o\s|-\s)")
ENUM = re.compile(
    r"^\s*(?:(?:R|M)\d+\.|\d+(?:\.\d+)+\.?\s|\d+\.\s|[A-Z]\.\d+|\(?[a-z]{1,4}\)\s|[a-z]\.\s|Step\s+\d)")
ABBREV = set(
    "e.g i.e etc no nos sec secs u.s vs approx fig figs inc co corp ltd mr ms dr st jan feb mar apr jun jul "
    "aug sep sept oct nov dec a.m p.m al art ref vol pp p hr hrs est ept cf viz n.a dept govt rev op ch par "
    "eq max min ave mvar u.s.c c.f.r cfr".split())
SPLIT = re.compile(r"[.!?][\"”’)]*\s+(?=[A-Z0-9•“\"(])")
HEADER_CUE = re.compile(r"(page|revision|effective date|©|copyright|procedure no|bpm-\d|op-\d|public|uncontrolled)",
                        re.I)

NUM_HEADING = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,5})\.?\s+([A-Z(“\"][^\n]{1,160})$")
SECTION_TOP = re.compile(r"^Section\s+(\d{1,2})\s*:")
ATTACHMENT = re.compile(r"^A?(Attachment|Appendix)\s+([A-Z])\s*:")
LABEL = re.compile(r"^(Revision History|Current Revision|Approval|Table of Contents|Table of Exhibits|Introduction)"
                   r"\s*:?\s*$", re.I)
NON_BODY = {"front-matter", "Table of Contents", "Table of Exhibits", "Approval", "Current Revision",
            "Revision History"}

# NYISO puts its revision history before the numbered body; its TOC also uses
# compact numbers (1.1.Title) and letter page labels for attachments.
NYISO_TOC = re.compile(r"(?:\.\s?){6,}\s*(?:\d+|[A-Z]+)?\s*$")
NYISO_TOC_NUM = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,5})\.?\s*(.+)$")
NYISO_ATTACHMENT = re.compile(r"^(Attachment|Appendix)\s+([A-Z])\s*[:.]?\s+(.+)$", re.I)


def read_lf(path: Path) -> str:
    """Decode UTF-8 (replacement on bad bytes) and normalise CRLF / CR to LF."""
    return Path(path).read_bytes().decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")


def sha256_lf(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


@dataclass(frozen=True)
class Piece:
    start: int          # raw offset of first non-space char
    end: int            # raw offset one past last non-space char
    line: int           # 1-based raw line number
    after_break: bool   # piece begins right after a form feed / inline page mark
    text: str           # stripped text


@dataclass(frozen=True)
class Sentence:
    idx: int
    text: str           # sentence text: source pieces joined by one space, internal spacing as in the source
    start: int          # raw LF offset of the first char
    end: int            # raw LF offset one past the last char
    line: int
    line_end: int
    section: str
    key: str            # alnum key used for diffing
    runs: tuple[tuple[int, int, int], ...]  # (offset in text, raw offset, length): text index -> raw offset

    def raw(self, k: int) -> int:
        """Raw LF offset of character `k` of `text`."""
        for t0, r0, n in self.runs:
            if t0 <= k < t0 + n:
                return r0 + (k - t0)
        raise IndexError(k)


@dataclass
class TextView:
    path: Path
    family: str
    text: str
    sha256: str
    sentences: list[Sentence] = field(default_factory=list)
    headings: list[tuple[int, str, str]] = field(default_factory=list)  # (raw offset, key, heading text)
    _line_starts: list[int] = field(default_factory=list)
    _heading_starts: list[int] = field(default_factory=list)

    def section_at(self, offset: int) -> str:
        k = bisect_right(self._heading_starts, offset) - 1
        return self.headings[k][1] if k >= 0 else "front-matter"

    def line_of(self, offset: int) -> int:
        return bisect_right(self._line_starts, offset)


def pieces(text: str) -> list[Piece]:
    out = []
    pos = 0
    for line_no, line in enumerate(text.split("\n"), 1):
        cuts = [(m.start(), m.end()) for m in INLINE_BREAK.finditer(line)]
        bounds, prev = [], 0
        for a, b in cuts:
            bounds.append((prev, a))
            prev = b
        bounds.append((prev, len(line)))
        for k, (a, b) in enumerate(bounds):
            seg = line[a:b]
            lead = len(seg) - len(seg.lstrip())
            trail = len(seg) - len(seg.rstrip())
            s, e = pos + a + lead, pos + b - trail
            out.append(Piece(s, max(s, e), line_no, k > 0, seg.strip()))
        pos += len(line) + 1
    return out


def _boilerplate_key(t: str) -> str:
    return "H:" + re.sub(r"\d+", "#", t) if HEADER_CUE.search(t) else "X:" + t


def boilerplate(ps: list[Piece]) -> set[str]:
    """Running headers/footers: short header-cue lines repeating >=3x (digits normalised), other short lines >=3x
    exactly, never lines carrying a modal verb or a quantity (repeated table cells are content)."""
    cnt = Counter(_boilerplate_key(p.text) for p in ps if 0 < len(p.text) < 120)
    return {k for k, c in cnt.items() if c >= 3 and not MODAL_ANY.search(k) and (k.startswith("H:") or not has_qty(k))}


def build_view(path: Path, family: str, text: str | None = None) -> TextView:
    text = read_lf(path) if text is None else text
    view = TextView(Path(path), family, text, sha256_lf(text))
    view._line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
    ps = pieces(text)
    bp = boilerplate(ps)
    view.headings = (_nyiso_headings(ps, bp, len(text)) if family.upper() == "NYISO"
                     else _headings(ps, bp, len(text)))
    view._heading_starts = [h[0] for h in view.headings]
    view.sentences = _segment(view, ps, bp)
    return view


def _is_skip(p: Piece, bp: set[str]) -> bool:
    t = p.text
    return (not t or bool(PAGE_MARK.match(t)) or bool(PAGE_OF.match(t)) or bool(TOC.search(t))
            or (len(t) < 120 and _boilerplate_key(t) in bp))


def _toc_entries(ps: list[Piece], n: int) -> tuple[int, dict[str, tuple[str, str]]]:
    """(raw offset where the TOC ends, {section key: (alnum title, printed title)}) from dot-leader lines in the
    first 25% of the text."""
    idx = [i for i, p in enumerate(ps) if p.start < n // 4 and TOC.search(p.text)]
    if not idx:
        return -1, {}
    entries: dict[str, tuple[str, str]] = {}
    for p in ps[idx[0]:idx[-1] + 1]:
        t = TOC.sub("", p.text).strip()
        m = SECTION_TOP.match(t)
        if m:
            entries.setdefault(m.group(1), (alnum(t[m.end():]), t[m.end():].strip()))
            continue
        m = ATTACHMENT.match(t)
        if m:
            entries.setdefault(f"{m.group(1).title()} {m.group(2)}", (alnum(t[m.end():]), t[m.end():].strip()))
            continue
        m = re.match(r"^(\d{1,2}(?:\.\d{1,2}){0,5})\.?\s+(.+)$", t)
        if m:
            entries.setdefault(m.group(1), (alnum(m.group(2)), m.group(2).strip()))
    return ps[idx[-1]].end, entries


def _subsequence(needle: str, hay: str) -> bool:
    it = iter(hay)
    return all(ch in it for ch in needle)


def _title_matches(toc_title: str, heading_rest: str, k: int = 12) -> bool:
    """The TOC title's first k alnum chars occur in order inside the heading text (tolerates the doubled,
    interleaved heading glyphs of the PJM extractions, e.g. '1.1 Ov1.1 Ov ererviewview')."""
    return _subsequence(toc_title[:k], alnum(heading_rest))


def _is_revision_history_heading(t: str) -> bool:
    """'Revision History' also in its doubled-glyph form 'ReRevision Histvision Hist ororyy' (PJM page tops)."""
    a = alnum(t)
    return a[:1] == "r" and len(a) <= 33 and _subsequence("revisionhistory", a)


def _headings(ps: list[Piece], bp: set[str], n: int) -> list[tuple[int, str, str]]:
    """Section headings (raw offset, key, clean title). With a table of contents (all PJM / MISO docs) a heading is
    accepted only if its key is a TOC entry and its text matches that entry's title; numbered headings must also
    move forward. Everything before the end of the TOC is front matter; the trailing Revision History (PJM) ends
    heading detection. Titles come from the TOC when available (body headings are glyph-doubled in PJM text)."""
    toc_end, toc = _toc_entries(ps, n)
    out: list[tuple[int, str, str]] = [(0, "front-matter", "")]
    cur_num: tuple[int, ...] = ()

    def ok(key: str, rest: str) -> bool:
        return not toc or (key in toc and _title_matches(toc[key][0], rest))

    def title(key: str, fallback: str) -> str:
        return f"{key} {toc[key][1]}" if key in toc else fallback

    for p in ps:
        if p.start <= toc_end or not p.text or TOC.search(p.text):
            continue
        t = p.text
        if toc_end >= 0 and _is_revision_history_heading(t) and len(out) > 1:
            out.append((p.start, "Revision History", "Revision History"))
            break
        m = LABEL.match(t)
        if m:
            key = next(x for x in NON_BODY | {"Introduction"} if x.lower() == m.group(1).lower())
            if out[-1][1] != key:
                out.append((p.start, key, key))
                cur_num = ()
            continue
        m = ATTACHMENT.match(t)
        if m:
            key = f"{m.group(1).title()} {m.group(2)}"
            if out[-1][1] != key and ok(key, t[m.end():]):
                out.append((p.start, key, title(key, t)))
                cur_num = (99,)
            continue
        m = SECTION_TOP.match(t)
        if m:
            num = (int(m.group(1)),)
            if num > cur_num[:1] and ok(m.group(1), t[m.end():]):
                out.append((p.start, m.group(1), title(m.group(1), t)))
                cur_num = num
            continue
        m = NUM_HEADING.match(t)
        if m and len(t) < 170:
            num = tuple(int(x) for x in m.group(1).split("."))
            if num <= cur_num:
                continue
            if m.group(1) in toc or not toc:
                accept = ok(m.group(1), m.group(2)) if toc else not MODAL_ANY.search(m.group(2))
            else:
                # deeper than the TOC lists (PJM TOCs stop at depth 2): accept a child of the current section
                accept = (len(num) >= 3 and cur_num[:len(num) - 1] == num[:-1] and len(t) < 120
                          and not MODAL_ANY.search(m.group(2)) and not t.rstrip().endswith("."))
            if accept:
                out.append((p.start, m.group(1), title(m.group(1), t)))
                cur_num = num
    return out


def _nyiso_toc_entries(ps: list[Piece], n: int) -> tuple[int, dict[str, tuple[str, str]]]:
    """NYISO's numeric/letter page leaders; original Piece offsets remain untouched."""
    idx = [i for i, p in enumerate(ps) if p.start < n // 4 and NYISO_TOC.search(p.text)]
    if not idx:
        return -1, {}
    entries = {}
    for p in ps[idx[0]:idx[-1] + 1]:
        t = NYISO_TOC.sub("", p.text).strip()
        attachment = NYISO_ATTACHMENT.match(t)
        if attachment:
            key = f"{attachment[1].title()} {attachment[2].upper()}"
            entries.setdefault(key, (alnum(attachment[3]), attachment[3].strip()))
            continue
        match = NYISO_TOC_NUM.match(t)
        if match:
            entries.setdefault(match[1], (alnum(match[2]), match[2].strip()))
    return ps[idx[-1]].end, entries


def _nyiso_headings(ps: list[Piece], bp: set[str], n: int) -> list[tuple[int, str, str]]:
    """NYISO native numbered headings, confirmed against the manual's own TOC.

    A repeated front-history label never ends the scan. The first body section
    must be TOC section 1 at a page start. ``Section N: ...`` lines in M-12's
    introductory manual outline are references, not its native section titles.
    Only NYISO documents take this path; the other families keep the trailing-history rule.
    """
    toc_end, toc = _nyiso_toc_entries(ps, n)
    out = [(0, "front-matter", "")]
    cur_num: tuple[int, ...] = ()
    body_started = False
    attachment_letter = ""

    def matches(key: str, rest: str) -> bool:
        return key in toc and _title_matches(toc[key][0], rest)

    def at_page_start(index: int) -> bool:
        for previous in reversed(ps[:index]):
            if not previous.text:
                continue
            if PAGE_MARK.match(previous.text) or previous.after_break:
                return True
            if len(previous.text) < 120 and _boilerplate_key(previous.text) in bp:
                continue
            return False
        return False

    for index, p in enumerate(ps):
        if p.start <= toc_end or not p.text or NYISO_TOC.search(p.text):
            continue
        t = p.text
        if _is_revision_history_heading(t):
            if body_started:
                out.append((p.start, "Revision History", "Revision History"))
                break
            if out[-1][1] != "Revision History":
                out.append((p.start, "Revision History", "Revision History"))
            continue
        match = NUM_HEADING.match(t)
        if not body_started:
            if not (match and match[1] == "1" and matches("1", match[2]) and at_page_start(index)):
                continue
            body_started = True
        attachment = NYISO_ATTACHMENT.match(t)
        if attachment:
            key = f"{attachment[1].title()} {attachment[2].upper()}"
            if attachment[2].upper() > attachment_letter and matches(key, attachment[3]):
                out.append((p.start, key, f"{key} {toc[key][1]}"))
                attachment_letter, cur_num = attachment[2].upper(), (99,)
            continue
        if not match or len(t) >= 170:
            continue
        num = tuple(int(value) for value in match[1].split("."))
        if num <= cur_num:
            continue
        if match[1] in toc:
            accept = matches(match[1], match[2])
        else:
            accept = (len(num) >= 3 and cur_num[:len(num) - 1] == num[:-1] and len(t) < 120
                      and not MODAL_ANY.search(match[2]) and not t.rstrip().endswith("."))
        if accept:
            title = f"{match[1]} {toc[match[1]][1]}" if match[1] in toc else t
            out.append((p.start, match[1], title))
            cur_num = num
    return out


def _segment(view: TextView, ps: list[Piece], bp: set[str]) -> list[Sentence]:
    blocks: list[list[Piece]] = []
    cur: list[Piece] = []
    pending_skip = False
    heading_starts = set(view._heading_starts)
    for p in ps:
        if p.after_break:
            pending_skip = True
        if _is_skip(p, bp):
            pending_skip = True
            continue
        is_heading = p.start in heading_starts
        new_block = bool(BULLET.match(p.text) or ENUM.match(p.text)) or is_heading
        if pending_skip and cur:
            last = cur[-1].text.rstrip()
            if (last and last[-1] in ".:;!?") or not p.text[0].islower():
                new_block = True
        pending_skip = False
        if new_block and cur:
            blocks.append(cur)
            cur = []
        cur.append(p)
        if is_heading:
            # a heading line is its own unit, so renumbered or retitled headings never glue onto body sentences
            blocks.append(cur)
            cur = []
    if cur:
        blocks.append(cur)

    out: list[Sentence] = []
    for blk in blocks:
        buf_parts: list[str] = []
        runs: list[tuple[int, int, int]] = []
        k = 0
        for p in blk:
            if buf_parts:
                buf_parts.append(" ")
                runs.append((k, p.start - 1, 1))
                k += 1
            buf_parts.append(p.text)
            runs.append((k, p.start, len(p.text)))
            k += len(p.text)
        buf = "".join(buf_parts)
        spans, start = [], 0
        for m in SPLIT.finditer(buf):
            end = m.start() + 1
            toks = buf[start:end].rsplit(None, 1)
            prev = toks[-1].lower().rstrip(".").lstrip("(\"“") if toks else ""
            if prev in ABBREV or (len(prev) == 1 and prev.isalpha()):
                continue
            spans.append((start, m.end()))
            start = m.end()
        spans.append((start, len(buf)))
        for a, b in spans:
            while a < b and buf[a].isspace():
                a += 1
            while b > a and buf[b - 1].isspace():
                b -= 1
            if a >= b:
                continue
            sub = _sub_runs(runs, a, b)
            ra, rb = sub[0][1], sub[-1][1] + sub[-1][2]
            if view.line_of(rb - 1) - view.line_of(ra) > 8 and b - a > 800:
                # merged table / figure text: fall back to one unit per source piece
                for p in blk:
                    if ra <= p.start < rb:
                        out.append(_sentence(view, len(out), p.text, ((0, p.start, len(p.text)),)))
                continue
            out.append(_sentence(view, len(out), buf[a:b], sub))
    return out


def _sub_runs(runs: list[tuple[int, int, int]], a: int, b: int) -> tuple[tuple[int, int, int], ...]:
    """Runs restricted to buffer slice [a, b), re-based so the slice starts at text offset 0."""
    out = []
    for t0, r0, n in runs:
        lo, hi = max(a, t0), min(b, t0 + n)
        if lo < hi:
            out.append((lo - a, r0 + (lo - t0), hi - lo))
    return tuple(out)


def _sentence(view: TextView, idx: int, s: str, runs: tuple[tuple[int, int, int], ...]) -> Sentence:
    ra, rb = runs[0][1], runs[-1][1] + runs[-1][2]
    return Sentence(idx, s, ra, rb, view.line_of(ra), view.line_of(rb - 1), view.section_at(ra), alnum(s), runs)


def raw_span(sent: Sentence, a: int, b: int) -> tuple[int, int]:
    """Raw LF offsets for the text slice [a, b) of `sent`."""
    return sent.raw(a), sent.raw(b - 1) + 1

"""The corpus's single PDF -> text recipe, shared by every family builder.

The geometry layer is a small content-stream walker on top of pypdf 6.x (no pdfplumber or pdfminer dependency):
it yields positioned, coloured text runs and thin filled/stroked rules. The walker reuses pypdf's layout-mode ``Font``/``TextStateParams`` (private
API, pinned by the ``PYPDF_TESTED`` assertion below) for glyph decoding and advance widths.

Recipe (``pdf_to_text`` / ``pdf_to_pages``; ``RECIPE_VERSION`` names it in every manifest row):
  1. walk each page; drop faux-bold overprints (a heading painted twice 0.28 pt apart) and runs tagged as
     running header/footer (``/Artifact <</Type /Pagination>>`` marked content, as Word writes them);
  2. join runs into lines in content-stream order; each line keeps its baseline and font height;
  3. strip untagged running headers/footers by position (``strip_running_lines``): only lines in a page's
     margin band or recurring there are candidates, and a candidate survives when the TOC lists it as a
     heading or when body text follows it at line spacing;
  4. keep blank (whitespace-only) lines as single paragraph breaks, none at a page's start or end; join
     pages with ``=== PAGE n ===`` marker lines; UTF-8 with LF; hashes after CRLF->LF.
Redline PDFs go through the same steps with a change status per run (``redline_lines``): one
``RedlineStyle`` model covers PJM red text, the NERC dark reds and colour-only insertions, and MISO's
per-author rule colours; ``detect_redline_spans`` groups the marked runs into spans.
"""
from __future__ import annotations

import bisect
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from typing import Any, Iterable

import pypdf
from pypdf import PdfReader
from pypdf._font import Font
from pypdf._text_extraction import mult
from pypdf._text_extraction._layout_mode._text_state_params import TextStateParams
from pypdf.generic import ContentStream, DictionaryObject

PYPDF_TESTED = "6."
assert pypdf.__version__.startswith(PYPDF_TESTED), f"pdf_text walker tested on pypdf {PYPDF_TESTED}x only"

PAGE_MARKER = "=== PAGE {n} ==="
PAGE_LINE = re.compile(r"=== PAGE (\d+) ===")
_PAGE_LINES = re.compile(r"^=== PAGE (\d+) ===$", re.M)
_IDENTITY = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0]
_FILL_OPS = {b"f", b"F", b"f*", b"B", b"B*", b"b", b"b*"}
_STROKE_OPS = {b"S", b"s", b"B", b"B*", b"b", b"b*"}
_PATH_OPS = {b"re", b"m", b"l", b"c", b"v", b"y", b"h", b"n"} | _FILL_OPS | _STROKE_OPS
_TEXT_STATE_OPS = {b"Tf", b"Tc", b"Tw", b"Tz", b"TL", b"Ts", b"Tr"}
_SHOW_OPS = {b"Tj", b"TJ", b"'", b'"'}
_MAX_FORM_DEPTH = 4


# ---------------------------------------------------------------- hashing / text hygiene / identity

def normalize_lf(text: str) -> str:
    """CRLF / CR -> LF (hash and write only normalised text)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def sha256_text_lf(text: str) -> str:
    return hashlib.sha256(normalize_lf(text).encode("utf-8")).hexdigest()


def squash_ws(text: str) -> str:
    """Remove all whitespace: robust key for 'same characters, different line/run breaks'."""
    return re.sub(r"\s+", "", text)


RECIPE_VERSION = f"procedures.corpus.pdf_text+pypdf-{pypdf.__version__}"


# ---------------------------------------------------------------- geometry records

@dataclass(frozen=True)
class TextRun:
    """One text-show operation (Tj/TJ/'/") placed on the page."""

    page: int
    seq: int
    x0: float
    x1: float
    y: float          # baseline (includes text rise)
    size: float       # effective font height
    text: str
    fill: tuple[float, ...]
    pagination: bool  # inside /Artifact <</Type /Pagination>> (running header/footer)
    char_x: tuple[float, ...] = ()  # page x where each character of ``text`` starts (empty if unknown)


@dataclass(frozen=True)
class Rule:
    """A thin painted shape (filled rect or stroked horizontal segment) in page space."""

    page: int
    x0: float
    x1: float
    y0: float
    y1: float
    color: tuple[float, ...]

    @property
    def height(self) -> float:
        return self.y1 - self.y0

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def y_mid(self) -> float:
        return (self.y0 + self.y1) / 2


@dataclass
class _GState:
    """Graphics state saved/restored by q/Q; includes the text state parameters (PDF 32000 §9.3)."""

    ctm: list[float] = field(default_factory=lambda: list(_IDENTITY))
    fill: tuple[float, ...] = (0.0, 0.0, 0.0)
    stroke: tuple[float, ...] = (0.0, 0.0, 0.0)
    line_width: float = 1.0
    font: Font | None = None
    size: float = 0.0
    Tc: float = 0.0
    Tw: float = 0.0
    Tz: float = 100.0
    TL: float = 0.0
    Ts: float = 0.0
    Tr: int = 0

    def copy(self) -> "_GState":
        return replace(self, ctm=list(self.ctm))


@dataclass
class WalkStats:
    """Counters that make degraded extraction visible instead of silent."""

    text_ops: int = 0
    unresolved_font_ops: int = 0
    uninterpretable_font_ops: int = 0
    invisible_text_ops: int = 0
    form_xobjects: int = 0
    char_x_mismatch: int = 0  # runs whose per-character x positions could not be aligned to the text
    rotated_pages: int = 0    # pages with /Rotate: their margin band is judged on unrotated y (none in the corpus)


def to_rgb(components: Iterable[float]) -> tuple[float, ...]:
    """Gray/RGB/CMYK component list -> rounded RGB triple (other spaces: rounded as given)."""
    c = [float(v) for v in components]
    if len(c) == 1:
        c = [c[0]] * 3
    elif len(c) == 4:
        k = c[3]
        c = [(1 - c[0]) * (1 - k), (1 - c[1]) * (1 - k), (1 - c[2]) * (1 - k)]
    return tuple(round(v, 3) for v in c)


def _apply(m: list[float], x: float, y: float) -> tuple[float, float]:
    return m[0] * x + m[2] * y + m[4], m[1] * x + m[3] * y + m[5]


def page_fonts(resources: Any) -> dict[str, Font]:
    """Layout-mode fonts declared in a resources dictionary."""
    fonts: dict[str, Font] = {}
    font_dict = resources.get("/Font", {}) if resources else {}
    for name in font_dict:
        fonts[name] = Font.from_font_resource(font_dict[name].get_object())
    return fonts


def _is_pagination(tag: Any, props: Any) -> bool:
    if tag != "/Artifact" or not isinstance(props, DictionaryObject):
        return False
    return props.get("/Type") == "/Pagination"


class _PageWalker:
    """Interpret one content stream (recursing into form XObjects) into runs and rules."""

    def __init__(self, reader: PdfReader, page_no: int, stats: WalkStats):
        self.reader = reader
        self.page_no = page_no
        self.stats = stats
        self.runs: list[TextRun] = []
        self.rules: list[Rule] = []
        self.images: list[tuple[float, float, float, float]] = []  # (x0, x1, y0, y1) of placed images

    def walk(self, content: Any, resources: Any, ctm: list[float], depth: int = 0) -> None:
        fonts = page_fonts(resources)
        xobjects = resources.get("/XObject", {}) if resources else {}
        gs = _GState(ctm=list(ctm))
        stack: list[_GState] = []
        marks: list[bool] = []  # marked-content stack; True = inside a pagination artifact
        path: dict[str, Any] = {"rects": [], "segs": [], "cur": (0.0, 0.0)}
        tm, tlm = list(_IDENTITY), list(_IDENTITY)
        for operands, op in ContentStream(content, self.reader, "bytes").operations:
            if op == b"q":
                stack.append(gs.copy())
            elif op == b"Q":
                gs = stack.pop() if stack else gs
            elif op in (b"BDC", b"BMC", b"EMC"):
                self._mark(op, operands, marks)
            elif op in _PATH_OPS:
                self._path(op, operands, gs, path)
            elif op in _TEXT_STATE_OPS:
                self._text_state(op, operands, gs, fonts)
            elif op in (b"BT", b"Tm", b"Td", b"TD", b"T*"):
                tm, tlm = self._text_position(op, operands, gs, tm, tlm)
            elif op in _SHOW_OPS:
                if op in (b"'", b'"'):
                    if op == b'"':
                        gs.Tw, gs.Tc = float(operands[0]), float(operands[1])
                    tm, tlm = self._text_position(b"T*", [], gs, tm, tlm)
                items = operands[0] if op == b"TJ" else [operands[-1]]
                tm = self._show(items, gs, tm, bool(marks and marks[-1]))
            elif op == b"Do":
                self._form(operands[0], xobjects, gs, depth)
            elif op == b"INLINE IMAGE":
                self._image(gs)
            else:
                self._graphics(op, operands, gs)

    @staticmethod
    def _graphics(op: bytes, operands: list[Any], gs: _GState) -> None:
        if op == b"cm":
            gs.ctm = mult([float(v) for v in operands], gs.ctm)
        elif op == b"w":
            gs.line_width = float(operands[0])
        elif op in (b"g", b"rg", b"k", b"sc", b"scn"):
            nums = [v for v in operands if isinstance(v, (int, float))]
            if nums:
                gs.fill = to_rgb(nums)
        elif op in (b"G", b"RG", b"K", b"SC", b"SCN"):
            nums = [v for v in operands if isinstance(v, (int, float))]
            if nums:
                gs.stroke = to_rgb(nums)

    @staticmethod
    def _mark(op: bytes, operands: list[Any], marks: list[bool]) -> None:
        if op == b"EMC":
            if marks:
                marks.pop()
            return
        props = operands[1] if op == b"BDC" and len(operands) > 1 else None
        marks.append(bool(marks and marks[-1]) or _is_pagination(operands[0], props))

    def _path(self, op: bytes, operands: list[Any], gs: _GState, path: dict[str, Any]) -> None:
        if op == b"re":
            x, y, w, h = (float(v) for v in operands)
            pts = [_apply(gs.ctm, px, py) for px, py in ((x, y), (x + w, y + h))]
            path["rects"].append((min(p[0] for p in pts), max(p[0] for p in pts),
                                  min(p[1] for p in pts), max(p[1] for p in pts)))
        elif op == b"m":
            path["cur"] = _apply(gs.ctm, float(operands[0]), float(operands[1]))
        elif op == b"l":
            nxt = _apply(gs.ctm, float(operands[0]), float(operands[1]))
            path["segs"].append((*path["cur"], *nxt))
            path["cur"] = nxt
        elif op in (b"c", b"v", b"y"):
            path["cur"] = _apply(gs.ctm, float(operands[-2]), float(operands[-1]))
        elif op != b"h":  # painting operator or n: emit thin shapes, then clear the path
            if op in _FILL_OPS:
                for x0, x1, y0, y1 in path["rects"]:
                    self.rules.append(Rule(self.page_no, x0, x1, y0, y1, gs.fill))
            if op in _STROKE_OPS:
                half = abs(gs.line_width * gs.ctm[3]) / 2
                for ax, ay, bx, by in path["segs"]:
                    if abs(ay - by) < 0.5:
                        mid = (ay + by) / 2
                        self.rules.append(Rule(self.page_no, min(ax, bx), max(ax, bx), mid - half, mid + half, gs.stroke))
            path["rects"], path["segs"] = [], []

    def _text_state(self, op: bytes, operands: list[Any], gs: _GState, fonts: dict[str, Font]) -> None:
        if op == b"Tf":
            gs.font = fonts.get(operands[0])
            gs.size = float(operands[1])
            if gs.font is None:
                self.stats.unresolved_font_ops += 1
        elif op == b"Tr":
            gs.Tr = int(operands[0])
        else:
            setattr(gs, op.decode(), float(operands[0]))

    @staticmethod
    def _text_position(op: bytes, operands: list[Any], gs: _GState,
                       tm: list[float], tlm: list[float]) -> tuple[list[float], list[float]]:
        if op == b"BT":
            return list(_IDENTITY), list(_IDENTITY)
        if op == b"Tm":
            tm = [float(v) for v in operands]
            return tm, list(tm)
        if op == b"T*":
            dx, dy = 0.0, -gs.TL
        else:
            dx, dy = float(operands[0]), float(operands[1])
            if op == b"TD":
                gs.TL = -dy
        tlm = mult([1.0, 0.0, 0.0, 1.0, dx, dy], tlm)
        return list(tlm), tlm

    def _show(self, items: list[Any], gs: _GState, tm: list[float], pagination: bool) -> list[float]:
        """Place one text-show operation; return the advanced text matrix."""
        self.stats.text_ops += 1
        if gs.font is None:
            return tm
        if not gs.font.interpretable:
            self.stats.uninterpretable_font_ops += 1
            return tm
        scale = gs.Tz / 100.0
        pieces: list[TextStateParams] = []
        char_x: list[float] = []
        end_tm = tm
        for item in items:
            if isinstance(item, (bytes, str)):
                tsp = TextStateParams(item, gs.font, gs.size, gs.Tc, gs.Tw, gs.Tz, gs.TL, gs.Ts, mult(tm, gs.ctm))
                pieces.append(tsp)
                glyphs = tsp._decoded_value or tsp.text  # pypdf adds Tc once per string; PDF adds it per glyph
                char_x.extend(self._glyph_x(glyphs, bool(tsp._decoded_value), gs, tm))
                advance = tsp.word_tx(item) + gs.Tc * max(len(glyphs) - 1, 0) * scale
                tm = mult([1.0, 0.0, 0.0, 1.0, advance, 0.0], tm)
                end_tm = tm
            else:
                tm = mult([1.0, 0.0, 0.0, 1.0, -float(item) / 1000.0 * gs.size * scale, 0.0], tm)
        text = "".join(p.text for p in pieces)
        if not text:
            return tm
        if gs.Tr in (3, 7):
            self.stats.invisible_text_ops += 1
            return tm
        if len(char_x) != len(text):
            self.stats.char_x_mismatch += 1
            char_x = []
        self.runs.append(TextRun(
            page=self.page_no, seq=len(self.runs), x0=pieces[0].tx, x1=mult(end_tm, gs.ctm)[4],
            y=pieces[0].ty, size=pieces[0].font_height or gs.size, text=text,
            fill=gs.fill, pagination=pagination, char_x=tuple(char_x),
        ))
        return tm

    @staticmethod
    def _glyph_x(glyphs: str, mapped: bool, gs: _GState, tm: list[float]) -> list[float]:
        """Page x of each output character of one string (a glyph mapped to n characters repeats its x)."""
        font = gs.font
        scale = gs.Tz / 100.0
        out: list[float] = []
        for g in glyphs:
            chars = font.character_map.get(g, g) if mapped else g
            out.extend([mult(tm, gs.ctm)[4]] * len(chars))
            width = font.space_width if g == font.space_char else font.get_text_width(g)
            word = gs.Tw if g == font.space_char else 0.0
            tm = mult([1.0, 0.0, 0.0, 1.0, (gs.size * width / 1000.0 + gs.Tc + word) * scale, 0.0], tm)
        return out

    def _form(self, name: Any, xobjects: Any, gs: _GState, depth: int) -> None:
        if name not in xobjects or depth >= _MAX_FORM_DEPTH:
            return
        xobj = xobjects[name].get_object()
        if xobj.get("/Subtype") == "/Image":
            self._image(gs)
            return
        if xobj.get("/Subtype") != "/Form":
            return
        self.stats.form_xobjects += 1
        matrix = [float(v) for v in xobj.get("/Matrix", _IDENTITY)]
        self.walk(xobj, xobj.get("/Resources"), mult(matrix, gs.ctm), depth + 1)

    def _image(self, gs: _GState) -> None:
        """An image paints the unit square under the current CTM; keep its page-space box."""
        pts = [_apply(gs.ctm, x, y) for x, y in ((0, 0), (1, 0), (0, 1), (1, 1))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        self.images.append((min(xs), max(xs), min(ys), max(ys)))


def walk_page(reader: PdfReader, page_no: int, stats: WalkStats | None = None) -> tuple[list[TextRun], list[Rule]]:
    """Runs and rules of page ``page_no`` (0-based), in content-stream order."""
    stats = stats if stats is not None else WalkStats()
    page = reader.pages[page_no]
    walker = _PageWalker(reader, page_no, stats)
    content = page.get_contents()
    if content is not None:
        walker.walk(content, page.get("/Resources"), list(_IDENTITY))
    return walker.runs, walker.rules



# ---------------------------------------------------------------- runs -> lines

def same_line(prev: TextRun, cur: TextRun) -> bool:
    """Same visual line: baseline within 0.6 x the larger font height and no big leftward jump."""
    tol = 0.6 * max(prev.size, cur.size, 1.0)
    return abs(cur.y - prev.y) <= tol and cur.x0 >= prev.x1 - 2.0 * max(prev.size, cur.size, 1.0)


def needs_space(prev: TextRun, cur: TextRun) -> bool:
    """Insert a space between two runs on one line when the visual gap is wider than ~0.15 em."""
    if prev.text.endswith((" ", "\t")) or cur.text.startswith((" ", "\t")):
        return False
    return cur.x0 - prev.x1 > 0.15 * max(prev.size, cur.size, 1.0)


def body_runs(runs: list[TextRun]) -> list[TextRun]:
    """Runs that are not tagged running header/footer."""
    return [r for r in runs if not r.pagination]


OLD_SIDE = frozenset({"plain", "marked", "delete"})
NEW_SIDE = frozenset({"plain", "marked", "insert"})


@dataclass
class Line:
    """One visual line: runs in content-stream order, with a change status per segment.

    ``segments`` are (status, text) with adjacent equal statuses merged; a clean page has only ``plain``.
    ``y`` is the baseline of the line's first run and ``size`` its largest font height.
    """

    page: int          # 1-based
    line: int          # 1-based within the page, stream order, numbered before running lines are dropped
    segments: list[tuple[str, str]]
    y: float = 0.0
    size: float = 0.0

    @property
    def text(self) -> str:
        return "".join(t for _, t in self.segments).rstrip()

    def markup(self) -> str:
        """Inline view: ``{+inserted+}`` ``[-deleted-]`` ``{?both?}``; plain/marked verbatim."""
        wrap = {"insert": "{+%s+}", "delete": "[-%s-]", "both": "{?%s?}"}
        return "".join(wrap[s] % t if s in wrap else t for s, t in self.segments)

    def side(self, keep: frozenset[str]) -> str:
        return "".join(t for s, t in self.segments if s in keep).rstrip()


def lines_of_page(runs: list[TextRun], page: int, status_of: dict[int, str] | None = None) -> list[Line]:
    """Runs of one page (stream order) -> lines; ``status_of`` maps run.seq -> change status (default plain)."""
    lines: list[Line] = []
    segs: list[tuple[str, str]] = []
    first: TextRun | None = None
    size = 0.0
    prev: TextRun | None = None

    def add(status: str, text: str) -> None:
        if segs and segs[-1][0] == status:
            segs[-1] = (status, segs[-1][1] + text)
        else:
            segs.append((status, text))

    for run in runs:
        if prev is not None and not same_line(prev, run):
            lines.append(Line(page, len(lines) + 1, segs, first.y, size))
            segs, first, size = [], None, 0.0
        elif prev is not None and needs_space(prev, run):
            add("plain", " ")
        first = first or run
        size = max(size, run.size)
        add("plain" if status_of is None else status_of[run.seq], run.text)
        prev = run
    if first is not None:
        lines.append(Line(page, len(lines) + 1, segs, first.y, size))
    return lines


def drop_overprint(runs: list[TextRun], max_dx: float = 1.5) -> list[TextRun]:
    """Drop runs that repeat the previous run's string at (almost) the same origin.

    Some generators fake bold by painting a heading twice 0.28 pt apart, which otherwise yields
    ``Section 1: OvSection 1: Ov ererviewview``.
    """
    kept: list[TextRun] = []
    prev: TextRun | None = None
    for run in runs:
        if (prev is not None and run.text == prev.text and abs(run.y - prev.y) < 0.5
                and abs(run.x0 - prev.x0) < max_dx):
            prev = run
            continue
        kept.append(run)
        prev = run
    return kept


# ---------------------------------------------------------------- running headers / footers

RUNNING_EDGE_LINES = 4    # lines nearest each page edge that may be running furniture
RUNNING_MIN_RUN = 3       # a furniture line recurs on at least this many pages ...
RUNNING_MIN_SHARE = 0.5   # ... and on half of all pages. F1's run_d1.strip_headers used 30%, which a table
                          # header repeated on 3 of 10 pages passes (NERC TOP-003-7 VSL table); per-section
                          # running titles need no text recurrence, they sit in the band (``_band_depth``)
MARGIN_MAX_SHARE = 0.2    # furniture lies within this share of the page height from the top or bottom edge
POS_TOL = 1.5             # pt: two lines this close in distance from the edge share a position
LINE_GAP = 2.0            # font heights: two lines this close belong to one block (a wrapped running title)
BODY_GAP = 3.0            # font heights: a line with a kept line this close below it heads body text
                          # (PJM paragraph spacing reaches 2.3, its running header sits 3.8 above the body)

# TOC entry: "2.3 Capacity Shortages ........ 22"; TOC_LINE also catches fused/unnumbered leader lines.
TOC_ENTRY = re.compile(r"^(?P<title>\S.*?)\s*(?:\.{4,}|…{2,}|(?:\. ){3,}\.?)\s*(?:\d{1,4}|[ivxlc]{1,6})$")
TOC_LINE = re.compile(r"(?:\.{4,}|…{2,}|(?:\. ){3,})\s*\d{0,4}\s*$|\.{8,}")
TOC_HEADING = re.compile(r"^\s*(?:Table of Contents|Contents|List of (?:Exhibits|Figures|Tables))\s*$", re.IGNORECASE)
PAGE_NUMBER_END = re.compile(r"\D\s\d{1,4}(?:\s+\d{1,4})?\s*$")  # ends in a page number (old+new may fuse)
TOC_PAGE_SHARE = 0.4      # dot-leader lines on a contents page
TOC_NUMBERED_SHARE = 0.5  # lines ending in a page number on a contents page without dot leaders


def _norm_line(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _mask_line(line: str) -> str:
    """Mask page numbers only: 'N of M', a trailing number or lower-case roman numeral, a leading number
    followed by a space. Section numbers ('3. Pre-Contingency State', '5. ...') stay distinct."""
    line = re.sub(r"\b\d+\s+of\s+\d+\b", "# of #", _norm_line(line))
    line = re.sub(r"(?:^|(?<=\s))(?:\d+|[ivxlc]{1,6})$", "#", line)
    return re.sub(r"^\d+(?=\s)", "#", line)


def toc_titles(pages: list[list[Line]]) -> frozenset[str]:
    """Whitespace-normalised titles of the document's dot-leader TOC entries."""
    return frozenset(_norm_line(m["title"]) for page in pages for ln in page
                     if (m := TOC_ENTRY.match(_norm_line(ln.text))))


@dataclass(frozen=True)
class _Placed:
    side: str   # top | bottom half of the page
    d: float    # distance from that edge (pt)


def _place(lines: list[Line], frame: tuple[float, float]) -> list[_Placed]:
    bottom, top = frame
    mid = (bottom + top) / 2
    return [_Placed("top", top - ln.y) if ln.y >= mid else _Placed("bottom", ln.y - bottom) for ln in lines]


def _edge_zone(placed: list[_Placed], height: float) -> set[int]:
    """Indices of the RUNNING_EDGE_LINES lines nearest each edge, within MARGIN_MAX_SHARE of the height."""
    out: set[int] = set()
    for side in ("top", "bottom"):
        near = sorted((p.d, i) for i, p in enumerate(placed) if p.side == side and p.d <= MARGIN_MAX_SHARE * height)
        out.update(i for _, i in near[:RUNNING_EDGE_LINES])
    return out


def _recurring(points: list[tuple[float, float, int]], share: float) -> list[tuple[float, float]]:
    """(distance, font height) of the points whose position (+-POS_TOL) is occupied on >= ``share`` pages."""
    pts = sorted(points)
    ds = [p[0] for p in pts]
    out = []
    for d, size, _ in pts:
        lo, hi = bisect.bisect_left(ds, d - POS_TOL), bisect.bisect_right(ds, d + POS_TOL)
        if len({pts[k][2] for k in range(lo, hi)}) >= share:
            out.append((d, size))
    return out


def _band_depth(seeds: list[tuple[float, float, int]], positions: list[tuple[float, float, int]],
                share: float) -> float | None:
    """How far the margin band reaches in from one edge (pt); None when that side has no furniture.

    The band reaches from the edge to the innermost recurring furniture position (a furniture line that
    also appears once as a body heading, like M-03's "Revision History", does not deepen it) and extends
    inward through edge-zone positions occupied on >= ``share`` pages that sit within LINE_GAP line heights
    of it (PJM prints the section title, which changes by section, under the manual title). ``seeds`` /
    ``positions``: (distance from the edge, font height, page) of furniture lines / of all edge-zone lines.
    """
    anchored = [d for d, _ in _recurring(seeds, share)]
    if not anchored:
        return None
    recurring = _recurring(positions, share)
    depth = max(anchored)
    while True:
        nxt = [d for d, size in recurring if depth + POS_TOL < d <= depth + LINE_GAP * size]
        if not nxt:
            return depth
        depth = min(nxt)


def _keep_lines(lines: list[Line], candidates: set[int], headings: set[int]) -> list[Line]:
    """Apply the guards to one page's candidates (``headings`` pass guard 1); return kept lines in stream order."""
    kept = [i not in candidates or i in headings for i in range(len(lines))]
    order = sorted(range(len(lines)), key=lambda i: lines[i].y)
    for k, i in enumerate(order):  # bottom-up: protection flows from body text to the lines stacked above it
        if kept[i]:
            continue
        below = [j for j in order[:k] if lines[j].y < lines[i].y - 0.6 * lines[i].size]
        if not below:
            continue
        y_next = max(lines[j].y for j in below)
        kept[i] = any(kept[j] and y_next - lines[j].y <= POS_TOL
                      and lines[i].y - lines[j].y <= BODY_GAP * max(lines[i].size, lines[j].size) for j in below)
    return [ln for ln, keep in zip(lines, kept) if keep]


def _paragraph_breaks(page: list[Line], kept: list[Line]) -> list[Line]:
    """``kept`` in stream order with the page's blank lines between them, runs collapsed to one blank line
    and none at the start or end."""
    keep = {id(ln) for ln in kept}
    out: list[Line] = []
    for ln in page:
        if ln.text.strip():
            if id(ln) in keep:
                out.append(ln)
        elif out and out[-1].text.strip():
            out.append(ln)
    while out and not out[-1].text.strip():
        out.pop()
    return out


def strip_running_lines(pages: list[list[Line]], frames: list[tuple[float, float]]) -> list[list[Line]]:
    """Per-page lines minus untagged running headers/footers (``frames``: page (bottom, top)); blank lines
    stay as single paragraph breaks (``_paragraph_breaks``) and play no part in the rules below.

    * furniture: a line whose text, page numbers masked (``_mask_line``), sits among the RUNNING_EDGE_LINES
      lines nearest the same edge (within MARGIN_MAX_SHARE of the page height) on >= 3 pages and >= 50% of pages;
    * margin band: from each edge to its innermost furniture line, extended inward through recurring positions
      stacked against it at line spacing (``_band_depth``);
    * candidates: the page's running block (lines in the band, and lines stacked under the top band within
      LINE_GAP font heights: the wrapped tail of a long running title) and furniture lines in an edge zone
      outside that block (a footer that moves, or a body heading worded like a running title);
    * guards, a candidate is kept when (1) it is such a furniture line outside the block and the TOC lists
      its text (M-03's body heading "Revision History" is also the running title of 60 pages), or (2) a kept
      line follows it within BODY_GAP font heights (a heading or table header over its body; bottom-up).
    Body headings that recur anywhere on the page ("PJM Actions" on ~20% of M-13 pages) are never candidates.
    """
    if len(pages) != len(frames):
        raise ValueError(f"{len(pages)} pages but {len(frames)} page frames")
    full = pages
    pages = [[ln for ln in page if ln.text.strip()] for page in pages]
    placed = [_place(lines, frame) for lines, frame in zip(pages, frames)]
    zones = [_edge_zone(pl, top - bottom) for pl, (bottom, top) in zip(placed, frames)]
    masked = [[_mask_line(ln.text) for ln in lines] for lines in pages]
    share = max(RUNNING_MIN_RUN, RUNNING_MIN_SHARE * len(pages))
    key_pages: dict[tuple[str, str], set[int]] = defaultdict(set)
    for p, (pl, zone) in enumerate(zip(placed, zones)):
        for i in zone:
            key_pages[(pl[i].side, masked[p][i])].add(p)
    furniture = {k for k, v in key_pages.items() if len(v) >= share}
    depth: dict[str, float | None] = {}
    for side in ("top", "bottom"):
        seeds, positions = [], []
        for p, (lines, pl, zone) in enumerate(zip(pages, placed, zones)):
            for i in zone:
                if pl[i].side == side:
                    positions.append((pl[i].d, lines[i].size, p))
                    if (side, masked[p][i]) in furniture:
                        seeds.append((pl[i].d, lines[i].size, p))
        depth[side] = _band_depth(seeds, positions, share)
    toc = toc_titles(pages)
    out = []
    for p, (lines, pl, zone) in enumerate(zip(pages, placed, zones)):
        block = set()  # the page's running header/footer block: band lines + the wrapped tail under the top band
        for i, pos in enumerate(pl):
            if depth[pos.side] is not None and pos.d <= depth[pos.side] + POS_TOL:
                block.add(i)
        reach = depth["top"]
        for d, i in sorted((pl[i].d, i) for i in range(len(lines)) if pl[i].side == "top"):
            if reach is not None and reach + POS_TOL < d <= reach + LINE_GAP * lines[i].size:
                block.add(i)
                reach = d
        recurring = {i for i in zone if (pl[i].side, masked[p][i]) in furniture} - block
        heading = {i for i in recurring if _norm_line(lines[i].text) in toc}
        out.append(_paragraph_breaks(full[p], _keep_lines(lines, block | recurring, heading)))
    return out


# ---------------------------------------------------------------- redline decorations

STRIKE_BAND = (0.10, 0.55)      # rule centre above baseline, in font heights
UNDERLINE_BAND = (-0.45, 0.10)  # rule centre relative to baseline, in font heights
MAX_RULE_HEIGHT = 2.0           # pt; thicker shapes are fills/boxes, not decorations
MIN_RUN_COVER = 0.5             # share of a run's width a rule must cover
MIN_RULE_TEXT_COVER = 0.8       # share of a rule's width that must lie over text on that line
PALETTE_RULE_TEXT_COVER = 0.3   # same, once rules are pre-filtered to change colours (a struck table row)
PALETTE_TOL = 0.1               # max per-channel distance to a palette colour


def rule_band(rule: Rule, run: TextRun) -> str | None:
    """'strike' / 'underline' if the rule sits in that vertical band of the run's line (x ignored)."""
    if rule.height > MAX_RULE_HEIGHT or rule.width < 0.8:
        return None
    rel = (rule.y_mid - run.y) / max(run.size, 1.0)
    if STRIKE_BAND[0] < rel < STRIKE_BAND[1]:
        return "strike"
    if UNDERLINE_BAND[0] < rel <= UNDERLINE_BAND[1]:
        return "underline"
    return None


def decoration_kind(rule: Rule, run: TextRun) -> str | None:
    """'strike' or 'underline' if ``rule`` decorates ``run`` geometrically, else None."""
    if run.x1 <= run.x0:
        return None
    overlap = min(rule.x1, run.x1) - max(rule.x0, run.x0)
    if overlap <= 0 or overlap / (run.x1 - run.x0) < MIN_RUN_COVER:
        return None
    return rule_band(rule, run)


def split_runs_at_rules(runs: list[TextRun], rules: list[Rule], tol: float = 0.5) -> list[TextRun]:
    """Split runs where a strike/underline rule starts or ends inside them; renumber ``seq`` in order.

    Word sometimes shows an inserted and a deleted word in one text operation
    (``Exhibit Exhibit`` = underlined + struck); run-level ``decorate`` would give the whole
    run one kind. Cuts fall on the first character starting at or after the rule edge, using
    ``char_x``; runs without ``char_x`` are kept whole.
    """
    out: list[TextRun] = []
    for run in runs:
        cuts = set()
        if run.char_x:
            for rule in rules:
                if rule.x1 <= run.x0 or rule.x0 >= run.x1 or rule_band(rule, run) is None:
                    continue
                for edge in (rule.x0, rule.x1):
                    if run.x0 + tol < edge < run.x1 - tol:
                        cut = next((i for i, x in enumerate(run.char_x) if x >= edge - tol), len(run.text))
                        if 0 < cut < len(run.text):
                            cuts.add(cut)
        bounds = [0, *sorted(cuts), len(run.text)]
        for a, b in zip(bounds, bounds[1:]):
            x1 = run.char_x[b] if b < len(run.text) else run.x1
            out.append(replace(run, seq=len(out), x0=run.char_x[a] if run.char_x else run.x0, x1=x1,
                               text=run.text[a:b], char_x=run.char_x[a:b]))
    return out


def _text_cover(rule: Rule, runs: list[TextRun]) -> float:
    """Share of the rule's x-extent lying under the given runs (union of intervals)."""
    spans = sorted((max(r.x0, rule.x0), min(r.x1, rule.x1)) for r in runs if r.x1 > rule.x0 and r.x0 < rule.x1)
    covered, end = 0.0, rule.x0
    for a, b in spans:
        a = max(a, end)
        if b > a:
            covered += b - a
            end = b
    return covered / rule.width if rule.width > 0 else 0.0


def decorate(runs: list[TextRun], rules: list[Rule],
             min_text_cover: float = MIN_RULE_TEXT_COVER) -> dict[int, dict[str, tuple[float, ...]]]:
    """Map run.seq -> {kind: rule colour} for strike/underline rules that sit over text.

    A rule counts only if >= ``min_text_cover`` of its width lies over runs it decorates with the same
    kind; this rejects table borders and separators, which extend past the text. With rules pre-filtered
    to change colours the cover can be lower (a struck table row draws one rule across the gap between cells).
    """
    out: dict[int, dict[str, tuple[float, ...]]] = {}
    for rule in rules:
        if rule.height > MAX_RULE_HEIGHT or rule.width < 0.8:
            continue
        by_kind: dict[str, list[TextRun]] = {}
        for run in runs:
            kind = decoration_kind(rule, run)
            if kind:
                by_kind.setdefault(kind, []).append(run)
        for kind, hit in by_kind.items():
            if _text_cover(rule, hit) >= min_text_cover:
                for run in hit:
                    out.setdefault(run.seq, {})[kind] = rule.color
    return out


def rgb_hex(color: tuple[float, ...]) -> str:
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in color[:3])


def is_red(color: tuple[float, ...]) -> bool:
    """Change red: #ff0000 (PJM, NERC), the dark reds #c00000 / #b5082e (NERC postings), #d13438 (a MISO
    author colour). PJM's orange heading colour (1, 0.6, 0) is not red."""
    return len(color) >= 3 and color[0] >= 0.6 and color[1] <= 0.25 and color[2] <= 0.25


def in_palette(color: tuple[float, ...], palette: tuple[tuple[float, ...], ...], tol: float = PALETTE_TOL) -> bool:
    rgb = color if len(color) == 3 else tuple(color[:1] * 3)
    return any(max(abs(a - b) for a, b in zip(rgb, colour)) <= tol for colour in palette)


@dataclass(frozen=True)
class RedlineStyle:
    """How one owner marks changes; the one model behind every family's redline detection.

    ``red``: any ``is_red`` colour is a change colour; ``palette``: further change colours (per-author
    colours), matched within PALETTE_TOL. ``marked_text_only`` (PJM): only change-coloured runs can be
    changes (black struck text is formatting there). ``palette_rules_only`` (MISO): rules outside the change
    colours are dropped before decorating, because the clean PDFs use black strike-through and hyperlink
    underlines as formatting. ``colour_only_insert`` (NERC PRC-023-6): change-coloured text without a rule
    is an insertion; confirm per document against the clean versions.
    """

    red: bool = True
    palette: tuple[tuple[float, ...], ...] = ()
    marked_text_only: bool = False
    palette_rules_only: bool = False
    colour_only_insert: bool = False

    def is_change_colour(self, color: tuple[float, ...]) -> bool:
        return (self.red and is_red(color)) or in_palette(color, self.palette)


def run_status(run: TextRun, deco: dict[str, tuple[float, ...]], style: RedlineStyle) -> str:
    """Change status of one run: plain | insert | delete | both | marked.

    Change-coloured rules decide when present (a struck hyperlink keeps its own blue underline). Otherwise a
    strike-through of any colour is a deletion, and an underline marks an insertion only under
    change-coloured text (black and hyperlink underlines are formatting). Change-coloured text without a
    rule is ``marked`` (kept on both sides) or, under ``colour_only_insert``, an insertion.
    """
    if not run.text.strip():
        return "plain"
    coloured = style.is_change_colour(run.fill)
    if style.marked_text_only and not coloured:
        return "plain"
    strike_c, under_c = deco.get("strike"), deco.get("underline")
    pal_s = strike_c is not None and style.is_change_colour(strike_c)
    pal_u = under_c is not None and style.is_change_colour(under_c)
    if pal_s or pal_u:
        strike, under = pal_s, pal_u
    else:
        strike, under = strike_c is not None, under_c is not None and coloured
    if strike and under:
        return "both"
    if strike:
        return "delete"
    if under:
        return "insert"
    if coloured:
        return "insert" if style.colour_only_insert else "marked"
    return "plain"


# ---------------------------------------------------------------- the recipe

def page_frame(page: Any) -> tuple[float, float]:
    """(bottom, top) of a page's media box in user space."""
    return float(page.mediabox.bottom), float(page.mediabox.top)


def _page_lines(reader: PdfReader, page_no: int, stats: WalkStats, style: RedlineStyle | None) -> list[Line]:
    runs, rules = walk_page(reader, page_no, stats)
    runs = drop_overprint(runs)
    status_of = None
    if style is not None:
        if style.palette_rules_only:
            rules = [r for r in rules if style.is_change_colour(r.color)]
        runs = split_runs_at_rules(runs, rules)
        deco = decorate(runs, rules, PALETTE_RULE_TEXT_COVER if style.palette_rules_only else MIN_RULE_TEXT_COVER)
        status_of = {r.seq: run_status(r, deco.get(r.seq, {}), style) for r in runs}
    return lines_of_page(body_runs(runs), page_no + 1, status_of)


def document_lines(path: Any, style: RedlineStyle | None = None, stats: WalkStats | None = None) -> list[list[Line]]:
    """Kept lines per page (index = page - 1) after steps 1-3 of the recipe; ``style`` adds change statuses."""
    reader = PdfReader(str(path))
    stats = stats if stats is not None else WalkStats()
    stats.rotated_pages += sum(1 for p in reader.pages if int(p.get("/Rotate", 0) or 0) % 360)
    pages = [_page_lines(reader, i, stats, style) for i in range(len(reader.pages))]
    return strip_running_lines(pages, [page_frame(p) for p in reader.pages])


def pdf_to_pages(path: Any, stats: WalkStats | None = None) -> list[str]:
    """Per-page text (index = page - 1) of the recipe."""
    return ["\n".join(ln.text for ln in page) for page in document_lines(path, None, stats)]


def join_pages(page_texts: list[str], marker: str = PAGE_MARKER) -> str:
    """Pages joined with 1-based marker lines; LF only; trailing newline."""
    parts = []
    for i, text in enumerate(page_texts, start=1):
        parts.append(marker.format(n=i))
        if text:
            parts.append(text)
    return normalize_lf("\n".join(parts)) + "\n"


def pdf_to_text(path: Any, stats: WalkStats | None = None, marker: str = PAGE_MARKER) -> str:
    """The corpus text of one PDF: ``pdf_to_pages`` joined with ``=== PAGE n ===`` marker lines."""
    return join_pages(pdf_to_pages(path, stats), marker)


def split_pages(text: str) -> list[str]:
    """Inverse of ``join_pages`` with the default marker: page texts, index = page - 1."""
    parts = _PAGE_LINES.split(text)
    numbers = [int(n) for n in parts[1::2]]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("page markers are not 1..n in order")
    return [body.strip("\n") for body in parts[2::2]]


def body_text(text: str) -> str:
    """Text without its ``=== PAGE n ===`` marker lines."""
    return "\n".join(line for line in text.split("\n") if not PAGE_LINE.fullmatch(line))


def body_chars(text: str) -> int:
    """Characters outside the marker lines (0 means an image-only PDF)."""
    return sum(len(line) for line in text.split("\n") if not PAGE_LINE.fullmatch(line))


def cover_text(path: Any) -> str:
    """Page 1 with its running header kept (covers print revision and effective date in header blocks that
    are tagged as pagination artifacts or stripped as running lines in the corpus text)."""
    runs, _ = walk_page(PdfReader(str(path)), 0)
    return "\n".join(ln.text for ln in lines_of_page(drop_overprint(runs), 1))


# ---------------------------------------------------------------- redline spans

SPAN_KINDS = frozenset({"insert", "delete", "both", "marked"})


def redline_lines(path: Any, style: RedlineStyle, stats: WalkStats | None = None) -> list[Line]:
    """All kept non-blank body lines of a redline PDF with per-segment change status (the recipe with ``style``)."""
    return [ln for page in document_lines(path, style, stats) for ln in page if ln.text.strip()]


def sides(lines: list[Line]) -> tuple[str, str]:
    """(pre-change text, post-change text) rebuilt from the markup: plain + marked + delete / + insert."""
    old = "\n".join(ln.side(OLD_SIDE) for ln in lines)
    new = "\n".join(ln.side(NEW_SIDE) for ln in lines)
    return old, new


@dataclass
class Span:
    page: int
    kind: str
    line_start: int
    line_end: int
    text: str
    toc: bool = False
    old_start: int = 0  # offsets in the whitespace-free old side (start == end for inserts)
    old_end: int = 0
    new_start: int = 0  # offsets in the whitespace-free new side (start == end for deletes)
    new_end: int = 0
    checks: dict[str, bool] = field(default_factory=dict)


def merge_spans(lines: list[Line]) -> tuple[list[Span], str, str]:
    """One span per maximal run of same-status changed segments within a page (whitespace-only plain
    segments and line breaks do not split it), plus the whitespace-free old and new sides it indexes into."""
    spans: list[Span] = []
    old: list[str] = []
    new: list[str] = []
    n_old = n_new = 0
    cur: Span | None = None
    parts: list[str] = []

    def close() -> None:
        nonlocal cur, parts
        if cur is not None:
            cur.text = re.sub(r"\s+", " ", "".join(parts)).strip()
            cur.old_end, cur.new_end = n_old, n_new
            spans.append(cur)
        cur, parts = None, []

    for ln in lines:
        if cur is not None and cur.page != ln.page:
            close()
        for status, text in ln.segments:
            if status == "plain" and not text.strip():
                if cur is not None:
                    parts.append(text)
                continue
            if cur is not None and status != cur.kind:
                close()
            if status in SPAN_KINDS and cur is None:
                cur = Span(ln.page, status, ln.line, ln.line, "", old_start=n_old, new_start=n_new)
            if cur is not None:
                parts.append(text)
                cur.line_end = ln.line
            sq = squash_ws(text)
            if status in OLD_SIDE:
                old.append(sq)
                n_old += len(sq)
            if status in NEW_SIDE:
                new.append(sq)
                n_new += len(sq)
        if cur is not None:
            parts.append(" ")
    close()
    return spans, "".join(old), "".join(new)


def _full(ln: Line) -> str:
    return "".join(t for _, t in ln.segments).rstrip()


def toc_keys(lines: list[Line]) -> set[tuple[int, int]]:
    """(page, line) of table-of-contents lines: dot-leader lines, and every line of a contents page.

    A contents page has >= 40% dot-leader lines, or >= 50% lines ending in a page number when it carries a
    contents heading or directly follows a contents page (TOCs set with tab leaders). Word regenerates the
    TOC on every revision, so its entries show up as marked changes whose only content is page numbers.
    """
    by_page: dict[int, list[Line]] = defaultdict(list)
    for ln in lines:
        by_page[ln.page].append(ln)
    keys: set[tuple[int, int]] = set()
    prev_toc_page = -1
    for page in sorted(by_page):
        page_lines = by_page[page]
        texts = [_full(ln) for ln in page_lines]
        leader = [ln for ln, t in zip(page_lines, texts) if TOC_LINE.search(t)]
        numbered = sum(bool(PAGE_NUMBER_END.search(t)) for t in texts)
        headed = any(TOC_HEADING.match(t) for t in texts[:6]) or prev_toc_page == page - 1
        whole = (len(leader) >= TOC_PAGE_SHARE * len(page_lines)
                 or (headed and numbered >= TOC_NUMBERED_SHARE * len(page_lines)))
        if whole:
            prev_toc_page = page
        keys.update((ln.page, ln.line) for ln in (page_lines if whole else leader))
    return keys


def mark_toc(spans: list[Span], keys: set[tuple[int, int]]) -> None:
    """Flag spans that touch a table-of-contents line."""
    for span in spans:
        span.toc = any((span.page, n) in keys for n in range(span.line_start, span.line_end + 1))


def span_record(span: Span, index: int, by_line: dict[tuple[int, int], Line]) -> dict[str, Any]:
    """JSON record of one span; ``context`` is its lines in inline markup (<= 600 characters)."""
    context = " / ".join(by_line[(span.page, n)].markup().strip()
                         for n in range(span.line_start, span.line_end + 1) if (span.page, n) in by_line)
    return {"page": span.page, "span": index, "kind": span.kind, "line_start": span.line_start,
            "line_end": span.line_end, "text": span.text, "n_chars": len(squash_ws(span.text)), "toc": span.toc,
            "context": context[:600]}


def redline_records(lines: list[Line]) -> list[dict[str, Any]]:
    """Span records of redline lines (``span`` numbers the spans within each page)."""
    spans, _, _ = merge_spans(lines)
    mark_toc(spans, toc_keys(lines))
    by_line = {(ln.page, ln.line): ln for ln in lines}
    per_page: Counter = Counter()
    out = []
    for span in spans:
        out.append(span_record(span, per_page[span.page], by_line))
        per_page[span.page] += 1
    return out


def detect_redline_spans(path: Any, style: RedlineStyle, stats: WalkStats | None = None) -> list[dict[str, Any]]:
    """Changed spans of a redline PDF: {page (1-based), span, kind: insert|delete|both|marked, text, toc, ...}."""
    return redline_records(redline_lines(path, style, stats))


# ---------------------------------------------------------------- image pages

IMAGE_MIN_AREA = 0.03  # share of the page an image must cover to count (drops logos, bullets)


def image_pages(path: Any, page_texts: list[str]) -> list[dict[str, Any]]:
    """Pages that place at least one image covering >= IMAGE_MIN_AREA of the page.

    ``page_texts`` are the per-page texts from ``pdf_to_pages`` (``n_chars`` counts their non-whitespace
    characters). Returns {page (1-based), n_chars, n_images, img_area_share}.
    """
    reader = PdfReader(str(path))
    if len(page_texts) != len(reader.pages):
        raise ValueError(f"{path}: {len(page_texts)} page texts for {len(reader.pages)} pages")
    out = []
    for i, page in enumerate(reader.pages):
        box = page.mediabox
        area = float(box.width) * float(box.height)
        walker = _PageWalker(reader, i, WalkStats())
        content = page.get_contents()
        if content is not None:
            walker.walk(content, page.get("/Resources"), list(_IDENTITY))
        shares = [(x1 - x0) * (y1 - y0) / area for x0, x1, y0, y1 in walker.images]
        big = [s for s in shares if s >= IMAGE_MIN_AREA]
        if big:
            out.append({
                "page": i + 1, "n_chars": len(squash_ws(page_texts[i])), "n_images": len(big),
                "img_area_share": round(min(sum(big), 1.0), 3),
            })
    return out

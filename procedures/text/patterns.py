"""Shared regexes and text helpers of the item construction (the owner-written branch pairs S2 and the unrevised
control sentences S3; also used by the M4 memory and the scorer's verbatim checks).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Strict normative modals (the S3 definition of a normative sentence). will / should are deliberately not included:
# PJM states many duties with will / should, but the precision of such matches was not measured.
NORM_MODAL = re.compile(
    r"\b(shall not|shall|must not|must|may not|is required|are required|prohibited|not permitted)\b", re.I)
# a clause "looks like a main clause" when it carries a finite verb / modal

WORDNUM = (r"(?:a|an|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|fifteen|twenty|twenty-four"
           r"|thirty|forty|forty-five|sixty|ninety)")
WORD2NUM = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
            "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
            "twenty-four": 24, "thirty": 30, "forty": 40, "forty-five": 45, "sixty": 60, "ninety": 90}
# electrical units case-sensitive; word units case-insensitive (MISO writes "Days", "Business Days")
UNIT = (r"(?:MWh|MW|MVAR|MVAr|Mvar|MVA|kV|Hz|%|p\.u\.|pu|ms|°\s?[CF]"
        r"|(?i:percent|per[- ]unit|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?|days?|degrees?"
        r"|weeks?|months?|years?|cycles?))")
NUM = r"(?<![\w.])[±+-]?\d{1,3}(?:,\d{3})*(?:\.\d+)?|(?<![\w.])\.\d+"
QUALIFIER = r"(?i:business|calendar|consecutive|clock|operating)"
QTY = re.compile(
    rf"(?:\(?(?:{NUM})\)?|(?i:{WORDNUM})\s+\(\d+\))\s*-?\s*(?:(?P<qual>{QUALIFIER})\s+)?"
    rf"(?P<unit>{UNIT})(?![A-Za-z])")
QTY_WORD = re.compile(
    rf"\b(?:within|no later than|not later than|at least|no less than|not less than|no more than|not more than"
    rf"|up to|in excess of|greater than|less than|exceed(?:s|ing)?|every|each|per)\s+(?P<num>{WORDNUM})\s+"
    rf"(?:(?P<qual>business|calendar|consecutive|clock)\s+)?(?P<unit>seconds?|minutes?|hours?|days?|weeks?|months?"
    rf"|years?)\b", re.I)

UNIT_CANON = {
    "mwh": "MWh", "mw": "MW", "mvar": "MVAR", "mva": "MVA", "kv": "kV", "hz": "Hz", "%": "%", "percent": "%",
    "p.u.": "pu", "pu": "pu", "per unit": "pu", "per-unit": "pu", "ms": "ms", "millisecond": "ms",
    "second": "s", "sec": "s", "minute": "min", "min": "min", "hour": "h", "hr": "h", "day": "day",
    "week": "week", "month": "month", "year": "year", "cycle": "cycle", "degree": "degree",
}
DIMENSION = {"MW": "P", "MWh": "E", "MVAR": "Q", "MVA": "S", "kV": "V", "Hz": "f", "%": "pct", "pu": "pu",
             "ms": "t", "s": "t", "min": "t", "h": "t", "day": "t", "week": "t", "month": "t_cal", "year": "t_cal",
             "cycle": "cyc", "degree": "deg", "°C": "temp", "°F": "temp"}


@dataclass(frozen=True)
class Quantity:
    start: int          # offset within the searched string
    end: int
    surface: str        # verbatim text, e.g. "fifteen (15) Days"
    value: str          # decimal string, e.g. "15"
    unit: str           # canonical unit, e.g. "day"
    qualifier: str      # "business" / "calendar" / ... or ""

    @property
    def dimension(self) -> str:
        return DIMENSION.get(self.unit, self.unit)

    @property
    def gold_unit(self) -> str:
        return f"{self.qualifier} {self.unit}" if self.qualifier else self.unit


def _canon_unit(raw: str) -> str:
    u = re.sub(r"\s+", " ", raw.strip())
    if u.startswith("°"):
        return "°" + u[-1].upper()
    low = u.lower()
    if low not in UNIT_CANON and low.endswith("s"):
        low = low[:-1]
    return UNIT_CANON.get(low, low)


NUMWORD = (r"(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen"
           r"|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
           r"|thousand)")
LEFT_WORDNUM = re.compile(rf"(?i:\b{NUMWORD}(?:[\s-]+(?:and\s+)?{NUMWORD})*)(?:\s+(?i:percent))?\s*$")
RIGHT_CLOSE = re.compile(r"\)(?:\s*(?i:percent\b|%))?")


def _extend(text: str, a: int, b: int) -> tuple[int, int]:
    """Widen a "(digits ...)" quantity to its spelled-out twin: "one-hundred fifteen (115%) percent",
    "fifty percent (50%)", "one hundred twenty (120) days"."""
    if "(" not in text[a:b]:
        return a, b
    m = LEFT_WORDNUM.search(text[max(0, a - 60):a])
    if m:
        a = max(0, a - 60) + m.start()
    if text[a:b].count("(") > text[a:b].count(")"):
        r = RIGHT_CLOSE.match(text, b)
        if r:
            b = r.end()
    return a, b


def find_quantities(text: str) -> list[Quantity]:
    """All numeric quantities with a unit, left to right, non-overlapping. "Day-Ahead"/"Hour-Ahead" market
    names are not quantities. Spans are widened to the spelled-out twin of a parenthesised number."""
    out: list[Quantity] = []
    for m in QTY.finditer(text):
        if re.match(r"-?\s*[Aa]head", text[m.end():m.end() + 7]):
            continue
        head = text[m.start():m.start("qual") if m.group("qual") else m.start("unit")]
        dig = re.search(r"\((\d+)\)", head)
        if dig:
            value = dig.group(1)
        else:
            value = re.sub(r"[()\s,±+]", "", head).rstrip("-")
        if not re.fullmatch(r"-?\d*\.?\d+", value):
            continue
        a, b = _extend(text, m.start(), m.end())
        out.append(Quantity(a, b, text[a:b], value, _canon_unit(m.group("unit")), (m.group("qual") or "").lower()))
    for m in QTY_WORD.finditer(text):
        s, e = m.start("num"), m.end("unit")
        if any(q.start < e and s < q.end for q in out):
            continue
        out.append(Quantity(s, e, text[s:e], str(WORD2NUM[m.group("num").lower()]), _canon_unit(m.group("unit")),
                            (m.group("qual") or "").lower()))
    out.sort(key=lambda q: q.start)
    return out


def alnum(s: str) -> str:
    """Lower-cased alphanumeric stream (the verbatim-match convention: spacing, case and punctuation never matter)."""
    return re.sub(r"[^0-9a-z]", "", s.lower())


def norm_words(s: str) -> str:
    """Lower-case, every non-alphanumeric run -> one space; used for token-boundary containment tests."""
    return " ".join(re.sub(r"[^0-9a-z]+", " ", s.lower()).split())


def contains_phrase(haystack: str, needle: str) -> bool:
    n = norm_words(needle)
    return bool(n) and f" {n} " in f" {norm_words(haystack)} "


STOPWORDS = frozenset(
    "a an the of to in on for by with at from as and or nor but if when where whenever that this these those such "
    "its their his her it they them is are was were be been being shall must will may should can could would has "
    "have had do does did not no any all each every other than then there which who whom whose within".split())


def balanced(s: str) -> bool:
    """Parentheses balanced and never closing before opening."""
    depth = 0
    for c in s:
        depth += {"(": 1, ")": -1}.get(c, 0)
        if depth < 0:
            return False
    return depth == 0

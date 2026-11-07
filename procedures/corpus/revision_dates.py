"""Date parsing and the as-of rule for revision-based frames: revision pairs (S1) and as-of date x branch items (L2).

Rule (fixed before any model output):
  * A version's window is [effective(rev), effective(next rev) - 1 day]; the effective date is the one printed
    on the version's own cover. The owner log / external metadata are used only when no cover is on disk
    (e.g. PJM M-03 Rev 66, M-13 Rev 98) and the source is recorded.
  * If no successor is known, the window ends on the date the version was verified current (external
    metadata, `current_on`); if that is unknown too the window is open and the side gets no as-of date.
  * as_of = start + floor((end - start) / 2) days  (window midpoint, deterministic).
  * Date conflict = the owner log and the cover disagree for a revision (e.g. M-03 Rev 70: cover 2026-03-04,
    Rev 71 log 2026-02-19). The cover wins for the window; the disputed interval [min, max) of the two dates
    is recorded, and an as-of date falling inside a disputed interval is excluded; a date outside both candidate
    dates, which the window midpoint normally is, is kept.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, timedelta

MONTHS = {m: i for i, ms in enumerate(
    [("jan", "january"), ("feb", "february"), ("mar", "march"), ("apr", "april"), ("may",), ("jun", "june"),
     ("jul", "july"), ("aug", "august"), ("sep", "sept", "september"), ("oct", "october"), ("nov", "november"),
     ("dec", "december")], 1) for m in ms}
_MON = r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|June?|July?|Aug(?:ust)?|Sept?(?:ember)?|Oct(?:ober)?" \
       r"|Nov(?:ember)?|Dec(?:ember)?)"
_PATTERNS = [
    # March 4, 2026 / Sept. 1 2025 / July 01 - 2025 / Dec 01-2024
    (re.compile(rf"\b{_MON}\.?\s*-?\s*(\d{{1,2}})(?:st|nd|rd|th)?\s*[,-]?\s*(\d{{4}})\b", re.I), ("mon", "d", "y")),
    # DEC-15-2025 / SEP-1-2025 / SEP-1-24
    (re.compile(rf"\b{_MON}-(\d{{1,2}})-(\d{{4}}|\d{{2}})\b", re.I), ("mon", "d", "y")),
    # 2026-03-04
    (re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), ("y", "m", "d")),
    # 03/04/2026, 3/5/26, 09-01-2025 (North-American month first)
    (re.compile(r"\b(\d{1,2})[/-](\d{1,2})[/-](\d{4}|\d{2})\b"), ("m", "d", "y")),
]


TWO_DIGIT_YEAR_PIVOT = 70   # owner logs go back to the late 1990s (PJM M-11 Rev 1: 1997); nothing in the corpus is dated 2070+


def expand_year(year: int) -> int:
    """Two-digit years: 70..99 -> 1970..1999, 00..69 -> 2000..2069 (TWO_DIGIT_YEAR_PIVOT); four-digit years
    unchanged."""
    if year >= 100:
        return year
    return 1900 + year if year >= TWO_DIGIT_YEAR_PIVOT else 2000 + year


def parse_date(s: str | None) -> date | None:
    """First parseable date in `s`; None when nothing parses or the parse is not a valid calendar date."""
    if not s:
        return None
    best = None
    for rx, order in _PATTERNS:
        m = rx.search(s)
        if m and (best is None or m.start() < best[0].start()):
            best = (m, order)
    if best is None:
        return None
    m, order = best
    parts = dict(zip(order, m.groups()))
    month = MONTHS[parts["mon"].lower().rstrip(".")] if "mon" in parts else int(parts["m"])
    year = expand_year(int(parts["y"]))
    try:
        return date(year, month, int(parts["d"]))
    except ValueError:
        return None


@dataclass(frozen=True)
class Window:
    start: date | None
    end: date | None          # inclusive; None = open / unknown
    start_source: str         # cover | owner_log | external | none
    end_source: str           # successor_cover | successor_owner_log | successor_external | owner_inactive_date |
                              # current_on | generation_reset / date_after_cover (off the ladder, no end) | none

    def contains(self, d: date) -> bool:
        return self.start is not None and self.end is not None and self.start <= d <= self.end


@dataclass(frozen=True)
class Conflict:
    rev: int
    cover: date
    log: date

    @property
    def interval(self) -> tuple[date, date]:
        """Disputed days [lo, hi): on these days the governing revision depends on which date is believed."""
        return (min(self.cover, self.log), max(self.cover, self.log))


def midpoint(w: Window) -> date | None:
    if w.start is None or w.end is None or w.end < w.start:
        return None
    return w.start + timedelta(days=(w.end - w.start).days // 2)


def as_of(w: Window, conflicts: list[Conflict]) -> tuple[date | None, list[str]]:
    """Window-midpoint as-of date for one side, plus flags. Returns (None, flags) when no admissible date exists."""
    flags = []
    if w.start is None:
        return None, ["window_start_unknown"]
    if w.end is None:
        return None, ["window_open"]
    if w.end < w.start:
        return None, ["window_empty"]
    d = midpoint(w)
    for c in conflicts:
        lo, hi = c.interval
        if lo <= d < hi:
            flags.append(f"asof_in_date_conflict(rev{c.rev})")
            return None, flags
        if w.start - timedelta(days=1) <= hi and lo <= w.end + timedelta(days=1):
            flags.append(f"window_touches_date_conflict(rev{c.rev})")
    if w.start_source != "cover" or w.end_source not in ("successor_cover", "current_on"):
        flags.append(f"window_bounds({w.start_source},{w.end_source})")
    return d, flags

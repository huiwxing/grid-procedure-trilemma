"""The as-of rule (an item asks about the midpoint of its revision's effective window): date parsing, window
midpoints, successor / current-on bounds, date conflicts."""
from datetime import date
from pathlib import Path

import pytest

from procedures.corpus.revision_dates import Conflict, Window, as_of, midpoint, parse_date
from procedures.corpus.ladder import Ladder, Version


@pytest.mark.parametrize("raw,expected", [
    ("Effective Date: March 4, 2026", date(2026, 3, 4)),
    ("Revision 70 (02/19/2026):", date(2026, 2, 19)),
    ("Effective Date: DEC-15-2025", date(2025, 12, 15)),
    ("Effective Date: SEP-1-2025", date(2025, 9, 1)),
    ("Effective Date: SEP-1-24", date(2024, 9, 1)),
    ("Effective Date: JUNE-06-2025", date(2025, 6, 6)),
    ("Effective Date: July 01 - 2025", date(2025, 7, 1)),
    ("Effective Date: Dec 01-2024", date(2024, 12, 1)),
    ("Effective Date: 09-01-2025", date(2025, 9, 1)),
    ("Effective Date: Oct-01-2025", date(2025, 10, 1)),
    ("Effective Date  3/05/26", date(2026, 3, 5)),
    ("2026-06-24", date(2026, 6, 24)),
])
def test_parse_date_formats(raw, expected):
    assert parse_date(raw) == expected


@pytest.mark.parametrize("raw", ["Effective Date: JUL-2024", "Effective Date: JUNDEC-135-2025", "", None, "02/30/2025"])
def test_parse_date_rejects_partial_garbled_or_invalid(raw):
    assert parse_date(raw) is None


def test_midpoint_is_floor_of_inclusive_window():
    assert midpoint(Window(date(2025, 11, 20), date(2026, 3, 3), "cover", "successor_cover")) == date(2026, 1, 10)
    assert midpoint(Window(date(2026, 1, 1), date(2026, 1, 1), "cover", "successor_cover")) == date(2026, 1, 1)


def test_open_or_unknown_window_gives_no_asof():
    assert as_of(Window(date(2025, 8, 1), None, "cover", "none"), []) == (None, ["window_open"])
    assert as_of(Window(None, date(2025, 8, 1), "none", "successor_cover"), [])[0] is None


def test_conflict_excludes_only_asof_inside_disputed_interval():
    rev70 = Conflict(70, cover=date(2026, 3, 4), log=date(2026, 2, 19))   # PJM M-03 Rev 70: cover and log disagree
    # Rev 69 window ends the day before the Rev 70 cover date; its midpoint is before the disputed days
    d, flags = as_of(Window(date(2025, 11, 20), date(2026, 3, 3), "cover", "successor_cover"), [rev70])
    assert d == date(2026, 1, 10)
    assert flags == ["window_touches_date_conflict(rev70)"]
    # a short window whose midpoint falls on a disputed day is excluded
    d, flags = as_of(Window(date(2026, 2, 10), date(2026, 3, 3), "cover", "successor_cover"), [rev70])
    assert d is None and flags == ["asof_in_date_conflict(rev70)"]


def test_non_cover_bounds_are_flagged_not_excluded():
    d, flags = as_of(Window(date(2023, 11, 15), date(2024, 5, 21), "cover", "successor_owner_log"), [])
    assert d == date(2024, 2, 17)
    assert flags == ["window_bounds(cover,successor_owner_log)"]


def _ladder() -> Ladder:
    lad = Ladder("PJM", "M-03")
    p = Path("x.txt")
    lad.versions = {
        "65": Version("65", p, cover_effective=date(2023, 11, 15), log_effective=date(2023, 11, 15)),
        "66": Version("66", None, log_effective=date(2024, 5, 22)),                  # text missing: owner log only
        "67": Version("67", p, cover_effective=date(2024, 11, 21), log_effective=date(2024, 11, 21)),
        "70": Version("70", p, cover_effective=date(2026, 3, 4), log_effective=date(2026, 2, 19)),
        "71": Version("71", p, cover_effective=date(2026, 5, 20), log_effective=date(2026, 5, 20)),
    }
    lad.current_on = date(2026, 9, 24)
    return lad


def test_ladder_window_uses_next_known_revision_even_without_text():
    w = _ladder().window("65")
    assert (w.start, w.end, w.start_source, w.end_source) == (
        date(2023, 11, 15), date(2024, 5, 21), "cover", "successor_owner_log")


def test_ladder_cover_date_wins_over_owner_log():
    lad = _ladder()
    assert lad.effective("70") == (date(2026, 3, 4), "cover")
    assert lad.window("67").end == date(2026, 3, 3)


def test_ladder_newest_on_disk_closes_at_current_on():
    w = _ladder().window("71")
    assert (w.end, w.end_source) == (date(2026, 9, 24), "current_on")

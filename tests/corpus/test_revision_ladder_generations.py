"""Revision ladders across a numbering reset: the IESO market-renewal manuals restart at 1.0 under the same doc_id as
the legacy manuals, so rev_key order alone put legacy 7.3 v50.0 (2024-12-04) after renewed v3.0 (2025-12-03):
reversed windows, and the legacy text read as current. Synthetic files with the real manifest dates; a normal ladder
must come out as before."""
import csv
from datetime import date
from pathlib import Path

import pytest

from procedures.corpus.ladder import Ladder, Version, VersionFile, build_ladders, flag_off_ladder, rev_key
from procedures.corpus.revisions import RevisionIndex

FIELDS = ["family", "doc_id", "revision", "doc_kind", "cover_effective_date", "text_path", "status", "inactive_date",
          "provenance_status", "fetched_at_utc", "gold_eligible", "automatic_frame_exclusion", "excluded",
          "exclude_reason"]
FETCHED = "2026-09-25T10:00:00Z"
# IESO rows of the corpus builder's raw manifest, dates and statuses as the builder wrote them (the cover dates are
# those of data/corpus/manifest.csv)
IESO = [
    ("IESO-MM-7.3", "3.0", "2025-12-03", "in_effect", ""),
    ("IESO-MM-7.3", "49.0", "2023-11-29", "superseded", "revision_number_reset"),
    ("IESO-MM-7.3", "50.0", "2024-12-04", "superseded", "revision_number_reset"),
    ("IESO-MM-7.4", "3.0", "2026-09-09", "in_effect", ""),
    ("IESO-MM-7.4", "41.0", "2022-09-14", "superseded", "revision_number_reset"),
]


def _manifest(tmp_path: Path, marker: bool = True, view: bool = False) -> Path:
    """marker: the manifest's automatic_frame_exclusion on the legacy rows; view: excluded / exclude_reason on them."""
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for doc, rev, eff, status, afe in IESO:
            name = f"{doc}_{rev}.txt"
            (tmp_path / name).write_text(f"{doc} version {rev}\nEffective {eff}\n", encoding="utf-8")
            w.writerow(dict(family="IESO", doc_id=doc, revision=rev, doc_kind="clean", cover_effective_date=eff,
                            text_path=name, status=status, provenance_status="owner_site", fetched_at_utc=FETCHED,
                            gold_eligible="true", automatic_frame_exclusion=afe if marker else "",
                            excluded="true" if view and afe else "false",
                            exclude_reason=f"expansion_sampling_source:{afe}" if view and afe else ""))
    return path


def test_the_marker_keeps_the_legacy_generation_off_the_ladder(tmp_path):
    index = RevisionIndex.from_manifest(_manifest(tmp_path))
    assert [(d, r.revision) for d in index.documents() for r in index.ladder(d)] == [("IESO-MM-7.3", "3.0"),
                                                                                      ("IESO-MM-7.4", "3.0")]
    renewed = index.ladder("IESO-MM-7.3")[0]
    assert (renewed.effective, renewed.window_end) == (date(2025, 12, 3), date(2026, 9, 25))   # current on the fetch


def test_excluded_rows_never_enter_a_ladder(tmp_path):
    index = RevisionIndex.from_manifest(_manifest(tmp_path, marker=False, view=True))
    assert [r.revision for d in index.documents() for r in index.ladder(d)] == ["3.0", "3.0"]


def test_generation_reset_without_the_marker_is_an_explicit_error(tmp_path):
    """The marker missing: rev_key order would have given renewed 3.0 the window [2025-12-03, 2023-11-28] and left
    legacy 50.0 open to today. The ladder does not guess which generation governs: it stops and names the revisions
    and the remedy."""
    with pytest.raises(ValueError) as err:
        RevisionIndex.from_manifest(_manifest(tmp_path, marker=False))
    msg = str(err.value)
    assert "IESO-MM-7.3" in msg and "generation_reset" in msg and "r49.0" in msg and "r3.0" in msg
    assert "revision_number_reset" in msg


def test_one_day_cover_inversion_inside_a_generation_is_not_silently_resolved(tmp_path):
    """r70 (in effect) dated one day before r69 (blank status). Guessing would have moved r70 off the ladder and given
    r69 a window; the ladder must refuse instead."""
    files = _files(tmp_path, "PJM", "M-14D", [("68", "2025-06-01", "superseded"), ("69", "2025-12-17", ""),
                                               ("70", "2025-12-16", "in_effect")])
    with pytest.raises(ValueError) as err:
        build_ladders(files, [])
    assert "r70" in str(err.value) and "r69" in str(err.value)


def _files(tmp_path: Path, family: str, doc: str, revs: list[tuple[str, str, str]]) -> list[VersionFile]:
    out = []
    for rev, eff, status in revs:
        p = tmp_path / f"{doc}_{rev}.txt"
        p.write_text(f"{doc} {rev}\n", encoding="utf-8")
        out.append(VersionFile(family, doc, rev, p, date.fromisoformat(eff) if eff else None, status=status,
                               verified_current_on=date(2026, 9, 24) if status == "in_effect" else None))
    return out


def test_normal_ladders_are_unchanged(tmp_path):
    """Integer labels that sort differently as text ("9" < "13"), MISO labels with an undated newest revision:
    order = rev_key order, windows chained by the next cover date, nothing flagged. (PJM ladders also read the
    owner log.)"""
    files = (_files(tmp_path, "CAISO", "CAISO-OP-3100", [("9", "2023-11-15", "superseded"), ("13", "2024-11-21", "superseded"),
                                              ("71", "2026-05-20", "in_effect")])
             + _files(tmp_path, "MISO", "MISO-BPM-015", [("31", "2025-06-01", "superseded"),
                                                         ("32", "2025-12-15", "in_effect"), ("33", "", "")]))
    ladders = build_ladders(files, [])
    caiso, miso = ladders[("CAISO", "CAISO-OP-3100")], ladders[("MISO", "MISO-BPM-015")]
    assert caiso.order() == sorted(caiso.versions, key=rev_key) == ["9", "13", "71"]
    assert miso.on_disk() == ["31", "32", "33"]            # the undated 33 keeps its rev_key place
    assert not caiso.off_ladder and not miso.off_ladder
    assert (caiso.window("9").end, caiso.window("13").end) == (date(2024, 11, 20), date(2026, 5, 19))
    assert caiso.current_rev() == "71" and caiso.window("71").end == date(2026, 9, 24)
    assert miso.window("31").end == date(2025, 12, 14)
    # an undated revision on disk with status "" is the newest in effect, as before; it has no window
    assert miso.current_rev() == "33" and miso.window("33").start is None


def test_current_is_none_when_a_later_dated_revision_has_taken_effect():
    """The newest text on disk is not current when the owner log / external metadata date a successor that took
    effect before the day the owner site was read; a successor announced for later does not close it."""
    lad = Ladder("MISO", "BPM-001")
    p = Path("x.txt")
    lad.versions = {"20": Version("20", p, cover_effective=date(2024, 6, 1)),
                    "21": Version("21", p, cover_effective=date(2025, 6, 6)),
                    "22": Version("22", None, external=date(2026, 6, 6))}
    assert lad.current_rev() is None
    assert lad.window("21").end == date(2026, 6, 5)
    lad.versions["21"].verified_current_on = date(2026, 1, 15)      # 22 announced after that day
    assert lad.current_rev() == "21"
    lad.versions["22"].status = "future_enforcement"
    lad.versions["21"].verified_current_on = None
    assert lad.current_rev() == "21"


def test_misread_log_year_does_not_become_the_successor():
    """An owner-log revision dated after a higher-numbered cover (PJM M-11 Rev 0 "05/01/97" read as 2097 before the
    two-digit-year pivot) is kept off the ladder: in (date, rev_key) order it would otherwise end the current
    revision's window in 2097. No Conflict: the cover dates win."""
    lad = Ladder("PJM", "M-11")
    p = Path("x.txt")
    lad.versions = {"0": Version("0", None, log_effective=date(2097, 5, 1)),
                    "136": Version("136", p, cover_effective=date(2025, 10, 1)),
                    "137": Version("137", p, cover_effective=date(2026, 7, 28),
                                   verified_current_on=date(2026, 9, 25))}
    flags = flag_off_ladder(lad)
    assert flags == {"0": ("date_after_cover", date(2025, 10, 1), "136")}
    assert lad.order() == ["136", "137"] and lad.window("0").end_source == "date_after_cover"
    lad.current_on = date(2026, 9, 25)
    assert lad.current_rev() == "137"
    assert (lad.window("137").end, lad.window("137").end_source) == (date(2026, 9, 25), "current_on")

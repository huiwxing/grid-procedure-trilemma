"""Revision ladders: which versions of each document exist, where their text is, and their effective dates.

The ladders are built from `VersionFile` rows (procedures.corpus.revisions.RevisionIndex.from_manifest makes them
from the corpus manifest through its one column mapping, procedures.corpus.manifest.read_manifest): clean revision
texts that were ever (or will be) in effect (never_effective revisions are not on the ladder) and that no exclusion
column withholds: `excluded` (reason in `exclude_reason`) and `automatic_frame_exclusion` (e.g. revision_number_reset
on the IESO pre-renewal manuals that share a doc_id with the renewed ones). `build_ladders` adds owner-log dates,
external dates and date conflicts.
Revisions are labels ("71", "8.2", "6.1"); `rev_key` is the integer runs of the label. A ladder is ordered by
(effective date, rev_key) (`Ladder.order`); an undated revision keeps its place after its rev_key predecessor.
Generation guard: an on-disk revision dated before a lower-numbered on-disk revision belongs to another numbering
generation (IESO legacy 7.3 v50.0 of 2024-12-04 vs renewed v3.0 of 2025-12-03) or carries a wrong cover date; the
ladder does not guess which — `build_ladders` raises ValueError (`generation_reset`) until the manifest marks the
older generation (`automatic_frame_exclusion=revision_number_reset` / `excluded`) or the date is corrected
(a one-day cover-date inversion inside one generation must not silently move a revision off the ladder). An
off-disk revision (owner log / external metadata) dated after a higher-numbered on-disk revision's date is flagged
`date_after_cover` and kept off the ladder without a Conflict: the cover date wins over owner-log and external
dates, and a misread log year (the two-digit "97" read as 2097) must not become the successor that ends the
current window.

Window of a revision: [effective, end] with end = the earliest of
(the next dated revision's effective date - 1 day, next in ladder order) and the owner's `inactive_date`; the
current revision (the newest on-disk revision in effect: a later pending / future-enforcement revision does not end
it; a later-dated revision that has taken effect means no on-disk text is current) without a dated successor ends
on the date it was verified current (`current_on`: the manifest fetch date of an owner-site copy, else an external
`current_on` row). The manifest's `successor_check`, `gold_eligible` and status travel on the Version so that as-of
dates are never taken from windows the corpus could not confirm (the item construction refused them).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from procedures.corpus.manifest import canonical_title
from procedures.corpus.revision_dates import Conflict, Window, parse_date
from procedures.corpus.revision_owner_logs import MisoEntry, PjmBullet, parse_miso_history, parse_pjm_history
from procedures.corpus.revision_text import read_lf, sha256_lf


def rev_key(label: str) -> tuple[int, ...]:
    """Order of revision labels: "8.2" < "17.0" < "17.2"; "9" < "13"; "1" < "1.1"."""
    return tuple(int(x) for x in re.findall(r"\d+", str(label))) or (-1,)


def same_doc(a: str, b: str) -> bool:
    """Document ids across naming schemes: "MISO-BPM-015" (manifest), "BPM-015" (archive / external table)."""
    def core(d: str) -> str:
        return re.sub(r"^(?:MISO|NERC|CAISO|ISONE|ISO-NE|PJM)-", "", d.upper())
    return core(a) == core(b)


# statuses that were never rule text: kept off the ladder that revision pairs (S1) are built on
OFF_LADDER_STATUSES = ("never_effective", "informational")
IN_EFFECT_STATUSES = ("", "in_effect")
# a dated successor with one of these statuses has taken effect: it closes the window of every earlier revision
TAKEN_EFFECT_STATUSES = IN_EFFECT_STATUSES + ("superseded",)


@dataclass(frozen=True)
class VersionFile:
    family: str                 # PJM | MISO | NERC | CAISO | ISO-NE
    doc: str                    # M-03 | MISO-BPM-031 | NERC-IRO-008 | CAISO-OP-3100 (manifest doc id)
    rev: str
    path: Path
    effective_date: date | None = None   # manifest-supplied cover date (else parsed from the text)
    inactive_date: date | None = None    # owner's last effective day (manifest)
    successor_check: str = ""            # confirmed | gap_exact | gap_upper_bound | unconfirmed | "" (manifest)
    gold_eligible: bool = True
    status: str = ""                     # manifest status ("" / in_effect / superseded / future_enforcement ...)
    verified_current_on: date | None = None   # owner-site fetch date (manifest), used for the newest revision


@dataclass
class Version:
    rev: str
    path: Path | None = None
    sha256: str = ""
    cover_effective: date | None = None
    approval: date | None = None
    log_effective: date | None = None
    external: date | None = None         # successor-only date known from external metadata, no text on disk
    inactive: date | None = None
    successor_check: str = ""
    gold_eligible: bool = True
    status: str = ""
    verified_current_on: date | None = None


@dataclass
class Ladder:
    family: str
    doc: str
    title: str = ""
    versions: dict[str, Version] = field(default_factory=dict)
    current_on: date | None = None       # newest on-disk version verified current on this date
    conflicts: list[Conflict] = field(default_factory=list)
    admin_changes: list[PjmBullet] = field(default_factory=list)
    bullets: list[PjmBullet] = field(default_factory=list)
    miso_history: list[MisoEntry] = field(default_factory=list)
    history_source: str = ""
    # rev -> "generation_reset: ..." / "date_after_cover: ..." (flag_off_ladder); kept in `versions`, off the ladder
    off_ladder: dict[str, str] = field(default_factory=dict)

    def order(self) -> list[str]:
        """Revisions not flagged off the ladder, oldest first by (effective date, rev_key); an undated revision sorts
        with the date of its rev_key predecessor, so it keeps its place right after it."""
        keyed, carried = [], date.min
        for r in sorted(self.versions, key=rev_key):
            if r in self.off_ladder:
                continue
            carried = self.effective(r)[0] or carried
            keyed.append(((carried, rev_key(r)), r))
        return [r for _, r in sorted(keyed)]

    def on_disk(self) -> list[str]:
        return [r for r in self.order() if self.versions[r].path is not None]

    def current_rev(self) -> str | None:
        """Newest on-disk revision in effect (manifest status empty or in_effect) in ladder order; None when a
        later-dated revision that has taken effect (TAKEN_EFFECT_STATUSES, off-disk owner-log / external revisions
        included) closes its window. A successor dated after the day the owner site showed the revision current is
        announced, not in effect, and does not count."""
        order = self.order()
        live = [r for r in order if self.versions[r].path is not None and self.versions[r].status in IN_EFFECT_STATUSES]
        if not live:
            return None
        cur = live[-1]
        seen_current = self.versions[cur].verified_current_on
        for r in order[order.index(cur) + 1:]:
            d = self.effective(r)[0]
            if (d is not None and self.versions[r].status in TAKEN_EFFECT_STATUSES
                    and (seen_current is None or d <= seen_current)):
                return None
        return cur

    def effective(self, rev: str) -> tuple[date | None, str]:
        v = self.versions.get(rev)
        if v is None:
            return None, "none"
        if v.cover_effective:
            return v.cover_effective, "cover"
        if v.log_effective:
            return v.log_effective, "owner_log"
        if v.external:
            return v.external, "external"
        return None, "none"

    def window(self, rev: str) -> Window:
        start, s_src = self.effective(rev)
        if rev in self.off_ladder:
            return Window(start, None, s_src, self.off_ladder[rev].split(":", 1)[0])
        ends: list[tuple[date, str]] = []
        order = self.order()
        later = [r for r in order[order.index(rev) + 1:] if self.effective(r)[0] is not None]
        if later:
            nxt, n_src = self.effective(later[0])
            ends.append((nxt - timedelta(days=1), {"cover": "successor_cover", "owner_log": "successor_owner_log",
                                                   "external": "successor_external"}[n_src]))
        if self.versions[rev].inactive:
            ends.append((self.versions[rev].inactive, "owner_inactive_date"))
        if ends:
            end, e_src = min(ends)
            return Window(start, end, s_src, e_src)
        if self.current_on and rev == self.current_rev():
            return Window(start, self.current_on, s_src, "current_on")
        return Window(start, None, s_src, "none")


PJM_EFF = re.compile(r"Effective Date:\s*([A-Za-z]+ \d{1,2}, \d{4}|\d{1,2}/\d{1,2}/\d{4})")
PJM_APPROVAL = re.compile(r"Approval Date:\s*(\d{1,2}/\d{1,2}/\d{4})")
MISO_EFF = re.compile(r"Effective Date:\s*([^\n]{4,24})")


def read_external(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def build_ladders(files: list[VersionFile], external: list[dict]) -> dict[tuple[str, str], Ladder]:
    ladders: dict[tuple[str, str], Ladder] = {}
    texts: dict[tuple[str, str, str], str] = {}
    for vf in files:
        lad = ladders.setdefault((vf.family, vf.doc), Ladder(vf.family, vf.doc))
        text = read_lf(vf.path)
        texts[(vf.family, vf.doc, vf.rev)] = text
        v = Version(vf.rev, vf.path, sha256_lf(text), inactive=vf.inactive_date, successor_check=vf.successor_check,
                    gold_eligible=vf.gold_eligible, status=vf.status, verified_current_on=vf.verified_current_on)
        head = text[:1500]
        if vf.family == "PJM":
            m = PJM_EFF.search(head)
            v.cover_effective = vf.effective_date or (parse_date(m.group(1)) if m else None)
            m = PJM_APPROVAL.search(text[:20000])
            v.approval = parse_date(m.group(1)) if m else None
        else:
            m = MISO_EFF.search(head)
            v.cover_effective = vf.effective_date or (parse_date(m.group(1)) if m else None)
        if vf.rev in lad.versions:
            raise ValueError(f"two texts for {vf.family} {vf.doc} r{vf.rev}: {lad.versions[vf.rev].path}, {vf.path}")
        lad.versions[vf.rev] = v
    for (fam, doc), lad in ladders.items():
        flag_off_ladder(lad)                   # on cover dates: the owner history is read from the newest generation
        newest = lad.on_disk()[-1]
        text = texts[(fam, doc, newest)]
        lad.title = title_of(fam, doc)
        lad.history_source = lad.versions[newest].path.name
        if fam == "PJM":
            lad.bullets = parse_pjm_history(text, pjm_manual(doc))
            for b in lad.bullets:
                if b.entry_kind == "revision" and b.revision_number is not None:
                    v = lad.versions.setdefault(str(b.revision_number), Version(str(b.revision_number)))
                    v.log_effective = v.log_effective or parse_date(b.date_raw)
                elif b.entry_kind == "administrative_change":
                    lad.admin_changes.append(b)
        elif fam == "MISO":
            lad.miso_history = parse_miso_history(text)
            for e in lad.miso_history:
                if same_doc(e.doc_number.rsplit("-r", 1)[0], doc) and e.date is not None:
                    v = lad.versions.setdefault(str(e.rev), Version(str(e.rev)))
                    v.log_effective = v.log_effective or e.date

    def ladder_of(r: dict) -> Ladder | None:
        return next((x for (f, d), x in sorted(ladders.items()) if f == r["family"] and same_doc(d, r["doc"])), None)

    for r in external:
        lad = ladder_of(r)
        if lad is not None and r["kind"] != "current_on":
            lad.versions.setdefault(r["rev"], Version(r["rev"])).external = parse_date(r["effective_date"])
    for lad in ladders.values():
        flag_off_ladder(lad)                   # again with every owner-log / external date
        lad.conflicts = [Conflict(v.rev, v.cover_effective, v.log_effective) for v in lad.versions.values()
                         if v.cover_effective and v.log_effective and v.cover_effective != v.log_effective]
        lad.versions = dict(sorted(lad.versions.items(), key=lambda kv: rev_key(kv[0])))
        cur = lad.current_rev()
        if cur is not None and lad.versions[cur].verified_current_on:
            lad.current_on = lad.versions[cur].verified_current_on
    for r in external:                         # after every date is known: current_rev() is final
        lad = ladder_of(r)
        if lad is not None and r["kind"] == "current_on" and r["rev"] == lad.current_rev() and lad.current_on is None:
            lad.current_on = parse_date(r["effective_date"])
    return ladders


def flag_off_ladder(lad: Ladder) -> dict[str, tuple[str, date, str]]:
    """Recompute `lad.off_ladder` from the ladder's current dates (idempotent); returns rev -> (flag, the other
    revision's date, the other revision).
      generation_reset: walking the on-disk revisions in rev_key order, one dated before a lower-numbered on-disk
        revision -> ValueError (the manifest must mark the older generation; nothing is inferred from dates);
      date_after_cover: an off-disk revision dated after a higher-numbered on-disk revision (kept off the ladder)."""
    flags: dict[str, tuple[str, date, str]] = {}
    covers: list[tuple[str, date]] = []
    for r in sorted((r for r, v in lad.versions.items() if v.path is not None), key=rev_key):
        d = lad.effective(r)[0]
        if d is None:
            continue
        top = max(covers, key=lambda x: x[1], default=None)
        if top is not None and d < top[1]:
            # Never decide a numbering reset from dates alone: a one-day cover-date inversion inside one
            # generation would be read as a reset and could hand a wrong revision a window.
            raise ValueError(
                f"{lad.family} {lad.doc}: on-disk r{r} (effective {d}) is dated before lower-numbered r{top[0]} "
                f"({top[1]}): a numbering reset or a wrong cover date. Mark the older generation in the manifest "
                f"(automatic_frame_exclusion=revision_number_reset, or excluded) or correct the date; the ladder "
                f"does not guess (flag generation_reset).")
        covers.append((r, d))
    for r, v in lad.versions.items():
        d = lad.effective(r)[0]
        if v.path is None and d is not None:
            earlier = [(h, dh) for h, dh in covers if rev_key(h) > rev_key(r) and dh < d]
            if earlier:
                h, dh = min(earlier, key=lambda x: rev_key(x[0]))
                flags[r] = ("date_after_cover", dh, h)
    lad.off_ladder = {r: f"{flag}: effective {lad.effective(r)[0]} vs r{other} ({od})"
                      for r, (flag, od, other) in sorted(flags.items(), key=lambda kv: rev_key(kv[0]))}
    return flags


def pjm_manual(doc: str) -> str:
    """Manual id the PJM history parser writes into its rows: M-03 / PJM-M03 -> M-03."""
    return "M-" + re.search(r"(\d+)", doc).group(1).zfill(2)


def title_of(family: str, doc: str) -> str:
    """Question {document} name without a revision (an as-of question must not name the version): the canonical
    title rule of every item (procedures.corpus.manifest.canonical_title); "-IP" documents are implementation
    plans."""
    if doc.endswith("-IP"):
        return canonical_title(family, doc[:-3], "", "impl_plan")
    return canonical_title(family, doc, "", "clean")

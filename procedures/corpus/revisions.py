"""Revision ladders for memory building: which revision texts exist per document, their effective windows, and which
of them are in effect on an item's as-of date.

The memory system holds the whole revision ladder plus the effective-date manifest; the retrieval (M2),
constraint-sentence (M4) and summary (M5) memories read only the revisions in effect on the item's as-of date. The
window rule is the one the revision-pair (S1), as-of x branch (L2) and as-of x branch x exception (L3) frames used
to put as-of dates on questions, reused rather than restated:
  * column mapping: `procedures.corpus.manifest.read_manifest` (the one place manifest column names are mapped);
  * rows kept: clean revisions and implementation plans (`procedures.corpus.manifest.USABLE_KINDS`; an
    implementation plan is its own document "<doc>-IP", as in the item construction) that are not `excluded`, carry
    no `automatic_frame_exclusion` and are not off the ladder (`procedures.corpus.ladder.OFF_LADDER_STATUSES`:
    never_effective, informational);
  * windows: `procedures.corpus.ladder.build_ladders` + `Ladder.window`:
    [effective, min(next dated revision's effective - 1 day, owner inactive date)], the current revision closed on the
    date it was verified current; both ends inclusive. A window with an unknown start never matches; an unknown end
    is open.
Memories and their cache keys name revisions by the hash of the LF text on disk; when it is not the manifest's
`sha256_lf` (the text the paper's runs read), a warning says so and the text is used as it is.

Paths: a manifest row's `text_path` and an item's corpus paths are relative to the repository root
("data/corpus/<family>/<file>.txt"). A relative path is looked up from the manifest's directory upwards, never above
the repository root when the manifest lies inside it.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
import warnings

from procedures import ROOT
from procedures.corpus.manifest import USABLE_KINDS, is_true, read_manifest
from procedures.corpus.revision_dates import parse_date
from procedures.corpus.ladder import OFF_LADDER_STATUSES, VersionFile, build_ladders, read_external, rev_key, title_of
from procedures.corpus.revision_text import read_lf, sha256_lf
from procedures.items.template import revision_label

IMPL_PLAN = "impl_plan"


@dataclass(frozen=True)
class RevisionText:
    """One revision text on a document's ladder."""
    family: str
    doc: str                    # document id (the items' `doc`): "M-13", "MISO-BPM-015", "NERC-FAC-011-IP"
    revision: str
    title: str                  # question-style document name without a revision (ladder.title_of)
    effective: date | None
    window_end: date | None     # inclusive; None = open
    path: Path                  # absolute
    source_path: str            # the manifest's text_path as written (provenance)
    sha256_lf: str

    def text(self) -> str:
        return _text(self.path, self.sha256_lf)

    def header(self) -> str:
        eff = f" (effective {self.effective.isoformat()})" if self.effective else ""
        return f"{self.title}, {revision_label(self.revision)}{eff}"

    def in_effect_on(self, day: date) -> bool:
        return (self.effective is not None and self.effective <= day
                and (self.window_end is None or day <= self.window_end))


@lru_cache(maxsize=64)
def _text(path: Path, sha: str) -> str:
    text = read_lf(path)
    if sha256_lf(text) != sha:
        raise ValueError(f"{path}: the text changed since the revision index read it (LF sha256 {sha256_lf(text)}, "
                         f"then {sha})")
    return text


def _norm(path: Path) -> str:
    return os.path.normcase(str(Path(path).resolve()))


def _search_dirs(base: Path) -> tuple[Path, ...]:
    """`base` and its parents, up to the repository root when `base` lies inside it."""
    base = Path(base).resolve()
    dirs = (base, *base.parents)
    root = Path(ROOT).resolve()
    return dirs[:dirs.index(root) + 1] if root in dirs else dirs


def _resolve(rel: str, base: Path) -> Path:
    """A manifest or item path (module docstring): absolute, or relative to `base` or one of its parents up to the
    repository root."""
    p = Path(rel.replace("\\", "/"))
    if p.is_absolute():
        return p
    dirs = _search_dirs(base)
    for d in dirs:
        if (d / p).is_file():
            return (d / p).resolve()
    raise FileNotFoundError(f"path {rel!r} not found under {base} or a parent directory up to {dirs[-1]}")


class RevisionIndex:
    """Documents -> revision ladders (revision order), with effective windows."""

    def __init__(self, ladders: dict[str, tuple[RevisionText, ...]], base: Path):
        self._ladders = ladders
        self.base = base
        self._by_path = {_norm(r.path): r for revs in ladders.values() for r in revs}

    @classmethod
    def from_manifest(cls, manifest: Path, external: Path | None = None, docs=None) -> RevisionIndex:
        """Load the corpus manifest (CSV or JSONL); `external` = data/corpus/version_dates_external.csv (successor /
        current_on dates without text on disk), optional. `docs` (document ids as the items name them, e.g. the items'
        `doc`s) keeps only those documents' ladders; None = every document. A ladder is built from its own document's
        rows and the external dates alone (procedures.corpus.ladder.build_ladders), so a kept ladder is the one the
        whole manifest gives, and rows of other documents are neither read nor checked."""
        wanted = None if docs is None else frozenset(docs)
        manifest = Path(manifest).resolve()
        files: list[VersionFile] = []
        source: dict[tuple[str, str, str], tuple[str, str]] = {}
        for r in read_manifest(manifest):
            kind = r.get("kind")
            # a manifest row the corpus marks automatic_frame_exclusion (e.g. revision_number_reset on the IESO
            # pre-renewal manuals) never enters a memory ladder
            if (kind not in USABLE_KINDS or is_true(r.get("excluded", "")) or r.get("automatic_frame_exclusion")
                    or r.get("status") in OFF_LADDER_STATUSES):
                continue
            doc = r["doc"] + ("-IP" if kind == IMPL_PLAN and not r["doc"].endswith("-IP") else "")
            if wanted is not None and doc not in wanted:
                continue
            key = (r["family"], doc, r["revision"])
            if key in source:
                raise ValueError(f"{manifest}: two usable rows for {key}")
            source[key] = (r["path"], r.get("sha256_lf", ""))
            fetched = parse_date(r.get("fetched_at_utc", "")[:10])
            files.append(VersionFile(
                r["family"], doc, r["revision"], _resolve(r["path"], manifest.parent),
                parse_date(r.get("effective_date") or ""), parse_date(r.get("inactive_date") or ""),
                r.get("successor_check", ""), is_true(r.get("gold_eligible", "true")), r.get("status", ""),
                fetched if r.get("provenance_status") == "owner_site" else None))
        ladders: dict[str, tuple[RevisionText, ...]] = {}
        for (family, doc), lad in sorted(build_ladders(files, read_external(external) if external else []).items()):
            if doc in ladders:
                raise ValueError(f"{manifest}: document id {doc!r} is used by two families")
            revs = []
            for rev in lad.on_disk():
                v = lad.versions[rev]
                src, sha = source[(family, doc, rev)]
                if sha and sha != v.sha256:
                    warnings.warn(f"{v.path}: the text differs from the one the manifest records (LF sha256 "
                                  f"{v.sha256[:12]}, recorded {sha[:12]}); memories are built from it as it is",
                                  stacklevel=2)
                window = lad.window(rev)
                revs.append(RevisionText(family, doc, rev, title_of(family, doc), window.start, window.end,
                                         Path(v.path).resolve(), src, v.sha256))
            ladders[doc] = tuple(revs)
        return cls(ladders, manifest.parent)

    def documents(self) -> list[str]:
        return sorted(self._ladders)

    def ladder(self, doc: str) -> tuple[RevisionText, ...]:
        if doc not in self._ladders:
            raise KeyError(f"document {doc!r} is not in the manifest ladders")
        return self._ladders[doc]

    def in_effect(self, doc: str, day: date) -> list[RevisionText]:
        return [r for r in self.ladder(doc) if r.in_effect_on(day)]

    def by_path(self, path: str) -> RevisionText:
        """The revision whose text file is `path`: absolute, or relative to the manifest directory or a parent up to
        the repository root (module docstring)."""
        key = _norm(_resolve(path, self.base))
        if key not in self._by_path:
            raise KeyError(f"{path!r} is not a revision text on any ladder")
        return self._by_path[key]


@lru_cache(maxsize=4)
def load_index(manifest: str, external: str | None = None) -> RevisionIndex:
    """Process-wide cached `RevisionIndex.from_manifest`, keyed by the resolved paths."""
    return RevisionIndex.from_manifest(Path(manifest), Path(external) if external else None)


def as_index(manifest) -> RevisionIndex:
    """`build_memory`'s manifest argument: a RevisionIndex, or the path of a manifest file (loaded without external
    dates; build a RevisionIndex with `external` to use them)."""
    if isinstance(manifest, RevisionIndex):
        return manifest
    if isinstance(manifest, (str, os.PathLike)):
        return load_index(str(Path(manifest).resolve()))
    raise TypeError(f"manifest must be a RevisionIndex or a manifest path, got {type(manifest).__name__}")


@dataclass(frozen=True)
class AsOf:
    day: date
    source: str                         # as_of_date | effective_date
    revisions: tuple[RevisionText, ...]


def asof_revisions(item: dict, index: RevisionIndex) -> AsOf:
    """Revisions of the item's document in effect on its as-of date: the question's `as_of_date`, else (a question
    that names its revision in its header, "Under {document} (Revision {rev}, effective {date})") the named
    revision's `effective_date`."""
    for name in ("as_of_date", "effective_date"):
        if item.get(name):
            day = date.fromisoformat(item[name])
            return AsOf(day, name, tuple(index.in_effect(item["doc"], day)))
    raise ValueError(f"item {item.get('item_id')!r} has neither as_of_date nor effective_date")


def in_ladder_order(revisions) -> list[RevisionText]:
    """Distinct revisions in (document, revision label) order."""
    return sorted(set(revisions), key=lambda r: (r.doc, rev_key(r.revision), r.revision))

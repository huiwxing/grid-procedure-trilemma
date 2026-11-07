"""procedures.corpus.revisions.RevisionIndex: a manifest row marked automatic_frame_exclusion (the IESO pre-renewal
generation) or excluded never enters a memory ladder (otherwise the generation guard raises on the manifest), the
corpus paths the items record resolve in this checkout, never in a neighbouring one, and a text other than the one the
manifest records is warned about and read as it is."""
import csv
import hashlib
from pathlib import Path

import pytest

from procedures.corpus.revisions import RevisionIndex

FIELDS = ["family", "doc_id", "revision", "doc_kind", "cover_effective_date", "text_path", "status", "inactive_date",
          "provenance_status", "fetched_at_utc", "gold_eligible", "automatic_frame_exclusion", "excluded",
          "exclude_reason"]
ROWS = [
    ("IESO-MM-7.3", "3.0", "2025-12-03", "in_effect", "", "false"),
    ("IESO-MM-7.3", "49.0", "2023-11-29", "superseded", "revision_number_reset", "false"),
    ("IESO-MM-7.3", "50.0", "2024-12-04", "superseded", "revision_number_reset", "false"),
    ("IESO-MM-7.10", "2.0", "2025-12-03", "in_effect", "", "false"),
    ("IESO-MM-7.10", "1.0", "2024-11-11", "superseded", "", "true"),
]


def _manifest(tmp_path: Path, marker: bool) -> Path:
    path = tmp_path / "manifest.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        for doc, rev, eff, status, afe, excluded in ROWS:
            name = f"{doc}_{rev}.txt"
            (tmp_path / name).write_text(f"{doc} version {rev}\nEffective {eff}\nbody\n", encoding="utf-8")
            w.writerow(dict(family="IESO", doc_id=doc, revision=rev, doc_kind="clean", cover_effective_date=eff,
                            text_path=name, status=status, provenance_status="owner_site",
                            fetched_at_utc="2026-09-25T10:00:00Z", gold_eligible="true",
                            automatic_frame_exclusion=afe if marker else "", excluded=excluded,
                            exclude_reason="manual" if excluded == "true" else ""))
    return path


def test_withheld_rows_never_enter_a_memory_ladder(tmp_path: Path):
    index = RevisionIndex.from_manifest(_manifest(tmp_path, marker=True))
    assert [t.revision for t in index.ladder("IESO-MM-7.3")] == ["3.0"]
    assert [t.revision for t in index.ladder("IESO-MM-7.10")] == ["2.0"]   # the excluded 1.0 is out


def test_raw_generation_reset_without_marker_is_an_error(tmp_path: Path):
    with pytest.raises(ValueError):
        RevisionIndex.from_manifest(_manifest(tmp_path, marker=False))


def _checkout(root: Path, text: str) -> Path:
    """A synthetic checkout: one revision text and a manifest naming it relative to the checkout."""
    data = text.encode("utf-8")
    (root / "data" / "corpus" / "test").mkdir(parents=True)
    (root / "data" / "corpus" / "test" / "r1.txt").write_bytes(data)
    manifest = root / "data" / "corpus" / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["sha256_lf"], lineterminator="\n")
        w.writeheader()
        w.writerow(dict(family="TEST", doc_id="TEST-OP-1", revision="1", doc_kind="clean",
                        cover_effective_date="2024-01-01", text_path="data/corpus/test/r1.txt", status="in_effect",
                        provenance_status="owner_site", fetched_at_utc="2026-09-25T10:00:00Z", gold_eligible="true",
                        excluded="false", sha256_lf=hashlib.sha256(data).hexdigest()))
    return manifest


def test_corpus_paths_resolve_up_to_the_repository_root_and_never_above_it(tmp_path, monkeypatch):
    """A corpus path is relative to the repository root ("data/corpus/..."): it is looked up from the manifest's
    directory up to the repository root, and a file above the repository root is never picked up."""
    from procedures.corpus import revisions
    checkout = tmp_path / "grid-procedure-trilemma"
    index_manifest = _checkout(checkout, "Rule text of this checkout.\n")
    monkeypatch.setattr(revisions, "ROOT", checkout)
    index = RevisionIndex.from_manifest(index_manifest)
    got = index.by_path("data/corpus/test/r1.txt")
    assert got.path == (checkout / "data" / "corpus" / "test" / "r1.txt").resolve()
    assert got.text() == "Rule text of this checkout.\n" and got.source_path == "data/corpus/test/r1.txt"
    assert index.by_path("data\\corpus\\test\\r1.txt") is got       # recorded with backslashes
    (tmp_path / "data" / "corpus" / "test").mkdir(parents=True)
    (tmp_path / "data" / "corpus" / "test" / "outside.txt").write_bytes(b"above the repository root\n")
    with pytest.raises(FileNotFoundError, match="outside.txt"):
        index.by_path("data/corpus/test/outside.txt")


def test_a_text_other_than_the_recorded_one_is_warned_about_and_read_as_it_is(tmp_path, monkeypatch):
    from procedures.corpus import revisions
    checkout = tmp_path / "grid-procedure-trilemma"
    manifest = _checkout(checkout, "Rule text as the runs read it.\n")
    rebuilt = "Rule text as rebuilt today.\n"
    (checkout / "data" / "corpus" / "test" / "r1.txt").write_bytes(rebuilt.encode("utf-8"))
    monkeypatch.setattr(revisions, "ROOT", checkout)
    with pytest.warns(UserWarning, match="differs from the one the manifest records"):
        index = RevisionIndex.from_manifest(manifest)
    (rev,) = index.ladder("TEST-OP-1")
    assert rev.text() == rebuilt and rev.sha256_lf == hashlib.sha256(rebuilt.encode("utf-8")).hexdigest()

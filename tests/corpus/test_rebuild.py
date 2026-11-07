"""scripts/corpus/rebuild.py offline: a kept download under data/raw is converted with the row's recipe without any
request, the text is written to its text_path whether or not it is the text the paper's runs used, and the row says
which ("same text" / "differs from the text the runs used"). Synthetic PDFs come from tests/corpus/synthetic_pdf.py."""
from __future__ import annotations

import hashlib
import importlib.util

import pytest
from synthetic_pdf import make_pdf, text_op

from procedures import ROOT
from procedures.corpus.pdf_text import RECIPE_VERSION, pdf_to_text, sha256_text_lf

_spec = importlib.util.spec_from_file_location("rebuild_script", ROOT / "scripts" / "corpus" / "rebuild.py")
rebuild = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rebuild)


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A repository layout in tmp_path; any request fails the test."""
    def refuse(*args, **kwargs):
        raise AssertionError("the rebuild of a kept download sent a request")
    monkeypatch.setattr(rebuild, "ROOT", tmp_path)
    monkeypatch.setattr(rebuild, "RAW_DIR", tmp_path / "data" / "raw")
    monkeypatch.setattr(rebuild, "get", refuse)
    return tmp_path


def manifest_row(root, recorded_text_sha: str | None = None) -> tuple[dict, str]:
    raw = root / "data" / "raw" / "test" / "doc.pdf"
    raw.parent.mkdir(parents=True)
    make_pdf(raw, [text_op(72, 700, "The operator shall act within 5 minutes.")])
    text = pdf_to_text(raw)
    return {"family": "TEST", "doc_id": "TEST-1", "revision": "1", "doc_kind": "clean",
            "provenance_status": "owner_site", "text_path": "data/corpus/test/doc.txt",
            "raw_path": "data/raw/test/doc.pdf", "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
            "zip_sha256": "", "zip_member": "", "pdf_container_repair": "", "conversion_input_sha256": "",
            "source_url": "https://example.invalid/doc.pdf", "fetched_at_utc": "2026-09-24T10:00:00Z",
            "text_recipe": RECIPE_VERSION, "sha256_lf": recorded_text_sha or sha256_text_lf(text)}, text


def test_a_kept_download_rebuilds_offline_and_the_text_is_the_runs(root):
    row, text = manifest_row(root)
    got = rebuild.rebuild_row(row, force=False)
    assert (got["status"], got["text"]) == ("rebuilt", rebuild.SAME) and got["source"].startswith("kept download")
    assert (root / row["text_path"]).read_text(encoding="utf-8") == text
    assert rebuild.rebuild_row(row, force=False) == {**{k: got[k] for k in ("family", "doc_id", "revision",
                                                                             "text_path")},
                                                      "status": "present", "text": rebuild.SAME}
    assert rebuild.rebuild_row(row, force=True)["status"] == "rebuilt"


def test_a_text_that_differs_from_the_runs_is_written_and_reported(root):
    row, text = manifest_row(root, recorded_text_sha="0" * 64)
    got = rebuild.rebuild_row(row, force=False)
    assert (got["status"], got["text"]) == ("rebuilt", rebuild.DIFFERS)
    assert (root / row["text_path"]).read_text(encoding="utf-8") == text
    assert not list(root.rglob("*.unverified"))
    assert rebuild.rebuild_row(row, force=False)["text"] == rebuild.DIFFERS          # rebuilt again, not kept


def test_redline_and_foreign_text_copies_are_not_rebuilt(root):
    row, _ = manifest_row(root)
    assert rebuild.rebuild_row({**row, "doc_kind": "redline"}, force=False)["status"] == "skipped"
    assert rebuild.rebuild_row({**row, "provenance_status": "archive_copy_unverified"},
                               force=False)["status"] == "not_rebuildable"

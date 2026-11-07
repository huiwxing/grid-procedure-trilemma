"""procedures.corpus.manifest: the manifest's rows with the pipeline's field names, and the name a question gives a
document (from the manifest fields, never the free-text title)."""
import csv
import json

from procedures.corpus.manifest import canonical_title, is_true, read_manifest

FIELDS = ["family", "doc_id", "doc", "revision", "doc_kind", "cover_effective_date", "text_path", "status", "title",
          "excluded"]
ROWS = [
    dict(family="MISO", doc_id="MISO-BPM-015", revision="33", doc_kind="clean", cover_effective_date="2026-06-01",
         text_path="BPM-015_r33.txt", title="BPM-015 Transmission Operations (free text)", excluded="false"),
    dict(family="NERC", doc_id="NERC-FAC-011", revision="4", doc_kind="implementation_plan", status="effective",
         cover_effective_date="2024-04-01", text_path="fac-011-ip.txt", excluded="true"),
    dict(family="PJM", doc_id="PJM-M13", doc="M-13", revision="98", doc_kind="clean", status="superseded",
         text_path="m13_rev98.txt", excluded="false"),
]
WANT = [
    {"family": "MISO", "doc": "MISO-BPM-015", "revision": "33", "kind": "clean", "effective_date": "2026-06-01",
     "path": "BPM-015_r33.txt", "excluded": "false"},
    {"family": "NERC", "doc": "NERC-FAC-011", "revision": "4", "kind": "impl_plan", "status": "in_effect",
     "effective_date": "2024-04-01", "path": "fac-011-ip.txt", "excluded": "true"},
    {"family": "PJM", "doc": "M-13", "revision": "98", "kind": "clean", "status": "superseded", "path": "m13_rev98.txt",
     "excluded": "false"},
]


def test_canonical_titles_name_the_document_not_the_free_text_title():
    assert canonical_title("PJM", "PJM-M03", "71", "clean") == "PJM Manual 03: Transmission Operations"
    assert canonical_title("MISO", "MISO-BPM-015", "33", "clean") == "MISO BPM-015"
    assert canonical_title("CAISO", "CAISO-OP-3100", "8.2", "clean") == "CAISO Operating Procedure 3100"
    assert canonical_title("ISO-NE", "ISONE-OP-19", "14", "clean") == "ISO-NE Operating Procedure OP-19"
    assert canonical_title("NERC", "NERC-TOP-003", "6.1", "clean") == "NERC Reliability Standard TOP-003-6.1"
    assert canonical_title("NERC", "NERC-FAC-011", "4", "impl_plan") == "NERC Implementation Plan for FAC-011-4"


def test_read_manifest_maps_the_columns_of_a_csv_or_jsonl_manifest(tmp_path):
    """text_path -> path, cover_effective_date -> effective_date, doc_kind -> kind (implementation_plan ->
    impl_plan), status effective -> in_effect, doc before doc_id; empty cells omitted, the free-text title ignored."""
    csv_path = tmp_path / "manifest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows({k: r.get(k, "") for k in FIELDS} for r in ROWS)
    jsonl_path = tmp_path / "manifest.jsonl"
    jsonl_path.write_bytes("".join(json.dumps(r) + "\n" for r in ROWS).encode("utf-8"))
    assert read_manifest(csv_path) == WANT == read_manifest(jsonl_path)
    assert [is_true(v) for v in ("true", " TRUE ", "false", "", None)] == [True, True, False, False, False]

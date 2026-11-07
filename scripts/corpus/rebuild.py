"""Rebuild the procedure texts from the corpus manifest's pointers.

    py -3.12 scripts/corpus/rebuild.py --items data/items/questions.jsonl   # the paper's documents
    py -3.12 scripts/corpus/rebuild.py --all                                               # every manifest row
    py -3.12 scripts/corpus/rebuild.py --family PJM --doc PJM-M-13                            # a subset

The procedure texts are not redistributed (the owners allow pointers only; the manifest's `terms_flag` column records
each owner's terms). The manifest records, per revision, the owner URL it was fetched from (or the Internet Archive
capture), the sha256 of the bytes fetched (`raw_sha256`; for a ZIP member also the ZIP's `zip_sha256` and the member
name), any container repair applied before conversion (`pdf_container_repair`), the conversion recipe (`text_recipe`)
and the sha256 of the LF text the paper's runs read (`sha256_lf`). This script replays that chain row by row and
writes every converted text to its `text_path` (data/corpus/<family>/<file>.txt):

* the bytes: the copy already downloaded under data/raw/ (the row's `raw_path`) when it holds the recorded bytes, else
  the owner's URL; when the owner's copy has changed since the manifest was written, the captures of that URL in the
  Internet Archive are tried, nearest to the recorded fetch date first, for the recorded bytes, and failing that the
  owner's current copy is converted (the row says so). Downloads are kept under data/raw/, so a second rebuild runs
  offline;
* the text: converted with the row's recipe (procedures.corpus.pdf_text, or procedures.corpus.caiso for the CAISO
  change notices) and written to `text_path`; the summary says per revision whether it is the text the paper's runs
  used ("same text") or not ("differs from the text the runs used"). A text already at `text_path` that is the runs'
  is kept (--force rebuilds it); any other is rebuilt.

Redline rows (change-span files, not texts) are skipped. Two manifest rows (`provenance_status`
archive_copy_unverified) are text copies made by another extraction tool (pdftotext) that this repository does not
include; they are reported as not rebuildable. No item of the paper reads them.

The summary, <out-dir>/rebuild_<sha12>.json (default out/corpus/), lists every row with its outcome. Requests are
sequential with a pause between them and the project User-Agent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from procedures import ROOT, stored  # noqa: E402
from procedures.corpus import caiso  # noqa: E402
from procedures.corpus.fetch import http_get  # noqa: E402
from procedures.corpus.pdf_text import RECIPE_VERSION, WalkStats, pdf_to_text, sha256_text_lf  # noqa: E402
from procedures.corpus.wayback import WAYBACK, cdx_query, decoded_payload  # noqa: E402
from procedures.corpus.frozen_manifest import DEFAULT_MANIFEST  # noqa: E402
from procedures.items.publish import display  # noqa: E402

PDF_RECIPE = "procedures.corpus.pdf_text"
NOTICE_RECIPE = "procedures.corpus.caiso.html_to_text+parse_notice"
NERC_STUB = b"<html><head><title>NERC</title></head></html>"
RAW_DIR = ROOT / "data" / "raw"
SAME, DIFFERS = "same text", "differs from the text the runs used"
PAUSE_SECONDS = 1.0
RETRIES = 4
BACKOFF_SECONDS = (5, 20, 60, 0)     # the Internet Archive answers 503 under load; wait before retrying
CAPTURES_TRIED = 6


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def get(url: str) -> tuple[str, bytes]:
    """(final URL, body) with retries and a pause; raises after RETRIES failures."""
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            status, final, _, body = http_get(url, timeout=180.0)
            time.sleep(PAUSE_SECONDS)
            if status == 200:
                return final, body
            last = RuntimeError(f"HTTP {status}")
        except Exception as exc:    # noqa: BLE001 - every transport error is retried the same way
            last = exc
        time.sleep(BACKOFF_SECONDS[attempt - 1])
    raise RuntimeError(f"{url}: {last}")


def archive_candidates(url: str, fetched_at: str) -> list[tuple[str, str]]:
    """(timestamp, original) of the Internet Archive captures of `url`, nearest to `fetched_at` first. When the CDX
    index does not answer (it returns 503 under load), the fallback is the archive's own nearest-capture redirect:
    a request for the fetch date's timestamp, then for the same day of each of the three preceding years."""
    anchor = fetched_at.replace("-", "").replace(":", "").replace("T", "").rstrip("Z")[:14].ljust(14, "0")
    try:
        _, answer = get(cdx_query(url))
        rows = json.loads(answer.decode("utf-8"))
        captures = [(r[0], r[1]) for r in rows[1:]]
        return sorted(captures, key=lambda c: abs(int(c[0][:14]) - int(anchor)))
    except RuntimeError as exc:
        print(f"    CDX index unavailable ({str(exc)[-60:]}); asking the archive for the nearest captures", flush=True)
        year = int(anchor[:4])
        return [(f"{year - k}{anchor[4:]}", url) for k in range(4)]


def raw_file(row: dict) -> Path:
    """Where the row's raw bytes are kept: its `raw_path` under the repository, else data/raw/<family>/<sha12>.bin."""
    if row["raw_path"] and not Path(row["raw_path"]).is_absolute():
        return ROOT / row["raw_path"]
    return RAW_DIR / row["family"].lower() / f"{row['raw_sha256'][:12]}.bin"


def cached(path: Path, sha: str) -> bytes | None:
    """The bytes of `path` when it exists and holds the bytes of sha256 `sha` (a download kept from an earlier
    rebuild)."""
    if path.is_file():
        data = path.read_bytes()
        if sha256(data) == sha:
            return data
    return None


def keep(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def fetch_zip_member(row: dict) -> tuple[bytes, str, list[str]]:
    url = row["source_url"]
    archive = RAW_DIR / "zips" / f"{row['zip_sha256'][:12]}.zip"
    notes = []
    data = cached(archive, row["zip_sha256"])
    if data is None:
        _, data = get(url)
        if sha256(data) == row["zip_sha256"]:
            keep(archive, data)
        else:
            notes.append(f"the ZIP at {url} now hashes to {sha256(data)[:12]}, recorded {row['zip_sha256'][:12]}")
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        member = zf.read(row["zip_member"])
    return member, f"ZIP {url} member {row['zip_member']}", notes


def fetch_bytes(row: dict) -> tuple[bytes, str, list[str]]:
    """(the row's raw bytes, where they came from, notes): the recorded bytes when a source still gives them (a kept
    download, the owner's URL, an archive capture), else the owner's current copy (noted)."""
    url = row["source_url"]
    data = cached(raw_file(row), row["raw_sha256"])
    if data is not None:
        return data, f"kept download {display(raw_file(row), ROOT)}", []
    if row["zip_sha256"]:
        data, obtained, notes = fetch_zip_member(row)
    else:
        _, data = get(url)
        if "web.archive.org/web/" in url:
            data, _ = decoded_payload(data)
        obtained, notes = url, []
    if sha256(data) == row["raw_sha256"]:
        keep(raw_file(row), data)
        return data, obtained, notes
    if row["zip_sha256"] or "web.archive.org/web/" in url:
        return data, obtained, notes + [f"the bytes now hash to {sha256(data)[:12]}, recorded {row['raw_sha256'][:12]}"]
    tried = []
    for timestamp, original in archive_candidates(url, row["fetched_at_utc"])[:CAPTURES_TRIED]:
        capture = WAYBACK.format(timestamp=timestamp, original=original)
        try:
            _, payload = get(capture)
        except RuntimeError as exc:
            tried.append(f"{timestamp}: {exc}")
            continue
        payload, _ = decoded_payload(payload)
        if sha256(payload) == row["raw_sha256"]:
            keep(raw_file(row), payload)
            return payload, f"{capture} (the owner's copy at {url} now hashes to {sha256(data)[:12]})", []
        tried.append(f"{timestamp}: {sha256(payload)[:12]}")
    return data, url, [f"the owner's copy now hashes to {sha256(data)[:12]}, recorded {row['raw_sha256'][:12]}, and no "
                       f"archive capture tried has the recorded bytes ({tried or 'none found'}); the current copy is "
                       "converted"]


def conversion_input(row: dict, raw: bytes) -> tuple[bytes, list[str]]:
    """The bytes the recipe converts: the raw bytes, less the HTML stub in front of the PDF header that the row's
    `pdf_container_repair` records."""
    notes = []
    if row["pdf_container_repair"].startswith("removed exact NERC HTML stub"):
        offset = raw.find(b"%PDF")
        if offset > 0:
            if raw[:offset] != NERC_STUB:
                notes.append("the bytes before the PDF header are not the recorded NERC HTML stub; cut there anyway")
            raw = raw[offset:]
    if row["conversion_input_sha256"] and sha256(raw) != row["conversion_input_sha256"]:
        notes.append(f"conversion input hashes to {sha256(raw)[:12]}, recorded {row['conversion_input_sha256'][:12]}")
    return raw, notes


def convert(row: dict, raw: bytes, data: bytes) -> str:
    """The text of the row's recipe over `data` (the conversion input of the raw bytes `raw`); the PDF recipe reads a
    file: the kept download, or for a repaired container its conversion input kept beside it."""
    recipe = row["text_recipe"]
    if recipe.startswith(PDF_RECIPE):
        path = raw_file(row)
        if data != raw or cached(path, sha256(data)) is None:
            path = RAW_DIR / row["family"].lower() / f"{sha256(data)[:12]}.pdf"
            if cached(path, sha256(data)) is None:
                keep(path, data)
        return pdf_to_text(path, WalkStats())
    if recipe.startswith(NOTICE_RECIPE):
        page = caiso.html_to_text(data)
        notice_date, entries = caiso.parse_notice(page)
        body = "\n".join(f"{e['procedure']} {e['title']}, version {e['version']}, effective {e['effective']}. "
                         f"{e['summary']}" for e in entries)
        return f"Notice {notice_date}\n{body}\n"
    raise RuntimeError(f"recipe not reproducible here: {recipe[:60]}")


def compared(text: str, row: dict) -> str:
    """SAME when `text` is the one the paper's runs read (the row's sha256_lf), else DIFFERS."""
    return SAME if sha256_text_lf(text) == row["sha256_lf"] else DIFFERS


def rebuild_row(row: dict, force: bool) -> dict:
    out = {"family": row["family"], "doc_id": row["doc_id"], "revision": row["revision"], "text_path": row["text_path"]}
    if row["doc_kind"] == "redline":
        return {**out, "status": "skipped", "note": "a redline row names its change-span file, not a text; the "
                "revision ladders read only the clean text of a revision"}
    if row["provenance_status"] == "archive_copy_unverified":
        return {**out, "status": "not_rebuildable", "note": "a pdftotext layout text, not made by a recipe of this "
                "repository; no released item reads it"}
    target = ROOT / row["text_path"]
    if target.exists() and not force and compared(target.read_text(encoding="utf-8"), row) == SAME:
        return {**out, "status": "present", "text": SAME}
    try:
        raw, obtained, notes = fetch_bytes(row)
        data, more = conversion_input(row, raw)
        text = convert(row, raw, data)
    except Exception as exc:    # noqa: BLE001 - every failure is a row outcome
        return {**out, "status": "failed", "note": str(exc)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
    notes = notes + more
    return {**out, "status": "rebuilt", "text": compared(text, row), "source": obtained,
            **({"note": "; ".join(notes)} if notes else {})}


def wanted_rows(rows: list[dict], a) -> list[dict]:
    keep_rows = rows
    if a.items:
        # every revision of every document the items read: the pipeline builds each document's whole revision ladder
        # (effective windows need the neighbouring revisions), so the ladder's other texts are rebuilt too
        texts = set()
        for it in stored.read_jsonl(a.items):
            for p in it.get("m0_paths", []) + [it.get("evidence_path", ""), it.get("base_doc_path", "")]:
                if p:
                    texts.add(p.replace("\\", "/"))
        documents = {r["doc_id"] for r in keep_rows if r["text_path"] in texts}
        keep_rows = [r for r in keep_rows if r["doc_id"] in documents]
    if a.family:
        keep_rows = [r for r in keep_rows if r["family"] in a.family]
    if a.doc:
        keep_rows = [r for r in keep_rows if r["doc_id"] in a.doc]
    return keep_rows[: a.limit] if a.limit else keep_rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--items", type=Path, help="rebuild the revision ladders of the documents this item file reads")
    ap.add_argument("--all", action="store_true", help="every manifest row")
    ap.add_argument("--family", action="append", help="restrict to an owner family (repeatable)")
    ap.add_argument("--doc", action="append", help="restrict to a document id (repeatable)")
    ap.add_argument("--limit", type=int, help="stop after this many rows")
    ap.add_argument("--force", action="store_true", help="rebuild texts that are already present")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "corpus",
                    help="directory of the rebuild summary (default out/corpus)")
    a = ap.parse_args(argv)
    if not (a.items or a.all or a.family or a.doc):
        ap.error("give --items, --all, --family or --doc")
    with a.manifest.open(encoding="utf-8", newline="") as fh:
        rows = wanted_rows(list(csv.DictReader(fh)), a)
    results = []
    for i, row in enumerate(rows, 1):
        r = rebuild_row(row, a.force)
        results.append(r)
        print(f"[{i}/{len(rows)}] {r['status']:15s} {r.get('text', ''):35s} {row['family']:6s} {row['doc_id']} "
              f"{row['revision']}" + (f"  {r['note'][:110]}" if r.get("note") else ""), flush=True)
    counts, texts = {}, {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if "text" in r:
            texts[r["text"]] = texts.get(r["text"], 0) + 1
    summary = {"manifest": display(a.manifest, ROOT), "recipe_here": RECIPE_VERSION,
               "finished_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "counts": counts,
               "texts": texts, "rows": results}
    data = (json.dumps(summary, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    path = a.out_dir / f"rebuild_{sha256(data)[:12]}.json"
    path.write_bytes(data)
    print(json.dumps({"counts": counts, "texts": texts}), "->", display(path, ROOT))
    return 1 if counts.get("failed") else 0


if __name__ == "__main__":
    raise SystemExit(main())

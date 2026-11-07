"""The corpus manifest's rows as the pipeline reads them.

The manifest (data/corpus/manifest.csv for the paper's runs) has one row per document file: the owner family, the
document id and revision, the kind of file, the owner URL or archive capture it was fetched from and when, the sha256
of the LF text (`sha256_lf`) and its path (`text_path`, relative to the repository root), the conversion recipe
(`text_recipe`), the owner's terms (`terms_flag`), the provenance (`provenance_status`: owner_site = fetched from the
owner, verified_wayback = an Internet Archive capture whose bytes match the archive's digest, archive_copy_unverified
= neither; only the first two may feed a gold) and exclusion flags (`excluded`, `exclude_reason`, `gold_eligible`,
`automatic_frame_exclusion`).

`read_manifest(path)` reads a manifest (CSV or JSONL) and is the one place where manifest column names are mapped:
`text_path` -> path, `doc` (PJM) else `doc_id` -> doc, `revision`, `cover_effective_date` -> effective_date,
`doc_kind` -> kind (implementation_plan -> impl_plan), `status` (effective -> in_effect; a row without status is in
effect: the MISO and PJM rows carry none and the revision ladder decides current vs older); every other column is
kept as it is. Empty cells are omitted, and the free-text `title` column is ignored: a question names a document by
`canonical_title` (from family / doc / revision / kind), never by the free text ("3100 Establishing System
Operating Limits ..." is not how a question names CAISO 3100). `USABLE_KINDS` are the kinds of rows that hold rule
text: clean revisions and implementation plans (an implementation plan is its own document, doc + "-IP", so it never
competes with the standard of the same revision for "current").
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

MANIFEST_ALIASES = {"text_path": "path", "cover_effective_date": "effective_date", "doc_kind": "kind"}
MANIFEST_IGNORED = ("title",)
USABLE_KINDS = ("clean", "impl_plan")
PJM_TITLES = {"03": "PJM Manual 03: Transmission Operations", "13": "PJM Manual 13: Emergency Operations"}
KIND_MAP = {"implementation_plan": "impl_plan"}
STATUS_MAP = {"effective": "in_effect"}


def read_manifest(path: Path) -> list[dict]:
    """Manifest rows with the pipeline's field names (module docstring); empty cells are omitted."""
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        with path.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    out = []
    for raw in rows:
        r = {MANIFEST_ALIASES.get(k, k): v for k, v in raw.items()
             if v not in (None, "") and k not in MANIFEST_IGNORED and k not in ("doc", "doc_id")}
        doc = raw.get("doc") or raw.get("doc_id")
        if doc:
            r["doc"] = doc
        if "kind" in r:
            r["kind"] = KIND_MAP.get(r["kind"], r["kind"])
        if "status" in r:
            r["status"] = STATUS_MAP.get(r["status"], r["status"])
        out.append(r)
    return out


def is_true(value) -> bool:
    return str(value).strip().lower() == "true"


def canonical_title(family: str, doc: str, revision: str, kind: str) -> str:
    """Question {document} name from a manifest row's fields: "PJM Manual 03: Transmission Operations",
    "MISO BPM-015", "CAISO Operating Procedure 3100", "ISO-NE Operating Procedure OP-19", "NERC Reliability Standard
    FAC-011-4", "NERC Implementation Plan for FAC-011-4"."""
    f = family.upper().replace("-", "")
    d = re.sub(r"^(?:MISO|NERC|CAISO|ISONE|ISO-NE|PJM)-", "", doc.upper())
    if f.startswith("PJM"):
        m = re.search(r"M-?(\d+)([A-Z]?)\b", d)
        num = (m.group(1).zfill(2) + m.group(2)) if m else d
        return PJM_TITLES.get(num, f"PJM Manual {num}")
    if f.startswith("MISO"):
        return f"MISO {d}"
    if f.startswith("CAISO"):
        return f"CAISO Operating Procedure {re.sub(r'^OP-', '', d)}"
    if f.startswith("ISONE"):
        return f"ISO-NE Operating Procedure {d}"
    if f.startswith("NERC"):
        std = f"{d}-{revision}" if revision else d
        return f"NERC Implementation Plan for {std}" if kind == "impl_plan" else f"NERC Reliability Standard {std}"
    if f == "AESO":
        return f"AESO ISO Rules Section {re.sub(r'^AESO-ISO-', '', doc, flags=re.I)}"
    return doc

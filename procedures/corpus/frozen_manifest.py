"""The corpus manifest a reader run reads.

The paper's runs read data/corpus/manifest.csv (DEFAULT_MANIFEST). A run manifest records the corpus manifest it was
planned with (`corpus_manifest`: its path and sha256), and the run identity binds it: a run planned on an edited
manifest is a new run. Whether a revision text is the one the runs read is checked per text
(procedures.corpus.revisions: the manifest's `sha256_lf`).

* `manifest_record(path)`: {path, sha256} as a run manifest records it (path relative to the repository root).
* `run_corpus_manifest(run_dir)`: the corpus manifest a finished run recorded, else the default: what a zero-call pass
  over the run (the sanity check, a rescore, its memory request map) reads by default.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from procedures import ROOT
from procedures.items.publish import display

CORPUS_DIR = ROOT / "data" / "corpus"
DEFAULT_MANIFEST = CORPUS_DIR / "manifest.csv"


def manifest_record(path: Path) -> dict:
    return {"path": display(path, ROOT), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def run_corpus_manifest(run_dir: Path) -> Path:
    recorded = json.loads((Path(run_dir) / "manifest.json").read_text(encoding="utf-8")).get("corpus_manifest")
    return DEFAULT_MANIFEST if recorded is None else ROOT / recorded["path"]

"""Re-score an existing run from its own call ledgers, without any call.

`rescore_run` reads the run directory as it is, so a run can be re-scored after the panel file or the prompt
templates have changed (both are part of a run's identity, so re-planning would give a different run):

* the cells: the rows file the run's last live invocation wrote (one row per reader cell, with the cell's plan:
  stage, budget, feasibility, planned tokens, memory key), in its order;
* the answers: each row's `provenance.request_hash` joined to the latest record of that reader's call ledger
  (read__<reader>.jsonl). A row whose hash has no record, a record of another item, of another stage (unless the
  row shares an earlier cell's request) or with another outcome than the row recorded raises; answered ledger records
  no row names are counted;
* the items: the run's item file: every item a row names must be in it, and items that do not hash to the run
  manifest's `items_sha256` (the items the run was planned with) are warned about;
* the readers: the current panel profiles; a fingerprint or user_message_suffix other than the run's is warned
  about.

It scores them with the current procedures.scoring.score_path (procedures.matrix.run_matrix.score_cells), which applies

* the answer-field mapping (MAPPING_RULE: an answer left blank in the content but given as exactly one JSON object in
  the reasoning field is scored on that object) for the readers whose panel entry sets
  `answer_from_reasoning_field` (procedures.readers.parse.answer_from_reasoning), and
* the blind arbitration decisions of a decisions file (arb_id -> bool, the frozen scorer's `decisions` argument), which
  must name this run and an arbitration queue file of this run directory (looked up by its file name; a queue that
  does not have the sha256 the decisions file records is warned about); every decision must resolve a row, or the
  file does not belong to this scoring.

The rows are attributed to the run's memory request map as run_matrix's scoring does, so that the analysis can charge
each LLM-built memory its build and compression cost; a run has one map (memory_requests_<sha12>.json as a run
writes it, memory_requests.json in the released runs), and every row names it (run_matrix.name_memory_requests). A
run with an LLM-built condition uses the map its directory already holds (the one its live scoring wrote), whose
memory keys must be exactly the LLM-built cache keys the run's manifest bound; a run without one has its memories
re-planned from its manifest (run_matrix.replan_memories over `corpus_index`, each checked against the bound cache
key; a run whose memories no longer re-plan to those keys is refused) and the map written. The rescoring record says
which.

Outputs, named after their content like run_matrix's, in `out_dir` (default: the run directory): new rows_ /
summary_ files and, when some answer still needs arbitration, an arbitration_ queue (the old files stay), the memory
request map when it is written by a re-plan, and rescore_<sha12>.json, the rescoring manifest: every input with its
sha256 (source rows, call ledgers, items, panel, decisions), the frozen scorer hashes of the run and now, the rules
applied, the output names, every cell whose verdict changed with the reason (answer_from_reasoning_field,
arbitration, or other), and accuracy per reader x condition before and after (repeat 0, answered cells). One line
(mode "rescore") is appended to the invocations.jsonl of `out_dir`. Nothing is sent, no key is read.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import warnings

from procedures.items.publish import display
from procedures.items.frozen import jsonl_bytes
from procedures import stored
from procedures.readers.ledger import read_ledger, utc_now
from procedures import ROOT
from procedures.memory import build_memory as default_build_memory
from procedures.memory.requests import FILE_STEM as REQUESTS_STEM
from procedures.memory.requests import load_memory_requests, memory_requests, write_memory_requests

from procedures.readers.cells import LLM_BUILT
from procedures.matrix.run_matrix import SCORER_FILES, _json_bytes, _write_addressed, corpus_text, lf_sha256, name_memory_requests, reader_digest, replan_memories, run_manifest, score_cells, write_scored
from procedures.scoring.score_path import row_memory_empty

SCHEMA = "procedures.scoring.rescore.v1"
MAPPING_RULE = ("answer_from_reasoning_field: for a reader whose panel entry sets it, an answered record whose content "
                "is blank and whose reasoning_content, stripped, parses as exactly one JSON object is scored on that "
                "object and flagged; every other record is scored on its content")
DECISIONS_RULE = "blind arbitration decisions (arb_id -> bool) passed to the frozen scorer's score_rows (N11)"
VERDICT_KEYS = ("correct", "status", "error_class", "pred", "pred_unit", "arb_id")


def file_sha256(path: Path) -> str:
    return stored.sha256(path)


def read_jsonl(path: Path) -> list[dict]:
    return stored.read_jsonl(path)


def scoring_source(run_dir: Path) -> dict:
    """The last live invocation of the run that wrote scored outputs (its rows are the run's cells)."""
    lines = read_jsonl(Path(run_dir) / "invocations.jsonl")
    live = [x for x in lines if x.get("outcome") == "live" and x.get("outputs")]
    if not live:
        raise ValueError(f"{run_dir}: no live invocation with scored outputs to re-score")
    return live[-1]


def load_decisions(path: Path, run_id: str, run_dir: Path) -> tuple[dict, dict]:
    """(arb_id -> bool, record) of a decisions file for `run_id`, whose queue file is the file of that name in
    `run_dir` (a warning when it does not have the sha256 the decisions file records)."""
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if doc.get("run_id") != run_id:
        raise ValueError(f"{path}: decisions for run {doc.get('run_id')!r}, not {run_id}")
    named = Path(doc["queue_file"]).name
    queue = Path(run_dir) / named
    if not queue.is_file():
        raise FileNotFoundError(f"{path}: queue file {named} is not in {run_dir}")
    if file_sha256(queue) != doc["queue_sha256"]:
        warnings.warn(f"{path}: queue file {queue.name} does not have the recorded sha256 "
                      f"{doc['queue_sha256'][:12]}; the decisions are applied by arb_id", stacklevel=2)
    decisions = doc["decisions"]
    if not decisions or any(not isinstance(v, bool) for v in decisions.values()):
        raise ValueError(f"{path}: decisions must be a non-empty arb_id -> bool map")
    queued = {r["arb_id"] for r in read_jsonl(queue)}
    if set(decisions) - queued:
        raise ValueError(f"{path}: decisions for arb_ids not in the queue: {sorted(set(decisions) - queued)}")
    record = {"path": display(path, ROOT), "sha256": file_sha256(path), "queue_file": doc["queue_file"],
              "queue_sha256": doc["queue_sha256"], "n_arbiters": doc.get("n_arbiters"),
              "decisions": dict(sorted(decisions.items()))}
    return decisions, record


def cell_meta(row: dict) -> dict:
    """The cell plan a ledger row carries, in the form procedures.readers.cells.ReaderCell.meta gives it (rows
    written before the compression memories (M6) existed have no query cost: 0 tokens, no usd; rows written before
    the empty-memory flag: score_path.row_memory_empty; rows written before the route-context feasibility check: no
    route context or infeasible reason, None)."""
    access, sus, prov = row["access"], row["sustainability"], row["provenance"]
    return {"item_id": row["item_id"], "stage": row["stage"], "repeat_idx": row["repeat_idx"], "form": row["form"],
            "representation": row["representation"], "budget_tokens": row["budget_tokens"],
            "builder_profile_id": row["builder_profile_id"], "context_ceiling": access["local_context_ceiling"],
            "planned_input_tokens": access["planned_input_tokens"], "feasible": access["feasible_at_B"],
            "route_context": access.get("route_context"), "infeasible_reason": access.get("infeasible_reason"),
            "memory_tokens": sus["memory_tokens"], "build_tokens": sus["build_tokens"],
            "query_tokens": sus.get("query_memory_tokens", 0), "query_usd": access.get("query_memory_usd"),
            "memory_cache_key": prov["memory_cache_key"], "revisions_used": prov["revisions_used"],
            "memory_empty": row_memory_empty(row)}


def _key(row: dict) -> tuple:
    return row["profile_id"], row["stage"], row["item_id"], row["repeat_idx"]


def _verdict(row: dict) -> dict | None:
    cap = row["capability"]
    return None if cap is None else {k: cap[k] for k in VERDICT_KEYS}


def accuracy_table(rows: list[dict]) -> dict:
    """reader|stage -> answered cells, correct, acc over repeat 0 (the summaries' capability population)."""
    out: dict[str, dict] = {}
    for r in rows:
        if r["repeat_idx"] == 0 and r["outcome"] == "observed":
            t = out.setdefault(f"{r['profile_id']}|{r['stage']}", {"n": 0, "correct": 0})
            t["n"] += 1
            t["correct"] += bool(r["capability"]["correct"])
    return {k: {**v, "acc": v["correct"] / v["n"]} for k, v in sorted(out.items())}


def changes(before: list[dict], after: list[dict], decisions: dict) -> dict:
    """Cells whose verdict changed, each with its reasons: answer_from_reasoning_field (the new row read the answer
    from the reasoning field), arbitration (the row's arb_id has a decision), else other."""
    old = {_key(r): r for r in before}
    listed, reasons = [], Counter()
    for r in after:
        prior = old[_key(r)]
        if _verdict(prior) == _verdict(r):
            continue
        why = [name for name, hit in (
            ("answer_from_reasoning_field", bool(r["capability"] and r["capability"]["answer_from_reasoning_field"])),
            ("arbitration", bool(r["capability"] and r["capability"]["arb_id"] in decisions))) if hit] or ["other"]
        reasons.update(why)
        listed.append({"profile_id": r["profile_id"], "stage": r["stage"], "item_id": r["item_id"],
                       "repeat_idx": r["repeat_idx"], "reasons": why, "before": _verdict(prior),
                       "after": _verdict(r)})
    return {"cells": len(listed), "by_reason": dict(sorted(reasons.items())),
            "by_condition": dict(sorted(Counter(f"{c['profile_id']}|{c['stage']}" for c in listed).items())),
            "became_correct": sum(bool(c["after"]["correct"]) and not (c["before"] or {}).get("correct")
                                  for c in listed),
            "list": listed}


def run_memory_requests(run_dir: Path, manifest: dict, by_id: dict, corpus_index, build_memory,
                        out_dir: Path | None = None) -> dict | None:
    """{file, keys, from} of the run's memory request map, None for a run without an LLM-built memory: the map in the
    run directory ("from": "run directory"), its keys checked against the manifest's LLM-built cache keys, else one
    written from a re-plan into `out_dir` (default: the run directory; "from": "re-plan", needs `corpus_index`)."""
    bound = {key for slot, key in manifest["memory_cache_keys"].items()        # "<item id>|<rep>@<budget>[/...]"
             if slot.split("|", 1)[1].split("@", 1)[0] in LLM_BUILT}
    if not bound:
        return None
    found = sorted(Path(run_dir).glob(f"{REQUESTS_STEM}*.json"))
    if len(found) > 1:
        raise ValueError(f"{run_dir}: several memory request maps {[p.name for p in found]}")
    if found:
        keys = set(load_memory_requests(found[0])["memories"])
        if keys != bound:
            raise ValueError(f"{found[0].name}: its memories are not the run's LLM-built memories "
                             f"({len(keys - bound)} extra, {len(bound - keys)} missing)")
        return {"file": found[0].name, "keys": keys, "from": "run directory"}
    if corpus_index is None:
        raise ValueError(f"{run_dir.name} has LLM-built conditions and no memory request map: re-planning it needs "
                         "the corpus index")
    memories = replan_memories(manifest, by_id, corpus_index, build_memory).values()
    return {"file": write_memory_requests(Path(out_dir or run_dir), memories),
            "keys": set(memory_requests(memories)["memories"]), "from": "re-plan"}


def rescore_run(run_dir: Path, items: list[dict], panel, *, decisions_path: Path | None = None,
                text_of=corpus_text, corpus_index=None, build_memory=default_build_memory,
                out_dir: Path | None = None) -> dict:
    """Re-score `run_dir` (module docstring); the outputs go to `out_dir` (default: the run directory)."""
    started = utc_now()
    run_dir = Path(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = run_manifest(run_dir)
    source = scoring_source(run_dir)
    before = read_jsonl(run_dir / source["outputs"]["rows"])
    run_id = stored.recorded_run_id(run_dir, before)
    by_id = {it["item_id"]: it for it in sorted(items, key=lambda it: it["item_id"])}
    missing = sorted({r["item_id"] for r in before} - set(by_id))
    if missing:
        raise ValueError(f"{run_id}: {len(missing)} item ids of its rows are not in the item file, e.g. {missing[:3]}")
    if hashlib.sha256(jsonl_bytes(list(by_id.values()))).hexdigest() != manifest["items_sha256"]:
        warnings.warn(f"the items differ from the ones {run_id} was planned with (its manifest's items_sha256)",
                      stacklevel=2)
    readers = {}
    for pid, want in manifest["readers"].items():
        pp = panel.get(pid)
        if (pp.fingerprint, pp.user_message_suffix) != (want["fingerprint"], want["user_message_suffix"]):
            warnings.warn(f"{pid}: the panel profile differs from the one {run_id} read with (fingerprint or "
                          "user_message_suffix)", stacklevel=2)
        readers[pid] = pp
    decisions, decisions_record = ({}, None) if decisions_path is None else load_decisions(decisions_path, run_id,
                                                                                            run_dir)
    requests = run_memory_requests(run_dir, manifest, by_id, corpus_index, build_memory, out)
    latest, ledgers = {}, {}
    for pid in readers:
        path = run_dir / f"read__{pid}.jsonl"
        header, records = read_ledger(path)
        if header["profile_fingerprint"] != manifest["readers"][pid]["fingerprint"]:
            raise ValueError(f"{path}: ledger of another profile identity")
        latest[pid] = {r["request_hash"]: r for r in records}
        ledgers[path.name] = {"sha256": file_sha256(path), "records": len(records)}
    cells: dict[str, list] = {pid: [] for pid in readers}
    named: dict[str, set] = {pid: set() for pid in readers}
    for row in before:
        pid, request = row["profile_id"], row["provenance"]["request_hash"]
        shared = bool(row["provenance"].get("shared_request"))
        record = None
        if request is not None:
            record = latest[pid].get(request)
            if record is None:
                raise ValueError(f"{pid} {row['stage']} {row['item_id']}: request {request} not in the ledger")
            if record["item_id"] != row["item_id"] or record["outcome"] != row["outcome"] or\
                    (not shared and record["stage"] != row["stage"]):
                raise ValueError(f"{pid} {row['stage']} {row['item_id']}: the ledger record of {request} is not "
                                 "the one the row scored")
            named[pid].add(request)
        cells[pid].append((cell_meta(row), record, shared))
    for pid in readers:
        ledgers[f"read__{pid}.jsonl"]["observed_unreferenced"] = sum(
            r["outcome"] == "observed" and h not in named[pid] for h, r in latest[pid].items())
    repeats = (manifest.get("repeats") or {}).get("k", 1)
    rows, summaries, answered = score_cells(run_id, by_id, readers, cells, text_of, repeats, decisions)
    unused = sorted(set(decisions) - {s["arb_id"] for s in answered if s["arb_id"]})
    if unused:
        raise ValueError(f"decisions resolve no row of {run_id}: {unused}")
    if requests is not None:
        rows = name_memory_requests(rows, requests["file"], requests["keys"])
    outputs = {**write_scored(out, run_id, rows, summaries, answered),
               **({"memory_requests": requests["file"]} if requests else {})}
    table_before, table_after = accuracy_table(before), accuracy_table(rows)
    scorer_now = {rel: lf_sha256(ROOT / rel) for rel in SCORER_FILES}
    record = {
        "schema": SCHEMA, "run_id": run_id,
        "run_manifest_sha256": file_sha256(run_dir / "manifest.json"),
        "source": {"rows": source["outputs"]["rows"], "rows_sha256": file_sha256(run_dir / source["outputs"]["rows"]),
                   "summary": source["outputs"].get("summary"), "invocation_finished_at": source.get("finished_at")},
        "items_sha256": manifest["items_sha256"], "ledgers": ledgers,
        "panel": {"path": display(panel.path, ROOT), "sha256": panel.sha256,
                  "answer_from_reasoning_field": {pid: pp.answer_from_reasoning_field for pid, pp in readers.items()}},
        "decisions": decisions_record,
        "memory_requests": None if requests is None else {"file": requests["file"], "from": requests["from"],
                                                          "memories": len(requests["keys"])},
        "scorer_sha256": {"run": manifest["scorer_sha256"], "now": scorer_now,
                          "changed": sorted(k for k in scorer_now if scorer_now[k] != manifest["scorer_sha256"].get(k))},
        "reader_digest": reader_digest(),
        "rules": {"answer_field": MAPPING_RULE, "decisions": DECISIONS_RULE if decisions else None},
        "outputs": outputs,
        "arbitration_queue_after": len(read_jsonl(out / outputs["arbitration"])) if outputs["arbitration"] else 0,
        "changes": changes(before, rows, decisions),
        "accuracy": {k: {"before": table_before.get(k), "after": v} for k, v in table_after.items()},
    }
    name = _write_addressed(out, "rescore", ".json", _json_bytes(record))
    with open(out / "invocations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps({"started_at": started, "finished_at": utc_now(), "mode": "rescore",
                                 "outcome": "rescored", "reader_digest": record["reader_digest"], "rescore": name,
                                 "outputs": outputs}, ensure_ascii=False, sort_keys=True) + "\n")
    return {"run_id": run_id, "rescore": name, **record}

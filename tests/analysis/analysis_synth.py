"""Synthetic reader-matrix runs for the analysis tests: items, rows, ledgers, summary and manifest written the way
procedures.matrix.run_matrix lays out a run directory, with every number chosen by the test."""
from __future__ import annotations

from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
from pathlib import Path

from procedures.analysis.data import RUN_SCHEMA
from procedures.items.frozen import jsonl_bytes
from procedures.readers import price_call
from procedures.readers.ledger import SCHEMA as LEDGER_SCHEMA
from procedures.memory.requests import FILE_STEM, SCHEMA, map_bytes

READER_B = "B1_dsv41flash_hf_deepinfra"
READERS_C = ("C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low")
BUILDER = "B1_dsv41flash_direct_nothink"
READER_C2 = "tierC2_qwen3_32b_hf_nscale_nothink"      # access tier C2: in the constrained panel
READER_C3 = "tierC3_gptoss120b_hf_deepinfra_low"      # access tier C3: the intermediate step, never pooled
READER_A = "A2_sonnet5_nothink"                       # the cloud comparator (Claude Sonnet 5): in no tier
READERS_B = (READER_B, "B2_qwen3_235b_2507_hf_deepinfra", "tierB_llama33_70b_hf_ovhcloud")   # main run's 3 B readers
TIER = {READER_A: "A", **{p: "B" for p in READERS_B}, **{p: "C" for p in (*READERS_C, READER_C2, READER_C3)}}
M5 = f"M5@4000/{BUILDER}"
M6 = f"M6@4000/{READERS_C[0]}"
OPTS = ["open the breaker", "hold the schedule", "none of these"]
WHEN = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)       # a Saturday: no peak window


def _item(item_id: str, typ: str, family: str, doc: str, **extra) -> dict:
    base = {"item_id": item_id, "type": typ, "family": family, "doc": doc, "question": f"Question {item_id}?",
            "evidence_path": f"data/corpus/test/{doc}.txt", "admissible": True, "split": "eval"}
    if extra.pop("side", False):
        gold = extra.pop("gold", OPTS[0])
        base.update(answer_type="closed_set", options=OPTS, gold={"value": gold, "unit": None})
    else:
        base.update(answer_type="numeric", gold={"value": extra.pop("gold", "15"), "unit": "min"})
    return {**base, **extra}


def make_items() -> list[dict]:
    """L3 (as-of date x branch x exception): 6 units x 2 cells (U1-U3 abstracted, U4-U6 transparent; U1 b is an
    exception cell, U2 b a not_in_effect cell); S2 (owner-written branch pairs): one pair; S3 (unrevised control
    sentences): one value item in each of two documents. 16 items."""
    items = []
    for u in range(1, 7):
        layer = "abstracted" if u <= 3 else "transparent"
        family, doc = ("PJM", "M-01") if u % 2 else ("MISO", "BPM-001")
        for branch in "ab":
            kind = "exception" if (u, branch) == (1, "b") else "rule"
            gold = "not_in_effect" if (u, branch) == (2, "b") else OPTS[0 if branch == "a" else 1]
            items.append(_item(f"u{u}{branch}", "L3-side", family, doc, side=True, gold=gold, unit_id=f"U{u}",
                               rule2_spirit=layer, cell_kind=kind, branch=branch))
    items += [_item("p1a", "S2-side", "PJM", "M-01", side=True, gold=OPTS[0], pair_id="P1", side_label="a"),
              _item("p1b", "S2-side", "PJM", "M-01", side=True, gold=OPTS[1], pair_id="P1", side_label="b"),
              _item("s3x", "S3-value", "PJM", "M-01"), _item("s3y", "S3-value", "MISO", "BPM-001")]
    return items


POPULATION = {"u1a", "u1b", "u2a", "u2b"}          # the population of main_run_items


def main_run_items() -> list[dict]:
    """make_items with the tags of the main run's item file: U1, U2 = the population (ops / technical, abstracted,
    load_bearing, primary); U2 same_rule medium; U1 a template family with U3 (outside the population: market) and an
    exemption cell (u1b, the subtype sits on one cell); controls elsewhere."""
    out = []
    tags = {"U1": dict(domain="ops", k6_exception_literal="exception", template_family="twin"),
            "U2": dict(domain="technical", k6_exception_literal="none", same_rule_confidence="medium"),
            "U3": dict(domain="market", k6_exception_literal="none", template_family="twin")}
    for it in make_items():
        u = it.get("unit_id")
        if u in tags:
            it = dict(it, analysis_stratum="primary", load_bearing=True, k6_exception_confidence=None,
                      subtype="exemption_outcome" if it["item_id"] == "u1b" else None,
                      cue_relation=None, owner_restates_standard=None, **{"same_rule_confidence": None, **tags[u]})
        elif u:
            it = dict(it, analysis_stratum="earlier_batches")
        else:
            it = dict(it, analysis_stratum="controls")
        out.append(it)
    return out


def chance(item: dict) -> Fraction:
    return Fraction(1, len(item["options"])) if item.get("options") and item["gold"]["value"] != "not_in_effect"\
        else Fraction(0)


def usage(input_uncached: int, output: int, input_cached: int = 0) -> dict:
    total = input_uncached + input_cached
    return {"complete": True, "input_total": total, "input_uncached": input_uncached, "input_cached": input_cached,
            "cache_write": 0, "cache_write_1h": 0, "output_total": output, "output_visible": output, "reasoning": 0,
            "reasoning_tokens_estimate": None, "missing": [], "inconsistent": []}


def call_record(profile, request_hash: str, item_id: str, stage: str, u: dict, book, **extra) -> dict:
    price = price_call(profile, u, book=book, observed_at=WHEN)
    return {"kind": "call", "schema": LEDGER_SCHEMA, "request_hash": request_hash, "item_id": item_id,
            "stage": stage, "outcome": "observed", "finish_reason": "stop", "usage": u, "usd": price["usd"],
            "usd_counted": price["usd"], "price": {"price_key": profile.price_key, **price},
            "attempts": [{"started_at": WHEN.isoformat()}], **extra}


def header(profile_id: str, book) -> dict:
    return {"kind": "header", "schema": LEDGER_SCHEMA, "profile_id": profile_id,
            "price_book_sha256": book.sha256}


def row(run_id: str, pid: str, cond: str, item: dict, correct: bool, *, error_class: str | None = None,
        other_subclass: str | None = None, status: str = "predicted", leak: bool = False, pred=None,
        matched_option=None, c_on_nie: bool = False, repeat: int = 0, tokens: tuple = (1000, 50),
        build_tokens: int = 0, memory_key: str | None = None, revisions=None, ref_input: int | None = None,
        from_reasoning: bool = False, query_tokens: int | None = None, requests_file: str | None = None,
        memory_empty: bool | None = False, not_sent: str | None = None) -> dict:
    """One scored row; `memory_empty` None leaves the field out (a row written before rows recorded an empty
    memory); `not_sent` = an infeasible reason: the cell was recorded and not sent (no capability, no request)."""
    rep, _, rest = cond.partition("@")
    budget, _, builder = rest.partition("/")
    if not correct and error_class is None and status == "predicted":
        error_class, other_subclass = "E-other", "value_in_doc"
    request_hash = hashlib.sha256(f"{pid}|{cond}|{item['item_id']}|{repeat}".encode()).hexdigest()
    out = {
        "run_id": run_id, "profile_id": pid, "item_id": item["item_id"], "stage": f"read/{cond}",
        "repeat_idx": repeat, "form": "side" if item.get("options") else "value", "representation": rep,
        "budget_tokens": None if budget == "full" else int(budget), "builder_profile_id": builder or None,
        "stratum": item["type"].split("-")[0], "type": item["type"], "family": item["family"],
        "cell_kind": item.get("cell_kind"), "outcome": "observed",
        **({} if memory_empty is None else {"memory_empty": memory_empty}),
        "capability": {"correct": correct, "status": status, "error_class": None if correct else error_class,
                       "other_subclass": other_subclass, "abstained": status == "abstained", "truncated": False,
                       "pred": pred if pred is not None else ("x" if not correct else item["gold"]["value"]),
                       "matched_option": matched_option, "c_on_nie": c_on_nie, "reasoning_leak": leak,
                       "answer_from_reasoning_field": from_reasoning},
        "access": {"tier": TIER[pid], "slot": pid[:2], "local_context_ceiling": 32_768 if TIER[pid] == "C" else None,
                   "planned_input_tokens": tokens[0], "feasible_at_B": True, "requested_model": f"{pid}-model",
                   "served_model": f"{pid}-served", "observed_provider": "synthetic"},
        "sustainability": {"tokens_input_uncached": tokens[0], "tokens_input_cached": 0, "tokens_cache_write": 0,
                           "tokens_output": tokens[1], "tokens_output_visible": tokens[1], "tokens_reasoning": 0,
                           "reasoning_tokens_estimate": None,
                           "ref_input_tokens": tokens[0] if ref_input is None else ref_input,
                           "memory_tokens": 0, "build_tokens": build_tokens, "token_method": "tiktoken:o200k_base",
                           **({"query_memory_tokens": query_tokens} if query_tokens is not None else {})},
        "provenance": {"request_hash": request_hash,
                       "memory_cache_key": memory_key or hashlib.sha256(f"{cond}|{item['doc']}".encode()).hexdigest(),
                       "revisions_used": revisions if revisions is not None else [[item["doc"], "1"]],
                       **({"memory_requests": requests_file} if requests_file else {})},
    }
    if not_sent is not None:
        out.update(outcome="not_sent_infeasible", capability=None)
        out["access"].update(feasible_at_B=False, infeasible_reason=not_sent, requested_model=None, served_model=None,
                             observed_provider=None)
        out["provenance"]["request_hash"] = None
    return out


def summary_of(rows: list[dict], items: dict) -> dict:
    """The run summary's capability block as procedures.scoring.score_path writes it: the first read (repeat 0) only."""
    out = {}
    for r in rows:
        if r["repeat_idx"] == 0 and r["capability"] is not None:
            out.setdefault(f"{r['profile_id']}|{r['stage']}", []).append(r)
    conditions = {}
    for key, rs in out.items():
        n, correct = len(rs), sum(r["capability"]["correct"] for r in rs)
        ch = sum((chance(items[r["item_id"]]) for r in rs), Fraction(0))
        conditions[key] = {"capability": {"n": n, "correct": correct,
                                          "acc_chance_corrected": round(float((correct - ch) / (n - ch)), 4)}}
    return {"conditions": conditions}


def write_run(root: Path, rows_of, items: list[dict], panel, book, *, label: str = "syn",
              build_records: list[dict] | None = None, summary_override: dict | None = None,
              query_records: dict[str, list[dict]] | None = None, requests: dict | None = None,
              other_ledgers: dict[str, tuple[str, list[dict]]] | None = None) -> Path:
    """rows_of(run_id) -> rows. Writes manifest.json, invocations.jsonl, rows, summary and one read call ledger per
    profile (one priced call per row), plus build__BUILDER.jsonl when `build_records` is given,
    query__<compressor>.jsonl per `query_records` entry, any other call ledger `other_ledgers` (file stem ->
    (profile, records)), and the memory request map `requests` (memory key -> entry) as memory_requests_<sha12>.json."""
    ids = sorted({it["item_id"] for it in items})
    by_id = {it["item_id"]: it for it in items}
    manifest = json.dumps({"schema": RUN_SCHEMA, "label": label, "n_items": len(ids),
                           "items_sha256": hashlib.sha256(jsonl_bytes([by_id[i] for i in ids])).hexdigest()},
                          sort_keys=True).encode()
    run_id = f"{label}_{hashlib.sha256(manifest).hexdigest()[:12]}"
    run_dir = Path(root) / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "manifest.json").write_bytes(manifest)
    rows = rows_of(run_id)
    (run_dir / "rows_x.jsonl").write_bytes(jsonl_bytes(rows))
    summary = summary_override or summary_of(rows, by_id)
    (run_dir / "summary_x.json").write_text(json.dumps(summary), encoding="utf-8")
    (run_dir / "invocations.jsonl").write_text(
        json.dumps({"outcome": "live", "outputs": {"rows": "rows_x.jsonl", "summary": "summary_x.json"}}) + "\n",
        encoding="utf-8")
    ledgers: dict[str, list[dict]] = {}
    for r in rows:
        if r["provenance"]["request_hash"] is None:        # recorded, not sent: no ledger record
            continue
        pp = panel.get(r["profile_id"])
        s = r["sustainability"]
        ledgers.setdefault(r["profile_id"], []).append(
            call_record(pp, r["provenance"]["request_hash"], r["item_id"], r["stage"],
                        usage(s["tokens_input_uncached"], s["tokens_output"]), book))
    for pid, recs in ledgers.items():
        (run_dir / f"read__{pid}.jsonl").write_bytes(jsonl_bytes([header(pid, book), *recs]))
    if build_records is not None:
        (run_dir / f"build__{BUILDER}.jsonl").write_bytes(jsonl_bytes([header(BUILDER, book), *build_records]))
    for pid, recs in (query_records or {}).items():
        (run_dir / f"query__{pid}.jsonl").write_bytes(jsonl_bytes([header(pid, book), *recs]))
    for stem, (pid, recs) in (other_ledgers or {}).items():
        (run_dir / f"{stem}.jsonl").write_bytes(jsonl_bytes([header(pid, book), *recs]))
    if requests is not None:
        write_requests(run_dir, requests)
    return run_dir


def write_requests(run_dir: Path, memories: dict) -> str:
    data = map_bytes({"schema": SCHEMA, "memories": memories})
    name = f"{FILE_STEM}_{hashlib.sha256(data).hexdigest()[:12]}.json"
    (Path(run_dir) / name).write_bytes(data)
    return name


def m5_requests(records: list[dict], keys: dict[str, str]) -> dict:
    """memory key -> request map entry of a summary (M5) memory over its revision's build calls ("DOC@REV#..."): keys
    maps a memory key to its "DOC@REV"."""
    return {key: {"representation": "M5", "budget_tokens": 4000, "profile": BUILDER, "ledger": f"build__{BUILDER}",
                  "revisions": [rev.split("@")], "requests": [{"request_hash": r["request_hash"]} for r in records
                                                              if r["item_id"].split("#")[0] == rev]}
            for key, rev in keys.items()}


def build_calls(panel, book, per_revision: dict[str, list[tuple[int, int]]], imported: int = 0) -> list[dict]:
    """M5 sub-block build records: revision "DOC@REV" -> [(input tokens, output tokens) per call]; the first
    `imported` calls of each revision are marked imported."""
    pp = panel.get(BUILDER)
    out = []
    for rev, calls in per_revision.items():
        for k, (tin, tout) in enumerate(calls):
            extra = {"imported_from": "syn_source", "imported_usd_counted": 0.0} if k < imported else {}
            out.append(call_record(pp, hashlib.sha256(f"{rev}#{k}".encode()).hexdigest(), f"{rev}#0.{k}",
                                   "build/M5@28", usage(tin, tout), book, **extra))
    return out

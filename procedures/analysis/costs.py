"""Sustainability and access tables.

Every dollar figure is re-priced from the call ledgers' recorded usage on ONE price book
(procedures.readers.reprice.reprice_record: same instant as the run priced it), never the recorded `usd`: a report
never mixes the snapshot book with a corrected one (the billing reconciliation checks usage x price against the
provider invoice; until it passes, dollar figures are not results). The recorded dollars appear only as a diagnostic
beside the re-priced ones.

* `build_costs`: the LLM calls of every LLM-built memory in a run (summary M5 and typed constraint records M7, built
  once per revision by a builder; query-focused compression M6, one call per question by a compressor), by request
  identity: the run's memory request map (procedures.memory.requests; procedures.analysis.data.load_run) lists per
  memory cache_key its call ledger ("build__<builder>" for M5 / M7, "query__<compressor>" for M6) and request hashes;
  the memory's calls are the answered records of those hashes in that ledger (latest record per hash; imported
  records included: they built this memory, their spend was booked by the run they came from). The attribution is
  checked, not assumed: provider input + output tokens over the calls must equal what the harness recorded on every
  answered row reading the memory (build_tokens for M5 / M7, summed over every cap level sent; query_memory_tokens
  for M6). A memory only rows that were not sent refer to is not attributed: an M6 compression over its compressor's
  ceiling is planned and recorded, never made (its query_memory_tokens are the plan, no call exists). An M7 build is
  budget-free, so the memories of the budget ladder share their calls: each memory carries its whole build (the
  deployment view: what one deployed memory costs); `bill` is the other view, each answered ledger record once,
  imported ones apart (what the run paid, the view the billing reconciliation checks).
* `sustainability`: per reader x condition, provider token counts per query (input uncached / cached / cache write,
  output, reasoning as reported and reasoning as estimated from reasoning text, kept apart; each kind averaged over
  the calls that report it, with that count; a kind a provider does not report counts 0 in the read total); the
  build tokens of the M5 / M7 memories read, amortised over N queries per memory (one memory = one as-of revision set
  at one budget): per-query total = read tokens + query tokens + build tokens / N, N in AMORTISE_N; and an M6 cell's
  own compression tokens (query tokens: per question, never amortised, attributed by (question, compressor)).
* `access`: tier and its governance rule, context ceiling and feasibility at the budget, the cells not sent by
  reason (`not_sent_by_reason`: local ceiling, route context, route timeout, ...), endpoints, $ per 1k queries (read
  + compression, and with the build amortised at each N).
* `energy`: rank-only statements from a band file (data/analysis/energy_bands.json: joules per token, low / high, per
  profile, each band with its source). Energy per query = input tokens x input band + output tokens x output band
  (+ an M6 cell's compression tokens on the compressor's bands, + build tokens / N on the builder's bands); A < B is
  stated only when A's high end is below B's low end, i.e. at every band endpoint. An unset band yields no
  statement: no number is invented.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from procedures import ROOT
from procedures.items.publish import display
from procedures.readers.reprice import reprice_record
from procedures.readers.cells import BUILT_ONCE, QUERY_BUILT

from procedures.analysis.data import Obs, Run
from procedures.analysis.stats import r4

AMORTISE_N = (10, 30, 100, 300, 1_000, 3_000, 10_000)
TOKEN_KINDS = ("input_uncached", "input_cached", "cache_write", "output", "reasoning", "reasoning_estimate")
BANDS_SCHEMA = "procedures.analysis.energy_bands.v1"
BAND_UNITS = "J/token"
BAND_KINDS = ("input", "output")
TIER_GOVERNANCE = {       # access tiers are defined by governance (where the procedure text may go), not by price
    "A": "cloud closed frontier: data leave the utility (not usable for control-centre data under CIP-011 / security "
         "rules)",
    "B": "open weights on a multi-GPU server (bf16 weights > 80 GB); served here through an API as a capability proxy",
    "C": "open weights on one local machine (C1 consumer 24 GB <= ~20B, C2 workstation 48 GB, C3 one 80 GB card); "
         "context capped at 32k",
}
LEDGER_KINDS = ("build__", "query__")


def _groups(obs: list[Obs]) -> dict[tuple, list[Obs]]:
    out: dict[tuple, list[Obs]] = defaultdict(list)
    for o in obs:
        out[(o.profile_id, o.condition)].append(o)
    return dict(sorted(out.items()))


def _usage(record: dict) -> dict:
    u = record.get("usage") or {}
    if u.get("input_total") is None or u.get("output_total") is None:
        raise ValueError(f"memory call {record.get('request_hash')} has no provider token totals; its attribution "
                         "cannot be checked against the tokens the rows record")
    return u


def _price(record: dict, book) -> float | None:
    return reprice_record(record, book)["usd"]


# ---------------------------------------------------------------------- builds and compressions

def _recorded_tokens(o: Obs) -> int:
    return o.build_tokens if o.representation in BUILT_ONCE else o.query_tokens


def _cost(records: list[dict], book) -> dict:
    usages = [_usage(r) for r in records]
    prices = [_price(r, book) for r in records]
    return {"calls": len(records), "imported_calls": sum(bool(r.get("imported_from")) for r in records),
            "tokens": {"input_total": sum(u["input_total"] for u in usages),
                       "input_uncached": sum(u.get("input_uncached") or 0 for u in usages),
                       "input_cached": sum(u.get("input_cached") or 0 for u in usages),
                       "cache_write": sum(u.get("cache_write") or 0 for u in usages),
                       "output": sum(u["output_total"] for u in usages),
                       "reasoning": sum(u.get("reasoning") or 0 for u in usages),
                       "total": sum(u["input_total"] + u["output_total"] for u in usages)},
            "reasoning_unreported_calls": sum(u.get("reasoning") is None for u in usages),
            "usd": None if any(x is None for x in prices) else sum(prices),
            "unpriced_calls": sum(x is None for x in prices),
            "usd_recorded": sum(r.get("usd") or 0.0 for r in records)}


def _answered(ledger: dict) -> dict:
    return {h: r for h, r in ledger.items() if r.get("outcome") == "observed"}


def build_costs(runs: list[Run], obs: list[Obs], book) -> dict:
    """{"memories": memory cache key -> its calls' cost (deployment view), "unattributed_calls": per run, answered
    build / compression calls no memory of the run lists, "bill": per run and ledger, each answered call once}."""
    by_run = {r.run_id: r for r in runs}
    memories, unattributed, bill = {}, {}, {}
    for run_id in sorted({o.run_id for o in obs if o.representation in BUILT_ONCE | QUERY_BUILT}):
        run = by_run[run_id]
        if run.memory_requests is None:
            raise ValueError(f"{run_id}: LLM-built memories but no memory request map; re-score the run "
                             "(scripts/rescore.py), which writes the map")
        listed = run.memory_requests["memories"]
        mems: dict[str, tuple] = {}
        for o in obs:
            if o.run_id != run_id or o.representation not in BUILT_ONCE | QUERY_BUILT or not o.observed:
                continue
            entry = (o.representation, o.budget, o.builder, o.revisions, _recorded_tokens(o))
            if mems.setdefault(o.memory_cache_key, entry) != entry:
                raise ValueError(f"{run_id}: rows disagree about memory {o.memory_cache_key[:12]}")
        used: dict[str, set] = defaultdict(set)
        for key, (rep, budget, profile, revisions, recorded) in sorted(mems.items()):
            entry = listed.get(key)
            if entry is None:
                raise ValueError(f"{run_id}: memory {key[:12]} ({rep}@{budget}) is not in {run.memory_requests_file}")
            if (entry["representation"], entry["profile"]) != (rep, profile):
                raise ValueError(f"{run_id}: memory {key[:12]} is {rep} by {profile} in the rows but "
                                 f"{entry['representation']} by {entry['profile']} in the request map")
            ledger = _answered(run.ledgers.get(entry["ledger"], {}))
            recs = [ledger[q["request_hash"]] for q in entry["requests"] if q["request_hash"] in ledger]
            used[entry["ledger"]].update(r["request_hash"] for r in recs)
            cost = _cost(recs, book)
            if cost["tokens"]["total"] != recorded or (entry["requests"] and not recs):
                raise ValueError(f"{run_id}: memory {key[:12]} ({rep}@{budget}, {revisions}) records {recorded} "
                                 f"tokens but its {len(recs)} attributed calls in {entry['ledger']} hold "
                                 f"{cost['tokens']['total']}")
            memories[key] = {"run_id": run_id, "representation": rep, "budget": budget, "profile": profile,
                             "ledger": entry["ledger"], "revisions": [list(r) for r in revisions], **cost}
        stems = sorted(stem for stem in run.ledgers if stem.startswith(LEDGER_KINDS))
        unattributed[run_id] = {stem: len(set(_answered(run.ledgers[stem])) - used[stem]) for stem in stems}
        bill[run_id] = {}
        for stem in stems:
            answered = list(_answered(run.ledgers[stem]).values())
            own = [r for r in answered if not r.get("imported_from")]
            bill[run_id][stem] = {**_cost(own, book), "imported_records": len(answered) - len(own),
                                  "usd_counted_recorded": sum(float(r.get("usd_counted") or 0.0)
                                                              for r in run.ledgers[stem].values())}
    return {"memories": memories, "unattributed_calls": unattributed, "bill": bill}


# ---------------------------------------------------------------------- sustainability

def _per_query(answered: list[Obs], memories: dict, field) -> float:
    """Mean over the answered cells of `field` of the memory each read (a memory read by k cells counts k times)."""
    return float(np.mean([field(memories[o.memory_cache_key]) for o in answered])) if answered else 0.0


def sustainability(obs: list[Obs], memories: dict, void: dict) -> list[dict]:
    out = []
    for (pid, cond), rows in _groups(obs).items():
        answered = [o for o in rows if o.observed]
        n = len(answered)
        values = {k: [o.tokens[k] for o in answered if o.tokens[k] is not None] for k in TOKEN_KINDS}
        sums = {k: sum(v) for k, v in values.items()}
        read_in = sums["input_uncached"] + sums["input_cached"] + sums["cache_write"]
        read_q = (read_in + sums["output"]) / n if n else None
        first = rows[0]
        row = {"profile_id": pid, "condition": cond, "tier": first.tier, "arm": first.arm,
               "void": void.get((pid, cond)), "answered": n,
               "per_query": {k: r4(np.mean(v)) if v else None for k, v in values.items()},
               "calls_reporting": {k: len(v) for k, v in values.items()},
               "per_query_input_total": r4(read_in / n) if n else None,
               "per_query_read_total": r4(read_q), "totals": sums, "build": None, "query": None,
               "amortised_per_query": None}
        if not n:
            out.append(row)
            continue
        per_q = read_q
        if first.representation in QUERY_BUILT:
            q_total = _per_query(answered, memories, lambda m: m["tokens"]["total"])
            row["query"] = {
                "compressor": first.builder, "memories": len({o.memory_cache_key for o in answered}),
                "per_query_input": r4(_per_query(answered, memories, lambda m: m["tokens"]["input_total"])),
                "per_query_output": r4(_per_query(answered, memories, lambda m: m["tokens"]["output"])),
                "per_query_total": r4(q_total)}
            per_q = read_q + q_total
        if first.representation in BUILT_ONCE:
            used = sorted({o.memory_cache_key for o in answered})
            per_memory = [memories[k]["tokens"]["total"] for k in used]
            b_total = _per_query(answered, memories, lambda m: m["tokens"]["total"])
            row["build"] = {
                "builder": first.builder, "memories": len(used),
                "tokens_per_memory": {"mean": r4(np.mean(per_memory)), "min": min(per_memory), "max": max(per_memory)},
                "per_query_input": r4(_per_query(answered, memories, lambda m: m["tokens"]["input_total"])),
                "per_query_output": r4(_per_query(answered, memories, lambda m: m["tokens"]["output"])),
                "per_query_total": r4(b_total)}
            row["amortised_per_query"] = {str(N): r4(per_q + b_total / N) for N in AMORTISE_N}
            row["build_share"] = {str(N): r4((b_total / N) / (per_q + b_total / N)) for N in AMORTISE_N}
        else:
            row["amortised_per_query"] = {str(N): r4(per_q) for N in AMORTISE_N}
        out.append(row)
    return out


# ---------------------------------------------------------------------- access

def read_record(runs: dict[str, Run], o: Obs) -> dict:
    record = runs[o.run_id].ledgers.get(f"read__{o.profile_id}", {}).get(o.request_hash)
    if record is None:
        raise ValueError(f"{o.run_id}: answered row ({o.profile_id}, {o.condition}, {o.item_id}) has no ledger record")
    return record


def _mean_or_none(xs: list) -> float | None:
    return None if not xs or any(x is None for x in xs) else float(np.mean(xs))


def not_sent_by_reason(obs: list[Obs]) -> dict[str, int]:
    """Cells recorded and not sent, by reason (Obs.infeasible_reason; "None" for a row written before the field),
    sorted: route_context (the cell exceeds the context of the route serving the reader) and route_timeout (its M6
    compression timed out on every attempt) are absences of the same kind."""
    return dict(sorted(Counter(str(o.infeasible_reason) for o in obs if o.outcome == "not_sent_infeasible").items()))


def access(runs: list[Run], obs: list[Obs], memories: dict, book, void: dict) -> list[dict]:
    by_run = {r.run_id: r for r in runs}
    out = []
    for (pid, cond), rows in _groups(obs).items():
        answered = [o for o in rows if o.observed]
        records = [read_record(by_run, o) for o in answered]
        prices = [_price(r, book) for r in records]
        first = rows[0]
        read_mean = _mean_or_none(prices)
        query_mean = None
        if first.representation in QUERY_BUILT:
            query_mean = _mean_or_none([memories[o.memory_cache_key]["usd"] for o in answered])
        per_q = None if read_mean is None or (first.representation in QUERY_BUILT and query_mean is None)\
            else read_mean + (query_mean or 0.0)
        row = {"profile_id": pid, "condition": cond, "tier": first.tier, "slot": first.slot,
               "governance": TIER_GOVERNANCE.get(first.tier[:1]), "void": void.get((pid, cond)),
               "local_context_ceiling": first.context_ceiling, "cells": len(rows),
               "feasible": sum(o.feasible for o in rows),
               "not_sent_infeasible": sum(o.outcome == "not_sent_infeasible" for o in rows),
               "not_sent_by_reason": not_sent_by_reason(rows),
               "planned_input_max": max((o.planned_input_tokens or 0) for o in rows),
               "requested_models": sorted({o.requested_model for o in answered if o.requested_model}),
               "served_models": sorted({o.served_model for o in answered if o.served_model}),
               "providers": sorted({o.observed_provider for o in answered if o.observed_provider}),
               "answered": len(answered), "unpriced": sum(x is None for x in prices),
               "usd_per_1k_read": r4(1000 * read_mean) if read_mean is not None else None,
               "usd_per_1k_read_recorded": r4(1000 * np.mean([r.get("usd") or 0.0 for r in records]))
               if records else None,
               "usd_per_1k_query_memory": r4(1000 * query_mean) if query_mean is not None else None,
               "build_usd_per_memory": None, "usd_per_1k_at_N": None}
        if first.representation in BUILT_ONCE and answered and per_q is not None:
            used = sorted({o.memory_cache_key for o in answered})
            costs = [memories[k]["usd"] for k in used]
            if all(c is not None for c in costs):
                build_q = _per_query(answered, memories, lambda m: m["usd"])
                row["build_usd_per_memory"] = r4(np.mean(costs))
                row["usd_per_1k_at_N"] = {str(N): r4(1000 * (per_q + build_q / N)) for N in AMORTISE_N}
        elif per_q is not None:
            row["usd_per_1k_at_N"] = {str(N): r4(1000 * per_q) for N in AMORTISE_N}
        out.append(row)
    return out


def prices_as_recorded(book, runs: list[Run]) -> bool:
    """Every answered call of the runs' ledgers re-prices on `book` (procedures.readers.reprice) to the usd its ledger
    recorded: the book prices the runs as they were priced, whatever else differs in its file."""
    for run in runs:
        for ledger in run.ledgers.values():
            for rec in _answered(ledger).values():
                if not (rec.get("price") or {}).get("price_key"):
                    continue
                again, recorded = _price(rec, book), rec.get("usd")
                if (again is None) != (recorded is None) or (
                        again is not None and abs(again - recorded) > 1e-12 + 1e-9 * abs(recorded)):
                    return False
    return True


def book_identity(book, runs: list[Run], label: str) -> dict:
    """The book every $ of the report is priced on (its path relative to the repository root), the books the ledgers
    recorded their prices on, and whether this book prices every answered call as its ledger recorded it
    (`same_as_ledgers`, prices_as_recorded)."""
    recorded = sorted({(rec.get("price") or {}).get("price_book_sha256") for r in runs for led in r.ledgers.values()
                       for rec in led.values()} - {None})
    return {"label": label, "path": display(book.path, ROOT), "sha256": book.sha256, "fetched": book.fetched,
            "ledgers_priced_on": recorded, "same_as_ledgers": prices_as_recorded(book, runs)}


# ---------------------------------------------------------------------- energy

def load_bands(path: Path) -> dict:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != BANDS_SCHEMA or doc.get("units") != BAND_UNITS:
        raise ValueError(f"{path}: not a {BANDS_SCHEMA} file in {BAND_UNITS}")
    profiles = doc.get("profiles")
    if not isinstance(profiles, dict):
        raise ValueError(f"{path}: 'profiles' must map profile ids to bands")
    for pid, entry in profiles.items():
        for kind in BAND_KINDS:
            band = entry.get(kind)
            if band is None:
                continue
            lo, hi = band.get("low"), band.get("high")
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in (lo, hi)) or not 0 < lo <= hi:
                raise ValueError(f"{path}: {pid}.{kind} needs 0 < low <= high, got {band}")
            if not entry.get("source"):
                raise ValueError(f"{path}: {pid}.{kind} is set without a source")
    sha = hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {"path": display(path, ROOT), "sha256": sha, "profiles": profiles}


def _band(bands: dict, pid: str, kind: str):
    band = (bands["profiles"].get(pid) or {}).get(kind)
    return None if band is None else (band["low"], band["high"])


def energy(sus_rows: list[dict], bands: dict) -> dict:
    """Per N: the per-query energy interval of each readable condition whose token kinds (read input / output, and
    build input / output for an LLM-built memory) all have a band, and the orderings that hold at every band endpoint
    among them; conditions with an unset band are listed with the bands they lack."""
    by_n = {}
    for N in AMORTISE_N:
        cells, missing = {}, {}
        for s in sus_rows:
            label = f"{s['profile_id']}|{s['condition']}"
            if s["void"] or not s["answered"]:
                continue
            parts = [(s["per_query_input_total"], _band(bands, s["profile_id"], "input"), "read input"),
                     (s["per_query"]["output"], _band(bands, s["profile_id"], "output"), "read output")]
            if s["query"]:
                q = s["query"]
                parts += [(q["per_query_input"], _band(bands, q["compressor"], "input"), "compression input"),
                          (q["per_query_output"], _band(bands, q["compressor"], "output"), "compression output")]
            if s["build"]:
                b = s["build"]
                parts += [(b["per_query_input"] / N, _band(bands, b["builder"], "input"), "build input"),
                          (b["per_query_output"] / N, _band(bands, b["builder"], "output"), "build output")]
            unset = [name for _, band, name in parts if band is None]
            if unset:
                missing[label] = unset
                continue
            cells[label] = {"low": r4(sum(t * band[0] for t, band, _ in parts)),
                            "high": r4(sum(t * band[1] for t, band, _ in parts))}
        ranks = sorted(f"{a} < {b}" for a, ea in cells.items() for b, eb in cells.items()
                       if a != b and ea["high"] < eb["low"])
        by_n[str(N)] = {"joules_per_query": cells, "unbanded": missing, "rank_statements": ranks}
    return {"bands": {"path": bands["path"], "sha256": bands["sha256"]}, "by_N": by_n}

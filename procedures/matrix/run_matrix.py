"""Run the reader matrix: items x conditions (representation @ budget) x reader profiles.

Representations: B0 closed book, M0 full text, M2 BM25 retrieval, M4 constraint sentences (both zero-LLM), and the
LLM-built ones: M5 summary and M7 records (built once per revision by the run's builder, build ledger) and M6
query-focused compression (one call per question by a compressor, query ledger).

`run_matrix(items, conditions, reader_ids, corpus_index, build_memory=..., panel=..., label=..., cap_usd=...,
dry_run=..., build_only=..., reuse_build_from=..., repeats=..., approval=..., stage_cap_usd=...)`:

1. Plan: `build_memory(item, representation, budget, corpus_index, builder_profile=..., ledger=None)` for every
   item x condition (an LLM-built representation is planned as a dry run: nothing is sent), then one reader cell per
   (item, condition, reader) (procedures.readers.cells). M6 (query-focused compression,
   procedures.memory.compressor) is compressed per question by `Condition.compressor`, or, when the condition names
   none, by each reader for itself: one memory per (item, reader), slot "M6@B/<reader>" (`m6_slots`). A compression
   whose request its compressor cannot take (its access-tier context ceiling or its route context:
   procedures.readers.cells.infeasible_reason) is not sent (`compression_over_ceiling`) and its reader cells are
   infeasible, like a reader cell over its limits (reason "compression_" + the compressor's). A condition may name
   the readers it is for (the main grid reads M0@full with the B readers only); cells over the reader's local
   ceiling ("local_ceiling") or its route's context ("route_context") are recorded, not sent, and counted per stage
   and per reason (estimate `infeasible_by_stage` / `infeasible_by_reason`, summary outcome `not_sent_infeasible` and
   `infeasible_by_reason`, the row's access field `infeasible_reason`). A compression that timed out on every
   attempt (procedures.memory.compressor mode route_timeout: a legitimate "no memory" state like no_input) makes its
   reader cells infeasible with reason "route_timeout", counted in the same places (and per compressor in the
   estimate's `queries.route_timeout`); one whose latest query-ledger record timed out is settled at planning
   (`route_timeouts`) and never sent again.
   Repeats (`repeats` = k > 1): the items of `repeat_subset` (20% of the items, the ones with the smallest hash of
   their id) get k - 1 more cells per (condition, reader), repeat_idx 1..k-1: the same request re-sent (the request
   hash binds repeat_idx), after all repeat-0 cells. Memories are built once; summaries read capability from repeat 0
   and report per-item agreement across the repeats (procedures.scoring.score_path.repeat_agreement).
   `k1_slice`: the conditions of the 4k slice (`in_k1_slice`: B0, M0@full and every condition at 4k) re-read the 20%
   subset united with the primary population (`primary_population`: the items of
   procedures.analysis.spec.in_population, the analysis spec's own rule), the other conditions the 20% subset alone;
   the same k everywhere (`repeat_sets`). The slice's conditions and the population ids are part of the run identity
   (manifest `repeats`); `repeated_for` reads the repeated ids of a condition back from that record.
2. Identity: <out_root>/<label>_<sha12>/manifest.json (out_root default out/runs) binds items, conditions, reader and
   builder profiles (fingerprints, suffix), the repeats (k, share, seed, repeated item ids; with `k1_slice` also the
   slice's conditions and the population), panel, price book, corpus index, the corpus manifest file the caller names
   (`corpus_manifest`: its path and sha256, procedures.corpus.frozen_manifest.manifest_record), frozen scorer files,
   prompt templates, every memory cache_key and the text hash of every zero-LLM memory (an LLM-built memory is bound by
   its cache_key, since its planned text is a dry-run stand-in). Anything that changes a request changes the run id, so
   one run directory never mixes requests.
3. Estimate the calls still pending (request hash not yet answered in the reader's call ledger) under the LOW and
   HIGH pricing scenarios (procedures.readers.executor.scenario) with `run_cells(dry_run=True)`; written to
   estimate_<sha12>.json. A dry run stops here.
   The estimate adds the LLM builds still needed (`pending_builds`: procedures.memory's HIGH price of each planned
   M5 / M7 sub-block request not answered in the build ledger, level by level up its cap ladder; a level after an
   unanswered one is counted apart as a possible escalation, and the cap check counts it too), and the M6
   compressions still needed (`pending_queries`: the HIGH price of each distinct planned request not answered in its
   compressor's query ledger).
   `reuse_build_from` (another run directory of the same builder identity: profile + fingerprint, price book,
   user_message_suffix, else it refuses) counts every build request this run still needs that the other run's build
   ledger answered as an import instead of a send: identical request, identical record. A live run copies those
   records into its own build ledger before building, marked `imported_from: <run id>` and counted as no spend of
   this run (usd_counted 0, the source's count kept as imported_usd_counted); invocations.jsonl counts them as
   `imported`.
4. Live: refuse before building or sending anything if spent (all call ledgers of the run, plus the reservations
   their send logs hold open) + HIGH estimate of the pending reads and builds exceeds `cap_usd`; with `build_only`
   only the build stage is checked (spent + pending builds + possible escalations against `cap_usd`, and the build
   ledger's spent + the same against `stage_cap_usd` when given; the reader stage re-checks the whole later). With
   `approval` (procedures.readers.budget.Approval, the spend grant for a batch of paid runs; the CLI's --execute
   requires one) the run's label must be one of its arms and `cap_usd` at most that arm's cap; the same pre-flight
   amount must also fit the approval (committed over every run bound to it + this amount <= granted) and the arm; the
   run directory is then bound to the approval and every reservation of every stage is checked against it under its
   cross-process lock (procedures.readers.budget.ApprovalGuard). All live stages share one
   procedures.readers.executor.SpendGuard (circuit breaker + approval): a route closed by the breaker (RouteTripped)
   gets no new sends for the rest of the run, the other readers go on and the run then stops; a paused run
   (RunPaused) stops at once; in both cases requests in flight settle first and the same command resumes.
   `timeout_overrides` ({profile id: seconds}) raises the client read timeout of the named senders in every live
   stage (procedures.readers.executor.timeout_overrides: transport, not in the run identity; each invocation records
   it); `workers_overrides` ({profile id: n}) likewise raises the named senders' in-flight limit (max_workers, to at
   most procedures.readers.executor.WORKERS_OVERRIDE_MAX; `concurrency` still only lowers it). LLM-built memories are
   then built for real through a `procedures.memory.BuilderLedger` on build__<builder>.jsonl (the build calls go
   through procedures.readers.executor.run_cells inside procedures.memory; its cap = what is left after every other
   ledger and the pending reads), each must keep its planned cache_key and come back built live (a dry-run
   placeholder, e.g. from an unset builder credential, stops the run: the mechanism sanity check requires every
   representation to have really executed); then the M6 compressions go, compressor by compressor
   (`compress_live`), through a BuilderLedger on query__<compressor>.jsonl (the query ledger: a compression is
   per-question cost, not a build) with the cap left after every other ledger and the pending reads, under the same
   cache-key and live checks (live, no_input or route_timeout); a compressor whose route the breaker closes leaves
   its unanswered compressions as route_timeout memories for this invocation and the next compressor goes on; the
   trip is reported with the readers' at the end (RouteTripped, resumable). When compressions timed out in this
   invocation, the reader estimate is recomputed on the re-planned grid and written as a second estimate file
   (`after_compressions`, naming the planning one), which the invocation's log lines name. Then every reader sends
   its pending cells through run_cells into read__<reader>.jsonl with the cap left after every other ledger of the
   run.
   With `build_only` the run stops after the live M5 / M7 builds (no compression, no reader is called), so the
   summaries can be inspected before any read is paid for; the build ledger's cap is then the cap left after every
   other ledger, and at most `stage_cap_usd`; the full command then finds them answered in the build ledger.
   Resume = rerun the same command: answered hashes are skipped by the ledgers (reads and builds), never re-sampled.
   The planning pass cannot see built summaries (procedures.memory reads them back from the build ledger only when
   given it), so a resumed estimate still counts the LLM-built reader cells as pending: conservative, never low.
5. Score every answered record through procedures.scoring.score_path: rows_<sha12>.jsonl (one row per cell: its
   capability, access and sustainability fields + provenance), summary_<sha12>.json per reader x condition, and, when
   some answer needs it, the frozen scorer's blind arbitration queue arbitration_<sha12>.jsonl (with no answer to
   arbitrate there is no queue file and the invocation records the queue as null). When the run has an LLM-built memory
   (M5 / M6 / M7), the memory request map of its planned memories (procedures.memory.requests: memory cache_key ->
   the build / compression requests that made it, every cap level) is written as memory_requests_<sha12>.json and
   every row names it (provenance `memory_requests`, the file name on the rows of a listed memory, None on the
   others; `attribute_memory_requests`), so that the analysis can charge each LLM-built memory its build and
   compression cost. Outputs are named after their content; every invocation (dry, refused, live, stopped) appends
   one line to invocations.jsonl.
   `replan_memories` re-plans a finished run's memories from its manifest (no call), each checked against the cache
   key the manifest bound: procedures.scoring.rescore uses it to write the map of a run that has none.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import logging
import math
from pathlib import Path
import re
import warnings

from procedures.analysis.spec import in_population
from procedures.items.frozen import jsonl_bytes
from procedures.readers import SpendCapExceeded, load_price_book, run_cells, scenario, token_method
from procedures.readers.budget import ApprovalGuard
from procedures.readers.ledger import Ledger, LedgerError, SendLog, identity_clash, read_ledger, sends_path, utc_now
from procedures import ROOT
from procedures.readers.executor import RouteTripped, SpendGuard, guarded, harness_digest, ledger_header, raised_timeout, raised_workers, route_timed_out
from procedures.readers.executor import timeout_overrides as overriding_timeouts
from procedures.readers.executor import workers_overrides as overriding_workers
from procedures.memory import BuilderLedger
from procedures.memory.build import route_timeout_memory
from procedures.memory.summary import escalates
from procedures.memory.compressor import ROUTE_TIMEOUT
from procedures.memory.requests import memory_requests, write_memory_requests

from procedures.readers.cells import BUILT_ONCE, LLM_BUILT, QUERY_BUILT, build_cell, cell_request_hash, infeasible_reason
from procedures.readers.prompts import CLOSED_BOOK, template_digest
from procedures.scoring.score_path import blind_arbitration_queue, condition_summary, ledger_row, repeat_agreement, score_entries, with_memory_requests

logger = logging.getLogger(__name__)

SCHEMA = "procedures.matrix.run.v1"
RUNS_ROOT = ROOT / "out" / "runs"     # where a new run directory goes by default (the paper's runs are in data/runs)
LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
UNBUDGETED = frozenset({CLOSED_BOOK, "M0"})
SCORER_FILES = ("procedures/scoring/__init__.py", "procedures/scoring/parse.py", "procedures/scoring/normalize.py",
                "procedures/scoring/score.py", "procedures/items/frozen.py")
SCENARIOS = ("low", "high")
REPEAT_FRACTION = 0.2                 # a 20% subset is re-read (k = 3 in the paper's runs)
# the seed string of the paper's runs (manifest repeats.seed): hashed with each item id to pick the repeated subset, so
# changing it changes that subset
REPEAT_SEED = "tg.reader.repeat_subset"
K1_SLICE_BUDGET = 4_000               # the 4k slice re-reads the whole primary population


def repeat_subset(item_ids, fraction: float = REPEAT_FRACTION, seed: str = REPEAT_SEED) -> list[str]:
    """The item ids re-read by a repeated run, sorted: the round(fraction x n) ids (at least one) with the smallest
    sha256(seed|id). An exact share: a per-id hash threshold only approximates it (e.g. 31 of 102 items at 20%), and
    a grown item set is a new run anyway (items_sha256 is part of the run identity)."""
    if isinstance(fraction, bool) or not isinstance(fraction, (int, float)) or not 0 < fraction <= 1:
        raise ValueError(f"repeat fraction must be in (0, 1], got {fraction!r}")
    ids = sorted(set(item_ids), key=lambda i: hashlib.sha256(f"{seed}|{i}".encode("utf-8")).hexdigest())
    return sorted(ids[:max(1, math.floor(fraction * len(ids) + 0.5))]) if ids else []


def primary_population(items) -> list[str]:
    """The sorted item ids of the primary analysis population (procedures.analysis.spec.in_population:
    analysis_stratum primary, load_bearing, the procedures.analysis.spec.MAIN_ANALYSIS domains and sub-layer), the rule
    the analysis spec freezes."""
    return sorted(it["item_id"] for it in items if in_population(it))


def population_sha256(item_ids) -> str:
    """sha256 of the sorted ids joined by newlines (UTF-8)."""
    return hashlib.sha256("\n".join(sorted(item_ids)).encode("utf-8")).hexdigest()


def repeat_sets(by_id: dict, conditions: list, repeats: int, fraction: float = REPEAT_FRACTION,
                k1_slice: bool = False) -> tuple[dict[str, list[str]], dict]:
    """(condition key -> the ids re-read under it, sorted; the manifest `repeats` record). Without `k1_slice` every
    condition re-reads the 20% subset (repeat_subset; none when repeats = 1). With it the 4k-slice conditions
    (`in_k1_slice`) re-read the subset united with the primary population of `by_id` and the record adds the
    slice's condition keys and the population (ids, n, sha256); it needs repeats > 1, a population and a slice
    condition."""
    subset = repeat_subset(by_id, fraction) if repeats > 1 else []
    record = {"k": repeats, "fraction": fraction, "seed": REPEAT_SEED, "item_ids": subset}
    if not k1_slice:
        return {c.key: subset for c in conditions}, record
    if repeats < 2:
        raise ValueError(f"k1_slice re-reads the 4k slice: it needs repeats > 1, got {repeats}")
    population = primary_population(by_id.values())
    sliced = [c.key for c in conditions if in_k1_slice(c)]
    if not population or not sliced:
        raise ValueError(f"k1_slice needs a primary population ({len(population)} items) and a 4k-slice condition "
                         f"({sliced})")
    union = sorted(set(subset) | set(population))
    record.update(k1_slice=True, k1_slice_conditions=sliced, population_item_ids=population,
                  population_n=len(population), population_sha256=population_sha256(population))
    return {c.key: union if c.key in sliced else subset for c in conditions}, record


def repeated_for(record: dict, representation: str, budget_tokens: int | None) -> frozenset:
    """The item ids a run re-read under a (representation, budget), from its manifest `repeats` record (None: no
    repeats). A 4k-slice condition of a k1_slice run re-read the subset and the
    population; every other condition the subset. The slice is budget-wide, so the compressor of an M6 condition does
    not matter."""
    if not record:
        return frozenset()
    subset = frozenset(record.get("item_ids", ()))
    if not record.get("k1_slice"):
        return subset
    key = f"{representation}@{'full' if budget_tokens is None else budget_tokens}"
    if any(c.split("/", 1)[0] == key for c in record["k1_slice_conditions"]):
        return subset | frozenset(record["population_item_ids"])
    return subset


def repeat_summary(record: dict) -> dict:
    """The estimate's `repeats`: k and the 20% subset's size, plus, with k1_slice, the
    slice's condition keys and the number of items each of them re-reads."""
    out = {"k": record["k"], "items": len(record["item_ids"])}
    if record.get("k1_slice"):
        out["k1_slice"] = {"conditions": record["k1_slice_conditions"],
                           "items": len(set(record["item_ids"]) | set(record["population_item_ids"]))}
    return out


@dataclass(frozen=True)
class Condition:
    """One memory configuration. `budget_tokens` None = full (M0) or none (B0); `readers` None = every reader;
    `compressor` (M6 only) = the profile that compresses, None = each reader compresses for itself."""
    representation: str
    budget_tokens: int | None = None
    readers: tuple | None = None
    compressor: str | None = None

    def __post_init__(self):
        b = self.budget_tokens
        if b is not None and (isinstance(b, bool) or not isinstance(b, int) or b <= 0):
            raise ValueError(f"{self.representation}: budget must be a positive int or None, got {b!r}")
        if self.representation in UNBUDGETED and b is not None:
            raise ValueError(f"{self.representation} takes no budget")
        if self.readers is not None:
            object.__setattr__(self, "readers", tuple(self.readers))
        if self.compressor is not None and self.representation not in QUERY_BUILT:
            raise ValueError(f"{self.representation}: only {sorted(QUERY_BUILT)} take a compressor")

    @property
    def key(self) -> str:
        key = f"{self.representation}@{'full' if self.budget_tokens is None else self.budget_tokens}"
        return key + (f"/{self.compressor}" if self.compressor else "")

    def identity(self) -> dict:
        """The condition as the run identity binds it: its fields, `compressor` only when one is named, so a condition
        without a compressor has the same identity as before the field was added (the paper's manifests carry no
        compressor key for those conditions)."""
        out = asdict(self)
        if self.compressor is None:
            del out["compressor"]
        return out

    def memory_slot(self, reader_id: str) -> str:
        """The memory `reader_id` reads under this condition: an M6 condition without a named compressor has one per
        reader (each compresses for itself), "M6@B/<reader>"; every other condition one per item, its key."""
        if self.representation in QUERY_BUILT and self.compressor is None:
            return f"{self.key}/{reader_id}"
        return self.key


def in_k1_slice(cond: Condition) -> bool:
    """The 4k slice that a `k1_slice` run re-reads over the primary population too: the unbudgeted conditions
    (B0, M0@full) and every condition at K1_SLICE_BUDGET, whatever its builder or compressor."""
    return cond.budget_tokens is None or cond.budget_tokens == K1_SLICE_BUDGET


def parse_condition(text: str) -> Condition:
    """CLI form "REP[@BUDGET|@full][/COMPRESSOR][:reader1,reader2]", e.g. "M2@4000", "B0",
    "M0@full:B1_dsv41flash_hf_deepinfra", "M6@4000" (each reader compresses), "M6@4000/B1_dsv41flash_hf_deepinfra"."""
    spec, _, readers = text.partition(":")
    spec, _, compressor = spec.partition("/")
    rep, _, budget = spec.partition("@")
    return Condition(rep, None if budget in ("", "full") else int(budget),
                     tuple(readers.split(",")) if readers else None, compressor or None)


def corpus_text(rel: str) -> str:
    """Text of a corpus file named as the item files name it (data/corpus/<family>/<file>.txt, relative to the
    repository root; rebuilt from the manifest's pointers by scripts/corpus/rebuild.py)."""
    return (ROOT / rel).read_bytes().decode("utf-8")


def lf_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def reader_digest() -> str:
    """Content hash of this package's *.py files (procedures/matrix, LF-normalised), recorded per invocation."""
    base = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(base.glob("*.py")):
        h.update(path.name.encode() + b"\0" + path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def _json_bytes(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=1, sort_keys=True) + "\n").encode("utf-8")


def _write_addressed(run_dir: Path, stem: str, suffix: str, data: bytes) -> str:
    name = f"{stem}_{hashlib.sha256(data).hexdigest()[:12]}{suffix}"
    (run_dir / name).write_bytes(data)
    return name


def _latest(path: Path) -> dict[str, dict]:
    """request_hash -> latest record of a ledger file (later lines supersede earlier ones, as in
    procedures.readers.ledger)."""
    if not path.exists():
        return {}
    return {r["request_hash"]: r for r in read_ledger(path)[1]}


def _spent(run_dir: Path) -> dict[str, float]:
    """Committed spend per ledger file of the run: usd_counted over every line (a superseded no_response may still
    have been billed) plus the reservations its send log holds open (in flight, or left by an executor that died)."""
    out = {}
    for path in sorted(run_dir.glob("*__*.jsonl")):
        log = SendLog(sends_path(path))
        log.refresh()
        out[path.name] = sum(float(r["usd_counted"]) for r in read_ledger(path)[1]) + log.in_flight_usd()
    return out


def _answered(records: dict, request_hash: str) -> dict | None:
    record = records.get(request_hash)
    return record if record is not None and record["outcome"] == "observed" else None


def pending_builds(memories: dict, build_records: dict, source_records: dict | None = None) -> dict:
    """The LLM build calls the planned (dry-run) memories still need. procedures.memory's MemoryResult.build lists each
    sub-block's cap levels with their request hash, block key and HIGH price (`blocks[].levels`). A level is answered
    when its request has an observed record in the run's build ledger (`build_records`), or importable when it has one
    in `source_records` (another run's build ledger); the next level is needed only when that answer escalates
    (procedures.memory.summary.escalates). Returns {"live": {block key: (cap level, HIGH usd)} of the requests to send,
    "possible": the same for the levels above a live one (sent only if it comes back truncated),
    "imports": {request_hash: source record}}."""
    live, possible, imports = {}, {}, {}
    for m in memories.values():
        if m.representation not in BUILT_ONCE:
            continue
        for block in (m.build or {}).get("blocks", []):
            if block["mode"] != "dry_run":
                continue
            levels = block["levels"]
            for i, level in enumerate(levels):
                record = _answered(build_records, level["request_hash"])
                if record is None and source_records:
                    record = _answered(source_records, level["request_hash"])
                    if record is not None:
                        imports[level["request_hash"]] = record
                if record is None:
                    live[level["block_key"]] = (level["cap_level"], level["usd"])
                    possible.update((v["block_key"], (v["cap_level"], v["usd"])) for v in levels[i + 1:])
                    break
                if not escalates(record):
                    break
    return {"live": live, "possible": possible, "imports": imports}


def run_manifest(run_dir: Path) -> dict:
    """The manifest.json of a run directory, wherever the directory is and whatever its name."""
    path = Path(run_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"{run_dir} is not a run directory: it has no manifest.json")
    return json.loads(path.read_text(encoding="utf-8"))


def reuse_source(run_dir: Path, builder_id: str, header: dict) -> tuple[str, dict]:
    """(run id, request_hash -> latest record) of another run's build ledger, for `reuse_build_from`. Refuses unless the
    directory is a run (it has a manifest.json) and its build ledger of `builder_id` has the run identity of `header`
    (procedures.readers.ledger.identity_clash: profile id + fingerprint, price book, user_message_suffix)."""
    run_dir = Path(run_dir)
    run_manifest(run_dir)
    path = run_dir / f"build__{builder_id}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: {run_dir.name} has no build ledger of {builder_id} to reuse")
    source_header, records = read_ledger(path)
    clash = identity_clash(source_header, header)
    if clash:
        raise LedgerError(f"{path}: build ledger identity differs from this run's: {clash}; nothing is imported")
    return run_dir.name, {r["request_hash"]: r for r in records}


def imported(record: dict, run_id: str) -> dict:
    """A build record copied from run `run_id`: the same request and answer, marked `imported_from`, and no spend of
    the importing run (usd_counted 0; the source's count kept as imported_usd_counted), so no spend is counted twice
    across runs."""
    return {**record, "imported_from": run_id, "usd_counted": 0.0,
            "imported_usd_counted": record.get("imported_usd_counted", record["usd_counted"])}


def _check_memory(memory, item: dict, cond: Condition) -> None:
    if memory.representation != cond.representation or memory.budget_tokens != cond.budget_tokens:
        raise ValueError(f"{item['item_id']}: build_memory returned {memory.representation}@{memory.budget_tokens} "
                         f"for {cond.key}")
    if not isinstance(memory.cache_key, str) or not memory.cache_key:
        raise ValueError(f"{item['item_id']}: {cond.key} memory without a cache_key")


def m6_slots(conditions: list[Condition], reader_ids) -> list[tuple[Condition, str, str]]:
    """(condition, memory slot, compressor) of every M6 memory an item gets: one for a named compressor, else one per
    reader the condition is for (Condition.memory_slot)."""
    out = []
    for cond in conditions:
        if cond.representation not in QUERY_BUILT:
            continue
        for pid in ((None,) if cond.compressor else (cond.readers or tuple(reader_ids))):
            out.append((cond, cond.memory_slot(pid), cond.compressor or pid))
    return out


def checked_timeouts(overrides: dict | None, senders: dict) -> dict[str, float]:
    """{profile id: seconds} of the client read-timeout overrides (procedures.readers.executor.timeout_overrides),
    each naming a profile this run sends (reader, compressor or builder: a typo would silently change nothing) and
    raising its panel timeout. Transport only: not part of the run identity; recorded in invocations.jsonl."""
    stray = sorted(set(overrides or {}) - set(senders))
    if stray:
        raise ValueError(f"timeout overrides name profiles this run does not send: {stray} (senders: {sorted(senders)})")
    return {pid: raised_timeout(pid, seconds, senders[pid].client_profile().timeout)
            for pid, seconds in sorted((overrides or {}).items())}


def checked_workers(overrides: dict | None, senders: dict) -> dict[str, int]:
    """{profile id: n} of the in-flight limit overrides (procedures.readers.executor.workers_overrides), each naming
    a profile this run sends and raising its panel max_workers (at most WORKERS_OVERRIDE_MAX). Transport only: not
    part of the run identity; recorded in invocations.jsonl."""
    stray = sorted(set(overrides or {}) - set(senders))
    if stray:
        raise ValueError(f"workers overrides name profiles this run does not send: {stray} (senders: {sorted(senders)})")
    return {pid: raised_workers(pid, n, senders[pid].client_profile().max_workers)
            for pid, n in sorted((overrides or {}).items())}


def plan_compressions(by_id: dict, conditions: list[Condition], corpus_index, build_memory, reader_ids) -> dict:
    """(item_id, memory slot) -> M6 memory, planned as a dry run, for every item x M6 slot (`m6_slots`); the live
    compressions go through `compress_live`."""
    out = {}
    for cond, slot, compressor in m6_slots(conditions, reader_ids):
        for iid, item in by_id.items():
            memory = build_memory(item, cond.representation, cond.budget_tokens, corpus_index,
                                  compressor_profile=compressor)
            _check_memory(memory, item, cond)
            out[(iid, slot)] = memory
    return out


def plan_memories(by_id: dict, conditions: list[Condition], corpus_index, build_memory, builder_id: str | None,
                  ledger=None, *, reader_ids=()) -> dict:
    """(item_id, memory slot) -> memory for every item x condition (slot = the condition key; M6: `m6_slots`, planned
    as dry runs). A built-once representation (M5 / M7) gets `builder_id` and `ledger` (None = planned as a dry run:
    nothing is sent)."""
    out = {}
    for cond in conditions:
        if cond.representation in QUERY_BUILT:
            continue
        built = cond.representation in BUILT_ONCE
        for iid, item in by_id.items():
            memory = build_memory(item, cond.representation, cond.budget_tokens, corpus_index,
                                  builder_profile=builder_id if built else None, ledger=ledger if built else None)
            _check_memory(memory, item, cond)
            out[(iid, cond.key)] = memory
    return {**out, **plan_compressions(by_id, conditions, corpus_index, build_memory, reader_ids)}


def compression_over_ceiling(memories: dict, panel) -> dict:
    """(item_id, slot) -> reason of the M6 memories whose compression request (reference input tokens + output cap)
    its compressor cannot take (procedures.readers.cells.infeasible_reason: the local ceiling of its access tier,
    the compressor running where it is deployed, or its route's context)."""
    out = {}
    for key, m in memories.items():
        if m.representation not in QUERY_BUILT or m.build["mode"] == "no_input":
            continue
        reason = infeasible_reason(panel.get(m.build["compressor_profile"]), m.build["input_tokens"],
                                   m.build["max_output_tokens"])
        if reason is not None:
            out[key] = reason
    return out


def timed_out(memory) -> bool:
    """An M6 memory whose compression timed out (procedures.memory.compressor mode route_timeout)."""
    return memory.representation in QUERY_BUILT and (memory.build or {}).get("mode") == ROUTE_TIMEOUT


def route_timeouts(memories: dict, run_dir: Path) -> dict:
    """(item_id, slot) -> route_timeout memory (procedures.memory.build.route_timeout_memory) of every planned
    (dry-run) M6 memory whose compression request's latest record in its compressor's query ledger in `run_dir` timed
    out on every attempt (procedures.readers.executor.route_timed_out). Such a compression is settled: not sent again,
    its reader cells not sent."""
    ledgers: dict[str, dict] = {}
    out = {}
    for key, m in memories.items():
        if m.representation not in QUERY_BUILT or m.build["mode"] != "dry_run":
            continue
        compressor = m.build["compressor_profile"]
        path = Path(run_dir) / f"query__{compressor}.jsonl"
        if compressor not in ledgers:
            ledgers[compressor] = _latest(path)
        record = ledgers[compressor].get(m.build["request_hash"])
        if route_timed_out(record):
            out[key] = route_timeout_memory(m, path, record)
    return out


def compress_live(by_id: dict, conditions: list[Condition], corpus_index, build_memory, reader_ids, compressor: str,
                  ledger: BuilderLedger, planned: dict, skip=frozenset()) -> tuple[dict, str | None]:
    """((item_id, slot) -> memory of every M6 slot `compressor` makes, built live through its query `ledger`; the
    RouteTripped message that closed the compressor's route, or None). A planned memory already in route_timeout form
    (`route_timeouts`), or whose request timed out earlier in this pass (an identical request), is kept as such and not
    sent. A compression that times out on every attempt comes back as a route_timeout memory
    (procedures.memory.compressor). When the compressor's route closes (RouteTripped), the compressions its ledger has
    answered are still read back (nothing is sent) and every other one becomes a route_timeout memory for this
    invocation: its reader cells are not sent, and a rerun sends it again unless its own latest record timed out.
    `skip`: (item_id, slot) of the compressions over their compressor's context ceiling (not sent)."""
    out, closed = {}, None
    latest: dict = {}           # the query ledger's latest records, read again after a timeout or a trip
    for cond, slot, maker in m6_slots(conditions, reader_ids):
        if maker != compressor:
            continue
        for iid, item in by_id.items():
            key = (iid, slot)
            if key in skip:
                continue
            plan = planned[key]
            hash_ = (plan.build or {}).get("request_hash")
            if timed_out(plan):
                out[key] = plan
                continue
            if plan.build["mode"] != "no_input" and (route_timed_out(latest.get(hash_))
                                                     or (closed and _answered(latest, hash_) is None)):
                out[key] = route_timeout_memory(plan, ledger.path, latest.get(hash_))
                continue
            try:
                memory = build_memory(item, cond.representation, cond.budget_tokens, corpus_index,
                                      compressor_profile=compressor, ledger=ledger)
            except RouteTripped as exc:
                closed = str(exc)
                latest = _latest(Path(ledger.path))
                memory = route_timeout_memory(plan, ledger.path, latest.get(hash_))
            else:
                if timed_out(memory):
                    latest = _latest(Path(ledger.path))
            _check_memory(memory, item, cond)
            out[key] = memory
    return out, closed


def pending_queries(memories: dict, answered: dict, skip=frozenset()) -> dict[str, dict[str, float]]:
    """compressor -> {request_hash: HIGH usd} of the planned (dry-run) M6 compressions not answered in its query ledger
    (`answered`: compressor -> answered request hashes); identical requests count once; `skip` as compress_live."""
    out: dict[str, dict[str, float]] = {}
    for key, m in memories.items():
        if m.representation not in QUERY_BUILT or key in skip or m.build["mode"] != "dry_run":
            continue
        compressor = m.build["compressor_profile"]
        if m.build["request_hash"] not in answered.get(compressor, set()):
            out.setdefault(compressor, {})[m.build["request_hash"]] = m.build["usd"]
    return out


def plan_grid(by_id: dict, conditions: list[Condition], readers: dict, memories: dict,
              builder_id: str | None, repeats: int = 1, repeated=(), infeasible=None) -> dict[str, list]:
    """reader profile id -> its reader cells (procedures.readers.cells.ReaderCell): repeat 0 of every item
    (conditions in order, items by id), then repeats 1..k-1 of the `repeated` item ids in the same order;
    `repeated` is one id set for every condition or a mapping condition key -> ids (`repeat_sets`, every condition
    keyed). The stage names the builder (M5 / M7) or the compressor (M6); a cell whose memory is in `infeasible`
    ((item_id, slot) -> reason: a compression its compressor cannot take, `compression_over_ceiling`) is infeasible,
    reason "compression_<reason>"; a cell whose M6 memory timed out (procedures.memory.compressor mode route_timeout) is
    infeasible, reason ROUTE_TIMEOUT, and not memory_empty (it is not read)."""
    infeasible = infeasible or {}
    def profile(cond, pid):
        if cond.representation in BUILT_ONCE:
            return builder_id
        return (cond.compressor or pid) if cond.representation in QUERY_BUILT else None

    def cell(pid, pp, repeat_idx, cond, iid):
        slot = cond.memory_slot(pid)
        memory = memories[(iid, slot)]
        rc = build_cell(by_id[iid], memory, pp, repeat_idx=repeat_idx, builder_profile_id=profile(cond, pid))
        if timed_out(memory):
            return replace(rc, feasible=False, infeasible_reason=ROUTE_TIMEOUT, memory_empty=False)
        if (iid, slot) not in infeasible:
            return rc
        return replace(rc, feasible=False, infeasible_reason=f"compression_{infeasible[(iid, slot)]}")

    per_condition = repeated if isinstance(repeated, dict) else {c.key: repeated for c in conditions}
    wanted = {c.key: set(per_condition[c.key]) for c in conditions}

    def cells(pid, pp, repeat_idx):
        return [cell(pid, pp, repeat_idx, cond, iid)
                for cond in conditions if cond.readers is None or pid in cond.readers
                for iid in by_id if repeat_idx == 0 or iid in wanted[cond.key]]
    return {pid: [rc for r in range(repeats) for rc in cells(pid, pp, r)] for pid, pp in readers.items()}


def run_matrix(items: list[dict], conditions: list[Condition], reader_ids: list[str], corpus_index, *,
               build_memory, panel, label: str, cap_usd: float, corpus_index_sha256: str, dry_run: bool = True,
               builder_id: str | None = None, out_root: Path = RUNS_ROOT, book=None, transport=None, sleep=None,
               concurrency: int | None = None, text_of=None, send_infeasible: bool = False,
               build_only: bool = False, reuse_build_from: Path | None = None, repeats: int = 1,
               repeat_fraction: float = REPEAT_FRACTION, corpus_manifest: dict | None = None, approval=None,
               stage_cap_usd: float | None = None, k1_slice: bool = False,
               timeout_overrides: dict | None = None, workers_overrides: dict | None = None) -> dict:
    book = book or load_price_book()
    text_of = text_of or corpus_text
    if corpus_manifest is not None and set(corpus_manifest) != {"path", "sha256"}:
        raise ValueError(f"corpus_manifest must be {{path, sha256}}, got {sorted(corpus_manifest)}")
    if LABEL.fullmatch(label) is None:
        raise ValueError(f"label {label!r} must match {LABEL.pattern}")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")
    ids = [it["item_id"] for it in items]
    if not ids or not conditions or not reader_ids:
        raise ValueError("a matrix needs items, conditions and readers")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate item ids")
    if len({c.key for c in conditions}) != len(conditions):
        raise ValueError("duplicate conditions")
    readers = {pid: panel.get(pid) for pid in dict.fromkeys(reader_ids)}
    stray = {c.key: sorted(set(c.readers) - set(readers)) for c in conditions if c.readers}
    if any(stray.values()):
        raise ValueError(f"conditions name readers outside the run: {stray}")
    built_once = [c for c in conditions if c.representation in BUILT_ONCE]
    queried = [c for c in conditions if c.representation in QUERY_BUILT]
    if bool(built_once) != (builder_id is not None):
        raise ValueError("a builder profile is required exactly when a condition is built by the run's builder "
                         f"({sorted(BUILT_ONCE)})")
    if build_only and not built_once:
        raise ValueError(f"build_only needs a condition built by the run's builder ({sorted(BUILT_ONCE)})")
    if reuse_build_from is not None and not built_once:
        raise ValueError(f"reuse_build_from needs a condition built by the run's builder ({sorted(BUILT_ONCE)})")
    if stage_cap_usd is not None and (not build_only or isinstance(stage_cap_usd, bool)
                                      or not isinstance(stage_cap_usd, (int, float)) or not stage_cap_usd >= 0):
        raise ValueError(f"stage_cap_usd (a non-negative number) is for build_only runs, got {stage_cap_usd!r}")
    if approval is not None and label in approval.arms and cap_usd > approval.arms[label]:
        raise ValueError(f"cap ${cap_usd:.4f} exceeds arm {label!r} of approval {approval.approval_id!r} "
                         f"(${approval.arms[label]:.4f})")
    for pid in readers:
        compressions = [(c.budget_tokens, c.compressor or pid) for c in queried if c.readers is None or pid in c.readers]
        if len(set(compressions)) != len(compressions):
            raise ValueError(f"two M6 conditions give {pid} the same compressor at the same budget")
    compressors = {c: panel.get(c) for _, _, c in m6_slots(queried, list(readers))}
    builder = panel.get(builder_id) if builder_id else None
    senders = {**readers, **compressors, **({builder_id: builder} if builder else {})}
    timeouts = checked_timeouts(timeout_overrides, senders)
    in_flight = checked_workers(workers_overrides, senders)
    by_id = {it["item_id"]: it for it in sorted(items, key=lambda it: it["item_id"])}
    repeated, repeat_record = repeat_sets(by_id, conditions, repeats, repeat_fraction, k1_slice)
    memories = plan_memories(by_id, conditions, corpus_index, build_memory, builder_id, reader_ids=list(readers))
    planned = memories      # the memory request map is written from the plan (replan_memories gives the same map)
    over_ceiling = compression_over_ceiling(memories, panel)
    skip = frozenset() if send_infeasible else frozenset(over_ceiling)
    identity = {
        "schema": SCHEMA, "label": label, "items_sha256": hashlib.sha256(jsonl_bytes(list(by_id.values()))).hexdigest(),
        "n_items": len(by_id), "conditions": [c.identity() for c in conditions],
        "readers": {pid: {"fingerprint": pp.fingerprint, "tier": pp.tier, "user_message_suffix": pp.user_message_suffix}
                    for pid, pp in readers.items()},
        "repeats": repeat_record,
        "builder": {builder.profile_id: builder.fingerprint} if builder else None,
        "panel_sha256": panel.sha256, "price_book_sha256": book.sha256,
        "corpus_index_sha256": corpus_index_sha256,
        "scorer_sha256": {rel: lf_sha256(ROOT / rel) for rel in SCORER_FILES},
        "templates_sha256": template_digest(), "send_infeasible": send_infeasible,
        "memory_cache_keys": {f"{iid}|{key}": m.cache_key for (iid, key), m in sorted(memories.items())},
        "memory_text_sha256": {f"{iid}|{key}": hashlib.sha256(m.text.encode("utf-8")).hexdigest()
                               for (iid, key), m in sorted(memories.items())
                               if m.representation not in LLM_BUILT},
    }
    if queried:
        identity["compressors"] = {pid: pp.fingerprint for pid, pp in sorted(compressors.items())}
    if corpus_manifest is not None:
        identity["corpus_manifest"] = dict(corpus_manifest)
    manifest = _json_bytes(identity)
    run_id = f"{label}_{hashlib.sha256(manifest).hexdigest()[:12]}"
    run_dir = Path(out_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "manifest.json").write_bytes(manifest)
    ap = ApprovalGuard(approval, run_dir, label) if approval is not None else None
    invocation = {"started_at": utc_now(), "mode": "dry_run" if dry_run else "live", "cap_usd": cap_usd,
                  "reader_digest": reader_digest(), "infer_harness_digest": harness_digest(),
                  **({"approval_id": approval.approval_id} if approval is not None else {}),
                  **({"stage_cap_usd": stage_cap_usd} if stage_cap_usd is not None else {}),
                  **({"timeout_overrides": timeouts} if timeouts else {}),
                  **({"workers_overrides": in_flight} if in_flight else {})}

    def log(**fields) -> None:
        with open(run_dir / "invocations.jsonl", "a", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps({**invocation, **fields, "finished_at": utc_now()}, ensure_ascii=False,
                                    sort_keys=True) + "\n")

    def to_send(rcs: list, pp) -> list:
        return sendable_cells(rcs, pp, send_infeasible)

    settled = route_timeouts(memories, run_dir)     # compressions that timed out in an earlier invocation
    memories = {**memories, **settled}
    grid = plan_grid(by_id, conditions, readers, memories, builder_id, repeats, repeated, over_ceiling)
    estimates, pending_high = reader_estimates(grid, readers, run_dir, cap_usd, book, send_infeasible)
    source_id, source = None, {}
    if reuse_build_from is not None:
        try:
            source_id, source = reuse_source(reuse_build_from, builder_id, ledger_header(builder, book))
        except (OSError, ValueError, LedgerError) as exc:
            log(outcome="refused", reuse_build_from=str(reuse_build_from), error=repr(exc))
            raise
    reuse = {"imported_from": source_id} if source_id else {}
    builds = pending_builds(memories, _latest(run_dir / f"build__{builder_id}.jsonl") if builder_id else {}, source)
    imports = builds["imports"]
    build_high = sum(usd for _, usd in builds["live"].values())
    possible_high = sum(usd for _, usd in builds["possible"].values())
    queries = pending_queries(memories, {c: {h for h, r in _latest(run_dir / f"query__{c}.jsonl").items()
                                             if r["outcome"] == "observed"} for c in compressors}, skip)
    query_high = sum(sum(v.values()) for v in queries.values())
    spent = _spent(run_dir)
    estimate = {"run_id": run_id, "token_method": token_method(), "spent_usd": sum(spent.values()),
                "pending_usd_high": pending_high, "repeats": repeat_summary(repeat_record),
                "readers": estimates,
                "builds": {"builder": builder_id, "pending": len(builds["live"]),
                           "pending_by_cap_level": dict(sorted(Counter(str(level) for level, _ in
                                                                       builds["live"].values()).items())),
                           "usd_high": build_high, "escalations_possible": len(builds["possible"]),
                           "escalations_possible_usd_high": possible_high,
                           "imports": {"from": source_id, "records": len(imports), "finish_reasons": dict(sorted(
                               Counter(str(r.get("finish_reason")) for r in imports.values()).items()))}
                           if source_id else None},
                "queries": {"pending": sum(len(v) for v in queries.values()), "usd_high": query_high,
                            "by_compressor": {c: {"pending": len(v), "usd_high": sum(v.values())}
                                              for c, v in sorted(queries.items())},
                            "over_ceiling": len(over_ceiling), **route_timeout_counts(memories)}
                if queried else None,
                "note": "LLM-built memories are planned as dry runs: their reader cells are sized at the full budget "
                        "and stay pending in this estimate until built; the build cost is procedures.memory's own "
                        "HIGH estimate per sub-block request not yet answered in the build ledger (nor importable), "
                        "level by level up the cap ladder; a level above an unanswered one is a possible escalation, "
                        "counted apart and in the cap check. M6 compressions are per-question cost: "
                        "procedures.memory's HIGH estimate per distinct request not yet answered in its compressor's "
                        "query ledger."}
    estimate_name = _write_addressed(run_dir, "estimate", ".json", _json_bytes(estimate))
    report = {"run_id": run_id, "run_dir": str(run_dir), "mode": invocation["mode"], "estimate": estimate_name,
              "spent_usd": sum(spent.values()), "pending_usd_high": pending_high, "build_usd_high": build_high,
              "build_usd_high_possible": possible_high, "builds_pending": len(builds["live"]),
              "query_usd_high": query_high, "queries_pending": sum(len(v) for v in queries.values()),
              "imports": len(imports) if source_id else None, **route_timeout_counts(memories)}
    need = build_high + possible_high if build_only else pending_high + build_high + possible_high + query_high
    if ap is not None:
        report["approval"] = {"preflight_usd": need, "stage": "build" if build_only else "full", **ap.tally()}
    if dry_run:
        log(outcome="dry_run", estimate=estimate_name, **reuse, **({"importable": len(imports)} if source_id else {}))
        return report
    build_name = f"build__{builder_id}.jsonl"
    try:
        if build_only:
            stage_spent = spent.get(build_name, 0.0)
            if stage_cap_usd is not None and stage_spent + need > stage_cap_usd:
                raise SpendCapExceeded(f"build stage: spent ${stage_spent:.4f} + pending builds ${build_high:.4f} + "
                                       f"possible escalations ${possible_high:.4f} (HIGH) exceeds the stage cap "
                                       f"${stage_cap_usd:.4f}; nothing was built or sent")
            if sum(spent.values()) + need > cap_usd:
                raise SpendCapExceeded(f"build stage: spent ${sum(spent.values()):.4f} + pending builds "
                                       f"${build_high:.4f} + possible escalations ${possible_high:.4f} (HIGH) exceeds "
                                       f"cap ${cap_usd:.4f}; nothing was built or sent")
        elif sum(spent.values()) + need > cap_usd:
            raise SpendCapExceeded(f"spent ${sum(spent.values()):.4f} + pending reads ${pending_high:.4f} + pending "
                                   f"builds ${build_high:.4f} + possible escalations ${possible_high:.4f} + pending "
                                   f"compressions ${query_high:.4f} (HIGH) exceeds cap ${cap_usd:.4f}; nothing was "
                                   "built or sent")
        if ap is not None:
            ap.check(need, f"{'build stage' if build_only else 'pending reads + builds + compressions'} (HIGH)")
            ap.bind()
    except SpendCapExceeded:
        log(outcome="refused", estimate=estimate_name, **reuse)
        raise
    guard = SpendGuard(approval=ap)
    sent, tripped = {}, {}
    if source_id:
        reuse["imported"] = 0
    try:
        with guarded(guard), overriding_timeouts(timeouts), overriding_workers(in_flight):
            if built_once:
                if imports:
                    ledger = Ledger(run_dir / build_name, ledger_header(builder, book), cap_usd=cap_usd)
                    for record in imports.values():
                        ledger.append(imported(record, source_id))
                    reuse["imported"] = len(imports)
                others = sum(v for k, v in _spent(run_dir).items() if k != build_name)
                build_cap = cap_usd - others - (0.0 if build_only else pending_high)
                if stage_cap_usd is not None:
                    build_cap = min(build_cap, stage_cap_usd)
                live = plan_memories(by_id, built_once, corpus_index, build_memory, builder_id,
                                     BuilderLedger(run_dir / build_name, max(0.0, build_cap), transport=transport,
                                                   sleep=sleep))
                memories = {**memories, **_checked_live(live, memories, ("live",))}
                if build_only:
                    log(outcome="built", estimate=estimate_name, builds=_spent(run_dir), breaker=guard.breaker.state(),
                        **reuse)
                    return {**report, "mode": "build_only", "spent_usd": sum(_spent(run_dir).values()),
                            **({"approval": ap.tally()} if ap is not None else {})}
            for compressor in sorted(compressors):
                query_name = f"query__{compressor}.jsonl"
                others = sum(v for k, v in _spent(run_dir).items() if k != query_name)
                ledger = BuilderLedger(run_dir / query_name, max(0.0, cap_usd - others - pending_high),
                                       transport=transport, sleep=sleep)
                live, closed = compress_live(by_id, queried, corpus_index, build_memory, list(readers), compressor,
                                             ledger, memories, skip)
                if closed is not None:      # this compressor's route is closed; the other compressors go on
                    tripped[f"compressions by {compressor}"] = closed
                memories = {**memories, **_checked_live(live, memories, ("live", "no_input", ROUTE_TIMEOUT))}
            if built_once or queried:
                grid = plan_grid(by_id, conditions, readers, memories, builder_id, repeats, repeated, over_ceiling)
            if {k for k, m in memories.items() if timed_out(m)} != set(settled):
                # compressions timed out in this invocation: the reader estimate as the readers now stand (their
                # route_timeout cells are not sent), written beside the planning one and named by the log lines
                after, after_high = reader_estimates(grid, readers, run_dir, cap_usd, book, send_infeasible)
                estimate_name = _write_addressed(run_dir, "estimate", ".json", _json_bytes({
                    **estimate, "readers": after, "pending_usd_high": after_high,
                    "queries": {**estimate["queries"], **route_timeout_counts(memories)},
                    "after_compressions": {"planned_estimate": estimate_name,
                                           "note": "reader cells re-planned after the live compressions; builds and "
                                                   "queries as planned, except the route_timeout counts"}}))
            for pid, pp in readers.items():
                spent = _spent(run_dir)
                ledger_name = f"read__{pid}.jsonl"
                others = sum(v for k, v in spent.items() if k != ledger_name)
                try:
                    out = run_cells([rc.cell for rc in to_send(grid[pid], pp)], pp, cap_usd=max(0.0, cap_usd - others),
                                    ledger_path=run_dir / ledger_name, book=book, transport=transport, sleep=sleep,
                                    concurrency=concurrency, guard=guard)
                except RouteTripped as exc:     # this reader's route is closed; the other readers go on
                    tripped[pid] = str(exc)
                    continue
                sent[pid] = {"sent": out["sent"], "skipped_done": out["skipped_done"],
                             **({"recovered": len(out["recovered"])} if out["recovered"] else {})}
            if tripped:
                raise RouteTripped(f"routes closed by the circuit breaker in this run: {tripped}")
    except Exception as exc:
        log(outcome="stopped", estimate=estimate_name, sent=sent, error=repr(exc), breaker=guard.breaker.state(),
            **route_timeout_counts(memories), **reuse)
        raise
    outputs = score_run(run_id, run_dir, by_id, readers, grid, text_of, planned, repeats, send_infeasible)
    log(outcome="live", estimate=estimate_name, sent=sent, outputs=outputs, breaker=guard.breaker.state(),
        **route_timeout_counts(memories), **reuse)
    return {**report, "estimate": estimate_name, "sent": sent, "outputs": outputs,
            "spent_usd": sum(_spent(run_dir).values()), **route_timeout_counts(memories),
            **({"approval": ap.tally()} if ap is not None else {})}


def infeasible_by_reason(rcs: list) -> dict[str, int]:
    """Infeasible reader cells by reason (procedures.readers.cells.ReaderCell.infeasible_reason), sorted."""
    return dict(sorted(Counter(rc.infeasible_reason for rc in rcs if not rc.feasible).items()))


def route_timeout_counts(memories: dict) -> dict:
    """{"route_timeout": {compressor: M6 memories that timed out}} when there is one (the estimate's `queries`), else
    {} (an estimate without one keeps its bytes)."""
    counts = Counter(m.build["compressor_profile"] for m in memories.values() if timed_out(m))
    return {"route_timeout": dict(sorted(counts.items()))} if counts else {}


def sendable_cells(rcs: list, pp, send_infeasible: bool = False) -> list:
    """The reader cells a reader sends: the feasible ones (all with `send_infeasible`), one per distinct request."""
    return distinct_requests([rc for rc in rcs if rc.feasible or send_infeasible], pp)


def reader_estimates(grid: dict, readers: dict, run_dir: Path, cap_usd: float, book,
                     send_infeasible: bool = False) -> tuple[dict, float]:
    """(reader id -> the estimate's reader block, the HIGH usd of every pending read): cell counts (infeasible by
    stage and by reason, repeats, shared requests) and the LOW / HIGH price of the sendable requests not yet
    answered in the reader's ledger in `run_dir`."""
    estimates, pending_high = {}, 0.0
    for pid, pp in readers.items():
        answered = {h for h, r in _latest(Path(run_dir) / f"read__{pid}.jsonl").items() if r["outcome"] == "observed"}
        sendable = sendable_cells(grid[pid], pp, send_infeasible)
        pending = [rc.cell for rc in sendable if cell_request_hash(rc.cell, pp) not in answered]
        est = {s: run_cells(pending, pp, cap_usd=cap_usd, dry_run=True, book=book, assumptions=scenario(pp, s))
               for s in SCENARIOS}
        estimates[pid] = {"cells": len(grid[pid]), "infeasible": sum(not rc.feasible for rc in grid[pid]),
                          "infeasible_by_stage": dict(sorted(Counter(rc.cell.stage for rc in grid[pid]
                                                                     if not rc.feasible).items())),
                          "infeasible_by_reason": infeasible_by_reason(grid[pid]),
                          "repeat_cells": sum(rc.cell.repeat_idx > 0 for rc in grid[pid]),
                          "shared_requests": sum(rc.feasible or send_infeasible for rc in grid[pid]) - len(sendable),
                          "pending": len(pending),
                          **{f"usd_{s}": est[s]["usd"] for s in SCENARIOS},
                          "usd_upper_bound": est["high"]["usd_upper_bound"],
                          "input_tokens_high": est["high"]["input_tokens"],
                          "by_stage_high": {k: {"calls": v["calls"], "usd": v["usd"]}
                                            for k, v in sorted(est["high"]["by_stage"].items())}}
        pending_high += est["high"]["usd"]
    return estimates, pending_high


def _checked_live(live: dict, planned: dict, modes: tuple) -> dict:
    """The live LLM-built memories, each checked against its plan: the same cache_key, and built live (`modes`: M6
    also allows no_input, a question without retrieved chunks, and route_timeout, a compression that timed out on
    every attempt); else the run stops (the mechanism sanity check)."""
    drift = sorted(f"{iid}|{key}" for (iid, key), m in live.items() if m.cache_key != planned[(iid, key)].cache_key)
    if drift:
        raise ValueError(f"live LLM-built memories changed cache_key after planning: {drift[:5]}")
    unbuilt = sorted(f"{iid}|{key}: {m.notes}" for (iid, key), m in live.items()
                     if m.representation in LLM_BUILT and (m.build or {}).get("mode") not in modes)
    if unbuilt:
        raise RuntimeError(f"LLM-built memories were not built live (mechanism sanity check): {unbuilt[:5]}")
    return live


def distinct_requests(rcs: list, pp) -> list:
    """The first cell of each distinct request, in grid order. Cells with identical messages, generation and
    repeat_idx are one request (one request hash): e.g. M2 or M4 at 8k and 16k when every scored chunk or extracted
    sentence already fits in 8k. One send answers them all; run_cells refuses duplicate hashes. A cell whose memory is
    still planned (ReaderCell.memory_planned: an LLM-built memory not built yet) is its own request per memory
    (cache_key): the built texts can differ, so an estimate never merges two memories on their stand-in text (the
    stand-ins name their budget, so run_cells would refuse two that coincide)."""
    seen, out = set(), []
    for rc in rcs:
        request = (cell_request_hash(rc.cell, pp), rc.memory_cache_key if rc.memory_planned else None)
        if request not in seen:
            seen.add(request)
            out.append(rc)
    return out


def score_run(run_id: str, run_dir: Path, by_id: dict, readers: dict, grid: dict, text_of, memories: dict,
              repeats: int = 1, send_infeasible: bool = False) -> dict:
    """Join every reader cell with its latest ledger record, score, and write the outputs (named after their content),
    the rows attributed to the memory request map of `memories` (the run's planned memories: a planned memory lists
    every cap level of its plan, a live one only the levels it sent, and the map is one file per run whichever pass
    writes it). A cell over its ceiling stays unanswered even when an identical feasible cell was sent (feasibility is
    at the cell's own budget); a cell answered by an earlier identical request is marked `shared_request`."""
    cells = {}
    for pid, pp in readers.items():
        latest = _latest(run_dir / f"read__{pid}.jsonl")
        metas, owners = [], set()
        for rc in grid[pid]:
            request = cell_request_hash(rc.cell, pp)
            record = latest.get(request) if rc.feasible or send_infeasible else None
            metas.append((rc.meta(), record, record is not None and request in owners))
            if record is not None:
                owners.add(request)
        cells[pid] = metas
    rows, summaries, answered = score_cells(run_id, by_id, readers, cells, text_of, repeats)
    rows, requests_file = attribute_memory_requests(run_dir, rows, memories.values())
    return {**write_scored(run_dir, run_id, rows, summaries, answered),
            **({"memory_requests": requests_file} if requests_file else {})}


def attribute_memory_requests(run_dir: Path, rows: list, memories) -> tuple[list, str | None]:
    """(rows, map file name): with an LLM-built memory among `memories`, the map is written into `run_dir`
    (procedures.memory.requests.write_memory_requests) and every row names it through
    score_path.with_memory_requests; every row's LLM-built memory must be in the map. Without one: the rows as they
    are and None."""
    memories = list(memories)
    mapped = set(memory_requests(memories)["memories"])
    if not mapped:
        return rows, None
    name = write_memory_requests(run_dir, memories)
    return name_memory_requests(rows, name, mapped), name


def name_memory_requests(rows: list, name: str, mapped) -> list:
    """`rows` naming the map file `name` (score_path.with_memory_requests over its memory keys `mapped`); an LLM-built
    row whose memory is not in the map raises."""
    out = [with_memory_requests(r, name, mapped) for r in rows]
    missing = sorted({(r["stage"], r["provenance"]["memory_cache_key"][:12]) for r in out
                      if r["representation"] in LLM_BUILT and r["provenance"]["memory_requests"] is None})
    if missing:
        raise ValueError(f"LLM-built rows whose memory is not in {name}: {missing[:5]}")
    return out


def replan_memories(manifest: dict, by_id: dict, corpus_index, build_memory) -> dict:
    """(item_id, slot) -> the LLM-built memories of a finished run, re-planned as dry runs from its manifest (same
    conditions, readers, builder; `by_id` = its items by id, a warning when they do not hash to the manifest's
    items_sha256), each checked against the cache key the manifest bound: the key binds the revisions' text hashes,
    prompt, parameters and every sub-block key, so an equal key means equal requests. {} when the run has no LLM-built
    condition."""
    ordered = {iid: by_id[iid] for iid in sorted(by_id)}
    digest = hashlib.sha256(jsonl_bytes(list(ordered.values()))).hexdigest()
    if digest != manifest["items_sha256"]:
        warnings.warn(f"the items hash to {digest[:12]}; the run manifest records {manifest['items_sha256'][:12]}",
                      stacklevel=2)
    conditions = [Condition(**c) for c in manifest["conditions"] if c["representation"] in LLM_BUILT]
    if not conditions:
        return {}
    builder = next(iter(manifest["builder"])) if manifest.get("builder") else None
    memories = plan_memories(ordered, conditions, corpus_index, build_memory, builder,
                             reader_ids=list(manifest["readers"]))
    bound = manifest["memory_cache_keys"]
    wrong = sorted(f"{iid}|{slot}" for (iid, slot), m in memories.items() if bound.get(f"{iid}|{slot}") != m.cache_key)
    if wrong:
        raise ValueError(f"{len(wrong)} re-planned memories differ from the run's cache keys, e.g. {wrong[:3]}; the "
                         "code or corpus no longer plans this run's requests")
    return memories


def score_cells(run_id: str, by_id: dict, readers: dict, cells: dict, text_of, repeats: int = 1,
                decisions: dict | None = None) -> tuple[list, dict, list]:
    """(ledger rows, summaries, answered scored rows) of `cells` (reader id -> [(cell meta, latest ledger record or
    None, shared_request)] in grid order). Each reader's answers go through
    procedures.scoring.score_path.score_entries with its panel flag `answer_from_reasoning_field` and the blind
    arbitration `decisions` (arb_id -> bool); each reader x condition summary reads repeat 0 and, with repeats, adds
    `repeats` (score_path.repeat_agreement)."""
    rows, summaries, answered_rows = [], {}, []
    for pid, pp in readers.items():
        metas = cells[pid]
        sent = [(m, r) for m, r, _ in metas if r is not None]
        scored = score_entries([{"item": by_id[m["item_id"]], "record": r} for m, r in sent], text_of, decisions,
                               items=list(by_id.values()), answer_from_reasoning_field=pp.answer_from_reasoning_field)
        scored_by = {(m["stage"], m["item_id"], m["repeat_idx"]): s for (m, _), s in zip(sent, scored)}
        answered_rows += scored
        by_stage: dict[str, tuple[list, list, list]] = {}
        for m, r, shared in metas:
            s = scored_by.get((m["stage"], m["item_id"], m["repeat_idx"]))
            row = ledger_row(run_id, by_id[m["item_id"]], m, pp, r, s, shared_request=shared)
            rows.append(row)
            bucket = by_stage.setdefault(m["stage"], ([], [], []))
            bucket[2].append(row)
            if m["repeat_idx"] == 0:
                bucket[1].append(row)
                if s is not None:
                    bucket[0].append(s)
        for stage, (stage_scored, stage_rows, every_row) in by_stage.items():
            summary = condition_summary(stage_scored, stage_rows)
            if repeats > 1:
                summary["repeats"] = repeat_agreement(every_row, repeats)
            blank = sum(bool(r["capability"] and r["capability"]["reasoning_only"]
                             and not r["capability"]["answer_from_reasoning_field"]) for r in every_row)
            if blank:
                logger.warning("%s %s: %d answered cell(s) with blank content and non-blank reasoning_content not read "
                               "as an answer, scored E-parse on the content", pid, stage, blank)
            summaries[f"{pid}|{stage}"] = summary
    return rows, summaries, answered_rows


def write_scored(run_dir: Path, run_id: str, rows: list, summaries: dict, answered_rows: list) -> dict:
    """The rows / summary / blind arbitration queue files of a scoring pass, each named after its content; an empty
    queue is no file ("arbitration": None)."""
    queue = blind_arbitration_queue(answered_rows)
    return {"rows": _write_addressed(run_dir, "rows", ".jsonl", jsonl_bytes(rows)),
            "summary": _write_addressed(run_dir, "summary", ".json", _json_bytes({"run_id": run_id,
                                                                                   "conditions": summaries})),
            "arbitration": _write_addressed(run_dir, "arbitration", ".jsonl", jsonl_bytes(queue)) if queue else None}

"""Reader-matrix runs as analysis observations.

`load_run(run_dir, rows=None, summary=None)`: one run directory of procedures.matrix.run_matrix (data/runs/<run>/,
whatever its name): its manifest.json, and the rows and summary files the last live invocation in invocations.jsonl
names, unless `rows` names another rows file of the directory (e.g. the rows of a re-scoring); with `rows` and no
`summary` the run has no summary to check against (the report lists it). The run's id is the one its rows record
(`run_id`, the same on every row; the directory name for rows without one). Every call ledger (read__*.jsonl,
build__*.jsonl, query__*.jsonl: one record per API call) is read with procedures.readers.ledger.read_ledger, latest
record per request hash (as the ledgers resume). The memory request map (procedures.memory.requests: memory cache_key ->
the build / compression requests that made it) is the file the rows name (provenance `memory_requests`), else the one
memory request map of the directory (memory_requests.json, or the memory_requests_<sha12>.json a run writes), else
none; two candidates raise.

`observations(runs, items, panel)`: one `Obs` per row of every run, joined with its item. Every item id a run's rows
name must be in the item file; when those items, sorted by id, do not hash (procedures.items.frozen.jsonl_bytes) to
the run manifest's items_sha256 (the items the run was planned with), a warning says so and the analysis goes on. A
(profile, condition, item, repeat) present in two runs raises: runs are combined, never overlaid.

Per observation:
* condition = the reader stage without "read/" (procedures.readers.cells.stage_name): "B0@full", "M2@4000",
  "M5@4000/<builder profile>"; `arm` = closed-book (B0), full-text (M0), zero-LLM (a memory made without an LLM:
  retrieval M2, constraint sentences M4), or "<tier>-built" by the builder profile's panel tier (an LLM-built
  memory: summary M5, query-focused compression M6, typed constraint records M7).
* cluster (the bootstrap unit: cells that share a branch point, a pair or a document are resampled together): L3 ->
  its unit_id, S3 -> its document, S1 / S2 / L2 -> its pair_id; prefixed by the stratum. `family` (the document
  family: the procedure owner, e.g. AESO, NERC, PJM) is the second-level sensitivity cluster. `layer`: an L3 cell's
  second-reader label (rule2_spirit: abstracted / transparent), None elsewhere.
* chance = the frozen scorer's guessing rate of the item's gold (procedures.scoring.score.gold_spec over the frozen
  item mapping procedures.items.frozen.scorer_item: 1/k on a lettered closed set, 0 on not_in_effect and open
  answers), as the run summaries use it; `lettered` = the side form.
* `polarity_flip` (applicability flip, an overlay on the frozen error classes): a wrong predicted answer that says a
  rule applies where the gold is not_in_effect (not counting a "none of these" answer there, `c_on_nie`: a
  not_in_effect cell of a closed-set question answered with the "none of these" option), or says not_in_effect /
  "none of these" where a rule applies.
* `empty`: an answered cell scored unparseable whose answer content is blank in the read call ledger, after the
  reasoning-field mapping (some readers return the answer only in their reasoning field; when that field holds
  exactly one answer object the answer is taken from it), i.e. a blank content whose reasoning field did not hold
  exactly one answer object. `from_reasoning`: the row says its answer was taken from the reasoning field
  (`answer_from_reasoning_field`), a recorded panel deviation.
* `memory_empty`: the cell's memory had no text (a legitimate outcome of the representation, not a failure; its
  prompt carried procedures.readers.prompts.EMPTY_MEMORY), from the row's `memory_empty`
  (procedures.scoring.score_path.row_memory_empty for rows without the field). Such cells stay in their
  condition.
* `infeasible_reason`: why a cell was recorded and not sent (the row's access `infeasible_reason`: local_ceiling,
  route_context, route_timeout (its M6 compression timed out on every attempt), compression_<reason>); None for a
  sent cell and for a row without the field. Such cells are absences, disclosed by reason
  (procedures.analysis.report `absences`, procedures.analysis.costs.access `not_sent_by_reason`).
* `query_tokens`: an M6 cell's compression tokens (the row's sustainability `query_memory_tokens`), 0 elsewhere.
* `access_tier`: the reader profile's panel access tier (C1 one 24 GB consumer card, C2 one 48 GB workstation card,
  C3 one 80 GB data-centre card, for C readers; procedures.analysis.spec.tier_group splits C by it).
* the ruling tags the authors recorded on the item (procedures.analysis.spec.RULED_FIELDS):
  `analysis_stratum` (primary / controls / earlier_batches, from the main-run item file's composition), `domain`,
  `load_bearing`, `subtype`, `same_rule_confidence`, `k6_exception_literal`, `k6_exception_confidence`,
  `pin_confidence`, `normative_confidence`, `cue_relation`, `cue_pool`, `template_family`, `owner_restates_standard`;
  None where the item has no such field (pilot items, controls, items of the earlier L3 batches). `doc` and `unit_id`
  (None off L3) carry the document and rule-unit identities for the cluster sensitivities.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import warnings

from procedures.items.frozen import NONE_OF_THESE, jsonl_bytes
from procedures.items.frozen import scorer_item as frozen_scorer_item
from procedures import stored
from procedures.readers.ledger import read_ledger
from procedures.memory.requests import FILE_STEM as REQUESTS_STEM
from procedures.memory.requests import load_memory_requests
from procedures.readers.cells import QUERY_BUILT, parse_stage
from procedures.scoring.score_path import row_memory_empty
from procedures.scoring.normalize import norm_label
from procedures.scoring.score import NIE_LABEL, gold_spec

from procedures.analysis.spec import RULED_FIELDS

RUN_SCHEMA = "procedures.matrix.run.v1"    # the schema id of a run manifest (procedures.matrix.run_matrix)
STAGE_PREFIX = "read/"
LAYERS = ("abstracted", "transparent")
ARM_RANK = {"zero-LLM": 0, "C-built": 1, "B-built": 2, "A-built": 3}   # builder arms, toward the frontier


@dataclass(frozen=True)
class Run:
    run_id: str
    run_dir: Path
    manifest: dict
    rows: tuple
    rows_file: str
    summary: dict | None
    summary_file: str | None
    ledgers: dict               # ledger file stem -> {request_hash: latest record}
    memory_requests: dict | None = None     # procedures.memory.requests map, None when the run has none
    memory_requests_file: str | None = None


@dataclass(frozen=True)
class Obs:
    run_id: str
    profile_id: str
    tier: str
    slot: str | None
    condition: str
    representation: str
    budget: int | None
    builder: str | None
    arm: str
    item_id: str
    repeat_idx: int
    stratum: str
    layer: str | None
    cluster: str
    family: str | None
    lettered: bool
    chance: float
    gold_nie: bool
    outcome: str
    observed: bool
    correct: bool | None
    status: str | None
    error_class: str | None
    other_subclass: str | None
    c_on_nie: bool
    polarity_flip: bool
    reasoning_leak: bool
    feasible: bool
    infeasible_reason: str | None   # why a cell was not sent (access `infeasible_reason`); None when sent or unknown
    empty: bool
    memory_empty: bool
    from_reasoning: bool
    query_tokens: int
    context_ceiling: int | None
    planned_input_tokens: int | None
    tokens: dict                # input_uncached, input_cached, cache_write, output, reasoning, reasoning_estimate
    ref_input_tokens: int | None
    memory_cache_key: str
    build_tokens: int
    revisions: tuple            # ((doc, revision), ...)
    request_hash: str | None
    requested_model: str | None
    served_model: str | None
    observed_provider: str | None
    access_tier: str
    doc: str | None
    unit_id: str | None
    analysis_stratum: str | None
    domain: str | None
    load_bearing: bool | None
    subtype: str | None
    same_rule_confidence: str | None
    k6_exception_literal: str | None
    k6_exception_confidence: str | None
    pin_confidence: str | None
    normative_confidence: str | None
    cue_relation: str | None
    cue_pool: str | None
    template_family: str | None
    owner_restates_standard: str | None


def _read_jsonl(path: Path) -> list[dict]:
    return stored.read_jsonl(path)


def _requests_file(run_dir: Path, rows: list[dict]) -> str | None:
    named = sorted({r["provenance"].get("memory_requests") for r in rows} - {None})
    if len(named) > 1:
        raise ValueError(f"{run_dir}: rows name several memory request maps {named}")
    if named:
        return named[0]
    found = sorted(p.name for p in run_dir.glob(f"{REQUESTS_STEM}*.json"))
    if len(found) > 1:
        raise ValueError(f"{run_dir}: several memory request maps {found} and the rows name none")
    return found[0] if found else None


def load_run(run_dir: Path, rows: str | None = None, summary: str | None = None) -> Run:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValueError(f"{run_dir}: manifest schema {manifest.get('schema')!r} is not {RUN_SCHEMA}")
    live = [x for x in _read_jsonl(run_dir / "invocations.jsonl") if x.get("outcome") == "live" and x.get("outputs")]
    if not live:
        raise ValueError(f"{run_dir}: no live invocation with scored outputs in invocations.jsonl")
    outputs = live[-1]["outputs"]
    rows_file = outputs["rows"] if rows is None else rows
    summary_file = outputs.get("summary") if rows is None else summary
    if summary is not None and rows is None:
        raise ValueError(f"{run_dir}: a summary file is given only with the rows file it summarises")
    table = _read_jsonl(run_dir / rows_file)
    run_id = stored.recorded_run_id(run_dir, table)
    summary_doc = json.loads((run_dir / summary_file).read_text(encoding="utf-8")) if summary_file else None
    ledgers = {stored.plain_name(p)[:-len(".jsonl")]: {r["request_hash"]: r for r in read_ledger(p)[1]}
               for p in sorted([*run_dir.glob("*__*.jsonl"), *run_dir.glob("*__*.jsonl.gz")])}
    requests_file = _requests_file(run_dir, table)
    requests = load_memory_requests(run_dir / requests_file) if requests_file else None
    return Run(run_id, run_dir, manifest, tuple(table), rows_file, summary_doc, summary_file, ledgers, requests,
               requests_file)


def check_items(run: Run, by_id: dict) -> None:
    ids = sorted({r["item_id"] for r in run.rows})
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ValueError(f"{run.run_id}: {len(missing)} row item ids are not in the item file, e.g. {missing[:3]}")
    digest = hashlib.sha256(jsonl_bytes([by_id[i] for i in ids])).hexdigest()
    if digest != run.manifest["items_sha256"] or len(ids) != run.manifest["n_items"]:
        warnings.warn(f"{run.run_id}: the item file's items differ from the ones the run was planned with "
                      f"(items_sha256 {digest[:12]}, the run manifest records {run.manifest['items_sha256'][:12]})",
                      stacklevel=2)


def stratum_of(item: dict) -> str:
    return item["type"].split("-")[0]


def cluster_of(item: dict) -> str:
    stratum = stratum_of(item)
    field = {"L3": "unit_id", "S3": "doc"}.get(stratum, "pair_id")
    key = item.get(field)
    if not key:
        raise ValueError(f"{item['item_id']}: {stratum} item without {field} (its bootstrap cluster)")
    return f"{stratum}:{key}"


def layer_of(item: dict) -> str | None:
    if stratum_of(item) != "L3":
        return None
    layer = item.get("rule2_spirit")
    if layer not in LAYERS:
        raise ValueError(f"{item['item_id']}: L3 cell without a rule2_spirit label in {LAYERS}")
    return layer


def arm_of(representation: str, builder_tier: str | None) -> str:
    if representation == "B0":
        return "closed-book"
    if representation == "M0":
        return "full-text"
    return "zero-LLM" if builder_tier is None else f"{builder_tier}-built"


def polarity_flip(capability: dict, gold_nie: bool) -> bool:
    if capability["correct"] or capability["status"] != "predicted":
        return False
    if gold_nie:
        return not capability["c_on_nie"]
    return norm_label(capability["pred"]) == NIE_LABEL or capability["matched_option"] == NONE_OF_THESE


def _empty(run: Run, row: dict, cap: dict | None) -> bool:
    """An answered cell scored unparseable whose content is blank in the read ledger."""
    if cap is None or cap["status"] != "unparseable":
        return False
    record = run.ledgers.get(f"read__{row['profile_id']}", {}).get(row["provenance"]["request_hash"])
    if record is None:
        raise ValueError(f"{run.run_id}: answered row ({row['profile_id']}, {row['stage']}, {row['item_id']}) has no "
                         "read-ledger record")
    return not (record.get("text") or "").strip()


def _query_tokens(run: Run, row: dict, representation: str) -> int:
    tokens = row["sustainability"].get("query_memory_tokens")
    if representation in QUERY_BUILT and tokens is None:
        raise ValueError(f"{run.run_id}: {row['stage']} row without query_memory_tokens")
    return tokens or 0


def _obs(run: Run, row: dict, item: dict, panel) -> Obs:
    representation, budget, builder = parse_stage(row["stage"])
    cap = row["capability"]
    observed = row["outcome"] == "observed"
    if observed and cap is None:
        raise ValueError(f"{run.run_id}: observed row without capability ({row['profile_id']}, {row['item_id']})")
    gold_nie = item["gold"]["value"] == "not_in_effect"
    sus, acc = row["sustainability"], row["access"]
    return Obs(
        run_id=run.run_id, profile_id=row["profile_id"], tier=acc["tier"], slot=acc["slot"],
        condition=row["stage"][len(STAGE_PREFIX):], representation=representation, budget=budget, builder=builder,
        arm=arm_of(representation, panel.get(builder).tier if builder else None), item_id=row["item_id"],
        repeat_idx=row["repeat_idx"], stratum=stratum_of(item), layer=layer_of(item), cluster=cluster_of(item),
        family=item.get("family"), lettered=row["form"] == "side",
        chance=float(gold_spec(frozen_scorer_item(item)["gold"]).chance), gold_nie=gold_nie,
        outcome=row["outcome"], observed=observed, correct=cap["correct"] if cap else None,
        status=cap["status"] if cap else None, error_class=cap["error_class"] if cap else None,
        other_subclass=cap["other_subclass"] if cap else None, c_on_nie=bool(cap and cap["c_on_nie"]),
        polarity_flip=bool(cap) and polarity_flip(cap, gold_nie), reasoning_leak=bool(cap and cap["reasoning_leak"]),
        feasible=acc["feasible_at_B"], infeasible_reason=acc.get("infeasible_reason"),
        empty=observed and _empty(run, row, cap), memory_empty=row_memory_empty(row),
        from_reasoning=bool(cap and cap.get("answer_from_reasoning_field")),
        query_tokens=_query_tokens(run, row, representation), context_ceiling=acc["local_context_ceiling"],
        planned_input_tokens=acc["planned_input_tokens"],
        tokens={"input_uncached": sus["tokens_input_uncached"], "input_cached": sus["tokens_input_cached"],
                "cache_write": sus["tokens_cache_write"], "output": sus["tokens_output"],
                "reasoning": sus["tokens_reasoning"], "reasoning_estimate": sus["reasoning_tokens_estimate"]},
        ref_input_tokens=sus["ref_input_tokens"], memory_cache_key=row["provenance"]["memory_cache_key"],
        build_tokens=sus["build_tokens"], revisions=tuple(tuple(r) for r in row["provenance"]["revisions_used"]),
        request_hash=row["provenance"]["request_hash"], requested_model=acc["requested_model"],
        served_model=acc["served_model"], observed_provider=acc["observed_provider"],
        access_tier=panel.get(row["profile_id"]).access_tier, doc=item.get("doc"),
        unit_id=item.get("unit_id") if stratum_of(item) == "L3" else None,
        **{field: item.get(field) for field in RULED_FIELDS})


def observations(runs: list[Run], items: list[dict], panel) -> list[Obs]:
    by_id = {it["item_id"]: it for it in items}
    if len(by_id) != len(items):
        raise ValueError("duplicate item ids in the item file")
    out, seen = [], {}
    for run in runs:
        check_items(run, by_id)
        for row in run.rows:
            key = (row["profile_id"], row["stage"], row["item_id"], row["repeat_idx"])
            if key in seen:
                raise ValueError(f"{key} is in runs {seen[key]} and {run.run_id}: runs are combined, never overlaid")
            seen[key] = run.run_id
            out.append(_obs(run, row, by_id[row["item_id"]], panel))
    return out

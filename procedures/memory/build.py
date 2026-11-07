"""`build_memory`: the one entry point the reader harness calls to build the memory of an item.

    build_memory(item, representation, budget_tokens, manifest, *, builder_profile=None, compressor_profile=None,
                 ledger=None) -> MemoryResult

  item               an item row: `item_id`, `doc`, `question` (read by the query-dependent representations),
                     `as_of_date` or `effective_date`, optional `m0_paths` (read by the full text)
  representation     B0 closed book (empty) | M0 full text | M2 retrieval (BM25 chunks) | M4 constraint sentences
                     (regex) | M5 summary (section-wise LLM summary, built sub-block by sub-block) | M6 compression
                     (query-focused LLM compression of the M2 retrieval, per question) | M7 records (typed constraint
                     records, built sub-block by sub-block, picked per question by BM25)
  budget_tokens      int >= 1 for M2 / M4 / M5 / M6 / M7; None ("full") or an int for M0 and B0
  manifest           a RevisionIndex, or the path of the corpus manifest (procedures.corpus.revisions.as_index)
  builder_profile    M5 / M7 only (required): a pinned panel profile id, e.g. "B1_dsv41flash_direct_nothink"
  compressor_profile M6 only (required): the panel profile that compresses (procedures.matrix.run_matrix passes the
                     reader itself unless the condition names one)
  ledger             M5 / M7: a BuilderLedger on the build call ledger; M6: a BuilderLedger on the query call ledger;
                     None = dry run

Every token count uses procedures.readers.tokens.estimate_tokens and the result records `token_method`, which is also
part of every cache key. Passing builder_profile, compressor_profile or ledger with a representation they do not apply
to raises, so a build is never silently attributed to the wrong arm. Cost: M5 / M7 report their one-time build as
build_tokens; M6 reports its per-question compression as query_tokens (build_tokens 0).

Reasoning-field mapping (procedures.memory.summary.answer_text: some servers return a model's whole answer in the
reasoning field and leave the content blank): only for a live memory whose builder or compressor profile sets
`answer_from_reasoning_field`, the build dict carries `text_from_reasoning_field` (M5 / M7:
procedures.memory.summary.reasoning_field_report of its sub-blocks, also noted as "text_from_reasoning_field=<n>"; M6:
whether the compression was read from the reasoning field), and an M5 / M7 memory whose mapped share exceeds
procedures.memory.summary.REASONING_FIELD_WARN_SHARE logs a warning and notes REASONING_FIELD_WARNING (nothing is
voided). A mapped sub-block or compression also carries the flag `text_from_reasoning_field`. Dry runs, and the build
dict and notes of every other profile, carry none of this.

M6 build modes: live | dry_run | no_input | route_timeout (a compression whose request timed out at the provider,
procedures.readers.executor.route_timed_out; `route_timeout_memory` turns a planned M6 memory into that form without
sending).
"""
from __future__ import annotations

import logging
from collections import Counter
from dataclasses import replace

from procedures.readers.tokens import token_method

from procedures.memory.chunking import BOUNDARY_SLACK, CHUNK_MAX_TOKENS, PARAMS, query_text
from procedures.memory.fulltext import build_m0, m0_revisions
from procedures.memory.retrieval import build_m2
from procedures.memory.constraints import EXTRACTOR_ID, build_m4
from procedures.memory.summary import PARAMS as M5_PARAMS
from procedures.memory.summary import PROMPT_SHA, REASONING_FIELD_FLAG, REASONING_FIELD_WARN_SHARE, BuilderLedger, build_m5, panel, reasoning_field_report
from procedures.memory.compressor import PROMPT_SHA as M6_PROMPT_SHA
from procedures.memory.compressor import ROUTE_TIMEOUT as M6_ROUTE_TIMEOUT
from procedures.memory.compressor import build_m6
from procedures.memory.compressor import timed_out as m6_timed_out
from procedures.memory.records import PARAMS as M7_PARAMS
from procedures.memory.records import PROMPT_SHA as M7_PROMPT_SHA
from procedures.memory.records import build_m7
from procedures.memory.result import MemoryResult, cache_key
from procedures.corpus.revisions import as_index, asof_revisions, in_ladder_order

REPRESENTATIONS = ("B0", "M0", "M2", "M4", "M5", "M6", "M7")
BOUNDED = ("M2", "M4", "M5", "M6", "M7")
BUILT = ("M5", "M7")            # built once per revision by builder_profile (the build ledger)
QUERY_BUILT = ("M6",)           # made per question by compressor_profile (the query ledger)
REASONING_FIELD_WARNING = f"{REASONING_FIELD_FLAG}_over_{round(100 * REASONING_FIELD_WARN_SHARE)}pct"

log = logging.getLogger(__name__)
_warned: set = set()            # the builds already warned about (REASONING_FIELD_WARNING), per process


def _check(representation: str, budget_tokens, builder_profile, compressor_profile, ledger) -> None:
    if representation not in REPRESENTATIONS:
        raise ValueError(f"representation must be one of {REPRESENTATIONS}, got {representation!r}")
    if budget_tokens is not None and (isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int)
                                      or budget_tokens < 1):
        raise ValueError(f"budget_tokens must be a positive int or None, got {budget_tokens!r}")
    if representation in BOUNDED and budget_tokens is None:
        raise ValueError(f"{representation} needs a token budget")
    if representation in BUILT and not builder_profile:
        raise ValueError(f"{representation} needs builder_profile")
    if representation not in BUILT and builder_profile is not None:
        raise ValueError(f"builder_profile applies to {BUILT} only, not {representation}")
    if representation in QUERY_BUILT and not compressor_profile:
        raise ValueError(f"{representation} needs compressor_profile")
    if representation not in QUERY_BUILT and compressor_profile is not None:
        raise ValueError(f"compressor_profile applies to {QUERY_BUILT} only, not {representation}")
    if ledger is not None:
        if representation not in BUILT + QUERY_BUILT:
            raise ValueError(f"ledger applies to {BUILT + QUERY_BUILT} only, not {representation}")
        if not isinstance(ledger, BuilderLedger):
            raise TypeError(f"ledger must be a BuilderLedger or None, got {type(ledger).__name__}")


def _revisions_key(revisions) -> list:
    return [[r.doc, r.revision, r.sha256_lf] for r in revisions]


def _reasoning_field(representation: str, builder_profile: str, memory, where: str) -> tuple[dict, list[str]]:
    """({"text_from_reasoning_field": report} or {}, the notes it adds) of an M5 / M7 memory (module docstring)."""
    if memory.mode != "live" or not panel().get(builder_profile).answer_from_reasoning_field:
        return {}, []
    report = reasoning_field_report(memory.builds)
    notes = [f"{REASONING_FIELD_FLAG}={report[REASONING_FIELD_FLAG]}"]
    once = (representation, builder_profile, tuple(b.call.key for b in memory.builds))
    if report["over_warn_share"]:
        notes.append(REASONING_FIELD_WARNING)
    if report["over_warn_share"] and once not in _warned:     # one log line per build, not per question / budget
        _warned.add(once)
        log.warning("%s %s by %s: %d of %d sub-blocks answered from the reasoning field (share %.4f > %.2f; "
                    "reported, nothing voided)", representation, where, builder_profile,
                    report[REASONING_FIELD_FLAG], report["subblocks"], report["share"], REASONING_FIELD_WARN_SHARE)
    return {REASONING_FIELD_FLAG: report}, notes


def build_memory(item: dict, representation: str, budget_tokens: int | None, manifest, *,
                 builder_profile: str | None = None, compressor_profile: str | None = None,
                 ledger: BuilderLedger | None = None) -> MemoryResult:
    _check(representation, budget_tokens, builder_profile, compressor_profile, ledger)
    method = token_method()
    common = {"representation": representation, "budget_tokens": budget_tokens, "token_method": method}
    if representation == "B0":
        return MemoryResult("", "B0", budget_tokens, [], [], 0, cache_key({**common}), "closed_book", method, 0)
    index = as_index(manifest)
    if representation == "M0":
        revisions, how = m0_revisions(item, index)
        text, prov, tokens = build_m0(revisions, budget_tokens)
        key = cache_key({**common, "revisions": _revisions_key(revisions)})
        note = how if revisions else f"{how}; no revision in effect"
        return MemoryResult(text, "M0", budget_tokens, [(r.doc, r.revision) for r in revisions], prov, 0, key,
                            note, method, tokens)
    asof = asof_revisions(item, index)
    revisions = in_ladder_order(asof.revisions)
    used = [(r.doc, r.revision) for r in revisions]
    where = f"as_of {asof.day.isoformat()} ({asof.source})"
    if not revisions:
        return MemoryResult("", representation, budget_tokens, [], [], 0,
                            cache_key({**common, "doc": item["doc"], "as_of": asof.day.isoformat()}),
                            f"{where}; no revision in effect", method, 0)
    if representation in ("M2", "M4"):
        query = query_text(item["question"])
        if representation == "M2":
            text, prov, tokens, note = build_m2(revisions, item["question"], budget_tokens)
            params = {"chunk_max_tokens": CHUNK_MAX_TOKENS, "k1": PARAMS.k1, "b": PARAMS.b}
        else:
            text, prov, tokens, note = build_m4(revisions, item["question"], budget_tokens)
            params = {"extractor": EXTRACTOR_ID, "k1": PARAMS.k1, "b": PARAMS.b}
        key = cache_key({**common, "revisions": _revisions_key(revisions), "query": query,
                         "params": {**params, "boundary_slack": BOUNDARY_SLACK}})
        return MemoryResult(text, representation, budget_tokens, used, prov, 0, key, f"{where}; {note}", method,
                            tokens)
    if representation == "M6":
        return _m6(item, budget_tokens, compressor_profile, ledger, revisions, where, asof, common)
    if representation == "M7":
        return _m7(item, budget_tokens, builder_profile, ledger, revisions, where, asof, common)
    m5 = build_m5(revisions, budget_tokens, builder_profile, ledger)
    trimmed = sum(m5.trimmed())
    history_tokens, history_share = m5.revision_history()
    live = m5.mode == "live"
    checks = m5.checks() if live else None
    levels = m5.cap_levels() if live else None
    build = {"mode": m5.mode, "builder_profile": builder_profile, "prompt_sha": PROMPT_SHA, "params": M5_PARAMS,
             "as_of": asof.day.isoformat(), "as_of_source": asof.source,
             "ledger": str(ledger.path) if ledger is not None else None, "trimmed_tokens": trimmed,
             "revision_history_tokens": history_tokens, "revision_history_share": history_share,
             "dropped_bullets": sum(m5.dropped), "checks": checks,
             "cap_levels": {str(k): v for k, v in levels.items()} if live else None,
             "still_capped": m5.still_capped() if live else None, "blocks": m5.info()}
    mapped, mapped_notes = _reasoning_field("M5", builder_profile, m5, where)
    build.update(mapped)
    key = cache_key({**common, "builder_profile": builder_profile, "params": M5_PARAMS,
                     "blocks": [list(b.call.keys) for b in m5.builds]})
    count = (f"sections={m5.section_count}; subblocks={len(m5.builds)}; revision_history_tokens={history_tokens}; "
             f"revision_history_share={history_share}")
    if m5.mode == "dry_run":
        notes = f"dry_run; {count}"
    else:
        finish = ",".join(f"{k}:{n}" for k, n in sorted(Counter(str(b.finish_reason) for b in m5.builds).items()))
        notes = "; ".join([where, "live", count, f"finish_reasons={finish}",
                           "cap_levels=" + ",".join(f"{k}:{n}" for k, n in levels.items()),
                           f"still_capped={m5.still_capped()}", f"trimmed_tokens={trimmed}",
                           f"dropped_bullets={sum(m5.dropped)}",
                           "checks=" + ",".join(f"{k}:{v}" for k, v in checks.items()), *m5.flags, *mapped_notes])
    return MemoryResult(m5.text, "M5", budget_tokens, used, m5.provenance, sum(b.build_tokens for b in m5.builds),
                        key, notes, method, m5.tokens, build)


def _counts(counter: dict) -> str:
    return ",".join(f"{k}:{n}" for k, n in sorted(counter.items()))


def _timed_out_notes(where: str, call: dict) -> str:
    if call["reason"] == M6_ROUTE_TIMEOUT:
        return (f"{where}; route_timeout: no answer after {len(call['attempts'])} attempts of unknown cost "
                f"(statuses {call['attempts']})")
    return f"{where}; route_timeout: not sent, the compressor's route was closed by the circuit breaker in this run"


def route_timeout_memory(memory: MemoryResult, ledger_path, record: dict | None = None) -> MemoryResult:
    """The route_timeout form (procedures.memory.compressor module docstring) of a planned M6 `memory` whose
    compression request, in the query ledger at `ledger_path`, timed out at the provider (`record`: its latest record,
    procedures.readers.executor.route_timed_out) or was left unsent after its compressor's route closed (`record`
    None): no text, no query tokens, the same cache key; the reader matrix does not send its reader cells."""
    if memory.representation not in QUERY_BUILT or (memory.build or {}).get("mode") in ("no_input", None):
        raise ValueError(f"only a planned M6 compression can time out, got {memory.representation} "
                         f"({(memory.build or {}).get('mode')})")
    build = {**m6_timed_out(memory.build, record), "mode": M6_ROUTE_TIMEOUT, "ledger": str(ledger_path)}
    notes = _timed_out_notes(f"as_of {build['as_of']} ({build['as_of_source']})", build)
    return replace(memory, text="", provenance=[], notes=notes, text_tokens=0, build=build, query_tokens=0)


def _m6(item: dict, budget: int, compressor_profile: str, ledger, revisions, where: str, asof,
        common: dict) -> MemoryResult:
    m6 = build_m6(revisions, item["item_id"], item["question"], budget, compressor_profile, ledger)
    build = {"mode": m6.mode, "ledger_role": "query", "compressor_profile": compressor_profile,
             "prompt_sha": M6_PROMPT_SHA, "params": m6.cache_key_inputs["params"], "as_of": asof.day.isoformat(),
             "as_of_source": asof.source, "ledger": str(ledger.path) if ledger is not None else None, **m6.call}
    if m6.mode == "no_input":
        notes = (f"{where}; no_input: " + (f"{m6.call['scored']} chunks scored above zero, none fits the input"
                                           if m6.call["scored"] else "no chunk scored above zero"))
    elif m6.mode == M6_ROUTE_TIMEOUT:
        notes = _timed_out_notes(where, m6.call)
    elif m6.mode == "dry_run":
        notes = f"dry_run; chunks={m6.call['chunks']}; room={m6.call['room']}"
    else:
        notes = "; ".join([where, "live", f"chunks={m6.call['chunks']}", f"room={m6.call['room']}",
                           f"finish_reason={m6.call['finish_reason']}", f"trimmed_tokens={m6.call['trimmed_tokens']}",
                           *m6.call["flags"]])
    return MemoryResult(m6.text, "M6", budget, [(r.doc, r.revision) for r in revisions], m6.provenance, 0,
                        cache_key({**common, **m6.cache_key_inputs}), notes, common["token_method"], m6.tokens, build,
                        m6.query_tokens)


def _m7(item: dict, budget: int, builder_profile: str, ledger, revisions, where: str, asof,
        common: dict) -> MemoryResult:
    calls, m7 = build_m7(revisions, item["question"], budget, builder_profile, ledger)
    live = m7.mode == "live"
    checks = m7.checks() if live else None
    levels = m7.cap_levels() if live else None
    records = m7.record_counts() if live else None
    build = {"mode": m7.mode, "ledger_role": "build", "builder_profile": builder_profile, "prompt_sha": M7_PROMPT_SHA,
             "params": M7_PARAMS, "as_of": asof.day.isoformat(), "as_of_source": asof.source,
             "ledger": str(ledger.path) if ledger is not None else None, "checks": checks,
             "cap_levels": {str(k): v for k, v in levels.items()} if live else None,
             "still_capped": m7.still_capped() if live else None, "records": records, "blocks": m7.info()}
    mapped, mapped_notes = _reasoning_field("M7", builder_profile, m7, where)
    build.update(mapped)
    key = cache_key({**common, "builder_profile": builder_profile, "params": M7_PARAMS,
                     "query": query_text(item["question"]),
                     "ranking": {"k1": PARAMS.k1, "b": PARAMS.b, "boundary_slack": BOUNDARY_SLACK},
                     "blocks": [list(c.keys) for c in calls]})
    if not live:
        notes = f"dry_run; subblocks={len(calls)}"
    else:
        notes = "; ".join([
            where, "live", f"subblocks={len(calls)}", f"records={records['valid']}",
            f"candidates={records['candidates']}", f"duplicates={records['duplicates']}",
            f"packed={records['packed']}", f"located={records['located']}",
            f"malformed={sum(records['malformed'].values())}", f"cut={records['cut']}",
            "arrays=" + _counts(records["arrays"]),
            "finish_reasons=" + _counts(Counter(str(b.finish_reason) for b in m7.builds)),
            "cap_levels=" + _counts(levels), f"still_capped={m7.still_capped()}",
            "checks=" + ",".join(f"{k}:{v}" for k, v in checks.items()), *m7.flags, *mapped_notes])
    return MemoryResult(m7.text, "M7", budget, [(r.doc, r.revision) for r in revisions], m7.provenance,
                        sum(b.build_tokens for b in m7.builds), key, notes, common["token_method"], m7.tokens, build)

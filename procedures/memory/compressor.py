"""Compression memory (M6): query-focused LLM compression, one request per question.

Input: the retrieval memory (M2) at INPUT_FACTOR x the budget, i.e. what a retriever hands the compressor: the as-of
revisions' paragraph chunks (procedures.memory.chunking) with a positive BM25 score against the question, packed in
rank order to INPUT_FACTOR x budget tokens and shown per revision (ladder order, under its header line) in document
order.

Request (COMPRESS_PROMPT, one user message): the full question, the passages, and at most W = floor(room x
WORDS_PER_TOKEN) words keeping every condition, value, unit, duration, exception and effective date verbatim and not
answering the question; room = budget - the memory's frame (the header lines of the revisions the passages came from).
The output cap is ceil(OUTPUT_CAP_FACTOR x room), plus the compressor's HIGH reasoning allowance when its profile
reasons. The memory is the frame, then the compression trimmed at a sentence boundary so the whole fits the budget
(procedures.memory.summary.trim_to_tokens). There is no cap ladder: the cap lies past the room, so the trim already
removes a truncated tail and a second request would pay for text that is trimmed away; a truncated answer is kept and
flagged `compress_truncated`.

Cost is per query (build tokens are one-time and amortised over the questions; a compression is paid on every
question): the request goes to the call ledger the caller gives (procedures.matrix.run_matrix: the query ledger
query__<compressor>.jsonl), its tokens are MemoryResult.query_tokens and build_tokens is 0. Identical requests (the
same question, chunk set, compressor and budget) are one ledger record, so the ledger is the cache.
  * live: a ledger (procedures.memory.summary.BuilderLedger) is given and the compressor's credential variable is set:
    the request goes through procedures.readers.executor.run_cells, the answer is read back by request hash with think
    blocks stripped; a request that timed out at the provider is a route_timeout memory (below); any other request
    without an observed answer, or an empty answer, raises (the compression did not happen: a failure of the
    mechanism sanity check). A compressor whose panel entry sets answer_from_reasoning_field has a blank content
    read from its reasoning field when the record finished "stop" / "end_turn" (procedures.memory.summary.answer_text:
    some servers return the whole answer in the reasoning field): the call carries `text_from_reasoning_field` (bool)
    and a mapped answer the flag of that name; other compressors are read from the content only.
  * dry_run: run_cells(dry_run=True) prices the request (HIGH scenario); the body is a placeholder; query_tokens =
    the request's reference tokens + the room.
  * no_input: no chunk scores above zero, or none fits INPUT_FACTOR x the budget: the memory is empty and nothing is
    sent.
  * route_timeout: the request was sent and ended without an answer after attempts of unknown cost, at least half of
    which timed out (procedures.readers.executor.route_timed_out: a client read timeout, 408 or 504, or a 500 at the
    provider's own cutoff, e.g. the Hugging Face router cutting a non-streamed answer at ~121 s; a request whose
    unknown-cost failures are mostly other faults, such as a quick 500 / 502 / 503 or a refused connection, stays
    no_response, raises as above and is sent again by a resumed run): like no_input a legitimate "no memory" state,
    with no text; the reader cells of the memory are not sent (procedures.matrix.run_matrix, infeasible reason
    "route_timeout"). `timed_out` gives the call description of such a memory (also for one the reader matrix leaves
    unsent after its compressor's route closed).
`plan` gives the request of one question without sending anything (the dry-run estimate of the reader matrix prices
its Cell); `from_records` (a query ledger's records) rebuilds a live memory from
the answered compression without sending anything (the mechanism sanity check of a run, procedures.matrix.sanity).

Cache key: (budget, token method, compressor profile, the question, the chunk set as sorted (doc_id, revision,
sha256_lf, char_start, char_end), prompt sha, params).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from procedures.readers import Cell, request_hash, run_cells
from procedures.readers.executor import route_timed_out
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens
from procedures.scoring.parse import TRUNCATION_REASONS

from procedures.memory.chunking import pack, paragraph_chunks, provenance, rank
from procedures.memory.chunking import render as render_units
from procedures.memory.summary import REASONING_FIELD_FLAG, BuilderLedger, answer_text, dry_reason, panel, trim_to_tokens
from procedures.corpus.revisions import RevisionText, in_ladder_order

COMPRESS_PROMPT = """You are compressing passages retrieved from a document into a short memory. An assistant will \
answer the question below from this memory alone, without seeing the document.

Write at most {words} words.
- Keep everything the question may turn on: each relevant rule with its conditions (when, if, unless, except, \
otherwise, in the event of) and each branch with its own outcome, and every threshold, limit, percentage, unit, \
duration, deadline, exception and effective date, copied verbatim from the passages.
- Use the document's own terms, role names and section numbers.
- Leave out what does not bear on the question.
- Do not answer the question, do not add anything the passages do not state, and never invent, estimate, round or \
convert a value.
- Plain text only: no preamble and no closing remarks.

Question: {question}

<passages>
{passages}
</passages>"""
INPUT_FACTOR = 4                # the passages: M2 chunks packed to this x the budget
WORDS_PER_TOKEN = 0.6           # the word allowance W = floor(room x this)
OUTPUT_CAP_FACTOR = 1.5         # max_output_tokens = ceil(this x room) (+ the reasoning allowance)
PROMPT_SHA = hashlib.sha256(json.dumps({"template": COMPRESS_PROMPT, "words_per_token": WORDS_PER_TOKEN},
                                       sort_keys=True).encode("utf-8")).hexdigest()
STAGE = "query/M6"
PARAMS = {"input": "M2 chunks (score > 0) packed to input_factor x budget", "input_factor": INPUT_FACTOR,
          "words_per_token": WORDS_PER_TOKEN, "output_cap_factor": OUTPUT_CAP_FACTOR,
          "fit": "frame + compression trimmed at a sentence boundary to the budget", "cap_ladder": "none"}
PLACEHOLDER = "[dry-run placeholder: no compression by {profile} was made for a {budget}-token memory]"
ROUTE_TIMEOUT = "route_timeout"         # mode of a compression that timed out at the provider (module docstring)


@dataclass(frozen=True)
class M6Memory:
    text: str
    tokens: int
    provenance: list                    # the input chunks (the memory's sources), rendered order
    mode: str                           # live | dry_run | no_input | route_timeout
    cache_key_inputs: dict              # what the memory's cache key names beside representation / budget / method
    call: dict                          # the request: request_hash, usd, input / output tokens, caps, finish, flags
    query_tokens: int


def _frame(revisions: list[RevisionText]) -> str:
    return "\n".join(f"[{rev.header()}]" for rev in revisions)


def fit(frame: str, body: str, budget: int) -> str:
    """The longest sentence-boundary prefix of `body` whose memory `frame` + newline + prefix fits `budget`."""
    if estimate_tokens(f"{frame}\n{body}") <= budget:
        return body
    lo, hi = 0, estimate_tokens(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if estimate_tokens(f"{frame}\n{trim_to_tokens(body, mid)}") <= budget:
            lo = mid
        else:
            hi = mid - 1
    return trim_to_tokens(body, lo)


def compress_cell(item_id: str, question: str, passages: str, budget: int, room: int, allowance: int) -> Cell:
    words = max(1, math.floor(room * WORDS_PER_TOKEN))
    prompt = COMPRESS_PROMPT.format(words=words, question=question, passages=passages)
    return Cell(item_id=item_id, stage=f"{STAGE}@{budget}", messages=({"role": "user", "content": prompt},),
                generation={"max_output_tokens": math.ceil(OUTPUT_CAP_FACTOR * room) + allowance},
                est_visible_tokens=room)


@dataclass(frozen=True)
class Plan:
    """One question's compression request before anything is sent (`plan`): the input chunks' provenance, the cache
    key inputs, how many chunks scored above zero, and, when a chunk fits the input (else no_input), the memory frame,
    the request Cell and its description `call` (request_hash, chunks, scored, passage_tokens, room, words,
    input_tokens, max_output_tokens, reasoning_allowance)."""
    sources: list
    inputs: dict
    scored: int
    frame: str | None = None
    cell: Cell | None = None
    call: dict | None = None


@lru_cache(maxsize=4096)
def retrieve(revisions: tuple, question: str, budget: int) -> tuple[tuple, int]:
    """(the input chunks: the M2 retrieval packed to INPUT_FACTOR x budget, how many chunks scored above zero). The
    same for every compressor, so it is computed once per (revisions, question, budget): planning a grid where every
    reader compresses for itself would otherwise repeat the packing per reader."""
    units = [u for r in revisions for u in paragraph_chunks(r)]
    ranked = [u for score, u in rank(units, question) if score > 0.0]
    chosen, _ = pack(ranked, INPUT_FACTOR * budget, lambda us: render_units(us, "paragraphs"))
    return tuple(chosen), len(ranked)


def plan(revisions: list[RevisionText], item_id: str, question: str, budget: int, compressor_profile: str) -> Plan:
    """The compression request of one question (module docstring); nothing is sent."""
    pp = panel().get(compressor_profile)
    chosen, scored = retrieve(tuple(revisions), question, budget)
    inputs = {"compressor_profile": compressor_profile, "query": question, "prompt_sha": PROMPT_SHA, "params": PARAMS,
              "chunks": sorted([u.revision.doc, u.revision.revision, u.revision.sha256_lf, u.char_start, u.char_end]
                               for u in chosen)}
    if not chosen:
        return Plan([], inputs, scored)
    frame = _frame(in_ladder_order(u.revision for u in chosen))
    room = budget - estimate_tokens(frame + "\n")
    if room < 1:
        raise ValueError(f"budget {budget} leaves no room for a compression after the revision headers")
    passages = render_units(chosen, "paragraphs")
    allowance = pp.expected_reasoning_tokens["high"] if pp.reasoning_expected else 0
    cell = compress_cell(item_id, question, passages, budget, room, allowance)
    sent = pp.messages_to_send(cell.messages)
    call = {"request_hash": request_hash(pp.profile_id, pp.fingerprint, sent, cell.generation, cell.repeat_idx,
                                         user_message_suffix=pp.user_message_suffix),
            "chunks": len(chosen), "scored": scored, "passage_tokens": estimate_tokens(passages), "room": room,
            "words": max(1, math.floor(room * WORDS_PER_TOKEN)), "input_tokens": estimate_message_tokens(sent),
            "max_output_tokens": cell.generation["max_output_tokens"], "reasoning_allowance": allowance}
    return Plan(provenance(chosen), inputs, scored, frame, cell, call)


def _no_input(p: Plan) -> M6Memory:
    return M6Memory("", 0, [], "no_input", p.inputs, {"chunks": 0, "scored": p.scored}, 0)


def timed_out(call: dict, record: dict | None) -> dict:
    """The call description of a route_timeout memory from its planned `call` (Plan.call, or a planned memory's): no
    answer, no priced cost; `attempts` = the statuses of the attempts `record` lists (a transport failure None) and
    `usd_pending_unknown` its unknown-cost bound, both empty / 0 when no record says the request was tried to the end
    (`record` None: left unsent after its route closed)."""
    tried = record if route_timed_out(record) else None
    return {**call, "usd": None, "reason": ROUTE_TIMEOUT if tried else "route_closed", "finish_reason": None,
            "output_tokens": None, "trimmed_tokens": 0, "flags": [],
            "attempts": [a.get("http_status") for a in tried["attempts"]] if tried else [],
            "usd_pending_unknown": tried.get("usd_pending_unknown", 0.0) if tried else 0.0}


def _timed_out(p: Plan, record: dict) -> M6Memory:
    return M6Memory("", 0, [], ROUTE_TIMEOUT, p.inputs, timed_out(p.call, record), 0)


def _from_record(p: Plan, budget: int, record: dict | None, where: str, ledger_path) -> M6Memory:
    """The live memory from the compression's ledger record (the latest for its request hash); a request that timed
    out at the provider (procedures.readers.executor.route_timed_out) gives a route_timeout memory; a record that is
    missing, or any other unanswered or empty one, raises (the compression did not happen)."""
    if route_timed_out(record):
        return _timed_out(p, record)
    if record is None or record.get("outcome") != "observed":
        raise RuntimeError(f"{where} got no answer ({None if record is None else record.get('outcome')}); "
                           f"see {ledger_path}")
    pp = panel().get(p.inputs["compressor_profile"])
    answer, mapped = answer_text(record, pp)
    if not answer:
        raise RuntimeError(f"{where} returned an empty answer; see {ledger_path}")
    body = fit(p.frame, answer, budget)
    text = f"{p.frame}\n{body}"
    flags = []
    if record.get("finish_reason") in TRUNCATION_REASONS:
        flags.append("compress_truncated")
    if body != answer:
        flags.append("trimmed_to_budget")
    if mapped:
        flags.append(REASONING_FIELD_FLAG)
    usage = record.get("usage") or {}
    if usage.get("input_total") is None or usage.get("output_total") is None:
        flags.append("query_tokens_estimated")
        tokens = p.call["input_tokens"] + estimate_tokens(answer)
    else:
        tokens = usage["input_total"] + usage["output_total"]
    call = {**p.call, "usd": record.get("usd"), "reason": "", "finish_reason": record.get("finish_reason"),
            "output_tokens": usage.get("output_total"),
            "trimmed_tokens": estimate_tokens(answer) - estimate_tokens(body), "flags": sorted(flags)}
    if pp.answer_from_reasoning_field:
        call[REASONING_FIELD_FLAG] = mapped
    return M6Memory(text, estimate_tokens(text), p.sources, "live", p.inputs, call, tokens)


def build_m6(revisions: list[RevisionText], item_id: str, question: str, budget: int, compressor_profile: str,
             ledger: BuilderLedger | None) -> M6Memory:
    """The memory for one question (module docstring)."""
    p = plan(revisions, item_id, question, budget, compressor_profile)
    if p.cell is None:
        return _no_input(p)
    pp = panel().get(compressor_profile)
    reason = dry_reason(pp, ledger)
    if reason:
        cap = ledger.cap_usd if ledger is not None else math.inf
        usd = run_cells([p.cell], pp, cap_usd=cap, dry_run=True)["usd"]
        text = f"{p.frame}\n" + PLACEHOLDER.format(profile=compressor_profile, budget=budget)
        if estimate_tokens(text) > budget:
            text = p.frame
        return M6Memory(text, estimate_tokens(text), p.sources, "dry_run", p.inputs,
                        {**p.call, "usd": usd, "reason": reason, "finish_reason": None, "output_tokens": None,
                         "trimmed_tokens": 0, "flags": []},
                        p.call["input_tokens"] + p.call["room"])
    report = run_cells([p.cell], pp, cap_usd=ledger.cap_usd, ledger_path=Path(ledger.path), transport=ledger.transport,
                       sleep=ledger.sleep)
    record = next((r for r in report["records"] if r["request_hash"] == p.call["request_hash"]), None)
    return _from_record(p, budget, record, f"query/M6 compression of {item_id} by {compressor_profile}", ledger.path)


def from_records(revisions: list[RevisionText], item_id: str, question: str, budget: int, compressor_profile: str,
                 records: dict, where) -> M6Memory:
    """The live memory build_m6 assembles, from `records` (request hash -> latest record of a query ledger, `where`
    names it); nothing is sent."""
    p = plan(revisions, item_id, question, budget, compressor_profile)
    if p.cell is None:
        return _no_input(p)
    return _from_record(p, budget, records.get(p.call["request_hash"]),
                        f"query/M6 compression of {item_id} by {compressor_profile}", where)

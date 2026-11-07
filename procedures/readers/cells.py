"""Reader cells: one procedures.readers Cell per (item, representation, budget, reader profile).

`build_cell` renders the prompt (prompts.build_messages) around a built memory (procedures.memory.build_memory's
MemoryResult: text, representation, budget_tokens, revisions_used, provenance, build_tokens, cache_key, notes) and
records beside the Cell what the call ledgers and the analysis tables need about the memory.

* stage = "read/{representation}@{budget or full}", plus "/{builder profile}" for an LLM-built representation
  (summary memories (M5), query-focused compression (M6), typed constraint records (M7); for M6 the profile is the
  compressor). The Cell's (item_id, stage) is what a call-ledger record carries; `parse_stage` reads it back.
* Per-question memory cost (M6, QUERY_BUILT, made by procedures.memory.compressor): the compression's tokens
  (MemoryResult.query_tokens) and, when it was made live, its priced cost (MemoryResult.build["usd"]) ride on the
  cell as `query_tokens` / `query_usd`, beside the one-time `build_tokens` of M5 / M7.
* Planning tokens (procedures.readers.tokens) are the prompt as rendered plus, for a budgeted representation, the
  unused part of the budget: a dry-run M5 memory is not materialised yet, and "fits at budget B" is the question the
  access table asks. The system message and the memory block form the cacheable prefix (`cache_key`, used by the LOW
  estimate).
* Feasibility (`feasible_at_B` in the access table), two limits checked in order (`infeasible_reason`):
  1. "local_ceiling": planned input + the profile's output cap over the local context ceiling of its access tier
     (`PanelProfile.access_tier`). The one-card tier has a 32k context ceiling: the study design sets it for C1 (one
     24 GB consumer card), and C2 (one 48 GB workstation card) and C3 (one 80 GB data-centre card) inherit it; A
     (cloud frontier models) and B (open-weight models on a multi-GPU server) have no local ceiling. 32k is read as
     32,768 tokens (our reading of "32k").
  2. "route_context": the route serving the profile cannot take the request:
     ceil(planned input x ROUTE_TOKEN_ALLOWANCE) + output cap over `PanelProfile.route_context` (the pinned HF
     route's context in its router snapshot; None = no route limit known). The allowance covers the route's own
     tokenizer and chat template against the o200k reference count (measured at 1.003-1.011 on the Llama-3.3-70B
     ovhcloud route).
  A cell over either is recorded, not sent; its reason rides on the cell and the access table row.
  A third reason is set by the reader matrix, not here: "route_timeout" (procedures.memory.compressor.ROUTE_TIMEOUT),
  a cell whose M6 memory is a compression that timed out on every attempt; recorded, not sent, and counted in the
  same places.
* `memory_empty`: a non-closed-book memory with no text (procedures.readers.prompts.is_empty_memory); its prompt
  carries the EMPTY_MEMORY line in the memory block, and the flag rides on the cell and its row (a legitimate reading
  of the representation, counted per condition by the analysis).
* `repeat_idx` > 0 marks a re-read of the same request (a repeat, from which the analysis reads per-item answer
  agreement): same messages, same stage, a different request hash (procedures.readers.ledger.request_hash binds
  repeat_idx).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re

from procedures.readers import Cell, PanelProfile, estimate_message_tokens, estimate_tokens, request_hash

from procedures.readers.prompts import CLOSED_BOOK, answer_form, build_messages, is_empty_memory

LLM_BUILT = frozenset({"M5", "M6", "M7"})
QUERY_BUILT = frozenset({"M6"})         # made per question by a compressor into the query ledger
BUILT_ONCE = LLM_BUILT - QUERY_BUILT    # built once per revision by the run's builder into the build ledger
ACCESS_CONTEXT_CEILING = {"A": None, "B": None, "C3": 32_768, "C2": 32_768, "C1": 32_768}
ROUTE_TOKEN_ALLOWANCE = (103, 100)     # route tokens per reference token, as an exact ratio (x 1.03)
LOCAL_CEILING, ROUTE_CONTEXT = "local_ceiling", "route_context"
STAGE = re.compile(r"read/(?P<rep>[A-Z][0-9]+)@(?P<budget>full|[1-9][0-9]*)(?:/(?P<builder>[A-Za-z0-9_.-]+))?")
QUESTION_MARK = "\n\nQuestion: "


def stage_name(representation: str, budget_tokens: int | None, builder_profile_id: str | None = None) -> str:
    if (representation in LLM_BUILT) != (builder_profile_id is not None):
        raise ValueError(f"{representation}: a builder profile is required exactly for LLM-built representations "
                         f"{sorted(LLM_BUILT)}")
    stage = f"read/{representation}@{'full' if budget_tokens is None else budget_tokens}"
    stage += f"/{builder_profile_id}" if builder_profile_id else ""
    if STAGE.fullmatch(stage) is None:
        raise ValueError(f"cannot name a reader stage for {representation!r}, {budget_tokens!r}, "
                         f"{builder_profile_id!r}")
    return stage


def parse_stage(stage: str) -> tuple[str, int | None, str | None]:
    m = STAGE.fullmatch(stage)
    if m is None:
        raise ValueError(f"not a reader stage: {stage!r}")
    return m["rep"], None if m["budget"] == "full" else int(m["budget"]), m["builder"]


def context_ceiling(pp: PanelProfile) -> int | None:
    if pp.access_tier not in ACCESS_CONTEXT_CEILING:
        raise ValueError(f"{pp.profile_id}: no context ceiling for access tier {pp.access_tier!r}")
    return ACCESS_CONTEXT_CEILING[pp.access_tier]


def infeasible_reason(pp: PanelProfile, planned_tokens: int, max_output_tokens: int) -> str | None:
    """Why a request of `planned_tokens` reference input and `max_output_tokens` output cannot be sent to `pp`:
    LOCAL_CEILING (planned + cap > the access tier's local ceiling), else ROUTE_CONTEXT (ceil(planned x 1.03) + cap >
    the route's context), else None (feasible). Integer arithmetic: a request exactly at a limit fits."""
    ceiling = context_ceiling(pp)
    if ceiling is not None and planned_tokens + max_output_tokens > ceiling:
        return LOCAL_CEILING
    if pp.route_context is not None:
        num, den = ROUTE_TOKEN_ALLOWANCE
        if -(-planned_tokens * num // den) + max_output_tokens > pp.route_context:
            return ROUTE_CONTEXT
    return None


def output_cap(pp: PanelProfile) -> int:
    cap = pp.spec.get("generation", {}).get("max_output_tokens")
    if isinstance(cap, bool) or not isinstance(cap, int) or cap <= 0:
        raise ValueError(f"{pp.profile_id}: reader profiles need an explicit max_output_tokens")
    return cap


@dataclass(frozen=True)
class ReaderCell:
    cell: Cell
    reader_profile_id: str
    item_id: str
    form: str
    representation: str
    budget_tokens: int | None
    builder_profile_id: str | None
    memory_cache_key: str
    memory_tokens: int
    build_tokens: int
    revisions_used: tuple
    memory_notes: object
    context_ceiling: int | None
    feasible: bool
    query_tokens: int = 0               # M6: the per-question compression's tokens (0 otherwise)
    query_usd: float | None = None      # M6 made live: the compression's priced cost
    memory_planned: bool = False        # an LLM-built memory still planned as a dry run: its text is a stand-in
    memory_empty: bool = False          # a non-closed-book memory with no text: the prompt carries EMPTY_MEMORY
    route_context: int | None = None    # the reader's route context (PanelProfile.route_context)
    infeasible_reason: str | None = None  # infeasible_reason, or "compression_" + the compressor's (run_matrix)

    def meta(self) -> dict:
        """JSON-able description of the cell (everything but the messages)."""
        return {"reader_profile_id": self.reader_profile_id, "item_id": self.item_id, "stage": self.cell.stage,
                "repeat_idx": self.cell.repeat_idx, "form": self.form, "representation": self.representation,
                "budget_tokens": self.budget_tokens, "builder_profile_id": self.builder_profile_id,
                "memory_cache_key": self.memory_cache_key, "memory_tokens": self.memory_tokens,
                "build_tokens": self.build_tokens, "revisions_used": [list(r) for r in self.revisions_used],
                "memory_notes": self.memory_notes, "planned_input_tokens": self.cell.est_input_tokens,
                "context_ceiling": self.context_ceiling, "route_context": self.route_context,
                "feasible": self.feasible, "infeasible_reason": self.infeasible_reason,
                "query_tokens": self.query_tokens, "query_usd": self.query_usd, "memory_empty": self.memory_empty}


def build_cell(item: dict, memory, reader: PanelProfile, *, builder_profile_id: str | None = None,
               repeat_idx: int = 0) -> ReaderCell:
    representation, budget = memory.representation, memory.budget_tokens
    if representation == CLOSED_BOOK and budget is not None:
        raise ValueError(f"{item['item_id']}: closed book has no budget, got {budget}")
    messages = build_messages(item, representation, memory.text)
    memory_tokens = estimate_tokens(memory.text)
    pad = 0 if budget is None else max(0, budget - memory_tokens)
    system, user = messages
    prefix = user["content"].split(QUESTION_MARK, 1)[0] if representation != CLOSED_BOOK else ""
    cell = Cell(item_id=item["item_id"], stage=stage_name(representation, budget, builder_profile_id),
                messages=messages, repeat_idx=repeat_idx, est_input_tokens=estimate_message_tokens(messages) + pad,
                cache_key=hashlib.sha256((system["content"] + "\0" + prefix).encode("utf-8")).hexdigest(),
                cache_prefix_tokens=estimate_message_tokens((system,)) + estimate_tokens(prefix) + pad)
    ceiling = context_ceiling(reader)
    reason = infeasible_reason(reader, cell.est_input_tokens, output_cap(reader))
    build_tokens = memory.build_tokens
    if isinstance(build_tokens, bool) or not isinstance(build_tokens, int) or build_tokens < 0:
        raise ValueError(f"{item['item_id']}: memory build_tokens must be a non-negative int, got {build_tokens!r}")
    query_tokens, query_usd = 0, None
    if representation in QUERY_BUILT:
        query_tokens = memory.query_tokens
        if memory.build["mode"] == "live":
            query_usd = memory.build["usd"]
    return ReaderCell(cell=cell, reader_profile_id=reader.profile_id, item_id=item["item_id"],
                      form=answer_form(item), representation=representation, budget_tokens=budget,
                      builder_profile_id=builder_profile_id, memory_cache_key=memory.cache_key,
                      memory_tokens=memory_tokens, build_tokens=build_tokens,
                      revisions_used=tuple(tuple(r) for r in memory.revisions_used), memory_notes=memory.notes,
                      context_ceiling=ceiling, feasible=reason is None, query_tokens=query_tokens, query_usd=query_usd,
                      memory_planned=representation in LLM_BUILT and (memory.build or {}).get("mode") == "dry_run",
                      memory_empty=representation != CLOSED_BOOK and is_empty_memory(memory.text),
                      route_context=reader.route_context, infeasible_reason=reason)


def cell_request_hash(cell: Cell, pp: PanelProfile) -> str:
    """The request_hash procedures.readers.executor.run_cells gives this cell under `pp` (the messages as sent, with
    the profile's user_message_suffix); tests pin it against the hash the call ledger records."""
    return request_hash(pp.profile_id, pp.fingerprint, pp.messages_to_send(cell.messages), cell.generation,
                        cell.repeat_idx, user_message_suffix=pp.user_message_suffix)

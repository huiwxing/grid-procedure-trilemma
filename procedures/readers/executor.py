"""Run grid cells against one pinned profile: dry-run estimates, or paid calls into the call ledger.

``run_cells(..., dry_run=True)`` never builds a client or touches the network; it
prices assumed token counts with the same ``price_call`` used for real calls.
Cells are first rewritten to the messages actually sent (the profile's
``user_message_suffix`` appended to the last user message); validation, hashing,
estimates and the call all use those.
A live run (``dry_run=False``) validates every request (adapter + trap checks),
checks the credential variable exists, takes the ledger's executor lock (one live sender per ledger), settles what a
dead executor left in flight (procedures.readers.ledger.Ledger.recover_unsettled: listed in the report as
``recovered``), refuses up front if committed + projected cost exceeds the cap (or the guard's approval), then sends
cells in order with at most ``workers(concurrency, client profile)`` in flight: the profile's ``max_workers`` (the
panel's per-route limit), lowered to ``concurrency`` when that is smaller. Budget safety:
  * each logical request reserves, before it is sent, the bound of every attempt the client may make
    (``max_attempts`` x ``upper_bound_usd``); each attempt settles its own share: a 2xx answer its priced usage (the
    bound when it cannot be priced), a 408 / 5xx / transport failure (cost unknown: a gateway timeout can be billed
    upstream) its bound as ``usd_pending_unknown``, any other status (429 and other 4xx: a definite refusal) nothing;
  * a ``Breaker`` shared by the run (``SpendGuard``) sees every request once, when it ends, by its final outcome
    (``request_cost`` over its attempts): 3 consecutive requests on one route (provider + wire model) that ended
    without an answer after attempts of unknown cost stop new sends to that route for the rest of the run
    (``RouteTripped``; an answered request resets the count), and more than 10% failed requests (unknown-cost or
    refused; a 429 is back-pressure, not a failure) among a route's first 20 requests pause the whole run
    (``RunPaused``). One request whose three attempts all time out is one failure, recorded
    ``no_response``; each attempt's reservation and pending_unknown are settled as above. An attempt that would start
    after a trip or pause is not sent, requests in flight settle before run_cells raises, the ledger stays resumable;
  * with an approval (procedures.readers.budget.ApprovalGuard on the guard), every reservation is also checked,
    under the approval's cross-process lock, against the approval's grant and the run's arm.
Retries/backoff are the client's (procedures.readers.client: 408/425/429/5xx and transport errors only).
Inside ``timeout_overrides({profile_id: seconds})`` the client of each named profile reads with that timeout instead
of its panel ``timeout`` (only raised, never lowered; transport like ``max_workers``: outside the fingerprint, the
request hash and the ledger header, so a resumed run keeps its identity).
Inside ``workers_overrides({profile_id: n})`` the named profile's ``max_workers`` is ``n`` (only raised, at most
WORKERS_OVERRIDE_MAX; a caller's ``concurrency`` still only lowers it; transport like the timeout: outside the
fingerprint, the request hash and the ledger header).
A cell's ``labels`` are copied into its ledger record as top-level fields (e.g. a build's
``cap_level``); they are never sent and are not part of the request hash.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
import hashlib
import http.client
import math
import os
from pathlib import Path
import re
import threading

from procedures.readers.client import Client, ConfigurationError
from procedures.readers.client import client as client_module
from procedures.readers.client.adapters import build_request
from procedures.readers.ledger import Ledger, SpendCapExceeded, canonical_messages, request_hash, utc_now
from procedures.readers.pricing import PriceBook, load_price_book, price_call
from procedures.readers.panel import PanelProfile, check_wire
from procedures.readers.tokens import estimate_message_tokens, token_method
from procedures.readers.usage import has_reasoning_text, message_reasoning_content, normalize_usage, provider_usage, synthetic_usage

CAP_INPUT_MARGIN = 1.25  # reference-tokenizer vs provider-tokenizer and chat-template slack (an assumed margin)
SCENARIOS = ("low", "high")
THINK_BLOCK = re.compile(r"<think>(.*?)(?:</think>|\Z)", re.S)  # inline reasoning, closed or cut off
# Top-level fields of a call-ledger record (run_cells' records, harness errors, procedures.readers.ledger, imports
# across runs): a cell label may not take one of these names.
RECORD_FIELDS = frozenset({
    "kind", "schema", "request_hash", "item_id", "stage", "repeat_idx", "params", "profile_id", "profile_fingerprint",
    "user_message_suffix", "requested_model", "served_model", "system_fingerprint", "observed_provider",
    "server_request_id", "outcome", "status", "finish_reason", "text", "reasoning_text", "reasoning_leak", "usage",
    "price", "usd", "usd_upper_bound", "usd_counted", "attempts", "error", "request", "response_headers",
    "raw_response", "raw_text", "ref_input_tokens", "token_method", "harness_digest", "started_at", "finished_at",
    "result", "imported_from", "imported_usd_counted", "send_id", "usd_reserved", "usd_pending_unknown"})
ROUTE_UNKNOWN_STREAK = 3        # consecutive requests failed at unknown cost that close a route for the run
EARLY_SENDS = 20                # a route's first requests watched for failures ...
EARLY_FAILURE_SHARE = 0.10      # ... more than this share of them failing pauses the whole run
ROUTE_TIMEOUT_STREAK = 6        # timeout-class requests (route_timed_out: an understood, disclosed absence) get
EARLY_TIMEOUT_SHARE = 0.50      # their own, looser thresholds (Breaker.observe says why)
TIMEOUT_OVERRIDE_FLOOR = 120.0  # a timeout override must exceed this (and the profile's own timeout): it only raises
WORKERS_OVERRIDE_MAX = 8        # a max_workers override lies in [the profile's own max_workers, this]


class RouteTripped(RuntimeError):
    """A route had ROUTE_UNKNOWN_STREAK consecutive requests fail at unknown cost: no new sends to it in this run."""


class RunPaused(RuntimeError):
    """More than EARLY_FAILURE_SHARE of a route's first EARLY_SENDS requests failed: the run pauses (resumable)."""


def attempt_cost(status: int | None) -> str:
    """What one sent attempt may cost: "answered" (2xx: its usage), "unknown" (408, 5xx, or no status = a transport
    failure: it may have been billed upstream), "rate_limited" (429) or "refused" (any other status); the last two
    are definite failures, not billed."""
    if status is None or status == 408 or 500 <= status <= 599:
        return "unknown"
    if 200 <= status < 300:
        return "answered"
    return "rate_limited" if status == 429 else "refused"


def request_cost(costs: list[str]) -> str:
    """How one request ended, from its attempts' costs (attempt_cost, in order), as the Breaker counts it: "answered"
    (its last attempt answered), "refused" (no answer, a definite refusal among its attempts), "unknown" (no answer,
    every failed attempt of unknown cost: 408 / 5xx / transport; a 429 in between is back-pressure, not a failure),
    "rate_limited" (no answer, 429s only) or "none" (no attempt was sent)."""
    if not costs:
        return "none"
    if costs[-1] == "answered":
        return "answered"
    if "refused" in costs:
        return "refused"
    return "unknown" if "unknown" in costs else "rate_limited"


def record_costs(record: dict) -> list[str]:
    """attempt_cost of each attempt a ledger call record lists (a transport failure has no http_status: unknown)."""
    return [attempt_cost(a.get("http_status")) for a in record.get("attempts") or []]


TIMEOUT_STATUSES = (408, 504)   # HTTP statuses of a timeout: request timeout, gateway timeout


PROVIDER_CUTOFF_SECONDS = 55.0   # an HTTP 500 that took at least this long is a provider-side timeout (see below)


def timeout_class(attempt: dict) -> bool:
    """Whether one attempt of a ledger call record failed by timing out: a client read timeout (a transport error,
    no http_status, whose message contains "timed out"), HTTP 408 / 504, or an HTTP 500 that took at least
    PROVIDER_CUTOFF_SECONDS (the nscale route answers "500 Error from model provider" at 60.8-61.3 s on every
    request its model cannot finish within 60 s, while its successful answers never exceed 60.9 s; such a 500 is the
    provider's own cutoff, not a fault). Any other failure (connection refused, DNS, IncompleteRead and other
    transport errors, a quick HTTP 500 / 502 / 503, ...) is not."""
    status = attempt.get("http_status")
    if status is not None:
        if status == 500:
            return float(attempt.get("elapsed_seconds") or 0.0) >= PROVIDER_CUTOFF_SECONDS
        return status in TIMEOUT_STATUSES
    error = attempt.get("error") or {}
    return error.get("kind") == "transport" and "timed out" in str(error.get("message") or "").lower()


def route_timed_out(record: dict | None) -> bool:
    """A call record of a request that was sent and ended without an answer after attempts of unknown cost only
    (request_cost "unknown"), the majority of which timed out (timeout_class: client read timeout, 408, 504, a 500
    at the provider's cutoff; a 429 in between is back-pressure and neutral). A request whose unknown-cost failures
    are mostly quick faults (500 / 502 / 503, connection refused, IncompleteRead) is not one: it stays no_response and
    a resumed run sends it again. A request cut by the circuit breaker (error kind "circuit_open") is not one either:
    it was not tried to the end. The majority rule: one quick 502 between two timeouts (seen on the nscale route: 500
    at 61 s, 502 at 11 s, 504 at 180 s) does not change the diagnosis of a request the route cannot finish."""
    if record is None or record.get("outcome") != "no_response"\
            or (record.get("error") or {}).get("kind") == "circuit_open"\
            or request_cost(record_costs(record)) != "unknown":
        return False
    attempts = record.get("attempts") or []
    unknown = [a for a, cost in zip(attempts, record_costs(record)) if cost == "unknown"]
    timed_out = sum(1 for a in unknown if timeout_class(a))
    return timed_out >= 1 and 2 * timed_out >= len(unknown)


class Breaker:
    """Per-route failure watch shared by every run_cells call of one run (module docstring): `observe` is called once
    per request that ended, with its request_cost."""

    def __init__(self):
        self._lock = threading.Lock()
        self._streak: Counter = Counter()
        self._timeout_streak: Counter = Counter()
        self._sends: Counter = Counter()
        self._early_failures: Counter = Counter()
        self._early_timeouts: Counter = Counter()
        self.tripped: dict[str, str] = {}
        self.paused: str | None = None

    def check(self, route: str) -> None:
        """Raise RunPaused / RouteTripped if nothing more may be sent to `route`."""
        with self._lock:
            if self.paused:
                raise RunPaused(self.paused)
            if route in self.tripped:
                raise RouteTripped(self.tripped[route])

    def blocked(self, route: str) -> bool:
        with self._lock:
            return bool(self.paused) or route in self.tripped

    def observe(self, route: str, cost: str) -> None:
        """`cost` is the request's class: request_cost's "answered" / "refused" / "unknown" / "rate_limited" /
        "none", or "timeout" for a no-answer request whose unknown-cost attempts mostly timed out (route_timed_out:
        an understood, disclosed absence). Quick faults keep the tight thresholds (ROUTE_UNKNOWN_STREAK,
        EARLY_FAILURE_SHARE); timeouts have their own, looser ones (ROUTE_TIMEOUT_STREAK, EARLY_TIMEOUT_SHARE), so a
        route with a known per-request cutoff (nscale: 60 s, about a sixth of Qwen3-32B's compressions) is not paused
        at every resume, while a route that times out on most requests still is."""
        with self._lock:
            self._sends[route] += 1
            if cost == "answered":
                self._streak[route] = 0
                self._timeout_streak[route] = 0
            elif cost == "unknown":
                self._streak[route] += 1
                if self._streak[route] >= ROUTE_UNKNOWN_STREAK and route not in self.tripped:
                    self.tripped[route] = (f"{route}: {self._streak[route]} consecutive unknown-cost failures "
                                           "(requests with no answer after 408 / 5xx / transport attempts); no new "
                                           "sends to this route in this run")
            elif cost == "timeout":
                self._timeout_streak[route] += 1
                if self._timeout_streak[route] >= ROUTE_TIMEOUT_STREAK and route not in self.tripped:
                    self.tripped[route] = (f"{route}: {self._timeout_streak[route]} consecutive timeout-class "
                                           "requests (route_timeout); no new sends to this route in this run")
            if cost in ("unknown", "refused") and self._sends[route] <= EARLY_SENDS:
                self._early_failures[route] += 1
                if self._early_failures[route] > EARLY_FAILURE_SHARE * EARLY_SENDS and self.paused is None:
                    self.paused = (f"{route}: {self._early_failures[route]} failed requests among its first "
                                   f"{self._sends[route]} requests (> {EARLY_FAILURE_SHARE:.0%} of {EARLY_SENDS}); "
                                   "the run is paused, resume with the same command once the route is understood")
            if cost == "timeout" and self._sends[route] <= EARLY_SENDS:
                self._early_timeouts[route] += 1
                if self._early_timeouts[route] > EARLY_TIMEOUT_SHARE * EARLY_SENDS and self.paused is None:
                    self.paused = (f"{route}: {self._early_timeouts[route]} timeout-class requests among its first "
                                   f"{self._sends[route]} requests (> {EARLY_TIMEOUT_SHARE:.0%} of {EARLY_SENDS}); "
                                   "the run is paused, resume with the same command once the route is understood")

    def state(self) -> dict:
        with self._lock:
            return {"sends": dict(self._sends), "early_failures": dict(self._early_failures),
                    "early_timeouts": dict(self._early_timeouts), "tripped": dict(self.tripped),
                    "paused": self.paused}


class SpendGuard:
    """What a live run shares across its run_cells calls: the Breaker and, for an approved paid run, the approval
    (procedures.readers.budget.ApprovalGuard) that every pre-flight and every reservation is checked against."""

    def __init__(self, approval=None):
        self.breaker = Breaker()
        self.approval = approval

    def preflight(self, usd: float, what: str) -> None:
        if self.approval is not None:
            self.approval.check(usd, what)

    def reserve(self, ledger: Ledger, usd: float, **kw):
        if self.approval is not None:
            return self.approval.reserve(ledger, usd, **kw)
        return ledger.reserve(usd, **kw)


_ACTIVE_GUARD: ContextVar = ContextVar("procedures.readers.executor.guard", default=None)


@contextmanager
def guarded(guard: SpendGuard):
    """Make `guard` the one every run_cells call inside the block uses when not given one (procedures.memory's
    builds and compressions call run_cells themselves)."""
    token = _ACTIVE_GUARD.set(guard)
    try:
        yield guard
    finally:
        _ACTIVE_GUARD.reset(token)


_TIMEOUT_OVERRIDES: ContextVar = ContextVar("procedures.readers.executor.timeouts", default=None)


def raised_timeout(profile_id: str, seconds, base: float = TIMEOUT_OVERRIDE_FLOOR) -> float:
    """`seconds` as a float if it raises a client read timeout of `base` seconds (and TIMEOUT_OVERRIDE_FLOOR), else
    ValueError: an override is for a route whose answers take longer than the panel's timeout, never to cut it."""
    floor = max(float(base), TIMEOUT_OVERRIDE_FLOOR)
    if (isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or seconds <= floor):
        raise ValueError(f"timeout override {profile_id}={seconds!r}: an override only raises the client read "
                         f"timeout; it must be a finite number of seconds > {floor:g}")
    return float(seconds)


def parse_timeout_override(text: str) -> tuple[str, float]:
    """"PROFILE_ID=SECONDS" -> (profile id, seconds), seconds checked by raised_timeout."""
    pid, sep, value = text.partition("=")
    pid = pid.strip()
    if not sep or not pid:
        raise ValueError(f"timeout override {text!r} must be PROFILE_ID=SECONDS")
    try:
        seconds = float(value)
    except ValueError:
        raise ValueError(f"timeout override {text!r}: {value!r} is not a number of seconds") from None
    return pid, raised_timeout(pid, seconds)


@contextmanager
def timeout_overrides(overrides: dict | None):
    """Every run_cells call inside the block sends profile `pid` with client read timeout `overrides[pid]` (module
    docstring); None or {} = no override."""
    checked = {pid: raised_timeout(pid, seconds) for pid, seconds in (overrides or {}).items()}
    token = _TIMEOUT_OVERRIDES.set(checked or None)
    try:
        yield checked
    finally:
        _TIMEOUT_OVERRIDES.reset(token)


_WORKERS_OVERRIDES: ContextVar = ContextVar("procedures.readers.executor.workers", default=None)


def raised_workers(profile_id: str, n, base: int = 1) -> int:
    """`n` if it is an int in [`base` (the profile's own max_workers), WORKERS_OVERRIDE_MAX], else ValueError: an
    override only raises a route's in-flight limit (a lower one is the caller's `concurrency`)."""
    if isinstance(n, bool) or not isinstance(n, int) or not max(int(base), 1) <= n <= WORKERS_OVERRIDE_MAX:
        raise ValueError(f"workers override {profile_id}={n!r}: an override only raises the profile's max_workers; "
                         f"it must be an int in [{max(int(base), 1)}, {WORKERS_OVERRIDE_MAX}] (lower it with "
                         "--concurrency)")
    return n


def parse_workers_override(text: str) -> tuple[str, int]:
    """"PROFILE_ID=N" -> (profile id, N), N checked by raised_workers (against 1; the profile's own max_workers is
    checked where the profile is known)."""
    pid, sep, value = text.partition("=")
    pid = pid.strip()
    if not sep or not pid:
        raise ValueError(f"workers override {text!r} must be PROFILE_ID=N")
    try:
        n = int(value)
    except ValueError:
        raise ValueError(f"workers override {text!r}: {value!r} is not an int") from None
    return pid, raised_workers(pid, n)


@contextmanager
def workers_overrides(overrides: dict | None):
    """Every run_cells call inside the block sends profile `pid` with at most `overrides[pid]` requests in flight
    (module docstring); None or {} = no override."""
    checked = {pid: raised_workers(pid, n) for pid, n in (overrides or {}).items()}
    token = _WORKERS_OVERRIDES.set(checked or None)
    try:
        yield checked
    finally:
        _WORKERS_OVERRIDES.reset(token)


def send_profile(pp: PanelProfile, client_profile):
    """The client profile `pp` is sent with: `client_profile`, its timeout and max_workers replaced when an override
    for `pp` is active (both are transport: the fingerprint, the request hash and the ledger header stay those of the
    panel)."""
    changes = {}
    seconds = (_TIMEOUT_OVERRIDES.get() or {}).get(pp.profile_id)
    if seconds is not None:
        changes["timeout"] = raised_timeout(pp.profile_id, seconds, client_profile.timeout)
    n = (_WORKERS_OVERRIDES.get() or {}).get(pp.profile_id)
    if n is not None:
        changes["max_workers"] = raised_workers(pp.profile_id, n, client_profile.max_workers)
    return replace(client_profile, **changes) if changes else client_profile


def workers(concurrency: int | None, client_profile) -> int:
    """Requests in flight at once: the profile's max_workers, or `concurrency` when given and smaller. A caller's
    concurrency never raises a route's limit (max_workers is transport: outside the fingerprint and request hash);
    only a workers override does, through the profile `send_profile` returns."""
    limit = client_profile.max_workers
    if concurrency is None:
        return limit
    if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency < 1:
        raise ValueError(f"concurrency must be a positive int or None, got {concurrency!r}")
    return min(concurrency, limit)


def route_of(client_profile) -> str:
    """provider:wire model, e.g. "huggingface:meta-llama/Llama-3.3-70B-Instruct:ovhcloud"."""
    return f"{client_profile.provider}:{client_profile.request_model}"


@dataclass(frozen=True)
class Cell:
    """One request. ``est_*`` and ``cache_*`` are planning hints used only by estimates."""
    item_id: str
    stage: str
    messages: tuple
    repeat_idx: int = 0
    generation: dict = field(default_factory=dict)
    est_input_tokens: int | None = None
    est_visible_tokens: int | None = None
    cache_key: str | None = None
    cache_prefix_tokens: int = 0
    labels: dict = field(default_factory=dict)   # copied into the ledger record; never sent, not in the request hash

    def __post_init__(self):
        object.__setattr__(self, "messages", tuple(canonical_messages(self.messages)))
        object.__setattr__(self, "generation", dict(self.generation))
        object.__setattr__(self, "labels", dict(self.labels))
        taken = sorted(k for k in self.labels if not isinstance(k, str) or k in RECORD_FIELDS)
        if taken:
            raise ValueError(f"cell labels may not be named like ledger record fields: {taken}")
        if not self.item_id or not self.stage:
            raise ValueError("cells need item_id and stage")
        for name in ("repeat_idx", "cache_prefix_tokens"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name in ("est_input_tokens", "est_visible_tokens"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True)
class Assumptions:
    """Token assumptions for estimates (the LOW and HIGH cost scenarios are built by ``scenario``)."""
    reasoning_tokens: int
    visible_out_tokens: int = 300
    tokenizer_factor: float = 1.0
    prefix_cache: bool = False
    contingency: float = 1.0


def scenario(pp: PanelProfile, name: str) -> Assumptions:
    """The two cost scenarios: LOW = prefix caching, reference tokens; HIGH = no caching, provider
    tokenizer factor, +10% contingency. Reasoning budgets come from the panel file."""
    if name not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    reasoning = pp.expected_reasoning_tokens[name]
    if name == "low":
        return Assumptions(reasoning_tokens=reasoning, prefix_cache=True)
    return Assumptions(reasoning_tokens=reasoning, tokenizer_factor=pp.tokenizer_factor, contingency=1.10)


def harness_digest() -> str:
    """Content hash of the inference layer source (LF-normalised), recorded with every call."""
    base = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(base.rglob("*.py")):
        h.update(path.relative_to(base).as_posix().encode() + b"\0")
        h.update(path.read_bytes().replace(b"\r\n", b"\n") + b"\0")
    return h.hexdigest()


def _effective_generation(pp: PanelProfile, cell: Cell) -> dict:
    merged = {**pp.spec.get("generation", {}), **cell.generation}
    return {k: v for k, v in merged.items() if v is not None}


def _input_tokens(cell: Cell) -> int:
    return cell.est_input_tokens if cell.est_input_tokens is not None else estimate_message_tokens(cell.messages)


def attempt_bound_usd(pp: PanelProfile, book: PriceBook, input_tokens: int, generation: dict) -> float:
    """Per-attempt worst case of a request with reference ``input_tokens`` and per-request ``generation``
    overrides: all input uncached (with margin) + the full output cap, at the dearest variant."""
    max_out = {**pp.spec.get("generation", {}), **generation}.get("max_output_tokens")
    if isinstance(max_out, bool) or not isinstance(max_out, int) or max_out <= 0:
        raise ConfigurationError(f"{pp.profile_id}: cannot bound cost without max_output_tokens")
    bound_in = math.ceil(input_tokens * pp.tokenizer_factor * CAP_INPUT_MARGIN)
    usage = synthetic_usage(input_uncached=bound_in, output_visible=max_out)
    return price_call(pp, usage, book=book)["usd"]


def upper_bound_usd(cell: Cell, pp: PanelProfile, book: PriceBook, input_tokens: int | None = None) -> float:
    """``attempt_bound_usd`` of one cell (reference tokens of its messages unless ``input_tokens`` is given)."""
    tokens = _input_tokens(cell) if input_tokens is None else input_tokens
    return attempt_bound_usd(pp, book, tokens, cell.generation)


def estimate_cells(cells, pp: PanelProfile, book: PriceBook, assumptions: Assumptions, *,
                   observed_at: datetime | None = None) -> dict:
    """Expected tokens and dollars under ``assumptions``; no network. Assumed output is clipped
    at the request's max_output_tokens (``clipped_calls`` counts where that happened)."""
    seen_prefixes: set[str] = set()
    factor = assumptions.tokenizer_factor
    totals = {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0, "usd_upper_bound": 0.0,
              "clipped_calls": 0}
    by_stage: dict[str, dict] = {}
    for cell in cells:
        base_tokens = _input_tokens(cell)
        upper = upper_bound_usd(cell, pp, book, base_tokens)  # also rejects a request without an output cap
        tokens = math.ceil(base_tokens * factor)
        prefix = min(math.ceil(cell.cache_prefix_tokens * factor), tokens)
        rates = book.rates(pp.price_key, observed_at=observed_at, input_tokens=tokens)
        write = cached = 0
        if assumptions.prefix_cache and cell.cache_key and prefix and rates.cached_input is not None:
            if cell.cache_key in seen_prefixes:
                cached = prefix
            else:
                write = prefix
                seen_prefixes.add(cell.cache_key)
        visible = cell.est_visible_tokens if cell.est_visible_tokens is not None else assumptions.visible_out_tokens
        visible, reasoning = math.ceil(visible * factor), math.ceil(assumptions.reasoning_tokens * factor)
        cap = _effective_generation(pp, cell).get("max_output_tokens")
        if visible + reasoning > cap:  # the provider cannot bill past the output cap
            totals["clipped_calls"] += 1
            reasoning = min(reasoning, cap)
            visible = cap - reasoning
        usage = synthetic_usage(input_uncached=tokens - write - cached, input_cached=cached, cache_write=write,
                                output_visible=visible, reasoning=reasoning)
        usd = price_call(pp, usage, book=book, observed_at=observed_at)["usd"] * assumptions.contingency
        stage = by_stage.setdefault(cell.stage, {"calls": 0, "input_tokens": 0, "output_tokens": 0, "usd": 0.0})
        for bucket in (totals, stage):
            bucket["calls"] += 1
            bucket["input_tokens"] += usage["input_total"]
            bucket["output_tokens"] += usage["output_total"]
            bucket["usd"] += usd
        totals["usd_upper_bound"] += upper
    return {"mode": "dry_run", "profile_id": pp.profile_id, "price_key": pp.price_key,
            "price_book_sha256": book.sha256, "token_method": token_method(),
            "assumptions": asdict(assumptions), **totals, "by_stage": by_stage}


def _reasoning_leak(pp: PanelProfile, result, usage: dict) -> bool:
    """For reasoning_expected=false: non-whitespace reasoning text (reasoning_content or an inline
    <think> block) or a reported reasoning count > 0. Whitespace-only reasoning is no leak."""
    if pp.reasoning_expected:
        return False
    inline = any(has_reasoning_text(block) for block in THINK_BLOCK.findall(result.text or ""))
    return has_reasoning_text(result.reasoning) or inline or bool(usage.get("reasoning"))


def _stub(cell: Cell, pp: PanelProfile, hash_: str) -> dict:
    """The identifying fields of a request's record, kept with its reservation (a recovered record gets them)."""
    return {"item_id": cell.item_id, "stage": cell.stage, "repeat_idx": cell.repeat_idx, "params": cell.generation,
            "profile_id": pp.profile_id, "user_message_suffix": pp.user_message_suffix, **cell.labels}


def _settlement(costs: list[str], bound: float, answered_usd: float | None) -> tuple[float, float]:
    """(usd_counted, usd_pending_unknown) of one request from its attempts' costs (attempt_cost, in order):
    the answer's priced usd (its bound when it could not be priced) + the bound of each unknown-cost attempt."""
    unknown = costs.count("unknown") * bound
    if costs and costs[-1] == "answered":
        if answered_usd is None:
            return bound + unknown, bound + unknown
        return answered_usd + unknown, unknown
    return unknown, unknown


def _record(cell, pp, fingerprint, hash_, result, bound, reservation, costs, ref_tokens, book, digest_) -> dict:
    spec = pp.spec
    raw_usage = provider_usage(spec["adapter"], result.raw_response, result.usage)
    usage = normalize_usage(spec["adapter"], spec.get("provider", ""), raw_usage,
                            reasoning_text_present=has_reasoning_text(result.reasoning),
                            reasoning_content=message_reasoning_content(spec["adapter"], result.raw_response))
    attempts = result.attempts
    observed = bool(attempts) and 200 <= (attempts[-1].get("http_status") or 0) < 300
    price = None
    if observed:
        price = price_call(pp, usage, book=book, observed_at=datetime.fromisoformat(attempts[-1]["started_at"]))
    counted, pending = _settlement(costs, bound, price["usd"] if observed else None)
    return {
        "kind": "call", "request_hash": hash_, "item_id": cell.item_id, "stage": cell.stage,
        "repeat_idx": cell.repeat_idx, "params": cell.generation,
        "profile_id": pp.profile_id, "profile_fingerprint": fingerprint,
        "user_message_suffix": pp.user_message_suffix,
        "requested_model": result.requested_model, "served_model": result.served_model,
        "system_fingerprint": result.system_fingerprint, "observed_provider": result.observed_provider,
        "server_request_id": result.server_request_id,
        "outcome": "observed" if observed else "no_response", "status": result.status,
        "finish_reason": result.finish_reason, "text": result.text, "reasoning_text": result.reasoning,
        "reasoning_leak": _reasoning_leak(pp, result, usage), "usage": usage, "price": price,
        "usd": price["usd"] if price else None, "usd_upper_bound": bound, "usd_reserved": reservation.usd,
        "usd_counted": counted, "usd_pending_unknown": pending,
        "attempts": attempts, "error": result.error, "request": result.request,
        "response_headers": result.response_headers, "raw_response": result.raw_response,
        "raw_text": result.raw_text if result.raw_response is None else None,
        "ref_input_tokens": ref_tokens, "token_method": token_method(), "harness_digest": digest_,
        "started_at": attempts[0]["started_at"] if attempts else None, "finished_at": utc_now(), **cell.labels,
    }


def ledger_header(pp: PanelProfile, book: PriceBook) -> dict:
    """Line 1 of a call ledger of `pp` priced on `book`: the run identity (procedures.readers.ledger) plus the
    profile spec, the harness digest and the token method."""
    return {"profile_id": pp.profile_id, "profile_fingerprint": pp.client_profile().fingerprint,
            "price_book_sha256": book.sha256, "profile_spec": pp.spec, "user_message_suffix": pp.user_message_suffix,
            "harness_digest": harness_digest(), "token_method": token_method()}


def run_cells(cells, profile: PanelProfile, *, cap_usd: float, concurrency: int | None = None,
              dry_run: bool = False, ledger_path: Path | None = None, book: PriceBook | None = None,
              assumptions: Assumptions | None = None, observed_at: datetime | None = None,
              transport=None, sleep=None, guard: SpendGuard | None = None) -> dict:
    pp, book = profile, book or load_price_book()
    cells = [replace(cell, messages=pp.messages_to_send(cell.messages)) for cell in cells]
    client_profile = pp.client_profile()
    fingerprint = client_profile.fingerprint
    hashes = []
    for cell in cells:  # validate every request before any estimate or paid call
        endpoint, payload = build_request(client_profile, list(cell.messages), cell.generation)
        check_wire(pp, endpoint, payload)
        hashes.append(request_hash(pp.profile_id, fingerprint, cell.messages, cell.generation, cell.repeat_idx,
                                   user_message_suffix=pp.user_message_suffix))
    if len(set(hashes)) != len(hashes):
        raise ValueError("duplicate requests in cells; distinct samples need distinct repeat_idx")
    assumptions = assumptions or scenario(pp, "high")
    if dry_run:
        estimate = estimate_cells(cells, pp, book, assumptions, observed_at=observed_at)
        estimate.update(cap_usd=cap_usd, exceeds_cap=estimate["usd"] > cap_usd)
        return estimate
    if ledger_path is None:
        raise ValueError("a live run needs ledger_path")
    key_env = client_profile.api_key_env
    if key_env and not os.environ.get(key_env):
        raise ConfigurationError(f"Set credential environment variable {key_env}")
    guard = guard or _ACTIVE_GUARD.get() or SpendGuard()
    route = route_of(client_profile)
    header = ledger_header(pp, book)
    ledger = Ledger(ledger_path, header, cap_usd=cap_usd)
    with ledger.executor():
        recovered = ledger.recover_unsettled()
        pending = [(c, h, estimate_message_tokens(c.messages)) for c, h in zip(cells, hashes)
                   if ledger.done(h) is None]
        projected = estimate_cells([replace(c, est_input_tokens=t) for c, _, t in pending], pp, book,
                                   assumptions)["usd"]
        committed = ledger.committed_usd()
        if committed + projected > cap_usd:
            raise SpendCapExceeded(f"projected ${projected:.4f} ({len(pending)} calls, {assumptions}) + committed "
                                   f"${committed:.4f} exceeds cap ${cap_usd:.4f}; nothing was sent")
        if pending:
            guard.preflight(projected, f"{pp.profile_id}: {len(pending)} pending calls (HIGH ${projected:.4f})")
            guard.breaker.check(route)
        sending = send_profile(pp, client_profile)   # resolved in the caller's context: sends run on worker threads
        sent, stopped = _send_all(pending, pp, sending, fingerprint, ledger, guard, route, book,
                                  concurrency, transport, sleep, header["harness_digest"])
    state = guard.breaker.state()
    if pending and guard.breaker.blocked(route):
        note = f" (stopped after {sent} of {len(pending)} pending calls)"
        if state["paused"]:
            raise RunPaused(state["paused"] + note)
        raise RouteTripped(state["tripped"][route] + note)
    if stopped is not None:
        raise SpendCapExceeded(f"stopped after {sent} of {len(pending)} pending calls: {stopped}")
    by_hash = {r["request_hash"]: r for r in ledger.records()}
    return {"mode": "live", "profile_id": pp.profile_id, "ledger": str(ledger.path), "sent": sent,
            "skipped_done": len(cells) - len(pending), "spent_usd": ledger.spent_usd(),
            "pending_unknown_usd": ledger.pending_unknown_usd(), "recovered": recovered, "route": route,
            "breaker": state, "records": [by_hash[h] for h in hashes]}


def _send_all(pending, pp, client_profile, fingerprint, ledger, guard, route, book, concurrency, transport, sleep,
              digest_) -> tuple[int, Exception | None]:
    """Send `pending` (cell, hash, reference tokens) with at most `workers(concurrency, client_profile)` in flight:
    (requests sent, the SpendCapExceeded that stopped the loop or None). Stops submitting on a failed send, a refused
    reservation or a blocked route; every submitted request settles before this returns; a failed send's exception is
    raised. The Breaker sees each request once, when the client returns it (request_cost of its attempts); a request
    cut by the Breaker itself, or stopped by a harness error, is not counted."""
    base_transport = transport or client_module.http_transport
    max_attempts = client_profile.max_attempts
    failed = threading.Event()

    def send(cell, hash_, reservation, bound, ref_tokens):
        log = []    # the attempts this request actually sent, with their cost class (attempt_cost)

        def attempt(url, headers, payload, timeout):
            guard.breaker.check(route)      # RouteTripped / RunPaused: not sent; propagates out of the client
            entry = {"attempt": len(log) + 1, "started_at": utc_now()}
            try:
                code, response_headers, raw = base_transport(url, headers, payload, timeout)
            except Exception as exc:    # re-raised: the client retries transport errors, anything else stops the run
                log.append({**entry, "cost": "unknown", "error": {"kind": "transport", "message": repr(exc)}})
                if isinstance(exc, http.client.HTTPException):
                    # A body that ends early (IncompleteRead) or a bad status line is a transport failure of this
                    # attempt, but http.client raises it as HTTPException, which the client does not retry (it
                    # retries URLError / TimeoutError / ConnectionError / OSError). Re-raise it in that family so it
                    # costs one retry instead of stopping the run (a chunked DeepSeek response once stopped a
                    # builder run 2,085 call records in).
                    raise ConnectionError(repr(exc)) from exc
                raise
            log.append({**entry, "cost": attempt_cost(code), "http_status": code})
            return code, response_headers, raw

        client = Client(client_profile, transport=attempt, sleep=sleep)
        result = None
        try:
            try:
                result = client.generate(list(cell.messages), cell.generation or None, request_id=hash_)
                record = _record(cell, pp, fingerprint, hash_, result, bound, reservation,
                                 [a["cost"] for a in log], ref_tokens, book, digest_)
                cost = request_cost([a["cost"] for a in log])
                guard.breaker.observe(route, "timeout" if cost == "unknown" and route_timed_out(record) else cost)
            except (RouteTripped, RunPaused) as exc:
                counted, pending_unknown = _settlement([a["cost"] for a in log], bound, None)
                record = {**_stub(cell, pp, hash_), "kind": "call", "request_hash": hash_,
                          "profile_fingerprint": fingerprint, "outcome": "no_response", "status": None, "usd": None,
                          "usd_upper_bound": bound, "usd_reserved": reservation.usd, "usd_counted": counted,
                          "usd_pending_unknown": pending_unknown,
                          "attempts": [{k: v for k, v in a.items() if k != "cost"} for a in log],
                          "error": {"kind": "circuit_open", "message": str(exc)}, "ref_input_tokens": ref_tokens,
                          "harness_digest": digest_, "finished_at": utc_now()}
            except Exception as exc:
                # Keep whatever came back (possibly a paid answer) before stopping the run; every attempt sent
                # counts its bound (its cost is not known here).
                ledger.append({**_stub(cell, pp, hash_), "kind": "call", "request_hash": hash_,
                               "outcome": "harness_error", "error": {"kind": "harness", "message": repr(exc)},
                               "result": result.to_dict() if result is not None else None, "usd": None,
                               "usd_upper_bound": bound, "usd_reserved": reservation.usd,
                               "usd_counted": bound * len(log), "usd_pending_unknown": bound * len(log),
                               "harness_digest": digest_, "finished_at": utc_now()}, reservation)
                raise
            ledger.append(record, reservation)
        except BaseException:
            failed.set()
            raise

    in_flight = workers(concurrency, client_profile)
    slots = threading.BoundedSemaphore(in_flight)
    stopped = None
    futures = []
    with ThreadPoolExecutor(max_workers=in_flight) as executor:
        for cell, hash_, ref_tokens in pending:
            slots.acquire()
            if failed.is_set() or guard.breaker.blocked(route):
                slots.release()
                break
            bound = upper_bound_usd(cell, pp, book, ref_tokens)
            try:
                reservation = guard.reserve(ledger, bound * max_attempts, request_hash=hash_,
                                            stub=_stub(cell, pp, hash_),
                                            detail={"attempt_bound_usd": bound, "max_attempts": max_attempts,
                                                    "route": route})
            except SpendCapExceeded as exc:
                slots.release()
                stopped = exc
                break
            future = executor.submit(send, cell, hash_, reservation, bound, ref_tokens)
            future.add_done_callback(lambda _: slots.release())
            futures.append(future)
    for future in futures:
        future.result()
    return len(futures), stopped

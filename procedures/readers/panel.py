"""The pinned model panel and the provider-trap checks.

``data/panel/panel.json`` (``panel_version`` 2026-09-26) holds 17 profiles:
the 7 panel models, the GPT-6 reasoning=none/T=0 control, 4 priced alternates, the 3
extension readers (Qwen3-32B, access tier C2; gpt-oss-120b, C3; Llama-3.3-70B on ovhcloud, B),
1 candidate (Llama-3.3-70B on deepinfra, the route the study's panel specification names: no
priced route in the router snapshot, never sent) and 1 negative control. Each ``client``
block is an inference profile of the HTTP client (procedures.readers.client).
13 profiles come from the original profile file ``data/panel/profiles.json``, every
field changed against it listed with its reason under the panel file's ``pilot_profiles``
field (the field changes against the original profile file); settings
of the other profiles (extensions and the candidate) that depart from the study's panel
specification are registered, each with its reason, in the panel's ``plan_deviations``.
The metadata around each ``client`` block (tier, access tier, slot, price key,
cost-scenario reasoning budget, route snapshot, optional ``user_message_suffix``,
``thinking_soft_switch`` and ``answer_from_reasoning_field``) is project-owned.

``answer_from_reasoning_field`` (true exactly when ``thinking_soft_switch``) is a
reader-side mapping of the response fields, not a request change:
procedures.scoring.score_path scores an answered response with a blank content on its
reasoning_content when that parses as exactly one JSON object (the nscale route's vLLM server
can return the whole answer there).

``tier`` is the governance tier A | B | C (A: cloud frontier models, the procedure text
leaves the operator's site; B: open-weight models that need a multi-GPU server; C:
open-weight models on one card); ``access_tier`` is A | B | C3 | C2 | C1, tier C split by
the card (C3 one 80 GB data-centre card, C2 one 48 GB workstation card, C1 one 24 GB
consumer card); ``slot`` is the panel position A1 A2 B1 B2 C1 C2 C3 (the panel's slots
C1/C2/C3 are all access tier C1; the extension readers fill access tiers C2 and C3).
Negative and candidate profiles are never sendable: ``PanelProfile.client_profile``
refuses them, so no run can send them.

``route_context`` (HF profiles) is the pinned route's context length in the router snapshot ``route_snapshot``
(tests check it offline against the snapshot): procedures.readers.cells refuses to send a request whose planned input
with the route tokenizer allowance plus the output cap exceeds it. A candidate (no priced route) has none. Non-HF
profiles have no snapshot and so no route context.

The client spec's ``max_workers`` is the most requests of the profile in flight at once
(procedures.readers.executor.run_cells never exceeds it; 1 for Llama-3.3-70B on ovhcloud: its gateway does not serve
concurrent 100k-token prefills). It is transport, not request: the client's fingerprint excludes
it, so neither the request hash nor the ledger identity depends on it.

``user_message_suffix`` is a panel-level request transform outside the client's
spec (so outside its fingerprint): ``PanelProfile.messages_to_send`` appends it to
the last user message, never to a system message. ``procedures.readers.executor.run_cells``
applies it before validation, hashing and sending; the ledger binds it into the run identity.

``check_wire`` inspects the exact wire payload the client's adapter builds and
raises ``TrapViolation`` if a known provider trap (a request setting the provider rejects,
ignores or silently defaults, or a missing pin) would be hit. ``load_panel`` runs it on every
profile, and ``procedures.readers.executor.run_cells`` runs it on every request before
anything is sent.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import json
from pathlib import Path
import re
import warnings

from procedures.readers.client import ConfigurationError, InferenceProfile
from procedures.readers.client.adapters import build_request
from procedures import ROOT
from procedures.readers.pricing import file_sha256_lf

DEFAULT_PANEL = ROOT / "data" / "panel" / "panel.json"
PANEL_SLOTS = ("A1", "A2", "B1", "B2", "C1", "C2", "C3")
ROLES = ("panel", "control", "alternate", "extension", "candidate", "negative")
ACCESS_TIERS = {"A": ("A",), "B": ("B",), "C": ("C1", "C2", "C3")}  # governance tier -> its access tiers
PROBE_MESSAGES = ({"role": "user", "content": "Return only the following word, without explanation: READY"},)
QWEN3_HYBRID = re.compile(r"Qwen/Qwen3-\d+B$")  # hybrid-thinking checkpoints (not the 2507 Instruct line)
QWEN3_SOFT_SWITCH = "/no_think"  # Qwen3's documented in-message switch
# HF routes that answer but do not forward chat_template_kwargs (measured in the connectivity smoke run before the main
# run: Qwen3-8B on nscale returned 104 reasoning tokens with enable_thinking=false and 6 with " /no_think")
ROUTES_IGNORING_CHAT_TEMPLATE_KWARGS = frozenset({"nscale"})
GPT_OSS = re.compile(r"openai/gpt-oss-\d+b$")  # reasoning cannot be disabled; the model card's default effort is medium


class PanelError(ValueError):
    """The panel file is inconsistent with its pins or roles."""


class TrapViolation(ValueError):
    """A request would hit a documented provider trap."""


@dataclass(frozen=True)
class PanelProfile:
    profile_id: str
    role: str
    tier: str | None
    slot: str | None
    price_key: str | None
    tokenizer_factor: float
    expected_reasoning_tokens: dict
    reasoning_expected: bool
    spec: dict
    notes: str
    user_message_suffix: str | None = None
    access_tier: str | None = None
    thinking_soft_switch: bool = False
    route_snapshot: str | None = None
    answer_from_reasoning_field: bool = False
    route_context: int | None = None

    def client_profile(self) -> InferenceProfile:
        if self.role == "negative":
            raise PanelError(f"{self.profile_id} is a negative control and must never be sent")
        if self.role == "candidate":
            raise PanelError(f"{self.profile_id} is a candidate without a priced route and must never be sent")
        return self._client_profile

    @cached_property
    def _client_profile(self) -> InferenceProfile:
        """The (frozen) client profile of `spec`, built once per panel profile: the planners ask for it per cell."""
        return InferenceProfile.from_dict(self.spec)

    @property
    def fingerprint(self) -> str:
        return self.client_profile().fingerprint

    def messages_to_send(self, messages) -> list:
        """Copies of ``messages`` with ``user_message_suffix`` appended to the last user message."""
        out = [dict(message) for message in messages]
        if self.user_message_suffix is None:
            return out
        for message in reversed(out):
            if message.get("role") == "user":
                message["content"] = message["content"] + self.user_message_suffix
                return out
        raise PanelError(f"{self.profile_id}: user_message_suffix needs a user message to append to")


@dataclass(frozen=True)
class Panel:
    path: Path
    sha256: str
    version: str
    price_snapshot: Path
    price_snapshot_sha256: str
    profiles: dict

    def get(self, profile_id: str) -> PanelProfile:
        if profile_id not in self.profiles:
            raise PanelError(f"unknown profile {profile_id!r}; choose from {sorted(self.profiles)}")
        return self.profiles[profile_id]

    def members(self) -> list:
        """The seven panel models in slot order."""
        by_slot = {p.slot: p for p in self.profiles.values() if p.role == "panel"}
        return [by_slot[slot] for slot in PANEL_SLOTS]


def _out_cap(payload: dict, key: str, profile_id: str) -> None:
    if key not in payload:
        raise TrapViolation(f"{profile_id}: every request needs an explicit output cap ({key})")


def check_wire(pp: PanelProfile, endpoint: str, payload: dict) -> None:
    spec, pid = pp.spec, pp.profile_id
    provider, model = spec.get("provider"), spec["model"]
    if provider == "openai":
        if "max_tokens" in payload:
            raise TrapViolation(f"{pid}: OpenAI rejects max_tokens for reasoning models; map to max_completion_tokens")
        _out_cap(payload, "max_completion_tokens", pid)
        if model.startswith("gpt-6"):
            effort = payload.get("reasoning_effort")
            if effort is None:
                raise TrapViolation(f"{pid}: GPT-6 reasoning_effort must be pinned (server default is medium)")
            if effort != "none" and {"temperature", "top_p"} & payload.keys():
                raise TrapViolation(f"{pid}: GPT-6 with reasoning_effort={effort} must not send temperature/top_p")
    elif provider == "anthropic":
        thinking = payload.get("thinking")
        if not isinstance(thinking, dict) or "type" not in thinking:
            raise TrapViolation(f"{pid}: Sonnet 5 thinking must be set explicitly (default is adaptive)")
        if {"temperature", "top_p", "top_k"} & payload.keys():
            raise TrapViolation(f"{pid}: Sonnet 5 returns HTTP 400 for non-default temperature/top_p/top_k")
        if pp.slot == "A2" and thinking["type"] != "disabled":
            raise TrapViolation(f"{pid}: panel slot A2 requires thinking disabled")
        _out_cap(payload, "max_tokens", pid)
    elif provider == "google-ai-studio":
        config = payload.get("generationConfig", {})
        level = config.get("thinkingConfig", {}).get("thinkingLevel")
        if level != "low":
            raise TrapViolation(f"{pid}: Gemini 3.8 Flash is pinned to thinkingLevel=low, got {level!r}")
        if config.get("temperature") != 1.0:
            raise TrapViolation(f"{pid}: Gemini 3 must run at temperature 1.0, got {config.get('temperature')!r}")
        _out_cap(config, "maxOutputTokens", pid)
    elif provider == "deepseek":
        mode = payload.get("thinking", {}).get("type") if isinstance(payload.get("thinking"), dict) else None
        if mode not in {"enabled", "disabled"}:
            raise TrapViolation(f"{pid}: DeepSeek thinking must be set explicitly (default is enabled)")
        if mode == "enabled" and ({"temperature", "top_p"} & payload.keys()
                                  or spec.get("capabilities", {}).get("temperature") is not False):
            raise TrapViolation(f"{pid}: DeepSeek ignores temperature in thinking mode; declare it unsupported")
        _out_cap(payload, "max_tokens", pid)
    elif provider == "huggingface":
        route = spec.get("hf_provider")
        if not route:
            raise TrapViolation(f"{pid}: HF profiles must pin hf_provider")
        if payload.get("model") != f"{model}:{route}":
            raise TrapViolation(f"{pid}: wire model {payload.get('model')!r} is not pinned to {route}")
        if QWEN3_HYBRID.match(model) and not pp.reasoning_expected:
            kwargs = payload.get("chat_template_kwargs", {})
            if kwargs.get("enable_thinking") is not False:
                raise TrapViolation(f"{pid}: Qwen3 hybrid model needs chat_template_kwargs.enable_thinking=false")
            if route in ROUTES_IGNORING_CHAT_TEMPLATE_KWARGS and\
                    not (pp.user_message_suffix or "").rstrip().endswith(QWEN3_SOFT_SWITCH):
                raise TrapViolation(f"{pid}: {route} does not forward chat_template_kwargs; Qwen3 needs the "
                                    f"{QWEN3_SOFT_SWITCH} soft switch as user_message_suffix")
        if GPT_OSS.match(model) and payload.get("reasoning_effort") is None:
            raise TrapViolation(f"{pid}: gpt-oss reasoning cannot be disabled; reasoning_effort must be pinned "
                                "(the default is medium)")
        _out_cap(payload, "max_tokens", pid)
    else:
        raise TrapViolation(f"{pid}: no trap rules for provider {provider!r}")
    if pp.user_message_suffix is not None:
        users = [m for m in payload.get("messages", []) if isinstance(m, dict) and m.get("role") == "user"]
        content = users[-1].get("content") if users else None
        if not isinstance(content, str) or not content.endswith(pp.user_message_suffix):
            raise TrapViolation(f"{pid}: user_message_suffix {pp.user_message_suffix!r} is not on the last user "
                                "message of the wire payload")
    if not endpoint.startswith("https://"):
        raise TrapViolation(f"{pid}: endpoint must be https, got {endpoint}")


def _profile(profile_id: str, raw: dict) -> PanelProfile:
    fields = ("role", "tier", "slot", "price_key", "tokenizer_factor",
              "expected_reasoning_tokens", "reasoning_expected", "client", "notes")
    missing = [f for f in fields if f not in raw]
    if missing:
        raise PanelError(f"{profile_id}: missing fields {missing}")
    if raw["role"] not in ROLES:
        raise PanelError(f"{profile_id}: role must be one of {ROLES}")
    reasoning = raw["expected_reasoning_tokens"]
    if set(reasoning) != {"low", "high"} or any(type(v) is not int or v < 0 for v in reasoning.values()):
        raise PanelError(f"{profile_id}: expected_reasoning_tokens needs non-negative int low/high")
    factor = raw["tokenizer_factor"]
    if isinstance(factor, bool) or not isinstance(factor, (int, float)) or not 0.5 <= factor <= 2.0:
        raise PanelError(f"{profile_id}: tokenizer_factor must be in [0.5, 2.0]")
    suffix = raw.get("user_message_suffix")
    if suffix is not None and (not isinstance(suffix, str) or not suffix.strip() or "\r" in suffix):
        raise PanelError(f"{profile_id}: user_message_suffix must be a non-blank string without CR")
    access = raw.get("access_tier")
    if raw["role"] != "negative" and access not in ACCESS_TIERS.get(raw["tier"], ()):
        raise PanelError(f"{profile_id}: access_tier {access!r} is not an access tier of tier "
                         f"{raw['tier']!r} {ACCESS_TIERS.get(raw['tier'], ())}")
    soft = raw.get("thinking_soft_switch", False)
    if not isinstance(soft, bool) or soft != (suffix or "").rstrip().endswith(QWEN3_SOFT_SWITCH):
        raise PanelError(f"{profile_id}: thinking_soft_switch must be true exactly when user_message_suffix ends "
                         f"with {QWEN3_SOFT_SWITCH}")
    mapped = raw.get("answer_from_reasoning_field", False)
    if not isinstance(mapped, bool) or mapped != soft:
        raise PanelError(f"{profile_id}: answer_from_reasoning_field must be true exactly when thinking_soft_switch is")
    route = raw.get("route_snapshot")
    if raw["role"] != "negative" and (raw["client"].get("provider") == "huggingface") != isinstance(route, str):
        raise PanelError(f"{profile_id}: route_snapshot is required for, and only for, HF profiles")
    if raw["role"] == "candidate" and raw["price_key"] is not None:
        raise PanelError(f"{profile_id}: a candidate has no priced route, so no price_key")
    context = raw.get("route_context")
    needs_context = isinstance(route, str) and raw["role"] not in ("negative", "candidate")
    if context is not None and (isinstance(context, bool) or not isinstance(context, int) or context <= 0):
        raise PanelError(f"{profile_id}: route_context must be a positive int")
    if needs_context != (context is not None):
        raise PanelError(f"{profile_id}: route_context is required for, and only for, sendable HF profiles "
                         "(the context length of the pinned route in its router snapshot)")
    return PanelProfile(profile_id=profile_id, role=raw["role"], tier=raw["tier"], slot=raw["slot"],
                        price_key=raw["price_key"], tokenizer_factor=float(factor),
                        expected_reasoning_tokens=dict(reasoning), reasoning_expected=bool(raw["reasoning_expected"]),
                        spec=raw["client"], notes=raw["notes"], user_message_suffix=suffix, access_tier=access,
                        thinking_soft_switch=soft, route_snapshot=route, answer_from_reasoning_field=mapped,
                        route_context=context)


def load_panel(path: Path = DEFAULT_PANEL) -> Panel:
    path = Path(path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    snapshot = ROOT / doc["price_snapshot"]["path"]
    snapshot_sha = file_sha256_lf(snapshot)
    if snapshot_sha != doc["price_snapshot"]["sha256_lf"]:
        warnings.warn(f"price snapshot {snapshot} changed since the panel was pinned (sha256 {snapshot_sha[:12]}, "
                      f"pinned {doc['price_snapshot']['sha256_lf'][:12]})", stacklevel=2)
    version = doc.get("panel_version")
    if not isinstance(version, str) or not version:
        raise PanelError(f"{path}: panel_version is required")
    profiles = {pid: _profile(pid, raw) for pid, raw in doc["profiles"].items()}
    for pp in profiles.values():
        if pp.role == "negative":
            try:
                InferenceProfile.from_dict(pp.spec)
            except ConfigurationError:
                continue
            raise PanelError(f"{pp.profile_id}: negative control was accepted by the client configuration")
        if pp.role != "candidate" and pp.price_key is None:
            raise PanelError(f"{pp.profile_id}: sendable profiles need a price_key")
        # a candidate is never sendable, but its request must already pass the trap checks
        endpoint, payload = build_request(InferenceProfile.from_dict(pp.spec), pp.messages_to_send(PROBE_MESSAGES))
        check_wire(pp, endpoint, payload)
    registered = (doc.get("plan_deviations") or {}).get("profiles", {})
    for pid, fields in registered.items():
        if pid not in profiles or profiles[pid].role not in ("extension", "candidate"):
            raise PanelError(f"plan_deviations names {pid!r}, which is not an extension or candidate profile")
        if not isinstance(fields, dict) or not fields or not all(
                isinstance(reason, str) and reason.strip() for reason in fields.values()):
            raise PanelError(f"plan_deviations[{pid!r}] must map each departing setting to a non-blank reason")
    slots = sorted(p.slot for p in profiles.values() if p.role == "panel")
    if slots != sorted(PANEL_SLOTS):
        raise PanelError(f"panel roles must fill slots {PANEL_SLOTS} exactly once, got {slots}")
    return Panel(path=path, sha256=file_sha256_lf(path), version=version, price_snapshot=snapshot,
                 price_snapshot_sha256=snapshot_sha, profiles=profiles)

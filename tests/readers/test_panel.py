"""Pinned panel: roles, pins, and every provider trap of the pilot profiles (data/panel/profiles.json: the provider
settings the panel started from) enforced on the wire payload."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest

from procedures.readers import PanelError, TrapViolation, check_wire, load_panel
from procedures.readers.client import ConfigurationError, InferenceProfile
from procedures.readers.client.adapters import build_request
from procedures.readers.panel import PROBE_MESSAGES

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "data" / "panel" / "profiles.json"


THIRD_B = "tierB_llama33_70b_hf_ovhcloud"   # the cheapest eligible B route (test below)
EXTENSION = {"tierC2_qwen3_32b_hf_nscale_nothink": "C2", "tierC3_gptoss120b_hf_deepinfra_low": "C3", THIRD_B: "B"}
CANDIDATE = "tierB_llama33_70b_hf_deepinfra"
HF_ROUTER = ROOT / "data" / "prices" / "router_snapshots" / "hf_router"
# How the panel's third B reader was chosen (cheapest eligible route), re-derived from the router snapshot.
# Published-checkpoint weights in GB (from the
# model cards, not in the snapshot); the tiers go by bf16 weights regardless of serving quantization and gpt-oss-120b
# (released in MXFP4 only, ~60-65 GB) is the C3 example, so the release precision is the one that counts.
EXCLUDED_FAMILIES = ("deepseek-ai/", "Qwen/")   # already on the panel as B1 / B2
CHECKPOINT_GB = {"meta-llama/Llama-3.1-8B-Instruct": 16.1, "meta-llama/Llama-3.3-70B-Instruct": 141.1,
                 "openai/gpt-oss-20b": 13.8, "openai/gpt-oss-120b": 65.3}
B_WEIGHTS_GB = 80          # tier B = multi-GPU, weights > 80 GB
MIN_CONTEXT = 65_536       # 64k: M0@full of the longest PJM manual (~92k tokens) needs more than 64k anyway
# Input share of the tokens a B reader is billed for in the main run's grid (HIGH scenario, reader cells + its own
# query-focused compressions, M6; the main grid's HIGH cost estimate: 39.22M in / 4.96M out for B2).
USUAL_INPUT_SHARE = 0.888


def test_panel_has_seven_members_in_the_seven_slots(panel):
    members = panel.members()
    assert [m.slot for m in members] == ["A1", "A2", "B1", "B2", "C1", "C2", "C3"]
    assert [m.tier for m in members] == ["A", "A", "B", "B", "C", "C", "C"]
    assert [m.access_tier for m in members] == ["A", "A", "B", "B", "C1", "C1", "C1"]   # slot names != access tiers
    assert len(panel.profiles) == 17 and panel.version == "2026-09-26"
    roles = sorted(p.role for p in panel.profiles.values())
    assert roles.count("control") == 1 and roles.count("alternate") == 4 and roles.count("negative") == 1
    assert {pid: p.access_tier for pid, p in panel.profiles.items() if p.role == "extension"} == EXTENSION
    assert [pid for pid, p in panel.profiles.items() if p.role == "candidate"] == [CANDIDATE]


def test_hf_routes_match_the_router_snapshot_and_the_price_book(panel, book):
    """Verified offline: every sendable HF profile's pinned route is live and priced in
    the 2026-09-24 router snapshot at the price book's in / out / context; the candidate has no priced route there
    (so it has no price key and is never sent)."""
    checked = set()
    for pid, pp in panel.profiles.items():
        if pp.role == "negative" or pp.spec["provider"] != "huggingface":
            continue
        path = ROOT / pp.route_snapshot
        assert path.parent == HF_ROUTER, pid
        data = json.loads(path.read_text(encoding="utf-8"))["data"]
        assert data["id"] == pp.spec["model"], pid
        routes = {r["provider"]: r for r in data["providers"]}
        route = routes.get(pp.spec["hf_provider"])
        checked.add(pid)
        if pp.role == "candidate":
            assert pp.price_key is None and (route is None or "pricing" not in route), pid
            assert pp.route_context is None, pid
            assert any("pricing" in r for r in routes.values())   # priced elsewhere: the project owner's route choice
            continue
        assert route is not None and route["status"] == "live" and "pricing" in route, pid
        entry = book.models[pp.price_key]
        assert pp.price_key == f"{pp.spec['model']}:{pp.spec['hf_provider']}"
        assert entry["in"] == pytest.approx(route["pricing"]["input"], rel=1e-9), pid
        assert entry["out"] == pytest.approx(route["pricing"]["output"], rel=1e-9), pid
        assert entry["context"] == route["context_length"], pid
        assert pp.route_context == route["context_length"], pid     # the route context ceiling
    assert checked == {"B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra", "C1_llama31_8b_hf_deepinfra",
                       "C2_qwen3_8b_hf_nscale_nothink", "C3_gptoss20b_hf_deepinfra_low", *EXTENSION, CANDIDATE}
    assert (book.models["Qwen/Qwen3-32B:nscale"]["in"], book.models["Qwen/Qwen3-32B:nscale"]["out"]) == (0.08, 0.25)
    assert (book.models["openai/gpt-oss-120b:deepinfra"]["in"],
            book.models["openai/gpt-oss-120b:deepinfra"]["out"]) == (0.037, 0.17)
    llama = book.models["meta-llama/Llama-3.3-70B-Instruct:ovhcloud"]
    assert (llama["in"], llama["out"], llama["context"]) == (0.74, 0.74, 131072)


def _snapshot_routes():
    for path in sorted(HF_ROUTER.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))["data"]
        for route in data["providers"]:
            yield data["id"], route


def test_third_b_reader_is_the_cheapest_eligible_route_in_the_router_snapshot(panel):
    """Offline: every open-weight model of the snapshot outside the DeepSeek / Qwen families is classified by its
    checkpoint size; the B ones' live, priced routes with >= 64k context are ranked by the blended price at the usual
    B-reader mix, and the panel's third B reader is the cheapest (Llama-3.3-70B@ovhcloud, the fallback named in
    advance, because no other model qualifies)."""
    models = {model for model, _ in _snapshot_routes() if not model.startswith(EXCLUDED_FAMILIES)}
    assert models == set(CHECKPOINT_GB)          # a new model in the snapshot has to be classified here first
    b_models = {m for m in models if CHECKPOINT_GB[m] > B_WEIGHTS_GB}
    assert b_models == {"meta-llama/Llama-3.3-70B-Instruct"}
    ranked, rejected = [], {}
    for model, route in _snapshot_routes():
        if model not in b_models:
            continue
        price, context = route.get("pricing"), route.get("context_length")
        if route["status"] != "live" or price is None or context is None or context < MIN_CONTEXT:
            rejected[route["provider"]] = (price is None, context)
            continue
        blend = USUAL_INPUT_SHARE * price["input"] + (1 - USUAL_INPUT_SHARE) * price["output"]
        ranked.append((blend, f"{model}:{route['provider']}", route.get("supports_structured_output")))
    ranked.sort()
    assert [(key, round(blend, 4)) for blend, key, _ in ranked] == [
        ("meta-llama/Llama-3.3-70B-Instruct:ovhcloud", 0.74), ("meta-llama/Llama-3.3-70B-Instruct:together", 1.04)]
    assert rejected == {"novita": (False, 12288), "featherless-ai": (True, None), "scaleway": (False, None)}
    assert ranked[0][2] is True                  # the route lists structured (JSON) output
    pp = panel.get(THIRD_B)
    assert (pp.role, pp.tier, pp.access_tier, pp.price_key) == ("extension", "B", "B", ranked[0][1])
    b_readers = {pid for pid, p in panel.profiles.items() if p.role in ("panel", "extension") and p.tier == "B"}
    assert b_readers == {"B1_dsv41flash_hf_deepinfra", "B2_qwen3_235b_2507_hf_deepinfra", THIRD_B}  # at least three


def test_a_candidate_is_never_sendable(panel, book):
    from procedures.readers import Cell, run_cells
    pp = panel.get(CANDIDATE)
    with pytest.raises(PanelError, match="candidate"):
        pp.client_profile()
    with pytest.raises(PanelError, match="candidate"):
        run_cells([Cell("i", "s", PROBE_MESSAGES)], pp, cap_usd=1.0, dry_run=True, book=book)


def _doc():
    return json.loads((ROOT / "data" / "panel" / "panel.json").read_text(encoding="utf-8"))


def _load(tmp_path, doc):
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return load_panel(path)


@pytest.mark.parametrize("pid, field, value, message", [
    ("C1_llama31_8b_hf_deepinfra", "access_tier", "B", "access_tier"),
    ("tierC2_qwen3_32b_hf_nscale_nothink", "access_tier", None, "access_tier"),
    ("C2_qwen3_8b_hf_nscale_nothink", "thinking_soft_switch", False, "thinking_soft_switch"),
    ("C1_llama31_8b_hf_deepinfra", "thinking_soft_switch", True, "thinking_soft_switch"),
    ("C2_qwen3_8b_hf_nscale_nothink", "answer_from_reasoning_field", False, "answer_from_reasoning_field"),
    ("C1_llama31_8b_hf_deepinfra", "answer_from_reasoning_field", True, "answer_from_reasoning_field"),
    ("C3_gptoss20b_hf_deepinfra_low", "route_snapshot", None, "route_snapshot"),
    ("A1_gpt6sol_reason_low", "route_snapshot", "data/x.json", "route_snapshot"),
    (CANDIDATE, "price_key", "openai/gpt-oss-120b:deepinfra", "candidate"),
    (THIRD_B, "route_context", None, "route_context"),                  # every sendable HF route has one
    (THIRD_B, "route_context", 0, "route_context"),
    (THIRD_B, "route_context", True, "route_context"),
    ("A1_gpt6sol_reason_low", "route_context", 131_072, "route_context"),   # no snapshot, no route context
    (CANDIDATE, "route_context", 131_072, "route_context"),             # no priced route
])
def test_panel_metadata_is_validated_at_load(tmp_path, pid, field, value, message):
    doc = _doc()
    doc["profiles"][pid][field] = value
    with pytest.raises(PanelError, match=message):
        _load(tmp_path, doc)


def test_panel_version_is_required(tmp_path):
    doc = _doc()
    del doc["panel_version"]
    with pytest.raises(PanelError, match="panel_version"):
        _load(tmp_path, doc)


# output caps raised so thinking/reasoning is not cut to finish_reason=length
RAISED_CAPS = {"C2_qwen3_8b_hf_nscale_nothink": (1_024, 2_048), "C3_gptoss20b_hf_deepinfra_low": (2_048, 4_096)}
ABSENT = "<absent>"
# Every field in which the panel differs from the pilot profiles: (pilot value, panel value).
EXPECTED_DEVIATIONS = {
    "Aalt_gemini38flash_low": {"client.generation.temperature": (ABSENT, 1.0),
                               "client.capabilities.temperature": (False, ABSENT)},
    # C2: the suffix was found in the connectivity smoke run; the nscale route returns the answer in reasoning_content
    "C2_qwen3_8b_hf_nscale_nothink": {"client.generation.max_output_tokens": RAISED_CAPS["C2_qwen3_8b_hf_nscale_nothink"],
                                      "user_message_suffix": (None, " /no_think"),
                                      "answer_from_reasoning_field": (None, True)},
    "C3_gptoss20b_hf_deepinfra_low": {"client.generation.max_output_tokens": RAISED_CAPS["C3_gptoss20b_hf_deepinfra_low"]},
}


def _leaves(value, prefix=""):
    if isinstance(value, dict) and value:
        out = {}
        for key, child in value.items():
            out.update(_leaves(child, f"{prefix}{key}."))
        return out
    return {prefix.rstrip("."): value}


def test_panel_differs_from_pilot_profiles_only_in_registered_fields(panel):
    """Field-by-field diff of every profile (client spec + panel-level user_message_suffix and
    answer_from_reasoning_field) against the pilot profile file."""
    pilot = json.loads(PILOT.read_text(encoding="utf-8"))["profiles"]
    registered = json.loads(panel.path.read_text(encoding="utf-8"))["pilot_profiles"]["deviations"]
    assert set(pilot) == {pid for pid, pp in panel.profiles.items() if pp.role not in ("extension", "candidate")}
    assert set(pilot).isdisjoint(EXTENSION) and CANDIDATE not in pilot    # the extension is not in the pilot file
    assert {pid: set(fields) for pid, fields in registered.items()} == \
        {pid: set(fields) for pid, fields in EXPECTED_DEVIATIONS.items()}
    assert all(isinstance(reason, str) and reason for fields in registered.values() for reason in fields.values())
    for pid, pp in panel.profiles.items():
        if pid not in pilot:
            continue
        mine = _leaves({"client": pp.spec, "user_message_suffix": pp.user_message_suffix,
                        "answer_from_reasoning_field": pp.answer_from_reasoning_field or None})
        theirs = _leaves({"client": pilot[pid], "user_message_suffix": None, "answer_from_reasoning_field": None})
        diff = {path: (theirs.get(path, ABSENT), mine.get(path, ABSENT)) for path in mine.keys() | theirs.keys()
                if theirs.get(path, ABSENT) != mine.get(path, ABSENT)}
        assert diff == EXPECTED_DEVIATIONS.get(pid, {}), pid
    assert "chat_template_kwargs.enable_thinking" in registered["C2_qwen3_8b_hf_nscale_nothink"]["user_message_suffix"]


def test_negative_control_is_refused_and_never_sendable(panel):
    negative = panel.get("NEGATIVE_hf_auto_routing")
    with pytest.raises(ConfigurationError, match="fixed provider"):
        InferenceProfile.from_dict(negative.spec)
    with pytest.raises(PanelError):
        negative.client_profile()


def test_wire_payloads_carry_the_pilot_profile_settings(panel):
    def wire(pid):
        pp = panel.get(pid)
        return build_request(pp.client_profile(), pp.messages_to_send(PROBE_MESSAGES))[1]
    a1 = wire("A1_gpt6sol_reason_low")
    assert a1["max_completion_tokens"] == 4096 and "max_tokens" not in a1 and "temperature" not in a1
    assert a1["reasoning_effort"] == "low"
    control = wire("A1_gpt6sol_none_T0")
    assert control["reasoning_effort"] == "none" and control["temperature"] == 0
    sonnet = wire("A2_sonnet5_nothink")
    assert sonnet["thinking"] == {"type": "disabled"} and not {"temperature", "top_p", "top_k"} & sonnet.keys()
    gemini = wire("Aalt_gemini38flash_low")["generationConfig"]
    assert gemini["thinkingConfig"] == {"thinkingLevel": "low"} and gemini["temperature"] == 1.0
    c2 = wire("C2_qwen3_8b_hf_nscale_nothink")
    assert c2["chat_template_kwargs"] == {"enable_thinking": False}
    assert c2["messages"] == [{"role": "user", "content": PROBE_MESSAGES[0]["content"] + " /no_think"}]
    soft = {"C2_qwen3_8b_hf_nscale_nothink", "tierC2_qwen3_32b_hf_nscale_nothink"}
    assert {pid for pid, pp in panel.profiles.items() if pp.thinking_soft_switch} == soft
    assert {pid for pid, pp in panel.profiles.items() if pp.answer_from_reasoning_field} == soft
    for pid, pp in panel.profiles.items():
        if pp.role != "negative" and pid not in soft:
            assert pp.user_message_suffix is None and pp.messages_to_send(PROBE_MESSAGES) == list(PROBE_MESSAGES)
    for pid, (_, cap) in RAISED_CAPS.items():
        assert wire(pid)["max_tokens"] == cap, pid
    assert wire("B1_dsv41flash_hf_deepinfra")["model"] == "deepseek-ai/DeepSeek-V4.1-Flash:deepinfra"
    # the extension readers get the treatment of their pilot-profile counterparts (thinking off / effort pinned,
    # same caps)
    q32 = wire("tierC2_qwen3_32b_hf_nscale_nothink")
    assert q32["model"] == "Qwen/Qwen3-32B:nscale" and q32["chat_template_kwargs"] == {"enable_thinking": False}
    assert q32["messages"][-1]["content"].endswith(" /no_think") and q32["max_tokens"] == 2048
    oss = wire("tierC3_gptoss120b_hf_deepinfra_low")
    assert oss["model"] == "openai/gpt-oss-120b:deepinfra" and oss["reasoning_effort"] == "low"
    assert oss["max_tokens"] == 4096 and oss["temperature"] == 0
    # the third B reader: not hybrid-thinking, not gpt-oss, so nothing to switch off; C1 Llama's settings
    llama = wire(THIRD_B)
    assert llama["model"] == "meta-llama/Llama-3.3-70B-Instruct:ovhcloud" and llama["max_tokens"] == 1024
    assert (llama["temperature"], llama["seed"]) == (0, 22)
    assert not {"chat_template_kwargs", "reasoning_effort", "thinking"} & llama.keys()
    assert llama["messages"] == list(PROBE_MESSAGES)


def test_plan_deviations_register_the_third_b_reader_and_are_validated_at_load(tmp_path, panel):
    registered = _doc()["plan_deviations"]["profiles"]
    assert set(registered) == {THIRD_B} and set(registered[THIRD_B]) == {"hf_provider", "price", "role"}
    assert "deepinfra" in registered[THIRD_B]["hf_provider"] and "0.74 / 0.74" in registered[THIRD_B]["price"]
    for bad in ({"B2_qwen3_235b_2507_hf_deepinfra": {"x": "panel slot"}}, {"no_such_profile": {"x": "y"}},
                {THIRD_B: {}}, {THIRD_B: {"price": "  "}}):
        doc = _doc()
        doc["plan_deviations"]["profiles"] = bad
        with pytest.raises(PanelError, match="plan_deviations"):
            _load(tmp_path, doc)


def test_smoke_plan_sends_one_capped_call_to_the_third_b_reader(panel, book):
    import importlib.util
    spec = importlib.util.spec_from_file_location("smoke_plan", ROOT / "scripts" / "smoke_plan.py")
    smoke = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(smoke)
    row = next(r for r in smoke.build_plan(panel, book) if r["profile_id"] == THIRD_B)
    assert row["action"] == "send 1 request" and row["wire_model"] == "meta-llama/Llama-3.3-70B-Instruct:ovhcloud"
    assert row["key_env"] == "HF_TOKEN" and "reasoning suppressed (reasoning_leak must be false)" in row["checks"]
    assert 0 < row["cap_usd"] == 3 * row["bound_usd"] < 0.002  # 512-token cap at 0.74 $/1M, all 3 attempts reserved


def test_deepseek_thinking_profile_refuses_temperature_before_sending(panel):
    pp = panel.get("B1_dsv41flash_direct_think")
    with pytest.raises(ConfigurationError, match="temperature"):
        build_request(pp.client_profile(), list(PROBE_MESSAGES), {"temperature": 0})


def _mutated(panel, pid, mutate):
    pp = panel.get(pid)
    spec = deepcopy(pp.spec)
    mutate(spec)
    return replace(pp, spec=spec)


def _check(pp):
    endpoint, payload = build_request(InferenceProfile.from_dict(pp.spec), pp.messages_to_send(PROBE_MESSAGES))
    check_wire(pp, endpoint, payload)


@pytest.mark.parametrize("pid, mutate, message", [
    ("A1_gpt6sol_reason_low", lambda s: s.pop("parameter_map"), "max_completion_tokens"),
    ("A1_gpt6sol_reason_low", lambda s: (s.pop("capabilities"), s["generation"].update(temperature=0)),
     "must not send temperature"),
    ("A1_gpt6sol_reason_low", lambda s: s["generation"].pop("reasoning_effort"), "must be pinned"),
    ("A2_sonnet5_nothink", lambda s: s.pop("extra_body"), "thinking must be set explicitly"),
    ("A2_sonnet5_nothink", lambda s: (s.pop("capabilities"), s["generation"].update(temperature=0)), "HTTP 400"),
    ("A2_sonnet5_nothink", lambda s: s["extra_body"].update(thinking={"type": "adaptive"}), "thinking disabled"),
    ("Aalt_gemini38flash_low", lambda s: s["generation"].pop("temperature"), "temperature 1.0"),
    ("Aalt_gemini38flash_low", lambda s: s["extra_body"]["generationConfig"]["thinkingConfig"].update(
        thinkingLevel="minimal"), "thinkingLevel=low"),
    ("B1_dsv41flash_direct_think", lambda s: s.pop("capabilities"), "declare it unsupported"),
    ("B1_dsv41flash_direct_nothink", lambda s: s.pop("extra_body"), "thinking must be set explicitly"),
    ("C2_qwen3_8b_hf_nscale_nothink", lambda s: s.pop("extra_body"), "enable_thinking=false"),
    ("C1_llama31_8b_hf_deepinfra", lambda s: s.pop("hf_provider"), "must pin hf_provider"),
    ("tierC2_qwen3_32b_hf_nscale_nothink", lambda s: s.pop("extra_body"), "enable_thinking=false"),
    ("tierC3_gptoss120b_hf_deepinfra_low", lambda s: s["generation"].pop("reasoning_effort"), "must be pinned"),
    ("C3_gptoss20b_hf_deepinfra_low", lambda s: s["generation"].pop("reasoning_effort"), "must be pinned"),
])
def test_each_trap_is_enforced(panel, pid, mutate, message):
    with pytest.raises(TrapViolation, match=message):
        _check(_mutated(panel, pid, mutate))


def test_qwen3_on_nscale_needs_the_soft_switch_and_it_must_reach_the_wire(panel):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    _check(pp)
    with pytest.raises(TrapViolation, match="does not forward chat_template_kwargs"):
        _check(replace(pp, user_message_suffix=None))
    endpoint, unsuffixed = build_request(pp.client_profile(), list(PROBE_MESSAGES))  # suffix never applied
    with pytest.raises(TrapViolation, match="not on the last user message"):
        check_wire(pp, endpoint, unsuffixed)
    gemini = replace(panel.get("Aalt_gemini38flash_low"), user_message_suffix=" /x")  # no messages list to verify
    with pytest.raises(TrapViolation, match="not on the last user message"):
        _check(gemini)


def test_messages_to_send_appends_to_the_last_user_message_only(panel):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    messages = ({"role": "system", "content": "sys"}, {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"},
                {"role": "assistant", "content": "prefill"})
    sent = pp.messages_to_send(messages)
    assert [m["content"] for m in sent] == ["sys", "q1", "a1", "q2 /no_think", "prefill"]
    assert messages[3]["content"] == "q2"  # the caller's messages are not mutated
    with pytest.raises(PanelError, match="needs a user message"):
        pp.messages_to_send(({"role": "system", "content": "sys"},))


@pytest.mark.parametrize("value", [42, "", "   ", " /no_think\r\n"])
def test_invalid_user_message_suffix_is_refused_at_load(tmp_path, value):
    doc = json.loads((ROOT / "data" / "panel" / "panel.json").read_text(encoding="utf-8"))
    doc["profiles"]["C2_qwen3_8b_hf_nscale_nothink"]["user_message_suffix"] = value
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(PanelError, match="user_message_suffix"):
        load_panel(path)


def test_a_changed_price_snapshot_is_warned_about(tmp_path):
    doc = json.loads((ROOT / "data" / "panel" / "panel.json").read_text(encoding="utf-8"))
    doc["price_snapshot"]["sha256_lf"] = "0" * 64
    path = tmp_path / "panel.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.warns(UserWarning, match="changed since the panel was pinned"):
        assert load_panel(path).profiles


ROUTE_CONTEXTS = {"B1_dsv41flash_hf_deepinfra": 1_048_576, "B2_qwen3_235b_2507_hf_deepinfra": 262_144,
                  THIRD_B: 131_072, "C1_llama31_8b_hf_deepinfra": 131_072, "C2_qwen3_8b_hf_nscale_nothink": 40_960,
                  "C3_gptoss20b_hf_deepinfra_low": 131_072, "tierC2_qwen3_32b_hf_nscale_nothink": 40_960,
                  "tierC3_gptoss120b_hf_deepinfra_low": 131_072}


def test_route_contexts_and_worker_limits_are_pinned(panel):
    """Every sendable HF profile carries its route's snapshot context; the Llama ovhcloud route sends one request at a
    time, every other profile keeps the client default of 4."""
    assert {pid: pp.route_context for pid, pp in panel.profiles.items() if pp.route_context is not None} == \
        ROUTE_CONTEXTS
    workers = {pid: pp.spec.get("max_workers") for pid, pp in panel.profiles.items() if pp.role != "negative"}
    assert workers.pop(THIRD_B) == 1 and set(workers.values()) == {InferenceProfile.max_workers} == {4}


def test_max_workers_changes_neither_the_request_hash_nor_the_ledger_identity(panel, book):
    """max_workers is transport, not request content: the vendored client's fingerprint excludes it, so the request
    hash, the call-ledger header identity and the reader cell's request hash of the Llama profile are the same at 1
    and at 4, and equal to the fingerprint its connectivity smoke ledger recorded."""
    from procedures.readers.ledger import IDENTITY_KEYS, request_hash
    from procedures.readers.executor import ledger_header
    one = panel.get(THIRD_B)
    four = replace(one, spec={**one.spec, "max_workers": 4})
    assert (one.client_profile().max_workers, four.client_profile().max_workers) == (1, 4)
    # the fingerprint the route's connectivity smoke ledger recorded
    assert one.fingerprint == four.fingerprint == "c86d4fbfe9094ae71780b4a36db4868a97abe810840ecb071c2921a7518b30b1"
    messages = list(PROBE_MESSAGES)
    assert request_hash(one.profile_id, one.fingerprint, messages, {}, 0) == \
        request_hash(four.profile_id, four.fingerprint, messages, {}, 0)
    h1, h4 = ledger_header(one, book), ledger_header(four, book)
    assert {k: h1[k] for k in IDENTITY_KEYS} == {k: h4[k] for k in IDENTITY_KEYS}
    changed = replace(one, spec={**one.spec, "timeout": 301, "max_workers": 2})
    assert changed.fingerprint == one.fingerprint
    semantic = replace(one, spec={**one.spec, "generation": {**one.spec["generation"], "seed": 23}})
    assert semantic.fingerprint != one.fingerprint              # the fingerprint does bind the request itself

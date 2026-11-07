"""Build-side reasoning-field mapping (procedures.memory.summary.answer_text): a builder or compressor whose panel
entry sets answer_from_reasoning_field (the Qwen3 profiles on nscale) has a blank content read from its reasoning
field, for summary memories (M5), constraint records (M7) and query-focused compressions (M6); every other profile is
read as before."""
from __future__ import annotations

import json
import logging
import urllib.request

import pytest

from test_memory import LONG, openai_body, write_corpus, write_long_corpus
from test_memory_m6_m7 import long_item, m6_item, rule_records
from procedures.readers.ledger import read_ledger
from procedures.memory import BuilderLedger, RevisionIndex, build_memory
from procedures.memory import summary as m5mod
from procedures.memory import records as m7mod
from procedures.memory.build import REASONING_FIELD_WARNING
from procedures.memory.summary import REASONING_FIELD_FLAG, REASONING_FIELD_WARN_SHARE, answer_text, reasoning_field_report

QWEN = "tierC2_qwen3_32b_hf_nscale_nothink"     # answer_from_reasoning_field: true
NOFLAG = "B1_dsv41flash_hf_deepinfra"           # the flag unset
SUMMARY = "- The operator shall reduce the transfer by 5 percent when the load exceeds 100 MW."
COMPRESSION = "The Transmission Owner shall restore the line within 30 minutes after the trip."


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import procedures.readers.client.client as client_module

    def refuse(*args, **kwargs):
        raise AssertionError("an offline test tried to open a network connection")
    monkeypatch.setattr(client_module, "http_transport", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


@pytest.fixture
def long_index(tmp_path) -> RevisionIndex:
    return write_long_corpus(tmp_path)


def test_the_panel_sets_the_flag_on_the_second_c_builder_only_among_the_builders():
    panel = m5mod.panel()
    assert panel.get(QWEN).answer_from_reasoning_field
    for pid in (NOFLAG, "B1_dsv41flash_direct_nothink", "C3_gptoss20b_hf_deepinfra_low"):
        assert not panel.get(pid).answer_from_reasoning_field


# ------------------------------------------------------------------------------------------------ answer_text
def rec(text, reasoning) -> dict:
    return {"outcome": "observed", "finish_reason": "stop", "text": text, "reasoning_text": reasoning}


@pytest.mark.parametrize("record, flagged, unflagged", [
    (rec("", '[{"source_span": "x"}]'), ('[{"source_span": "x"}]', True), ("", False)),       # JSON in reasoning
    (rec(None, "  - a bullet.\n"), ("- a bullet.", True), ("", False)),                        # plain text
    (rec("<think>\n\n</think>\n", "<think>x</think> - b."), ("- b.", True), ("", False)),      # think-stripped both
    (rec("", "  "), ("", False), ("", False)),                                                  # both blank
    (rec("", None), ("", False), ("", False)),
    (rec(" - kept.", "- ignored."), ("- kept.", False), ("- kept.", False)),                   # content wins
])
def test_answer_text_reads_the_reasoning_field_only_for_a_flagged_profile_and_a_blank_content(record, flagged,
                                                                                             unflagged):
    panel = m5mod.panel()
    assert answer_text(record, panel.get(QWEN)) == flagged
    assert answer_text(record, panel.get(NOFLAG)) == unflagged


@pytest.mark.parametrize("finish, flagged", [
    ("stop", ("- b.", True)),              # OpenAI shape: mapped
    ("end_turn", ("- b.", True)),          # Anthropic shape: mapped
    ("length", ("", False)),               # cut at the cap: not mapped
    ("max_tokens", ("", False)),
    ("content_filter", ("", False)),       # any other finish: not mapped
    (None, ("", False)),
])
def test_the_reasoning_field_is_read_only_after_a_stop_or_end_turn_finish(finish, flagged):
    """Tightened before any result of the main run was read: a blank content maps to the reasoning field only when
    the record finished "stop" / "end_turn"; otherwise the answer is empty. A profile without the flag never maps."""
    panel = m5mod.panel()
    record = {**rec("", "<think>x</think> - b."), "finish_reason": finish}
    assert answer_text(record, panel.get(QWEN)) == flagged
    assert answer_text(record, panel.get(NOFLAG)) == ("", False)
    content = {**record, "text": " - kept."}                     # a non-blank content is read whatever the finish
    assert answer_text(content, panel.get(QWEN)) == ("- kept.", False)


def test_a_length_finished_reasoning_answer_climbs_the_cap_ladder_and_is_empty_if_still_cut(long_index):
    """A level-1 answer with a blank content, its answer in the reasoning field and finish "length" is not mapped: it
    escalates as any truncated answer does; a level-2 "stop" answer in the reasoning field is used (mapped), a level-2
    answer still cut is an empty answer (raises, as before)."""
    pp = m5mod.panel().get(QWEN)
    calls = m5_calls(long_index, QWEN)
    base = records_for(calls, pp, [(SUMMARY, "")] * len(calls))
    first = calls[0]
    h1, h2 = (m5mod._request(pp, cell)[0] for cell in first.cells[:2])
    cut = {**rec("", SUMMARY), "finish_reason": "length", "usd": 0.0, "usage": {"input_total": 100, "output_total": 9}}
    records = {**base, h1: {**cut, "request_hash": h1}, h2: {**rec("", SUMMARY), "request_hash": h2, "usd": 0.0,
                                                           "usage": {"input_total": 100, "output_total": 20}}}
    builds = m5mod.answers_from_records(calls, pp, records, "mock")
    assert builds[0].level == 2 and builds[0].text == SUMMARY and REASONING_FIELD_FLAG in builds[0].flags
    assert [lv["finish_reason"] for lv in builds[0].levels] == ["length", "stop"]
    still_cut = {**records, h2: {**cut, "request_hash": h2}}
    with pytest.raises(RuntimeError, match="returned an empty answer at cap level 2"):
        m5mod.answers_from_records(calls, pp, still_cut, "mock")


# ------------------------------------------------------------------------------------------------ ledger read-back
def records_for(calls, pp, answers) -> dict:
    """request hash -> an observed level-1 record per call; answers[i] = (content, reasoning_text)."""
    out = {}
    for call, (text, reasoning) in zip(calls, answers):
        hash_ = m5mod._request(pp, call.cells[0])[0]
        out[hash_] = {**rec(text, reasoning), "request_hash": hash_, "usd": 0.0,
                      "usage": {"input_total": 100, "output_total": 20}}
    return out


def m5_calls(index, builder):
    return m5mod.plan([index.ladder(LONG)[0]], 2_000, builder)


def m7_calls(index, builder):
    return m7mod.plan([index.ladder(LONG)[0]], builder)


def test_m5_read_back_maps_json_and_plain_reasoning_and_refuses_both_blank(long_index):
    pp = m5mod.panel().get(QWEN)
    calls = m5_calls(long_index, QWEN)
    assert len(calls) >= 3
    answers = [("", '{"rule": "json in reasoning"}'), ("", SUMMARY)] + [(SUMMARY, "ignored")] * (len(calls) - 2)
    builds = m5mod.answers_from_records(calls, pp, records_for(calls, pp, answers), "mock")
    assert [b.text for b in builds[:3]] == ['{"rule": "json in reasoning"}', SUMMARY, SUMMARY]
    assert [REASONING_FIELD_FLAG in b.flags for b in builds] == [True, True] + [False] * (len(calls) - 2)
    report = reasoning_field_report(builds)
    assert report == {REASONING_FIELD_FLAG: 2, "subblocks": len(calls), "share": round(2 / len(calls), 4),
                      "over_warn_share": 2 / len(calls) > REASONING_FIELD_WARN_SHARE}
    blank = [("", "")] + answers[1:]
    with pytest.raises(RuntimeError, match="build/M5 of .* returned an empty answer at cap level 1"):
        m5mod.answers_from_records(calls, pp, records_for(calls, pp, blank), "mock")


def test_m7_read_back_maps_json_and_plain_reasoning_and_refuses_both_blank(long_index):
    pp = m5mod.panel().get(QWEN)
    calls = m7_calls(long_index, QWEN)
    part = [c.revision.text()[c.block.char_start:c.block.char_end] for c in calls]
    array = [json.dumps(rule_records(p)[:c.quota]) for p, c in zip(part, calls)]
    answers = [("", array[0]), ("", "This part lists transfer reductions.")] +\
              [(a, "ignored") for a in array[2:]]
    builds = m5mod.answers_from_records(calls, pp, records_for(calls, pp, answers), "mock")
    mapped = [REASONING_FIELD_FLAG in b.flags for b in builds]
    assert mapped == [True, True] + [False] * (len(calls) - 2)
    mem = m7mod.assemble(calls, builds, long_item()["question"], 2_000)
    assert mem.parsed[0].array == "strict" and len(mem.parsed[0].records) == calls[0].quota   # JSON from reasoning
    assert mem.parsed[1].array == "none" and mem.checks()["subblocks_without_array"] == 1     # prose: counted
    with pytest.raises(RuntimeError, match="build/M7 of .* returned an empty answer"):
        m5mod.answers_from_records(calls, pp, records_for(calls, pp, [("", None)] + answers[1:]), "mock")


@pytest.mark.parametrize("calls_of", [m5_calls, m7_calls])
def test_an_unflagged_builder_is_read_from_the_content_only(long_index, calls_of):
    pp = m5mod.panel().get(NOFLAG)
    calls = calls_of(long_index, NOFLAG)
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        m5mod.answers_from_records(calls, pp, records_for(calls, pp, [("", SUMMARY)] * len(calls)), "mock")
    builds = m5mod.answers_from_records(calls, pp, records_for(calls, pp, [(SUMMARY, "x")] * len(calls)), "mock")
    assert all(REASONING_FIELD_FLAG not in b.flags for b in builds) and builds[0].text == SUMMARY


# ------------------------------------------------------------------------------------------------ live, end to end
class ReasoningEndpoint:
    """Answers M5 (a bullet), M7 (the part's RULE records up to the quota) and M6 (a compression) requests; a request
    whose sub-block (section, part) is in `into_reasoning`, or any M6 request when `m6_into_reasoning`, gets a blank
    content and the answer in reasoning_content (as the nscale route's vLLM server returns it)."""

    def __init__(self, into_reasoning=(), m6_into_reasoning=False):
        self.into, self.m6, self.calls = set(into_reasoning), m6_into_reasoning, []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        content = payload["messages"][-1]["content"]
        if "<passages>" in content:
            answer, moved = COMPRESSION, self.m6
        else:
            k = int(content.split("\nSection: Section ", 1)[1].split("/", 1)[0]) - 1
            j = int(content.split("\nPart: Part ", 1)[1].split("/", 1)[0]) - 1
            moved = (k, j) in self.into
            if "JSON array of at most " in content:
                quota = int(content.split("JSON array of at most ", 1)[1].split(" ", 1)[0])
                answer = json.dumps(rule_records(content.split("<part>\n", 1)[1].rsplit("\n</part>", 1)[0])[:quota])
            else:
                answer = SUMMARY
        body = openai_body(answer, prompt=1_000, completion=50)
        body["model"] = "Qwen/Qwen3-32B"
        if moved:
            body["choices"][0]["message"] = {"role": "assistant", "content": "", "reasoning_content": answer}
        return 200, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


def live(tmp_path, transport, name) -> BuilderLedger:
    return BuilderLedger(tmp_path / name, cap_usd=5.0, transport=transport, sleep=lambda s: None)


def test_m7_live_build_by_qwen_reads_the_reasoning_field_through_the_ledger_and_reports_the_share(
        long_index, monkeypatch, tmp_path, caplog):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    calls = m7_calls(long_index, QWEN)
    assert len(calls) > 1 / REASONING_FIELD_WARN_SHARE                      # one mapped sub-block stays under 10%
    one = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=QWEN,
                       ledger=live(tmp_path, ReasoningEndpoint({(0, 0)}), "one.jsonl"))
    _, ledger = read_ledger(tmp_path / "one.jsonl")
    moved = [r for r in ledger if not (r["text"] or "").strip()]
    assert len(moved) == 1 and moved[0]["reasoning_text"].startswith("[")       # the ledger keeps the raw field
    assert one.build[REASONING_FIELD_FLAG] == {REASONING_FIELD_FLAG: 1, "subblocks": len(calls),
                                               "share": round(1 / len(calls), 4), "over_warn_share": False}
    assert one.build["records"]["arrays"] == {"strict": len(calls)}           # the mapped answer parsed like the rest
    assert f"{REASONING_FIELD_FLAG}=1" in one.notes and REASONING_FIELD_WARNING not in one.notes
    assert [b["flags"] for b in one.build["blocks"] if REASONING_FIELD_FLAG in b["flags"]] == [[REASONING_FIELD_FLAG]]
    with caplog.at_level(logging.WARNING, logger="procedures.memory.build"):
        every = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=QWEN,
                             ledger=live(tmp_path, ReasoningEndpoint({(c.section.index, c.block.index)
                                                                      for c in calls}), "all.jsonl"))
    assert every.build[REASONING_FIELD_FLAG]["share"] == 1.0 and every.build[REASONING_FIELD_FLAG]["over_warn_share"]
    assert REASONING_FIELD_WARNING in every.notes and every.text == one.text    # warned, not voided
    assert any("answered from the reasoning field" in r.message for r in caplog.records)


def test_m5_live_build_by_qwen_reads_the_reasoning_field(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    item = {**long_item(), "item_id": "m5"}
    mem = build_memory(item, "M5", 2_000, long_index, builder_profile=QWEN,
                       ledger=live(tmp_path, ReasoningEndpoint({(0, 0), (0, 1)}), "m5.jsonl"))
    report = mem.build[REASONING_FIELD_FLAG]
    assert report[REASONING_FIELD_FLAG] == 2 and report["subblocks"] == len(mem.build["blocks"])
    assert mem.build["checks"]["subblocks_without_bullet"] == 0 and SUMMARY[2:] in mem.text


def test_unflagged_live_builds_carry_no_mapping_fields(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    for rep in ("M5", "M7"):
        mem = build_memory(long_item(), rep, 2_000, long_index, builder_profile=NOFLAG,
                           ledger=live(tmp_path, ReasoningEndpoint(), f"{rep}.jsonl"))
        assert REASONING_FIELD_FLAG not in mem.build and REASONING_FIELD_FLAG not in mem.notes
    with pytest.raises(RuntimeError, match="returned an empty answer"):
        build_memory(long_item(), "M7", 2_000, long_index, builder_profile=NOFLAG,
                     ledger=live(tmp_path, ReasoningEndpoint({(0, 0)}), "refused.jsonl"))
    dry = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=QWEN)
    assert dry.build["mode"] == "dry_run" and REASONING_FIELD_FLAG not in dry.build


def test_m6_compression_by_qwen_reads_the_reasoning_field(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    index = RevisionIndex.from_manifest(write_corpus(tmp_path))
    mapped = build_memory(m6_item(), "M6", 400, index, compressor_profile=QWEN,
                          ledger=live(tmp_path, ReasoningEndpoint(m6_into_reasoning=True), "q1.jsonl"))
    assert mapped.text.endswith(COMPRESSION) and mapped.build[REASONING_FIELD_FLAG] is True
    assert REASONING_FIELD_FLAG in mapped.build["flags"] and REASONING_FIELD_FLAG in mapped.notes
    plain = build_memory(m6_item(), "M6", 400, index, compressor_profile=QWEN,
                         ledger=live(tmp_path, ReasoningEndpoint(), "q2.jsonl"))
    assert plain.text == mapped.text and plain.build[REASONING_FIELD_FLAG] is False
    assert REASONING_FIELD_FLAG not in plain.build["flags"]
    other = build_memory(m6_item(), "M6", 400, index, compressor_profile=NOFLAG,
                         ledger=live(tmp_path, ReasoningEndpoint(), "q3.jsonl"))
    assert REASONING_FIELD_FLAG not in other.build
    with pytest.raises(RuntimeError, match="empty answer"):
        build_memory(m6_item(), "M6", 400, index, compressor_profile=NOFLAG,
                     ledger=live(tmp_path, ReasoningEndpoint(m6_into_reasoning=True), "q4.jsonl"))

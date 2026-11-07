"""procedures.memory M7 (typed constraint records) and M6 (query-focused compression) offline: JSON parsing
robustness, record quotas and caps, source-span location, budget packing, the cap ladder, cache keys, the dry-run /
live / ledger-cache paths with mocked transports. Synthetic corpora from tests/memory/test_memory.py; no network."""
from __future__ import annotations

import json
import math
import re
import urllib.request
from fractions import Fraction
from types import SimpleNamespace

import pytest

from test_memory import DOC2, LONG, openai_body, write_corpus, write_long_corpus
from procedures.readers.ledger import read_ledger
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens
from procedures.memory import BuilderLedger, RevisionIndex, build_memory
from procedures.memory import records as m7mod
from procedures.memory.chunking import pack, paragraph_chunks, rank, render as render_units
from procedures.memory.summary import answers_from_records, panel, sections, subblocks
from procedures.memory.compressor import INPUT_FACTOR, OUTPUT_CAP_FACTOR as M6_CAP_FACTOR
from procedures.memory.compressor import WORDS_PER_TOKEN
from procedures.memory.records import RECORD_FIELDS, SOURCE_TOKENS_PER_RECORD, SPAN_MAX_CHARS, TOKENS_PER_RECORD, locate, parse_records, plan, record_cap, record_quota, record_units, render_record

BUILDER = "B1_dsv41flash_hf_deepinfra"
REASONER = "C3_gptoss20b_hf_deepinfra_low"
COMPRESSOR = "C1_llama31_8b_hf_deepinfra"
RULE = re.compile(r"When the load at substation (\S+) exceeds (\d+) MW for more than (\d+) minutes, the operator "
                  r"shall reduce the transfer by (\d+) percent unless the Reliability Coordinator directs otherwise\.")


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


def record(**fields) -> dict:
    return {name: fields.get(name) for name in RECORD_FIELDS}


GOOD = record(subject="operator", quantity="transfer reduction", value="5", unit="percent",
              condition="load over 100 MW", source_span="the operator shall reduce the transfer by 5 percent")


# ------------------------------------------------------------------------------------------------ parsing
def test_parse_takes_a_strict_array_and_counts_nothing_dropped():
    p = parse_records(json.dumps([GOOD, {**GOOD, "value": "6"}]))
    assert (p.array, p.cut, p.malformed, p.missing_fields) == ("strict", 0, (), 0)
    assert [r["value"] for r in p.records] == ["5", "6"] and p.records[0] == GOOD
    assert parse_records("[]") == m7mod.Parsed((), "strict", 0, (), 0)


@pytest.mark.parametrize("wrap, array", [
    ("```json\n{}\n```", "fenced"), ("```\n{}\n```", "fenced"), ("```json\n{}", "fenced"),
    ("Here are the records [as JSON]:\n{}\nDone.", "embedded"), ("<think>list [them]</think>\n{}", "strict"),
    ("Rules [1, 2] and [] below:\n{}", "embedded"),        # a list of numbers is not the array of records
])
def test_parse_finds_the_array_behind_fences_prose_and_think_blocks(wrap, array):
    p = parse_records(wrap.replace("{}", json.dumps([GOOD], indent=2)))
    assert p.array == array and p.records == (GOOD,) and p.cut == 0


def test_parse_salvages_the_complete_records_of_a_cut_array():
    whole = json.dumps([GOOD, {**GOOD, "value": "6"}, {**GOOD, "value": "7"}], indent=2)
    cut = whole[:whole.rindex('"value": "7"') + 6]                   # cut inside the third record
    p = parse_records(cut)
    assert (p.array, p.cut, [r["value"] for r in p.records]) == ("salvaged", 1, ["5", "6"])
    missing_comma = json.dumps([GOOD])[:-1] + " " + json.dumps(GOOD) + "]"
    q = parse_records(missing_comma)
    assert (q.array, q.cut, len(q.records)) == ("salvaged", 1, 1)
    inner = json.dumps([{**GOOD, "value": "[]"}, {**GOOD, "value": "8"}])
    r = parse_records(inner[:inner.rindex('"value": "8"')])          # cut; an empty list inside a string value
    assert (r.array, r.cut, [x["value"] for x in r.records]) == ("salvaged", 1, ["[]"])
    assert parse_records("The part states no rule.") == m7mod.Parsed((), "none", 0, (), 0)
    assert parse_records("No rule here: []") == m7mod.Parsed((), "embedded", 0, (), 0)


def test_parse_drops_malformed_elements_by_reason_and_reads_missing_fields_as_null():
    elements = ["a string", 5, [GOOD], {**GOOD, "subject": ["operator"]}, {**GOOD, "exception": True},
                {**GOOD, "unit": {"a": 1}}, {**GOOD, "source_span": None}, {**GOOD, "source_span": "   "},
                {**GOOD, "source_span": "x" * (SPAN_MAX_CHARS + 1)}, {**GOOD, "source_span": "x" * SPAN_MAX_CHARS},
                {"source_span": "only a span"}, {**GOOD, "value": 30}, {**GOOD, "value": 0.5, "extra": "ignored"}]
    p = parse_records(json.dumps(elements), "drop")       # the by-reason accounting under the "drop" span rule
    assert dict(p.malformed) == {"not_object": 3, "non_scalar_field": 3, "no_source_span": 2,
                                 "source_span_too_long": 1} and p.dropped == 9
    assert [r["value"] for r in p.records] == ["5", None, "30", "0.5"]       # numbers kept as their JSON text
    assert p.records[1] == record(source_span="only a span") and p.missing_fields == 1
    assert all(set(r) == set(RECORD_FIELDS) for r in p.records)


def test_span_too_long_switch_truncates_by_default_or_drops(long_index):
    # the rule fixed in advance: "truncate" (a development build dropped 18.9% > 10% under "drop")
    assert m7mod.SPAN_TOO_LONG == "truncate" and m7mod.PARAMS["span_too_long"] == "truncate"
    text = "The operator shall reduce the transfer by 5 percent. " * 6
    long = {**GOOD, "source_span": text.strip()}
    dropped = parse_records(json.dumps([GOOD, long]), "drop")
    assert dict(dropped.malformed) == {"source_span_too_long": 1} and dropped.truncated_spans == 0
    kept = parse_records(json.dumps([GOOD, long]))              # the default rule
    assert kept.malformed == () and kept.truncated_spans == 1 and len(kept.records) == 2
    assert kept.records[1]["source_span"] == text.strip()[:SPAN_MAX_CHARS]
    assert parse_records(json.dumps([GOOD, long]), "drop") == dropped
    rev = long_index.ladder(LONG)[0]
    body = rev.text()
    quote = body[body.index("When the load"):][:SPAN_MAX_CHARS + 40]
    cut = parse_records(json.dumps([{**GOOD, "source_span": quote}]), "truncate").records[0]["source_span"]
    assert locate(body, cut, 0, len(body))[2] == "verbatim"                 # a truncated quote is still placed
    with pytest.raises(ValueError, match="span rule"):
        parse_records("[]", "keep")


def test_render_record_is_one_line_of_the_non_null_fields_in_schema_order():
    line = render_record(record(value="30", subject="Transmission\n Owner", unit="", source_span="shall  restore"))
    assert line == "- subject: Transmission Owner | value: 30 | source_span: shall restore"
    assert "\n" not in render_record(record(condition="a\nb", source_span="c\n\nd"))


# ------------------------------------------------------------------------------------------------ quotas, caps, locate
def test_record_quota_is_body_tokens_over_125_rounded_half_up_with_a_floor_of_one():
    assert SOURCE_TOKENS_PER_RECORD == 125
    for body in range(0, 3_000, 7):
        want = max(1, math.floor(Fraction(body, SOURCE_TOKENS_PER_RECORD) + Fraction(1, 2)))
        assert record_quota(SimpleNamespace(body_tokens=body)) == want, body
    assert [record_quota(SimpleNamespace(body_tokens=b)) for b in (0, 62, 187, 188, 2_000)] == [1, 1, 1, 2, 16]


def test_record_caps_follow_the_shared_ladder_factors():
    for k in range(1, 40):
        assert record_cap(k) == math.ceil(2.0 * TOKENS_PER_RECORD * k) == 224 * k
        assert record_cap(k, 2) == math.ceil(5.0 * TOKENS_PER_RECORD * k) + TOKENS_PER_RECORD == 560 * k + 112
    with pytest.raises(ValueError, match="cap level"):
        record_cap(1, 3)


def test_locate_prefers_the_subblock_then_the_revision_then_whitespace_runs():
    text = "Intro. The unit shall stop within 5 minutes.\n\nPart two. The unit shall stop within 5 minutes. The unit\n" \
           "shall   ramp at 3 MW. End."
    span = "The unit shall stop within 5 minutes."
    first, second = text.index(span), text.index(span, 20)
    assert locate(text, span, second - 3, len(text)) == (second, second + len(span), "verbatim")   # in the block
    assert locate(text, f" {span} ", 0, 20) == (first, first + len(span), "verbatim")          # stripped
    assert locate(text, span, text.index("The unit\n"), len(text)) == (first, first + len(span), "verbatim")
    start, end, how = locate(text, "The unit shall ramp at 3 MW.", 0, len(text))
    assert how == "whitespace" and text[start:end].split() == "The unit shall ramp at 3 MW.".split()
    assert locate(text, "The unit must ramp at 3 MW.", 0, len(text)) is None


# ------------------------------------------------------------------------------------------------ mock endpoint
def rule_records(part: str) -> list[dict]:
    return [record(subject="the operator", quantity="transfer reduction", value=m[4], unit="percent",
                   condition=f"the load at substation {m[1]} exceeds {m[2]} MW for more than {m[3]} minutes",
                   exception="unless the Reliability Coordinator directs otherwise", source_span=m[0])
            for m in RULE.finditer(part)]


class Endpoint:
    """A builder / compressor endpoint. M7 record requests: `records_of(part text)` (default: one record per RULE
    sentence) up to the quota, source_span the sentence verbatim; `modes[(section, part)]` scripts a reply: "fenced",
    "prose" (no array), "bad" (four malformed elements first), "ws" (the first span with a doubled space), "reworded"
    (the first span not in the text), "dup" (the first record twice), "cut1" (cut mid-record, finish length, at cap
    level 1 only), "cut" (cut at both levels). M6 compression requests: `compression(prompt)` or the passages' first
    sentences, finish `compress_finish`. Keeps every payload and the latest reply per (section, part)."""

    def __init__(self, modes=None, compression=None, compress_finish="stop", allowance=0, records_of=None):
        self.modes, self.compression, self.compress_finish = modes or {}, compression, compress_finish
        self.allowance, self.records_of = allowance, records_of or rule_records
        self.calls, self.replies, self.levels = [], {}, []

    @staticmethod
    def where(content: str) -> tuple[int, int]:
        k = int(content.split("\nSection: Section ", 1)[1].split("/", 1)[0]) - 1
        return k, int(content.split("\nPart: Part ", 1)[1].split("/", 1)[0]) - 1

    def extract(self, content: str, max_tokens: int) -> tuple[str, str]:
        k, j = self.where(content)
        quota = int(content.split("JSON array of at most ", 1)[1].split(" ", 1)[0])
        level = 1 if max_tokens == record_cap(quota) + self.allowance else 2
        self.levels.append((k, j, level))
        part = content.split("<part>\n", 1)[1].rsplit("\n</part>", 1)[0]
        records = self.records_of(part)[:quota]
        mode = self.modes.get((k, j))
        if mode == "ws":
            records[0]["source_span"] = records[0]["source_span"].replace(" the operator", "  the operator", 1)
        elif mode == "reworded":
            records[0]["source_span"] = records[0]["source_span"].replace("shall reduce", "must reduce")
        elif mode == "dup":
            records = [records[0]] + records
        body = json.dumps(records, indent=2)
        if mode == "bad":
            body = json.dumps(["x", {"subject": ["a"], "source_span": "y"}, {"subject": "no span"},
                               {"source_span": "z" * (SPAN_MAX_CHARS + 1)}] + records, indent=2)
        elif mode == "fenced":
            body = f"```json\n{body}\n```"
        elif mode == "prose":
            body = "This part lists transfer reductions."
        finish = "stop"
        if mode == "cut" or (mode == "cut1" and level == 1):
            body, finish = body[:body.rindex('"exception"')], "length"
        self.replies[(k, j)] = body
        return body, finish

    def compress(self, content: str) -> tuple[str, str]:
        if self.compression is not None:
            return self.compression(content), self.compress_finish
        passages = content.split("<passages>\n", 1)[1].rsplit("\n</passages>", 1)[0]
        body = " ".join(line for line in passages.splitlines() if line and not line.startswith("["))
        return body, self.compress_finish

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        content = payload["messages"][-1]["content"]
        text, finish = (self.compress(content) if "<passages>" in content
                        else self.extract(content, payload["max_tokens"]))
        body = openai_body(text, prompt=estimate_message_tokens(payload["messages"]), completion=estimate_tokens(text),
                           finish=finish)
        return 200, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


def long_item(question: str = "By how many percent is the transfer reduced when the load at substation 3-2-1-0 "
                              "exceeds 100 MW?") -> dict:
    return {"item_id": "m7", "doc": LONG, "as_of_date": "2025-01-01",
            "question": f"Under {LONG} as in effect on 2025-01-01, {question[0].lower()}{question[1:]}"}


def live(tmp_path, transport, name="build.jsonl") -> BuilderLedger:
    return BuilderLedger(tmp_path / name, cap_usd=5.0, transport=transport, sleep=lambda s: None)


# ------------------------------------------------------------------------------------------------ M7
def test_m7_plan_is_the_m5_subblocks_with_budget_free_keys(long_index):
    rev = long_index.ladder(LONG)[0]
    calls = plan([rev], BUILDER)
    assert [(c.section, c.block) for c in calls] == [(s, b) for s in sections(rev) for b in subblocks(rev, s)]
    assert len(calls) >= 15 and all(c.quota == record_quota(c.block) and c.block_target == 112 * c.quota
                                    for c in calls)
    for c in calls:
        prompt = c.cell.messages[0]["content"]
        assert rev.text()[c.block.char_start:c.block.char_end] in prompt
        assert f"JSON array of at most {c.quota} records" in prompt and c.block.label() in prompt
        assert [x.generation["max_output_tokens"] for x in c.cells] == [record_cap(c.quota), record_cap(c.quota, 2)]
        assert c.cells[1].messages == c.cell.messages and c.cells[1].labels == {"cap_level": 2}
        assert c.cell.stage == f"build/M7@{112 * c.quota}" and c.cell.item_id.startswith(f"{LONG}@1#")
    keys = [k for c in calls for k in c.keys]
    assert len(set(keys)) == len(keys)
    a = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=BUILDER)
    b = build_memory(long_item("How long may the load exceed 104 MW?"), "M7", 8_000, long_index,
                     builder_profile=BUILDER)
    assert [s["levels"] for s in a.build["blocks"]] == [s["levels"] for s in b.build["blocks"]]   # one build, any query
    assert a.cache_key != b.cache_key and a.build_tokens == b.build_tokens > 0
    assert {c.key for c in plan([rev], "C1_llama31_8b_hf_deepinfra")}.isdisjoint(c.key for c in calls)


def test_m7_caps_add_the_reasoning_allowance_of_a_reasoning_builder(long_index):
    rev = long_index.ladder(LONG)[0]
    (plain, *_), (reasoning, *_) = plan([rev], BUILDER), plan([rev], REASONER)
    assert [c.generation["max_output_tokens"] for c in reasoning.cells] == [
        record_cap(reasoning.quota) + 3_000, record_cap(reasoning.quota, 2) + 3_000]
    assert plain.cells[1].est_visible_tokens == reasoning.cells[1].est_visible_tokens == record_cap(plain.quota)
    key = m7mod.record_key(rev, plain.section, plain.block, BUILDER, plain.quota, 1, record_cap(plain.quota))
    assert key == plain.key != m7mod.record_key(rev, plain.section, plain.block, BUILDER, plain.quota + 1, 1,
                                                record_cap(plain.quota))                 # the quota is in the key
    assert key != m7mod.record_key(rev, plain.section, plain.block, BUILDER, plain.quota, 1, record_cap(plain.quota) + 1)


def test_m7_dry_run_prices_every_level_and_sends_nothing(long_index):
    mem = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=BUILDER)
    calls = plan([long_index.ladder(LONG)[0]], BUILDER)
    assert mem.notes == f"dry_run; subblocks={len(calls)}" and mem.build["mode"] == "dry_run"
    assert mem.build["checks"] is None and mem.build["records"] is None and mem.provenance == []
    assert "dry-run placeholder: no records" in mem.text and mem.text_tokens <= 2_000
    for block, c in zip(mem.build["blocks"], calls):
        assert [v["block_key"] for v in block["levels"]] == list(c.keys) and all(v["usd"] > 0 for v in block["levels"])
        assert block["mode"] == "dry_run" and block["quota"] == c.quota and block["records"] is None
    assert mem.build_tokens == sum(estimate_message_tokens(c.cell.messages) + 112 * c.quota for c in calls)
    assert mem.query_tokens == 0


def test_m7_live_build_parses_locates_and_packs_records_and_the_ledger_is_the_cache(long_index, monkeypatch,
                                                                                     tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = Endpoint(modes={(0, 1): "ws", (0, 2): "reworded", (1, 0): "dup", (1, 1): "fenced"})
    ledger = live(tmp_path, transport)
    rev = long_index.ladder(LONG)[0]
    calls = plan([rev], BUILDER)
    mem = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert len(transport.calls) == len(calls) and mem.build["mode"] == "live"
    assert mem.text_tokens <= 2_000 and mem.text.startswith(f"[{rev.header()}]\n- subject: the operator | ")
    counts = mem.build["records"]
    assert counts["valid"] == sum(c.quota + ((c.section.index, c.block.index) == (1, 0)) for c in calls)
    assert counts["duplicates"] == 1 and counts["candidates"] == counts["valid"] - 1
    assert counts["arrays"] == {"fenced": 1, "strict": len(calls) - 1}
    assert counts["located"] == counts["candidates"] - 1 and counts["packed"] == len(mem.provenance) < 40
    assert mem.build["checks"] == {"subblocks_without_array": 0, "subblocks_not_stop": 0, "subblocks_over_quota": 0,
                                   "malformed_records": 0, "unlocated_records": 1, "over_budget": 0}
    assert "; live; " in mem.notes and "checks=subblocks_without_array:0" in mem.notes
    text = rev.text()
    for p in mem.provenance:
        a, b = p["block_span"]
        if p["located"]:
            assert a <= p["char_start"] < p["char_end"] <= b and RULE.fullmatch(" ".join(
                text[p["char_start"]:p["char_end"]].split()))
        else:
            assert p["char_start"] is None and p["char_end"] is None
    # every candidate, with its provenance, independently of packing
    answered = {r["request_hash"]: r for r in read_ledger(ledger.path)[1]}
    units, parsed, dups = record_units(calls, answers_from_records(calls, panel().get(BUILDER), answered, ledger.path))
    ws = next(u for u in units if u.order == (next(i for i, c in enumerate(calls)
                                                   if (c.section.index, c.block.index) == (0, 1)), 0))
    assert ws.provenance["match"] == "whitespace" and ws.provenance["located"]
    lost = next(u for u in units if not u.provenance["located"])
    assert "must reduce" in lost.text and lost.provenance["block_span"] == [calls[2].block.char_start,
                                                                            calls[2].block.char_end]
    # packing: rank order in, document order out
    ranked = [u for _, u in rank(units, long_item()["question"])]
    chosen, used = pack(ranked, 2_000, m7mod.render)
    assert mem.text == m7mod.render(chosen) and used == mem.text_tokens and ranked[0] in chosen
    lines = mem.text.splitlines()[1:]
    assert lines == [u.text for u in sorted(chosen, key=lambda u: u.order)]
    # the ledger is the cache: another question and budget send nothing
    other = build_memory(long_item("How long may the load exceed 104 MW?"), "M7", 8_000, long_index,
                         builder_profile=BUILDER, ledger=ledger)
    assert len(transport.calls) == len(calls) and other.build_tokens == mem.build_tokens
    assert other.text != mem.text and other.text_tokens <= 8_000 and other.build["records"]["packed"] > 40
    header, records = read_ledger(ledger.path)
    assert header["profile_id"] == BUILDER and {r["stage"].split("@")[0] for r in records} == {"build/M7"}


def test_m7_cap_ladder_resends_a_cut_answer_and_salvages_a_twice_cut_one(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = Endpoint(modes={(0, 0): "cut1", (0, 1): "cut", (0, 2): "prose", (0, 3): "bad"})
    ledger = live(tmp_path, transport)
    calls = plan([long_index.ladder(LONG)[0]], BUILDER)
    mem = build_memory(long_item(), "M7", 2_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert sorted(t for t in transport.levels if t[2] == 2) == [(0, 0, 2), (0, 1, 2)]
    assert len(transport.calls) == len(calls) + 2
    blocks = {(b["section_index"], b["subblock_index"]): b for b in mem.build["blocks"]}
    assert (blocks[(0, 0)]["cap_level"], blocks[(0, 0)]["finish_reason"], blocks[(0, 0)]["cut"]) == (2, "stop", 0)
    assert blocks[(0, 0)]["records"] == calls[0].quota
    assert (blocks[(0, 1)]["cap_level"], blocks[(0, 1)]["finish_reason"], blocks[(0, 1)]["array"],
            blocks[(0, 1)]["cut"]) == (2, "length", "salvaged", 1)
    assert blocks[(0, 1)]["records"] == calls[1].quota - 1                       # the cut record is dropped
    assert blocks[(0, 2)]["array"] == "none" and blocks[(0, 2)]["records"] == 0
    # under the default "truncate" rule the over-long span is cut and kept
    assert blocks[(0, 3)]["malformed"] == {"no_source_span": 1, "non_scalar_field": 1, "not_object": 1}
    assert mem.build["still_capped"] == 1 and mem.build["cap_levels"] == {"1": len(calls) - 2, "2": 2}
    # the kept over-long span of the "bad" block is synthetic text absent from the revision -> located=false, kept
    assert mem.build["checks"] == {"subblocks_without_array": 1, "subblocks_not_stop": 1, "subblocks_over_quota": 0,
                                   "malformed_records": 3, "unlocated_records": 1, "over_budget": 0}
    _, records = read_ledger(ledger.path)
    assert sorted(r.get("cap_level", 1) for r in records).count(2) == 2
    assert "build_truncated" in mem.notes and "cut=1" in mem.notes


def test_m7_without_an_answer_raises(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")

    def refuse(url, headers, payload, timeout):     # rate limited until the retries run out: no answer, no bill,
        return 429, {}, json.dumps({"error": "rate limited"}).encode("utf-8")   # and no circuit-breaker failure
    with pytest.raises(RuntimeError, match="build/M7 of .* got no answer"):
        build_memory(long_item(), "M7", 2_000, long_index, builder_profile=BUILDER, ledger=live(tmp_path, refuse))


# ------------------------------------------------------------------------------------------------ M6
@pytest.fixture
def small_index(tmp_path) -> RevisionIndex:
    return RevisionIndex.from_manifest(write_corpus(tmp_path))


def m6_item(item_id="c1", question="within how many minutes shall the Transmission Owner restore the line after "
                                   "the trip?") -> dict:
    return {"item_id": item_id, "doc": DOC2, "as_of_date": "2024-06-01",
            "question": f"Under TEST-OP-2 as in effect on 2024-06-01, {question} If no such requirement applies on "
                        "that date, answer not_in_effect."}


def test_m6_dry_run_prices_one_compression_of_the_m2_retrieval_at_four_times_the_budget(small_index):
    mem = build_memory(m6_item(), "M6", 100, small_index, compressor_profile=COMPRESSOR)
    rev = small_index.ladder(DOC2)[0]
    assert (mem.build["mode"], mem.build["ledger_role"], mem.build_tokens) == ("dry_run", "query", 0)
    units = [u for _, u in rank(list(paragraph_chunks(rev)), m6_item()["question"]) if _ > 0]
    chosen, _ = pack(units, INPUT_FACTOR * 100, lambda us: render_units(us, "paragraphs"))
    assert mem.build["chunks"] == len(chosen) > 0 and mem.provenance == [u.provenance() for u in
                                                                         sorted(chosen, key=lambda u: u.char_start)]
    room = 100 - estimate_tokens(f"[{rev.header()}]\n")
    assert (mem.build["room"], mem.build["words"]) == (room, math.floor(room * WORDS_PER_TOKEN))
    assert mem.build["max_output_tokens"] == math.ceil(M6_CAP_FACTOR * room) and mem.build["usd"] > 0
    assert mem.query_tokens == mem.build["input_tokens"] + room and mem.text_tokens <= 100
    assert mem.notes == f"dry_run; chunks={len(chosen)}; room={room}"
    reasoning = build_memory(m6_item(), "M6", 100, small_index, compressor_profile=REASONER)
    assert reasoning.build["max_output_tokens"] == math.ceil(M6_CAP_FACTOR * room) + 3_000


def test_m6_input_is_the_m2_retrieval_at_four_times_the_budget(long_index):
    assert INPUT_FACTOR == 4
    mem = build_memory(long_item(), "M6", 400, long_index, compressor_profile=COMPRESSOR)
    m2 = build_memory(long_item(), "M2", 1_600, long_index)
    fewer = build_memory(long_item(), "M2", 1_200, long_index)
    assert mem.provenance == m2.provenance and len(fewer.provenance) < len(m2.provenance) == mem.build["chunks"]
    assert mem.build["passage_tokens"] == m2.text_tokens and mem.build["scored"] > mem.build["chunks"]


def test_m6_cache_key_names_question_chunks_compressor_and_budget(small_index):
    base = build_memory(m6_item(), "M6", 100, small_index, compressor_profile=COMPRESSOR)
    assert base.cache_key == build_memory(m6_item("other-id"), "M6", 100, small_index,
                                          compressor_profile=COMPRESSOR).cache_key   # same request, same memory
    for other in (build_memory(m6_item(), "M6", 101, small_index, compressor_profile=COMPRESSOR),
                  build_memory(m6_item(), "M6", 100, small_index, compressor_profile=REASONER),
                  build_memory(m6_item(question="when is the deposit $150,000?"), "M6", 100, small_index,
                               compressor_profile=COMPRESSOR)):
        assert other.cache_key != base.cache_key
    assert base.build["request_hash"] == build_memory(m6_item("other-id"), "M6", 100, small_index,
                                                      compressor_profile=COMPRESSOR).build["request_hash"]


def test_m6_live_compression_is_trimmed_to_the_budget_and_cached_in_the_query_ledger(small_index, monkeypatch,
                                                                                     tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = Endpoint()
    ledger = live(tmp_path, transport, "query.jsonl")
    m2 = build_memory(m6_item(), "M2", INPUT_FACTOR * 100, small_index)
    mem = build_memory(m6_item(), "M6", 100, small_index, compressor_profile=COMPRESSOR, ledger=ledger)
    (sent,) = transport.calls
    prompt = sent["messages"][-1]["content"]
    assert m2.text in prompt and m6_item()["question"] in prompt and f"at most {mem.build['words']} words" in prompt
    assert sent["max_tokens"] == mem.build["max_output_tokens"] and mem.build["mode"] == "live"
    rev = small_index.ladder(DOC2)[0]
    assert mem.text.startswith(f"[{rev.header()}]\n") and mem.text_tokens <= 100
    assert mem.build["flags"] == ["trimmed_to_budget"] and mem.build["trimmed_tokens"] > 0
    body = mem.text.split("\n", 1)[1]
    assert body.endswith(".") and body in transport.compress(prompt)[0]
    _, (rec,) = read_ledger(ledger.path)
    assert mem.query_tokens == rec["usage"]["input_total"] + rec["usage"]["output_total"] and mem.build_tokens == 0
    assert mem.build["usd"] == rec["usd"] and rec["stage"] == "query/M6@100" and rec["item_id"] == "c1"
    again = build_memory(m6_item("c2"), "M6", 100, small_index, compressor_profile=COMPRESSOR, ledger=ledger)
    assert len(transport.calls) == 1 and again.text == mem.text                   # identical request: the ledger
    assert "; live; " in mem.notes and "trimmed_to_budget" in mem.notes


def test_m6_flags_a_truncated_answer_and_refuses_an_empty_one(small_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    cut = build_memory(m6_item(), "M6", 400, small_index, compressor_profile=COMPRESSOR,
                       ledger=live(tmp_path, Endpoint(compression=lambda p: "The owner shall restore it.",
                                                      compress_finish="length"), "q1.jsonl"))
    assert cut.build["flags"] == ["compress_truncated"] and cut.text.endswith("The owner shall restore it.")
    with pytest.raises(RuntimeError, match="empty answer"):
        build_memory(m6_item(), "M6", 400, small_index, compressor_profile=COMPRESSOR,
                     ledger=live(tmp_path, Endpoint(compression=lambda p: "<think>x</think>  "), "q2.jsonl"))


def test_m6_without_a_retrieved_chunk_is_empty_and_sends_nothing(small_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = Endpoint()
    mem = build_memory(m6_item(question="what colour is zebra plumage?"), "M6", 100, small_index,
                       compressor_profile=COMPRESSOR, ledger=live(tmp_path, transport))
    assert (mem.text, mem.build["mode"], mem.query_tokens, transport.calls) == ("", "no_input", 0, [])
    assert mem.notes.endswith("no_input: no chunk scored above zero") and mem.build["scored"] == 0
    tight = build_memory(m6_item(), "M6", 50, small_index, compressor_profile=COMPRESSOR, ledger=live(tmp_path, transport))
    assert (tight.text, tight.build["mode"], transport.calls) == ("", "no_input", [])   # the one chunk exceeds 4 x 50
    assert tight.notes.endswith("no_input: 1 chunks scored above zero, none fits the input")


def test_profiles_and_ledgers_apply_to_their_representations_only(small_index, tmp_path):
    it = m6_item()
    ledger = BuilderLedger(tmp_path / "x.jsonl", cap_usd=1.0)
    for rep, kw in (("M7", {}), ("M6", {}), ("M6", {"builder_profile": BUILDER}),
                    ("M7", {"builder_profile": BUILDER, "compressor_profile": COMPRESSOR}),
                    ("M2", {"compressor_profile": COMPRESSOR}), ("M2", {"ledger": ledger}), ("M6", {"ledger": ledger}),
                    ("M7", {"ledger": ledger})):
        with pytest.raises(ValueError):
            build_memory(it, rep, 400, small_index, **kw)
    with pytest.raises(TypeError):
        build_memory(it, "M6", 400, small_index, compressor_profile=COMPRESSOR, ledger="query.jsonl")

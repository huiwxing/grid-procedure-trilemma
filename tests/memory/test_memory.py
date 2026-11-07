"""procedures.memory offline: as-of selection through manifest windows, packing to budget, constraint-sentence (M4)
extraction recall, full-text (M0) revision selection, summary (M5) sections and sub-blocks (tiling, sentence /
heading / paragraph boundaries, proportional targets, bullet quotas), dry-run and mocked live sub-block builds through
run_cells, the mechanical checks, budget fitting, cache keys, determinism. Synthetic manifests only; no network."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import urllib.request
from dataclasses import replace
from fractions import Fraction
from pathlib import Path

import pytest

from procedures.text.exclusions import history_regions
from procedures.text.sentences import segment
from procedures.readers.ledger import read_ledger
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens, token_method
from procedures.memory import BuilderLedger, RevisionIndex, build_memory
from procedures.memory.chunking import query_text
from procedures.memory.chunking import sentences as chunk_sentences
from procedures.memory.constraints import constraint_sentences
from procedures.memory.summary import BULLET_WORDS, CAP_ESCALATION, MIN_BULLETS, OUTPUT_CAP_FACTOR, PARAMS, PART_MAX_TOKENS, PROMPT_SHA, SECTION_MAX_TOKENS, SNAP, SUBBLOCK_PROMPT, TARGET_FLOOR, TOKENS_PER_BULLET, Section, SubBlock, _parts, answers_from_records, block_cap, block_cell, block_key, block_weights, build_blocks, bullet_count, bullet_lines, drop_cut_bullet, escalates, fit, history_flags, panel, parse_call_id, plan, quota_split, render, sections, subblocks
from procedures.memory.result import cache_key

BUILDER = "B1_dsv41flash_hf_deepinfra"
NO_HISTORY = "revision_history_tokens=0; revision_history_share=0.0"   # M5 notes of a revision without revision history
DOC = "TEST-OP-1"
DOC2 = "TEST-OP-2"
COLUMNS = ("family", "doc_id", "revision", "doc_kind", "cover_effective_date", "inactive_date", "status", "text_path",
           "sha256_lf", "excluded", "gold_eligible", "provenance_status", "fetched_at_utc")

VEGETATION = "\n\n".join(
    f"Vegetation paragraph {i} describes trimming cycles for corridor segment {i} and the crews that clear brush "
    f"along access roads. Crews record the species, the height of regrowth and the condition of culverts near "
    f"segment {i}." for i in range(20))


def ladder_text(limit: int, extra: str = "") -> str:
    return (f"=== PAGE 1 ===\nTest Operating Procedure 1\n\n1.1 Ramp Limits\n\n"
            f"The ramp limit for the Test unit shall be {limit} MW per minute. The ramp limit applies to every "
            f"dispatch interval of the Test unit.{extra}\n\n{VEGETATION}\n")


PLANTED = [
    "The Transmission Owner shall restore the line within 30 minutes of the trip.",
    "If the frequency falls below the trigger, the operator must shed load.",
    "Generators will submit outage requests no later than 5 business days before the outage.",
    "The deposit shall be $150,000 for each request.",
    "Unless the Reliability Coordinator directs otherwise, the operator should hold the schedule.",
    "The operator may extend the study by three (3) months when data are late.",
    "Voltage must not drop below 95 percent of nominal.",
    "In the event of a system emergency, market participants are required to follow dispatch instructions.",
    "Except for black start units, each unit shall test its governor at least once every 24 months.",
    "Participants are prohibited from submitting more than two (2) offers per hour.",
]
DISTRACTORS = [
    "The control area covers 12,000 MW of load.",
    "The operator shall maintain a log.",
    "Staff will attend the annual training.",
    "If the weather is cold, demand is higher.",
    "The procedure was revised on May 20, 2020 to add a 15 minute ramp window.",
    "This section describes the purpose of the manual.",
]
CONSTRAINT_TEXT = ("2.1 Constraints\n\n" + " ".join(PLANTED[:3] + DISTRACTORS[:2]) + "\n\n"
                   + " ".join(DISTRACTORS[2:4] + PLANTED[3:7]) + "\n\n" + " ".join(PLANTED[7:] + DISTRACTORS[4:])
                   + "\n")


def write_corpus(root: Path) -> Path:
    """tmp/proj/data/corpus/manifest.csv + texts; returns the manifest path."""
    corpus = root / "proj" / "data" / "corpus"
    (corpus / "test").mkdir(parents=True)
    texts = {
        "r0.txt": ladder_text(5), "r1.txt": ladder_text(10), "r2.txt": ladder_text(20), "r3.txt": ladder_text(30),
        "r4.txt": ladder_text(40), "rx.txt": ladder_text(99), "ip3.txt": ladder_text(30, " The plan phases it in."),
        "r3.redline.txt": ladder_text(31), "op2.txt": CONSTRAINT_TEXT,
    }
    for name, text in texts.items():
        (corpus / "test" / name).write_bytes(text.encode("utf-8"))

    def row(doc, rev, kind, eff, name, status="", inactive="", excluded="false", prov="archive_copy_unverified"):
        text = texts[name]
        return {"family": "TEST", "doc_id": doc, "revision": rev, "doc_kind": kind, "cover_effective_date": eff,
                "inactive_date": inactive, "status": status, "text_path": f"data/corpus/test/{name}",
                "sha256_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(), "excluded": excluded,
                "gold_eligible": "true", "provenance_status": prov, "fetched_at_utc": "2026-09-24T10:00:00Z"}

    rows = [row(DOC, "0", "clean", "2019-01-01", "r0.txt", status="never_effective"),
            row(DOC, "1", "clean", "2020-01-01", "r1.txt", status="superseded", inactive="2020-09-30"),
            row(DOC, "2", "clean", "2021-01-01", "r2.txt", status="superseded"),
            row(DOC, "3", "clean", "2022-06-01", "r3.txt", status="effective", prov="owner_site"),
            row(DOC, "3", "redline", "2022-06-01", "r3.redline.txt"),
            row(DOC, "4", "clean", "", "r4.txt", status="pending_regulatory_approval"),
            row(DOC, "5", "clean", "2023-01-01", "rx.txt", excluded="true"),
            row(DOC, "3", "implementation_plan", "2022-06-01", "ip3.txt", status="effective", prov="owner_site"),
            row(DOC2, "1", "clean", "2024-01-01", "op2.txt", status="effective", prov="owner_site")]
    manifest = corpus / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return manifest


@pytest.fixture
def corpus(tmp_path) -> tuple[Path, RevisionIndex]:
    manifest = write_corpus(tmp_path)
    return tmp_path, RevisionIndex.from_manifest(manifest)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    import procedures.readers.client.client as client_module

    def refuse(*args, **kwargs):
        raise AssertionError("an offline test tried to open a network connection")
    monkeypatch.setattr(client_module, "http_transport", refuse)
    monkeypatch.setattr(urllib.request, "urlopen", refuse)


def item(question="Under TEST-OP-1 as in effect on 2021-06-01, what is the ramp limit for the Test unit? "
                  "If no such requirement applies on that date, answer not_in_effect.", doc=DOC, **fields):
    return {"item_id": "t1", "doc": doc, "question": question, **fields}


# ------------------------------------------------------------------------------------------------ as-of windows
@pytest.mark.parametrize("as_of, expected", [
    ("2020-06-30", ["1"]), ("2020-09-30", ["1"]), ("2020-10-01", []),   # r1 ends on its owner inactive date
    ("2020-12-31", []), ("2021-01-01", ["2"]), ("2022-05-31", ["2"]),    # r2 ends the day before r3 starts
    ("2022-06-01", ["3"]), ("2026-09-24", ["3"]),                        # current revision verified to the fetch date
    ("2026-09-25", []), ("2019-06-01", []),                              # after verification / never_effective r0
])
def test_asof_filtering_follows_manifest_windows(corpus, as_of, expected):
    _, index = corpus
    for rep in ("M2", "M4"):
        mem = build_memory(item(as_of_date=as_of), rep, 400, index)
        assert [rev for _, rev in mem.revisions_used] == expected
        assert {p["revision"] for p in mem.provenance} <= set(expected)
        if not expected:
            assert mem.text == "" and "no revision in effect" in mem.notes


def test_asof_memory_reads_only_the_revision_in_effect(corpus):
    _, index = corpus
    for as_of, limit in (("2020-03-01", 10), ("2021-06-01", 20), ("2024-01-01", 30)):
        mem = build_memory(item(as_of_date=as_of), "M2", 400, index)
        assert f"shall be {limit} MW per minute" in mem.text
        assert set(re.findall(r"shall be (\d+) MW", mem.text)) == {str(limit)}


def test_named_revision_question_uses_its_effective_date_and_ladder_filters(corpus):
    _, index = corpus
    mem = build_memory(item(question="Under TEST-OP-1 (Revision 2, effective 2021-01-01), what is the ramp limit?",
                            effective_date="2021-01-01"), "M4", 400, index)
    assert mem.revisions_used == [(DOC, "2")] and "(effective_date)" in mem.notes
    both = build_memory(item(as_of_date="2021-06-01", effective_date="2020-01-01"), "M4", 400, index)
    assert both.revisions_used == [(DOC, "2")] and "(as_of_date)" in both.notes   # the question's date wins
    assert [r.revision for r in index.ladder(DOC)] == ["1", "2", "3", "4"]   # r0 never_effective, r5 excluded, no redline
    assert index.ladder(DOC)[3].effective is None and index.in_effect(DOC, index.ladder(DOC)[2].effective)\
        == [index.ladder(DOC)[2]]
    ip = build_memory(item(doc=f"{DOC}-IP", as_of_date="2024-01-01"), "M2", 400, index)
    assert ip.revisions_used == [(f"{DOC}-IP", "3")] and "phases it in" in ip.text


def test_unknown_document_and_missing_dates_raise(corpus):
    _, index = corpus
    with pytest.raises(KeyError):
        build_memory(item(doc="NO-SUCH-DOC", as_of_date="2021-06-01"), "M2", 400, index)
    with pytest.raises(ValueError):
        build_memory(item(), "M2", 400, index)


# ------------------------------------------------------------------------------------------------ packing
@pytest.mark.parametrize("rep, budget", [("M2", b) for b in (300, 600, 1_200, 5_000)]   # chunks are ~256 tokens
                         + [("M4", b) for b in (40, 90, 200, 500)])
def test_packing_respects_budget_and_keeps_whole_sentences(corpus, rep, budget):
    _, index = corpus
    q = ("Under TEST-OP-2 as in effect on 2024-06-01, when a system emergency occurs, within how many minutes "
         "shall the Transmission Owner restore the line? If no such requirement applies on that date, answer "
         "not_in_effect.") if rep == "M4" else None
    it = item(question=q, doc=DOC2, as_of_date="2024-06-01") if q else item(as_of_date="2021-06-01")
    mem = build_memory(it, rep, budget, index)
    assert estimate_tokens(mem.text) == mem.text_tokens <= budget
    assert mem.provenance and all(set(p) == {"doc_id", "revision", "path", "char_start", "char_end"}
                                  for p in mem.provenance)
    rev = index.ladder(it["doc"])[-1 if rep == "M4" else 1]
    source = rev.text()
    sents = segment(source)
    starts, ends = {s.start for s in sents}, {s.end for s in sents}
    flat = re.sub(r"[^0-9a-z]", "", mem.text.lower())
    for p in mem.provenance:
        assert p["char_start"] in starts and p["char_end"] in ends      # never cut inside a sentence
        assert p["path"] == rev.source_path
        body = [s.text for s in sents if p["char_start"] <= s.start and s.end <= p["char_end"]]
        assert re.sub(r"[^0-9a-z]", "", " ".join(body).lower()) in flat


def test_budget_larger_than_the_revision_gives_ranked_content_not_full_text(corpus):
    _, index = corpus
    it = item(as_of_date="2021-06-01")
    full = build_memory(it, "M0", None, index)
    m2 = build_memory(it, "M2", 100_000, index)
    assert m2.text_tokens < full.text_tokens and "Vegetation paragraph 19" in full.text
    assert "Vegetation paragraph 19" not in m2.text and "shall be 20 MW" in m2.text
    m4 = build_memory(item(doc=DOC2, as_of_date="2024-06-01"), "M4", 100_000, index)
    assert "This section describes" not in m4.text
    assert m4.text.count("\n- ") == len(PLANTED) and m4.text.startswith("[TEST-OP-2")


def test_query_drops_the_template_frame():
    q = ("Under MISO BPM-015 (Revision 33, effective 2026-07-01), if the IC withdraws during Phase III, which of the "
         "following applies? (A) the IC funds the restudy (B) MISO refunds the deposit (C) none of these.")
    assert query_text(q) == "if the IC withdraws during Phase III, the IC funds the restudy MISO refunds the deposit ."
    assert query_text("Under PJM Manual 13: Emergency Operations as in effect on 2026-08-09, what is the limit? "
                      "If no such requirement applies on that date, answer not_in_effect.") == "what is the limit?"


# ------------------------------------------------------------------------------------------------ M4 extraction
def test_m4_extracts_every_planted_constraint_and_no_distractor(corpus):
    _, index = corpus
    extracted = [u.text for u in constraint_sentences(index.ladder(DOC2)[0])]
    recall = sum(s in extracted for s in PLANTED)
    assert recall == len(PLANTED) == 10
    assert sorted(extracted) == sorted(PLANTED)
    mem = build_memory(item(doc=DOC2, as_of_date="2024-06-01"), "M4", 2_000, index)
    assert all(s in mem.text for s in PLANTED) and not any(s in mem.text for s in DISTRACTORS)
    assert mem.build_tokens == 0


# ------------------------------------------------------------------------------------------------ M0
def test_m0_takes_the_pair_for_an_asof_unit_and_the_current_revision_otherwise(corpus):
    root, index = corpus
    pair = item(as_of_date="2021-06-01", m0_paths=[str(root / "proj/data/corpus/test/r2.txt"),
                                                   "proj/data/corpus/test/r1.txt"])
    mem = build_memory(pair, "M0", None, index)
    assert mem.revisions_used == [(DOC, "1"), (DOC, "2")] and mem.notes == "m0_paths"
    assert mem.text.index("Revision 1 (effective 2020-01-01)]") < mem.text.index("Revision 2 (effective 2021-01-01)]")
    assert "shall be 10 MW" in mem.text and "shall be 20 MW" in mem.text
    assert [(p["revision"], p["char_start"], p["char_end"]) for p in mem.provenance] ==\
        [("1", 0, len(ladder_text(10))), ("2", 0, len(ladder_text(20)))]
    single = build_memory(item(effective_date="2022-06-01"), "M0", None, index)
    assert single.revisions_used == [(DOC, "3")] and "shall be 30 MW" in single.text and "20 MW" not in single.text
    assert build_memory(pair, "M0", 10_000, index).text == mem.text
    with pytest.raises(ValueError):
        build_memory(pair, "M0", 50, index)                              # M0 is never truncated
    with pytest.raises(ValueError):
        build_memory({**pair, "m0_paths": ["proj/data/corpus/test/op2.txt"]}, "M0", None, index)
    with pytest.raises(FileNotFoundError):
        build_memory({**pair, "m0_paths": ["proj/data/corpus/test/missing.txt"]}, "M0", None, index)


def test_b0_is_empty(corpus):
    _, index = corpus
    mem = build_memory(item(), "B0", None, index)
    assert (mem.text, mem.provenance, mem.revisions_used, mem.build_tokens, mem.notes) == ("", [], [], 0, "closed_book")


# ------------------------------------------------------------------------------------------------ M5
def openai_body(text, prompt=5_000, completion=300, finish="stop"):
    return {"id": "cmpl-1", "model": "deepseek-ai/DeepSeek-V4.1-Flash",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion}}


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        if not self.responses:
            raise AssertionError("FakeTransport ran out of scripted responses")
        code, body = self.responses.pop(0)
        return code, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


ALL_CLEAR = "checks=subblocks_without_bullet:0,subblocks_not_stop:0,subblocks_over_quota:0,over_budget:0"


def test_m5_dry_run_without_ledger_estimates_the_build(corpus):
    _, index = corpus
    it = item(as_of_date="2021-06-01")
    mem = build_memory(it, "M5", 300, index, builder_profile=BUILDER)
    assert mem.notes == f"dry_run; sections=1; subblocks=1; {NO_HISTORY}"
    assert mem.build["mode"] == "dry_run" and mem.build["checks"] is None and "dry-run placeholder" in mem.text
    rev = index.ladder(DOC)[1]
    (sec,) = sections(rev)                                     # a short revision is one section of one sub-block
    (blk,) = subblocks(rev, sec)
    assert (sec.char_start, sec.char_end, sec.title) == (0, len(rev.text()), "1.1 Ramp Limits")
    assert (blk.char_start, blk.char_end, blk.label()) == (0, len(rev.text()), "Part 1/1: 1.1 Ramp Limits")
    t = 300 - estimate_tokens(f"[{rev.header()}]\n## Section 1/1: 1.1 Ramp Limits\n")
    n = bullet_count(t)
    cell = block_cell(rev, sec, blk, n)                        # a lone sub-block gets the section's whole allowance
    (call,) = mem.build["blocks"]
    assert (call["target_tokens"], call["quota"], call["span"]) == (t, n, [0, len(rev.text())]) and call["block_key"]
    assert mem.build_tokens == estimate_message_tokens(cell.messages) + TOKENS_PER_BULLET * n == call["build_tokens"]
    assert call["usd"] > 0 and call["reason"] == "no ledger" and mem.text_tokens <= 300
    assert f"at most {n} bullets" in cell.messages[0]["content"]
    assert ladder_text(20) in cell.messages[0]["content"]
    assert cell.generation == {"max_output_tokens": math.ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * n)}
    other = build_memory(it, "M5", 300, index, builder_profile="C1_llama31_8b_hf_deepinfra")
    assert other.cache_key != mem.cache_key


def test_m5_dry_run_memory_names_its_budget_so_equal_plans_stay_distinct(corpus, monkeypatch):
    """A revision whose plan does not depend on the budget (AESO 305.1: one revision-history section at the target
    floor at every budget) had four identical dry-run memories on the budget ladder, and the reader requests priced on
    them collided (run_cells: duplicate requests). The last line names the budget."""
    import procedures.memory.summary as m5
    _, index = corpus
    it = item(as_of_date="2021-06-01")
    real_plan = m5.plan
    monkeypatch.setattr(m5, "plan", lambda revisions, budget, builder: real_plan(revisions, 300, builder))
    m5._dry_m5.cache_clear()
    try:
        mems = {b: build_memory(it, "M5", b, index, builder_profile=BUILDER) for b in (300, 600)}
    finally:
        m5._dry_m5.cache_clear()
    bodies = {b: m.text.rsplit("\n", 1) for b, m in mems.items()}
    assert bodies[300][0] == bodies[600][0] and "dry-run placeholder" in bodies[300][0]       # the same stand-in
    assert [bodies[b][1] for b in (300, 600)] == [m5.DRY_BUDGET_LINE.format(budget=b) for b in (300, 600)]
    assert mems[300].text != mems[600].text and mems[300].text_tokens == estimate_tokens(mems[300].text)


def test_m5_dry_run_when_the_credential_is_unset(corpus, monkeypatch, tmp_path):
    _, index = corpus
    monkeypatch.delenv("HF_TOKEN", raising=False)
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=FakeTransport([]))
    mem = build_memory(item(as_of_date="2021-06-01"), "M5", 300, index, builder_profile=BUILDER, ledger=ledger)
    assert mem.notes == f"dry_run; sections=1; subblocks=1; {NO_HISTORY}"
    assert "HF_TOKEN" in mem.build["blocks"][0]["reason"]
    assert not ledger.path.exists() and ledger.transport.calls == []


def test_m5_live_build_goes_through_run_cells_and_is_read_back_from_the_ledger(corpus, monkeypatch, tmp_path):
    _, index = corpus
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    summary = "- The ramp limit for the Test unit shall be 20 MW per minute in every dispatch interval."
    transport = FakeTransport([(200, openai_body("<think>\n\n</think>\n" + summary))])
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    it = item(as_of_date="2021-06-01")
    mem = build_memory(it, "M5", 2_000, index, builder_profile=BUILDER, ledger=ledger)
    assert mem.text.endswith(summary) and "<think>" not in mem.text and mem.build["mode"] == "live"
    assert mem.build_tokens == 5_300 and mem.text_tokens <= 2_000
    assert mem.notes.endswith(f"; live; sections=1; subblocks=1; {NO_HISTORY}; finish_reasons=stop:1; "
                              f"cap_levels=1:1; still_capped=0; trimmed_tokens=0; dropped_bullets=0; {ALL_CLEAR}")
    assert (mem.build["cap_levels"], mem.build["still_capped"]) == ({"1": 1}, 0)
    assert mem.build["checks"] == {"subblocks_without_bullet": 0, "subblocks_not_stop": 0,
                                   "subblocks_over_quota": 0, "over_budget": 0}
    sent, call = transport.calls[0], mem.build["blocks"][0]
    assert ladder_text(20) in sent["messages"][0]["content"]
    assert sent["max_tokens"] == 56 * call["quota"] == call["max_output_tokens"]             # 2 x 28 tokens a bullet
    assert (call["bullets"], call["output_tokens"], call["finish_reason"]) == (1, 300, "stop")
    assert call["cap_level"] == 1 and [v["cap_level"] for v in call["levels"]] == [1]     # a stop: no level 2
    header, records = read_ledger(ledger.path)
    assert header["profile_id"] == BUILDER and len(records) == 1
    assert records[0]["request_hash"] == call["request_hash"]
    assert parse_call_id(records[0]["item_id"]) == (DOC, "2", 0, 0)
    again = build_memory(it, "M5", 2_000, index, builder_profile=BUILDER, ledger=ledger)   # ledger = cache
    assert again == mem and len(transport.calls) == 1


def test_m5_overlong_summary_is_trimmed_at_a_sentence_boundary_and_the_checks_record_it(corpus, monkeypatch,
                                                                                            tmp_path):
    _, index = corpus
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    long = " ".join(f"Rule {i} shall apply when condition {i} holds for {i} minutes." for i in range(300))
    transport = FakeTransport([(200, openai_body(long, finish="length"))] * 2)      # cap levels 1 and 2 both truncated
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport)
    mem = build_memory(item(as_of_date="2021-06-01"), "M5", 150, index, builder_profile=BUILDER, ledger=ledger)
    first, second = transport.calls                                                  # no level 3
    q = mem.build["blocks"][0]["quota"]
    assert first["messages"] == second["messages"] and (first["max_tokens"], second["max_tokens"]) == (
        block_cap(q), block_cap(q, 2)) == (56 * q, 140 * q + 112)
    assert "cap_levels=2:1; still_capped=1" in mem.notes and mem.build["still_capped"] == 1
    assert mem.text_tokens <= 150 and "trimmed_to_budget" in mem.notes and "build_truncated" in mem.notes
    assert mem.text.endswith("minutes.") and "Rule 0 shall apply" in mem.text
    cut = mem.build["trimmed_tokens"]
    assert cut == mem.build["blocks"][0]["trimmed_tokens"] > 0 and f"trimmed_tokens={cut}" in mem.notes
    assert "checks=subblocks_without_bullet:1,subblocks_not_stop:1,subblocks_over_quota:0,over_budget:0" in mem.notes


def test_m5_build_without_an_answer_raises(corpus, monkeypatch, tmp_path):
    _, index = corpus
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0,
                           transport=FakeTransport([(400, {"error": "bad request"})]))
    with pytest.raises(RuntimeError, match="got no answer"):
        build_memory(item(as_of_date="2021-06-01"), "M5", 300, index, builder_profile=BUILDER, ledger=ledger)


def test_m5_call_ids_name_the_section_and_the_subblock():
    assert parse_call_id("M-03@71#2.4") == ("M-03", "71", 2, 4)
    assert parse_call_id("NERC-FAC-011-IP@3#0.0") == ("NERC-FAC-011-IP", "3", 0, 0)
    for bad in ("M-03@71#2", "M-03@71", "M-03@71#a.1", "M-03@71#1.", "@71#1.1"):
        with pytest.raises(ValueError, match="sub-block call id"):
            parse_call_id(bad)


def test_m5_bullet_lines_count_marked_lines_only():
    text = "Intro line\n- a\n* b\n• c\n  ◦ d\n– e\n1. f\n2) g\n-no space\n\nClosing."
    assert bullet_lines(text) == ["a", "b", "c", "d", "e", "f", "g"]
    assert bullet_lines("") == [] and bullet_lines("Plain prose. More prose.") == []


# ------------------------------------------------------------------------------------------------ M5 sections
LONG = "TEST-LONG"      # ~32k o200k tokens, a numbered heading every ~1k tokens
FLAT = "TEST-FLAT"      # the same rules without any heading
RULE = ("When the load at substation {i}-{j}-{k}-{m} exceeds {mw} MW for more than {mins} minutes, the operator shall "
        "reduce the transfer by {pct} percent unless the Reliability Coordinator directs otherwise.")


def long_text(headings: bool) -> str:
    parts = ["=== PAGE 1 ===\nTest Long Manual\n"]
    for i in range(1, 9):
        if headings:
            parts.append(f"{i} Operating Area {i}\n")
        for j in range(1, 5):
            if headings:
                parts.append(f"{i}.{j} Procedure {i}.{j}\n")
            for k in range(6):          # paragraphs of 3 to 5 rules, so an ideal cut rarely falls on a paragraph start
                parts.append(" ".join(RULE.format(i=i, j=j, k=k, m=m, mw=100 + m, mins=5 + m, pct=m + 1)
                                      for m in range(3 + (i + j + k) % 3)) + "\n")
    return "\n".join(parts)


HIST = "TEST-HIST"      # LONG with a short document history up front and a revision history of ~40% at the end


def history_entry(n: int) -> str:
    return (f"Revision {n}: the procedure for operating area {1 + n % 8} was updated so that the operator shall notify "
            f"the Reliability Coordinator before the transfer changes, and the ramp limit changed from {n} MW to "
            f"{n + 5} MW.")


def hist_text() -> str:
    """A short "Document History" up front (ended by "Contents"); a body line without end punctuation that the
    segmenter glues to the "Revision History" heading (a sentence straddling the region start: not history by the
    frames rule, which tests a sentence's start); the revision history; a short glossary appendix after it (ended
    history region, too short for a span of its own, so it joins the last history section)."""
    front = "Document History\n\n" + "\n\n".join(history_entry(n) for n in range(4)) + "\n\nContents\n\n"
    history = ("The operating areas above are listed in the change record below\nRevision History\n\n"
               + "\n\n".join(history_entry(n) for n in range(520)) + "\n\nAppendix A Glossary\n\n"
               + "\n\n".join(f"Term {n} means the load at substation {n} while the transfer exceeds {100 + n} MW."
                             for n in range(75)) + "\n")
    return long_text(True).replace("Test Long Manual\n", "Test Long Manual\n\n" + front, 1) + history


def write_long_corpus(root: Path) -> RevisionIndex:
    corpus = root / "proj" / "data" / "corpus"
    (corpus / "long").mkdir(parents=True)
    rows = []
    for doc, text in ((LONG, long_text(True)), (FLAT, long_text(False)), (HIST, hist_text())):
        name = f"{doc.lower()}.txt"
        (corpus / "long" / name).write_bytes(text.encode("utf-8"))
        rows.append({"family": "TEST", "doc_id": doc, "revision": "1", "doc_kind": "clean",
                     "cover_effective_date": "2024-01-01", "inactive_date": "", "status": "effective",
                     "text_path": f"data/corpus/long/{name}", "excluded": "false", "gold_eligible": "true",
                     "sha256_lf": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                     "provenance_status": "owner_site", "fetched_at_utc": "2026-09-24T10:00:00Z"})
    manifest = corpus / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return RevisionIndex.from_manifest(manifest)


@pytest.fixture
def long_index(tmp_path) -> RevisionIndex:
    return write_long_corpus(tmp_path)


def long_item(doc=LONG) -> dict:
    return {"item_id": "long", "doc": doc, "as_of_date": "2025-01-01",
            "question": f"Under {doc} as in effect on 2025-01-01, by how many percent is the transfer reduced? "
                        "If no such requirement applies on that date, answer not_in_effect."}


class BlockSummarizer:
    """A builder endpoint for sub-block requests: answers each with bullets naming its section and part (at least one),
    `fill` x the sub-block's token target long (per section index in `fills`), never past the request's max_tokens;
    finish "length" when the fill is at the output cap factor. `modes` scripts a reply per (section, part): "prose" (no
    bullet marker), "extra" / "plus1" (quota + 2 / quota + 1 bullets), "cut" (finish "length"), "cutmid" (finish
    "length" and a last bullet broken off mid-sentence), "cut1" (finish "length" at cap level 1 only: "stop" once
    the request comes back at cap level 2). Keeps every payload, the latest reply per (section, part) and the cap level
    of each request (`levels`)."""

    def __init__(self, fill: float = 0.8, fills: dict | None = None, modes: dict | None = None):
        self.fill, self.fills, self.modes = fill, fills or {}, modes or {}
        self.calls, self.replies, self.levels = [], {}, []

    @staticmethod
    def where(content: str) -> tuple[int, int]:
        k = int(content.split("\nSection: Section ", 1)[1].split("/", 1)[0]) - 1
        return k, int(content.split("\nPart: Part ", 1)[1].split("/", 1)[0]) - 1

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        content = payload["messages"][-1]["content"]
        k, j = self.where(content)
        quota = int(content.split("as a list of at most ", 1)[1].split(" ", 1)[0])
        fill, mode = self.fills.get(k, self.fill), self.modes.get((k, j))
        level = 1 if payload["max_tokens"] == block_cap(quota) else 2
        self.levels.append((k, j, level))
        want = min(payload["max_tokens"], int(fill * TOKENS_PER_BULLET * quota))
        lines: list[str] = []
        while True:
            n = len(lines)
            line = f"- S{k + 1}.P{j + 1} rule {n}: load over {100 + n} MW for {n + 5} minutes."
            if mode in ("extra", "plus1"):
                if n == quota + (2 if mode == "extra" else 1):
                    break
            elif lines and estimate_tokens("\n".join(lines + [line])) > want:
                break
            lines.append(line)
        if mode == "cutmid":
            lines.append(f"- S{k + 1}.P{j + 1} rule {len(lines)}: load over")
        text = "\n".join(line[2:] if mode == "prose" else line for line in lines)
        self.replies[(k, j)] = text
        finish = ("length" if fill >= OUTPUT_CAP_FACTOR or mode in ("cut", "cutmid") or (mode == "cut1" and level == 1)
                  else "stop")
        body = openai_body(text, prompt=estimate_message_tokens(payload["messages"]), completion=estimate_tokens(text),
                           finish=finish)
        return 200, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


def block_lines(text: str, k: int, j: int) -> list[str]:
    """The bullet lines of sub-block (k, j) in a memory built by BlockSummarizer."""
    return [line for line in text.splitlines() if line.startswith(f"- S{k + 1}.P{j + 1} ")]


def test_m5_sections_tile_the_revision_and_never_split_a_sentence(long_index):
    for doc in (LONG, FLAT):
        rev = long_index.ladder(doc)[0]
        text, secs = rev.text(), sections(rev)
        total = sum(s.tokens for s in secs)
        size = total / len(secs)
        assert len(secs) == math.ceil(total / SECTION_MAX_TOKENS) >= 3
        assert "".join(text[s.char_start:s.char_end] for s in secs) == text          # the sections tile the text
        assert [(s.index, s.count) for s in secs] == [(k, len(secs)) for k in range(len(secs))]
        sents = segment(text)
        at = {s.start: i for i, s in enumerate(sents)}
        for sec in secs[1:]:
            assert sec.char_start in at                                              # cut at a sentence start
            i = at[sec.char_start]
            assert sents[i].block != sents[i - 1].block                              # ... that opens a paragraph
        assert all(abs(s.tokens - size) <= 2 * SNAP * size + 50 for s in secs)
        if doc == LONG:   # a heading always lies within reach of the ideal cut: every section opens on one
            assert all(re.match(r"\d+(\.\d+)? (Operating Area|Procedure) \d", text[s.char_start:]) for s in secs[1:])
            assert secs[0].title.startswith("1 Operating Area 1 to ")
            assert all(re.fullmatch(r"\d+(\.\d)? \S.* to \d+(\.\d)? \S.*", s.title) for s in secs)
        else:
            assert all(s.title == "" and s.label() == f"Section {s.index + 1}/{len(secs)}" for s in secs)


def test_m5_subblocks_tile_each_section_at_paragraph_starts(long_index):
    for doc in (LONG, FLAT, HIST):
        rev = long_index.ladder(doc)[0]
        text = rev.text()
        sents = segment(text)
        at = {s.start: i for i, s in enumerate(sents)}
        for sec in sections(rev):
            blocks = subblocks(rev, sec)
            assert "".join(text[b.char_start:b.char_end] for b in blocks) == text[sec.char_start:sec.char_end]
            assert [(b.index, b.count) for b in blocks] == [(j, len(blocks)) for j in range(len(blocks))]
            assert sum(b.history_tokens for b in blocks) == sec.history_tokens
            for b in blocks[1:]:
                i = at[b.char_start]                                                # a sentence start ...
                assert sents[i].block != sents[i - 1].block                         # ... opening a paragraph
            assert all(b.label() == f"Part {b.index + 1}/{b.count}" + (f": {b.title}" if b.title else "")
                       for b in blocks)
            if doc != HIST:
                total = sum(b.tokens for b in blocks)
                size = total / len(blocks)
                assert len(blocks) == math.ceil(total / PART_MAX_TOKENS) >= 5
                assert all(abs(b.tokens - size) <= 2 * SNAP * size + 50 for b in blocks)
                assert all((b.title == "") == (doc == FLAT) for b in blocks)


def test_m5_bullet_allowance_is_the_target_over_28_rounded_half_up_with_a_floor_of_3():
    assert (TOKENS_PER_BULLET, BULLET_WORDS, MIN_BULLETS) == (28, 25, 3)
    cases = {1: 3, 42: 3, 97: 3, 98: 4, 111: 4, 120: 4, 125: 4, 126: 5, 666: 24, 676: 24, 686: 25, 4_964: 177}
    assert {t: bullet_count(t) for t in cases} == cases             # 98 / 28 = 3.5 and 686 / 28 = 24.5 round up
    assert all(bullet_count(t) == max(3, int(t / 28 + 0.5)) for t in range(1, 6_000))


def test_m5_quotas_split_the_section_bullets_over_its_subblocks(long_index):
    assert quota_split(24, [1_000] * 6) == [4] * 6
    assert quota_split(10, [1, 1, 2]) == [3, 3, 5]                   # 2.5 rounds up; the shares may sum past N
    assert quota_split(5, [1, 1]) == [3, 3] and quota_split(4, [100] * 6) == [1] * 6            # the floor of 1
    assert quota_split(3, [0, 10]) == [1, 3] and quota_split(4, [0, 0]) == [1, 1]
    for n in range(1, 60):
        for w in ([1, 2, 3], [5, 0, 7, 11], [1_674, 1_948, 1_637, 1_444]):
            assert quota_split(n, w) == [max(1, math.floor(Fraction(n * x, sum(w)) + Fraction(1, 2))) for x in w]
    for doc in (LONG, HIST):
        rev = long_index.ladder(doc)[0]
        calls = plan([rev], 4_000, BUILDER)
        for sec in sections(rev):
            mine = [c for c in calls if c.section == sec]
            blocks = subblocks(rev, sec)
            weights = [b.tokens if sec.revision_history else b.body_tokens for b in blocks]
            assert [c.block for c in mine] == list(blocks) and block_weights(sec, blocks) == weights
            assert [c.quota for c in mine] == quota_split(bullet_count(mine[0].target), weights)
            for c in mine:
                assert c.cell.generation == {"max_output_tokens": 56 * c.quota}
                assert (c.cell.stage, c.cell.est_visible_tokens, c.block_target) == (
                    f"build/M5@{28 * c.quota}", 28 * c.quota, 28 * c.quota)
                assert c.cell.item_id == f"{doc}@1#{sec.index}.{c.block.index}"
        hist = [c for c in calls if c.section.revision_history]
        assert (doc == HIST) == bool(hist)
        assert all(c.target == TARGET_FLOOR and c.quota == 1 for c in hist)   # N = 4 over 5+ sub-blocks: 1 each
    body_section, history_section = Section(0, 1, 0, 100, "", 50, 10), Section(0, 1, 0, 100, "", 50, 40)
    mixed = (SubBlock(0, 2, 0, 50, "", 30, 30), SubBlock(1, 2, 50, 100, "", 20, 0))
    all_history = (SubBlock(0, 2, 0, 50, "", 30, 30), SubBlock(1, 2, 50, 100, "", 20, 20))
    assert block_weights(body_section, mixed) == [0, 20]                 # non-history tokens only
    assert block_weights(history_section, mixed) == [30, 20]              # a revision-history section: all tokens
    assert block_weights(body_section, all_history) == [30, 20]           # no non-history tokens: all tokens


def test_m5_prompt_sets_per_subblock_counts_not_tokens_and_keeps_the_output_cap(long_index):
    rev = long_index.ladder(LONG)[0]
    text = rev.text()
    calls = plan([rev], 4_000, BUILDER) + plan([rev], 300, BUILDER)                 # proportional targets and the floor
    assert len({c.quota for c in calls}) > 2 and min(c.quota for c in calls) == 1
    for c in calls:
        content = c.cell.messages[0]["content"]
        instructions, context = content.split("\nDocument: ", 1)
        q = c.quota
        assert f"as a list of at most {q} bullets" in instructions and f"Write at most {q} bullets" in instructions
        assert f"each at most {BULLET_WORDS} words" in instructions and "Cover the whole part" in instructions
        assert "token" not in instructions.lower() and str(c.target) not in instructions   # no token target stated
        assert context.startswith(f"{rev.header()}\nSection: {c.section.label()}\nPart: {c.block.label()}\n<part>\n")
        assert context.endswith(f"\n{text[c.block.char_start:c.block.char_end]}\n</part>")
        assert c.cell.generation == {"max_output_tokens": math.ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * q)}
    assert PROMPT_SHA == hashlib.sha256(json.dumps(
        {"template": SUBBLOCK_PROMPT, "tokens_per_bullet": 28, "bullet_words": 25, "min_bullets": 3},
        sort_keys=True).encode("utf-8")).hexdigest()                      # the constants are part of the prompt id
    assert {k: PARAMS[k] for k in ("length_control", "part_max_tokens", "tokens_per_bullet", "bullet_words",
                                   "min_bullets", "quota_slack")} == {
        "length_control": "subblock_quota", "part_max_tokens": 2_000, "tokens_per_bullet": 28, "bullet_words": 25,
        "min_bullets": 3, "quota_slack": 1}


def section_targets(calls, secs) -> list[int]:
    return [next(c.target for c in calls if c.section == s) for s in secs]


def test_m5_targets_are_proportional_to_section_size_and_sum_to_the_budget(long_index):
    rev = long_index.ladder(LONG)[0]
    secs = sections(rev)
    total = sum(s.tokens for s in secs)
    share = 4_000 - estimate_tokens(f"[{rev.header()}]\n" + "\n\n".join(f"## {s.label()}\n" for s in secs))
    targets = section_targets(plan([rev], 4_000, BUILDER), secs)
    assert targets == [max(TARGET_FLOOR, round(share * s.tokens / total)) for s in secs]
    assert abs(sum(targets) - share) <= len(secs)
    assert {c.target for c in plan([rev], 300, BUILDER)} == {TARGET_FLOOR}           # a share under the floor


def test_m5_revision_history_gets_the_floor_and_agrees_with_the_frames_rule(long_index):
    rev = long_index.ladder(HIST)[0]
    text, secs = rev.text(), sections(rev)
    sents = segment(text)
    regions = history_regions(text, rev.family)
    frames = {s.idx for s in sents if any(a <= s.start < b for a, b in regions)}
    assert frames == {i for i, h in enumerate(history_flags(rev)) if h}                 # the item builders' rule
    straddle = next(i for i, t in enumerate(sents) if t.text.endswith("change record below Revision History"))
    assert straddle not in frames and straddle + 1 in frames
    hist = [s for s in secs if s.revision_history]
    history_tokens = sum(s.history_tokens for s in secs)
    assert 0.38 <= history_tokens / sum(s.tokens for s in secs) <= 0.45                 # ~40% of the text
    assert len(hist) >= 2 and hist[0].char_start == sents[straddle + 1].start          # history sectioned on its own
    in_hist = {i for i, t in enumerate(sents) for s in hist if s.char_start <= t.start < s.char_end}
    assert all(i in in_hist or sents[i].start < secs[0].char_end for i in frames)
    assert all(s.history_tokens == 0 for s in secs[1:] if not s.revision_history)
    assert 0 < secs[0].history_tokens < secs[0].tokens / 2                               # the short front history
    assert 0 < hist[-1].body_tokens < hist[-1].tokens / 2                                # the glossary joined it
    share = 4_000 - estimate_tokens(f"[{rev.header()}]\n" + "\n\n".join(f"## {s.label()}\n" for s in secs))
    rest = share - TARGET_FLOOR * len(hist)
    body = sum(s.tokens - s.history_tokens for s in secs if not s.revision_history)
    calls = plan([rev], 4_000, BUILDER)
    targets = section_targets(calls, secs)
    assert targets == [TARGET_FLOOR if s.revision_history
                       else max(TARGET_FLOOR, round(rest * (s.tokens - s.history_tokens) / body)) for s in secs]
    assert abs(sum(targets) - share) <= len(secs)
    given = round(sum(t for t, s in zip(targets, secs) if s.revision_history) / sum(targets), 4)
    assert given < 0.1                                                                   # not its ~40% token share
    mem = build_memory(long_item(HIST), "M5", 4_000, long_index, builder_profile=BUILDER)
    assert (mem.build["revision_history_tokens"], mem.build["revision_history_share"]) == (history_tokens, given)
    assert mem.notes == (f"dry_run; sections={len(secs)}; subblocks={len(calls)}; "
                         f"revision_history_tokens={history_tokens}; revision_history_share={given}")
    assert [s["revision_history"] for s in mem.build["blocks"]] == [c.section.revision_history for c in calls]


def test_m5_build_blocks_builds_part_of_a_plan_through_run_cells(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    calls = plan([long_index.ladder(LONG)[0]], 4_000, BUILDER)
    transport = BlockSummarizer(fill=0.8)
    ledger = BuilderLedger(tmp_path / "partial.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    builds = build_blocks(calls[:3], BUILDER, ledger)                      # a partial build: the first three sub-blocks
    assert len(transport.calls) == 3 and [b.mode for b in builds] == ["live"] * 3
    assert [b.text for b in builds] == [transport.replies[(0, j)] for j in range(3)]
    assert sorted(p["max_tokens"] for p in transport.calls) == sorted(
        c.cell.generation["max_output_tokens"] for c in calls[:3])
    _, records = read_ledger(ledger.path)
    assert sorted(parse_call_id(r["item_id"])[2:] for r in records) == [(0, 0), (0, 1), (0, 2)]   # sent concurrently
    assert {b.request_hash for b in builds} == {r["request_hash"] for r in records}
    assert build_blocks(calls[:3], BUILDER, ledger) == builds and len(transport.calls) == 3    # resume: no resend
    dry = build_blocks(calls[:3], BUILDER, None)
    assert [b.mode for b in dry] == ["dry_run"] * 3
    assert [b.request_hash for b in dry] == [b.request_hash for b in builds]


def test_m5_builds_every_subblock_of_a_long_revision_within_its_output_cap(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    rev = long_index.ladder(LONG)[0]
    secs, text = sections(rev), rev.text()
    calls = plan([rev], 4_000, BUILDER)
    transport = BlockSummarizer(fill=0.8)
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert len(transport.calls) == len(calls) >= 5 * len(secs) >= 15
    sent = {BlockSummarizer.where(p["messages"][-1]["content"]): p for p in transport.calls}
    for call in calls:
        where = (call.section.index, call.block.index)
        payload, reply = sent[where], transport.replies[where]
        assert text[call.block.char_start:call.block.char_end] in payload["messages"][-1]["content"]
        assert payload["max_tokens"] == math.ceil(OUTPUT_CAP_FACTOR * TOKENS_PER_BULLET * call.quota)
        assert 0 < estimate_tokens(reply) <= payload["max_tokens"] and block_lines(mem.text, *where) ==\
            reply.splitlines()
    _, records = read_ledger(ledger.path)
    assert sorted(parse_call_id(r["item_id"]) for r in records) == sorted(
        (LONG, "1", c.section.index, c.block.index) for c in calls)
    assert mem.build_tokens == sum(r["usage"]["input_total"] + r["usage"]["output_total"] for r in records)
    assert mem.text_tokens <= 4_000 and mem.build["mode"] == "live" and mem.build["trimmed_tokens"] == 0
    heads = [mem.text.index(f"## {s.label()}\n") for s in secs]
    assert heads == sorted(heads) and mem.text.startswith(f"[{rev.header()}]\n## Section 1/")
    for s, start, end in zip(secs, heads, [*heads[1:], len(mem.text)]):             # a section = its sub-blocks
        body = mem.text[start:end].split("\n", 1)[1].strip()
        assert body == "\n".join(transport.replies[(s.index, b.index)] for b in subblocks(rev, s))
    assert mem.notes.endswith(f"live; sections={len(secs)}; subblocks={len(calls)}; {NO_HISTORY}; "
                              f"finish_reasons=stop:{len(calls)}; cap_levels=1:{len(calls)}; still_capped=0; "
                              f"trimmed_tokens=0; dropped_bullets=0; {ALL_CLEAR}")
    assert [p["char_start"] for p in mem.provenance] == [c.block.char_start for c in calls]
    again = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert again == mem and len(transport.calls) == len(calls)                     # the ledger is the cache


def test_m5_checks_count_missing_bullets_cut_answers_and_overlong_lists(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = BlockSummarizer(fill=0.8, modes={(0, 0): "prose", (0, 1): "extra", (1, 0): "cut", (1, 1): "plus1"})
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert mem.build["checks"] == {"subblocks_without_bullet": 1, "subblocks_not_stop": 1, "subblocks_over_quota": 1,
                                   "over_budget": 0}
    assert "checks=subblocks_without_bullet:1,subblocks_not_stop:1,subblocks_over_quota:1,over_budget:0" in mem.notes
    info = {(b["section_index"], b["subblock_index"]): b for b in mem.build["blocks"]}
    assert info[(0, 0)]["bullets"] == 0 and info[(0, 1)]["bullets"] == info[(0, 1)]["quota"] + 2
    assert info[(1, 1)]["bullets"] == info[(1, 1)]["quota"] + 1                     # quota + 1 is tolerated
    assert info[(1, 0)]["finish_reason"] == "length" and "build_truncated" in mem.notes
    assert "finish_reasons=length:1,stop:" in mem.notes


def test_m5_output_cap_is_twice_28_tokens_a_bullet_and_part_of_the_block_key(long_index, monkeypatch):
    assert OUTPUT_CAP_FACTOR == 2.0 and PARAMS["output_cap_factor"] == 2.0                  # (was 1.25)
    assert [block_cap(k) for k in (1, 4, 5, 142)] == [56, 224, 280, 7_952]
    assert all(block_cap(k) == math.ceil(2.0 * 28 * k) for k in range(1, 300))
    assert PARAMS["cut_bullet_rule"] == "drop_last_if_truncated_mid_sentence"
    assert [block_cap(k, 2) for k in (1, 3, 14, 142)] == [252, 532, 2_072, 19_992]           # the cap ladder's level 2
    assert all(block_cap(k, 2) == 140 * k + 112 for k in range(1, 300))
    with pytest.raises(ValueError, match="cap level"):
        block_cap(1, 3)                                                                     # no level 3
    assert PARAMS["cap_escalation"] == CAP_ESCALATION and "ceil(5.0 x 28 x quota) + 112" in CAP_ESCALATION
    rev = long_index.ladder(LONG)[0]
    c = plan([rev], 4_000, BUILDER)[0]
    assert c.cell.generation == {"max_output_tokens": block_cap(c.quota)} and c.cell.est_visible_tokens == 28 * c.quota
    up = c.cells[1]                                     # level 2: the same prompt, the level-2 cap, labelled
    assert (up.messages, up.item_id, up.stage) == (c.cell.messages, c.cell.item_id, c.cell.stage)
    assert up.generation == {"max_output_tokens": block_cap(c.quota, 2)} and up.labels == {"cap_level": 2}
    assert up.est_visible_tokens == block_cap(c.quota) and c.cell.labels == {}
    assert c.keys == tuple(block_key(rev, c.section, c.block, BUILDER, 4_000, c.target, c.quota, level)
                           for level in (1, 2)) and c.keys[0] != c.keys[1] and c.key == c.keys[0]
    import procedures.memory.summary as m5
    monkeypatch.setattr(m5, "OUTPUT_CAP_FACTOR", 1.25)
    assert block_key(rev, c.section, c.block, BUILDER, 4_000, c.target, c.quota) != c.key     # the cap is in the key
    monkeypatch.setattr(m5, "ESCALATED_CAP_EXTRA", 111)
    assert block_key(rev, c.section, c.block, BUILDER, 4_000, c.target, c.quota, 2) != c.keys[1]


def test_m5_cap_ladder_resends_only_truncated_level_1_answers_at_level_2(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    rev = long_index.ladder(LONG)[0]
    calls = plan([rev], 4_000, BUILDER)
    transport = BlockSummarizer(fill=0.8, modes={(0, 1): "cut1", (1, 2): "cut"})   # stop at level 2 / still cut
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert sorted(t for t in transport.levels if t[2] == 2) == [(0, 1, 2), (1, 2, 2)]    # level 2 only after a cut
    assert len(transport.calls) == len(calls) + 2
    by = {(c.section.index, c.block.index): c for c in calls}
    for payload in transport.calls:
        k, j = BlockSummarizer.where(payload["messages"][-1]["content"])
        c = by[(k, j)]
        assert payload["messages"][-1]["content"] == c.cell.messages[-1]["content"]      # the same prompt at each level
        assert payload["max_tokens"] in (block_cap(c.quota), block_cap(c.quota, 2))
    _, records = read_ledger(ledger.path)
    assert len(records) == len(calls) + 2                                           # both records stay in the ledger
    assert sorted(parse_call_id(r["item_id"])[2:] for r in records if r.get("cap_level") == 2) == [(0, 1), (1, 2)]
    assert all("cap_level" not in r for r in records if r["params"]["max_output_tokens"] == 56 * (
        by[parse_call_id(r["item_id"])[2:]].quota))                                  # level-1 records carry no label
    info = {(b["section_index"], b["subblock_index"]): b for b in mem.build["blocks"]}
    up, capped = info[(0, 1)], info[(1, 2)]
    assert (up["cap_level"], up["finish_reason"], [v["finish_reason"] for v in up["levels"]]) == (
        2, "stop", ["length", "stop"])                                               # the level that finished stop
    assert up["max_output_tokens"] == block_cap(up["quota"], 2)
    assert up["request_hash"] == up["levels"][1]["request_hash"]
    level2 = {r["request_hash"]: r for r in records}[up["request_hash"]]
    assert level2["cap_level"] == 2 and block_lines(mem.text, 0, 1) == level2["text"].splitlines()
    assert up["build_tokens"] == sum(v["build_tokens"] for v in up["levels"]) and up["usd"] == pytest.approx(
        sum(v["usd"] for v in up["levels"]))                                         # both calls are this block's cost
    assert (capped["cap_level"], capped["finish_reason"]) == (2, "length")          # no level 3: kept, counted
    assert all(b["cap_level"] == 1 and len(b["levels"]) == 1 for w, b in info.items() if w not in ((0, 1), (1, 2)))
    assert mem.build["cap_levels"] == {"1": len(calls) - 2, "2": 2} and mem.build["still_capped"] == 1
    assert f"cap_levels=1:{len(calls) - 2},2:2; still_capped=1" in mem.notes
    assert mem.build["checks"]["subblocks_not_stop"] == 1
    assert mem.build_tokens == sum(r["usage"]["input_total"] + r["usage"]["output_total"] for r in records)
    again = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    assert again == mem and len(transport.calls) == len(calls) + 2                  # the ledger is the cache
    assert not escalates({"outcome": "observed", "finish_reason": "stop"}) and not escalates(None)
    assert not escalates({"outcome": "no_response", "finish_reason": "length"})
    assert escalates({"outcome": "observed", "finish_reason": "length"})


def test_m5_a_truncated_level_1_answer_without_its_level_2_record_is_not_a_built_memory(long_index, monkeypatch,
                                                                                         tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    rev = long_index.ladder(LONG)[0]
    calls = plan([rev], 4_000, BUILDER)
    transport = BlockSummarizer(fill=0.8, modes={(0, 1): "cut1"})
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    head, *lines = ledger.path.read_text(encoding="utf-8").splitlines()               # a ledger from before the ladder
    kept = [x for x in lines if "cap_level" not in json.loads(x)]
    assert len(kept) == len(calls)
    ledger.path.write_text("\n".join([head, *kept]) + "\n", encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="no ledger record for the cap level 2 request"):
        answers_from_records(calls, panel().get(BUILDER), {r["request_hash"]: r for r in read_ledger(ledger.path)[1]},
                             ledger.path)


def test_m5_cut_bullet_rule_drops_only_a_truncated_last_bullet_broken_mid_sentence():
    assert drop_cut_bullet("- a b.\n- c d", "length") == ("- a b.", 1)
    assert drop_cut_bullet("- a b.\n- c d\n  continued", "length") == ("- a b.", 1)       # a wrapped last bullet
    assert drop_cut_bullet("- a b.\n- c d;", "length") == ("- a b.", 1)
    assert drop_cut_bullet("- only cut", "length") == ("", 1)
    for whole in ("- a b.\n- c d.", '- quoted "end."', "- a (see x.)", "- a?\n- b!"):         # ends a sentence
        assert drop_cut_bullet(whole, "length") == (whole, 0)
    assert drop_cut_bullet("- a b.\n- c d", "stop") == ("- a b.\n- c d", 0)                  # not truncated
    assert drop_cut_bullet("- a b.\n- c d", None) == ("- a b.\n- c d", 0)
    assert drop_cut_bullet("prose cut off", "length") == ("prose cut off", 0)                 # no bullet line


def test_m5_assembly_drops_a_bullet_cut_mid_sentence_and_records_it(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = BlockSummarizer(fill=0.8, modes={(0, 1): "cutmid", (1, 2): "cut"})
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    info = {(b["section_index"], b["subblock_index"]): b for b in mem.build["blocks"]}
    cut = transport.replies[(0, 1)].splitlines()
    assert cut[-1].endswith("load over") and block_lines(mem.text, 0, 1) == cut[:-1]      # the broken bullet is gone
    assert block_lines(mem.text, 1, 2) == transport.replies[(1, 2)].splitlines()          # length, but whole: kept
    assert {w: b["dropped_bullets"] for w, b in info.items() if b["dropped_bullets"]} == {(0, 1): 1}
    assert info[(0, 1)]["trimmed_tokens"] == 0 and mem.build["trimmed_tokens"] == 0      # a drop is not a trim
    assert mem.build["dropped_bullets"] == 1 and "dropped_bullets=1" in mem.notes and "dropped_cut_bullets" in mem.notes
    assert mem.build["checks"]["subblocks_not_stop"] == 2 and "trimmed_to_budget" not in mem.notes


def test_m5_assembly_trims_every_subblock_proportionally_at_sentence_boundaries(long_index, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "dummy-hf-token")
    transport = BlockSummarizer(fill=OUTPUT_CAP_FACTOR, fills={0: 0.9})   # all but section 1 run to the cap
    ledger = BuilderLedger(tmp_path / "b1.jsonl", cap_usd=1.0, transport=transport, sleep=lambda s: None)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER, ledger=ledger)
    info = mem.build["blocks"]
    assert mem.text_tokens <= 4_000 < estimate_tokens("\n".join(transport.replies.values()))
    assert "trimmed_to_budget" in mem.notes and "build_truncated" in mem.notes
    cut = mem.build["trimmed_tokens"]
    assert cut == sum(s["trimmed_tokens"] for s in info) > 0 and f"trimmed_tokens={cut}" in mem.notes
    for b in info:
        where = (b["section_index"], b["subblock_index"])
        reply, kept = transport.replies[where], "\n".join(block_lines(mem.text, *where))
        if where[0] == 0:                                                           # within its share: untouched
            assert kept == reply and b["trimmed_tokens"] == 0
        else:                   # every sub-block keeps its first bullets: the trim keeps each section's breadth
            assert kept and reply.startswith(kept) and kept.endswith("minutes.") and kept != reply
            assert b["kept_tokens"] == estimate_tokens(kept) < b["summary_tokens"]
    ratios = [b["kept_tokens"] / b["block_target_tokens"] for b in info if b["section_index"] > 0]
    longest = max(estimate_tokens(line) for r in transport.replies.values() for line in r.splitlines())
    assert max(ratios) - min(ratios) <= longest / min(b["block_target_tokens"] for b in info)


def test_m5_fit_caps_unequal_subblocks_at_one_multiple_of_their_targets(long_index):
    rev = long_index.ladder(LONG)[0]
    calls = [replace(c, quota=q) for c, q in zip(plan([rev], 4_000, BUILDER), (14, 35, 71))]
    bodies = [" ".join(f"Rule {k}-{n} applies for {n + 5} minutes." for n in range(c.block_target * 3 // 20))
              for k, c in enumerate(calls)]                                          # ~1.5 x each target
    assert fit(calls, bodies, 100_000) == bodies                                     # fits: nothing is cut
    kept = fit(calls, bodies, 3_000)
    assert estimate_tokens(render(_parts(calls, kept))) <= 3_000
    assert all(b.startswith(k) and k.endswith("minutes.") for b, k in zip(bodies, kept))
    ratios = [estimate_tokens(k) / c.block_target for k, c in zip(kept, calls)]
    assert max(ratios) - min(ratios) < 0.05 and max(ratios) < 1.0


def test_m5_cache_keys_are_per_subblock_and_differ_from_the_section_scheme(long_index):
    rev = long_index.ladder(LONG)[0]
    calls = plan([rev], 4_000, BUILDER)
    keys = [c.key for c in calls]
    assert len(set(keys)) == len(keys) and all(re.fullmatch(r"[0-9a-f]{64}", k) for k in keys)
    per_section = {hashlib.sha256(json.dumps(   # a per-section key without sub-block fields (it must differ)
        {"doc_id": rev.doc, "revision": rev.revision, "sha256_lf": rev.sha256_lf, "builder_profile": BUILDER,
         "budget": 4_000, "section_index": c.section.index,
         "section_span": [c.section.char_start, c.section.char_end], "revision_history": c.section.revision_history,
         "target_tokens": c.target, "prompt_sha": PROMPT_SHA, "token_method": token_method()},
        sort_keys=True).encode("utf-8")).hexdigest() for c in calls}
    assert not per_section & set(keys)
    mem = build_memory(long_item(), "M5", 4_000, long_index, builder_profile=BUILDER)
    assert [s["block_key"] for s in mem.build["blocks"]] == keys
    assert mem.notes == f"dry_run; sections={len(sections(rev))}; subblocks={len(calls)}; {NO_HISTORY}"
    assert mem.cache_key == cache_key({"representation": "M5", "budget_tokens": 4_000, "token_method": token_method(),
                                       "builder_profile": BUILDER, "params": PARAMS,
                                       "blocks": [list(c.keys) for c in calls]})      # every cap level's key
    assert [[v["block_key"] for v in s["levels"]] for s in mem.build["blocks"]] == [list(c.keys) for c in calls]
    for other in (plan([rev], 3_000, BUILDER), plan([rev], 4_000, "C1_llama31_8b_hf_deepinfra")):
        assert not {c.key for c in other} & set(keys)
    c = calls[1]
    same = block_key(rev, c.section, c.block, BUILDER, 4_000, c.target, c.quota)
    for moved in (replace(c.block, char_start=c.block.char_start + 1), replace(c.block, index=0)):
        assert block_key(rev, c.section, moved, BUILDER, 4_000, c.target, c.quota) != c.key == same
    assert block_key(rev, c.section, c.block, BUILDER, 4_000, c.target, c.quota + 1) != c.key
    flat = build_memory(long_item(FLAT), "M5", 4_000, long_index, builder_profile=BUILDER)
    assert flat.cache_key != mem.cache_key


def test_m5_sections_and_plans_are_deterministic(tmp_path):
    runs = []
    for k in range(2):
        sections.cache_clear()
        subblocks.cache_clear()
        chunk_sentences.cache_clear()
        index = write_long_corpus(tmp_path / f"run{k}")
        rev = index.ladder(LONG)[0]
        mem = build_memory(long_item(), "M5", 4_000, index, builder_profile=BUILDER)
        runs.append(([(s.char_start, s.char_end, s.title, s.tokens) for s in sections(rev)],
                     [(c.key, c.cell.messages) for c in plan([rev], 4_000, BUILDER)], mem.to_dict()))
    assert runs[0] == runs[1]


# ------------------------------------------------------------------------------------------------ determinism
def test_memories_are_deterministic_and_keyed_by_their_inputs(tmp_path):
    manifest = write_corpus(tmp_path)
    runs = []
    for _ in range(2):
        index = RevisionIndex.from_manifest(manifest)
        it = item(as_of_date="2021-06-01", m0_paths=["proj/data/corpus/test/r1.txt", "proj/data/corpus/test/r2.txt"])
        runs.append([build_memory(it, rep, b, index, **kw).to_dict() for rep, b, kw in (
            ("B0", None, {}), ("M0", None, {}), ("M2", 200, {}), ("M4", 200, {}),
            ("M5", 300, {"builder_profile": BUILDER}))])
    assert runs[0] == runs[1]
    keys = [m["cache_key"] for m in runs[0]]
    assert len(set(keys)) == len(keys) and all(re.fullmatch(r"[0-9a-f]{64}", k) for k in keys)
    index = RevisionIndex.from_manifest(manifest)
    other = build_memory(item(question="Under TEST-OP-1 as in effect on 2021-06-01, how long is a dispatch interval?",
                              as_of_date="2021-06-01"), "M2", 200, index)
    assert other.cache_key != runs[0][2]["cache_key"]


def test_arguments_are_checked(corpus):
    _, index = corpus
    it = item(as_of_date="2021-06-01")
    for rep, budget, kw in (("M9", 100, {}), ("M2", None, {}), ("M4", 0, {}), ("M2", 100, {"builder_profile": BUILDER}),
                            ("M5", 100, {}), ("M0", True, {})):
        with pytest.raises(ValueError):
            build_memory(it, rep, budget, index, **kw)
    with pytest.raises(TypeError):
        build_memory(it, "M5", 100, index, builder_profile=BUILDER, ledger="ledger.jsonl")
    with pytest.raises(TypeError):
        build_memory(it, "M2", 100, {"not": "an index"})

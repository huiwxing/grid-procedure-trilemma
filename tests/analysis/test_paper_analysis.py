"""The paper's analysis end to end on a synthetic copy of the paper's four runs (the B-built arm, the two C-built
arms of records memories (M7), the cloud comparator) written the way
procedures.matrix.run_matrix lays them out, with real summary (M5) / M7 plans over two synthetic revision texts, so
the memory read-back (procedures.analysis.readback) runs its build checks. Covered: the 17 primary comparisons, REF-B
(the open-weight reference: the B1 reader's chosen condition, M2@4000 here) chosen once and held under the
sensitivities, the outcomes of the constrained panel (tier C_load_bearing: the C1 and C2 readers) against REF-B
(verdict) and against the cloud comparator (reported), the k = 1 rule (a result read once is marked) on the 4k slice,
absences (route_context, route_timeout), the supply / power check block, the exploratory memory sensitivities
(build_truncated M5; M7 located >= 0.90), a tier on fewer than 3 models (AMBER) and comparisons the runs cannot give
(not computable, with what is missing). Zero network: sockets are blocked."""
from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
import socket

import pytest

import analysis_synth as S
from procedures.analysis.costs import load_bands
from procedures.analysis.data import load_run
from procedures.analysis.readback import memory_readback
from procedures.analysis.report import analyse, render_markdown
from procedures.analysis.spec import CONSTRAINED_TIER, make_paper_spec
from procedures.corpus.revision_text import read_lf, sha256_lf
from procedures.readers import load_panel, load_price_book
from procedures.memory import summary as m5, records as m7
from procedures.corpus.revisions import RevisionIndex, RevisionText

ROOT = Path(__file__).resolve().parents[2]
ITEMS = S.main_run_items()
BY_ID = {it["item_id"]: it for it in ITEMS}
POP = S.POPULATION
B1_DIRECT = S.BUILDER
GPTOSS20B, QWEN32B = S.READERS_C[2], S.READER_C2
C_READERS = (*S.READERS_C, S.READER_C2, S.READER_C3)          # the five C readers (C3 = gpt-oss-120b, apart)
TIERED = (*S.READERS_B, *C_READERS)
BUDGETS = (2000, 4000, 16000)
SUBSET = {"p1a", "s3x", "u4a"}              # the 20% re-read subset outside the population
TIMEOUT = {S.READERS_C[0]: {"u2a"}, QWEN32B: {"u4a", "s3y"}}   # M6 self-compressions that timed out
ROUTE_CONTEXT = {"s3x", "s3y"}              # Llama@ovhcloud M0@full cells over the route's context
TRUNCATED_M5 = ("M-01", 4000)               # its level-2 answer is still capped: build_truncated
UNLOCATED = {(GPTOSS20B, "BPM-001"): 1}     # records of that builder's M7 whose span is not in the text
PAPER_IDS = ["P0-B1", "P0-A", "REF-B", f"OUTCOME-{CONSTRAINED_TIER}", f"OUTCOME-{CONSTRAINED_TIER}-A",
            "dGap-M5-B", f"dGap-M5-{CONSTRAINED_TIER}", "dGap-M7-B", f"dGap-M7-{CONSTRAINED_TIER}", "dGap-M6-B",
            f"dGap-M6-{CONSTRAINED_TIER}", "interaction-M5", "interaction-M7", "H_M7", "H_S",
            "CBUILT-gptoss20b", "CBUILT-qwen32b"]


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise RuntimeError("network blocked in tests")
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(scope="module")
def panel():
    return load_panel()


@pytest.fixture(scope="module")
def book():
    return load_price_book()


@pytest.fixture(scope="module")
def bands():
    return load_bands(ROOT / "data" / "analysis" / "energy_bands.json")


# ---------------------------------------------------------------------- revisions and memories


def sentences(doc: str) -> list[str]:
    return [f"{doc} rule {k}: the operator shall open the breaker within {k + 5} minutes when the bus voltage exceeds "
            f"{100 + k} kV." for k in range(6)]


def revision_index(root: Path) -> RevisionIndex:
    ladders = {}
    for doc, family in (("M-01", "PJM"), ("BPM-001", "MISO")):
        path = root / f"{doc}.txt"
        path.write_text("\n\n".join(sentences(doc)) + "\n", encoding="utf-8", newline="\n")
        ladders[doc] = (RevisionText(family, doc, "1", f"Manual {doc}", date(2020, 1, 1), None, path.resolve(),
                                     f"corpus/{doc}.txt", sha256_lf(read_lf(path))),)
    return RevisionIndex(ladders, root)


def m7_answer(builder: str, doc: str) -> str:
    bad = UNLOCATED.get((builder, doc), 0)
    spans = sentences(doc)[:5 - bad] + [f"a clause {k} that {doc} does not contain" for k in range(bad)]
    return json.dumps([{"subject": f"rule {k}", "value": f"{k + 5} min", "source_span": s} for k, s in enumerate(spans)])


def ledger_of(pp, book, calls, answer, finishes, tokens=(100, 20)) -> tuple[list[dict], list[dict]]:
    """(the ledger records of `calls` by builder `pp`, the request map's request list, every cap level as a planned
    map lists it): per call the cap levels in `finishes` (one finish per level sent), each with `answer`."""
    records, listed = [], []
    for call in calls:
        for level, (key, cell) in enumerate(zip(call.keys, call.cells), start=1):
            h = m5._request(pp, cell)[0]
            listed.append({"request_hash": h, "cap_level": level, "block_key": key})
            if level <= len(finishes):
                records.append(S.call_record(pp, h, cell.item_id, cell.stage, S.usage(*tokens), book,
                                             finish_reason=finishes[level - 1], text=answer))
    return records, listed


def built(panel, book, index, builder: str) -> tuple[list[dict], dict, dict]:
    """(the build ledger records of `builder`, the request map entries of its memories, memory key -> build tokens):
    M7 at every budget for both revisions; M5 too for the B1 builder, TRUNCATED_M5 truncated at both cap levels."""
    pp = panel.get(builder)
    records, entries, tokens = [], {}, {}
    for doc in ("M-01", "BPM-001"):
        rev = index.ladder(doc)[0]
        recs, listed = ledger_of(pp, book, m7.plan([rev], builder), m7_answer(builder, doc), ["stop"])
        records += recs
        for b in BUDGETS:
            key = mem_key("M7", b, builder, doc)
            entries[key] = {"representation": "M7", "budget_tokens": b, "profile": builder,
                            "ledger": f"build__{builder}", "revisions": [[doc, "1"]], "requests": listed}
            tokens[key] = 120 * len(recs)
        if builder != B1_DIRECT:
            continue
        for b in BUDGETS:
            capped = (doc, b) == TRUNCATED_M5
            answer = f"- {doc} rule 0: open the breaker within 5 minutes" + ("" if capped else ".")
            recs, listed = ledger_of(pp, book, m5.plan([rev], b, builder), answer,
                                     ["length", "length"] if capped else ["stop"])
            records += recs
            key = mem_key("M5", b, builder, doc)
            entries[key] = {"representation": "M5", "budget_tokens": b, "profile": builder,
                            "ledger": f"build__{builder}", "revisions": [[doc, "1"]], "requests": listed}
            tokens[key] = 120 * len(recs)
    return records, entries, tokens


def mem_key(rep: str, budget, maker: str, doc_or_item: str) -> str:
    return hashlib.sha256(f"{rep}|{budget}|{maker}|{doc_or_item}".encode()).hexdigest()


# ---------------------------------------------------------------------- rows


def wrong(pid: str, cond: str) -> set:
    """The design: B1 M0@full misses u2b (0.75; REF-B = B1 M2@4000, 1.0); B0 misses the population; every tiered reader
    misses u1a on the B1-built M5@4000 (M-01, whose M5@4000 memory is build_truncated); the gpt-oss-20b-built M7@4000
    misses u2b (BPM-001, the revision gpt-oss-20b located below 0.90) and the Qwen3-32B-built one u1a; the rest
    right."""
    if cond == "B0@full":
        return set(POP)
    if (pid, cond) == (S.READER_B, "M0@full"):
        return {"u2b"}
    if cond == f"M5@4000/{B1_DIRECT}":
        return {"u1a"}
    if cond == f"M7@4000/{GPTOSS20B}":
        return {"u2b"}
    if cond == f"M7@4000/{QWEN32B}":
        return {"u1a"}
    return set()


def reads(pid: str, cond: str, item_id: str, slice_: list[str]) -> int:
    """3 reads for the population in the 4k slice (tiered readers) and for the 20% subset; 1 elsewhere; the A arm 1."""
    if pid == S.READER_A:
        return 1
    in_slice = any(x.replace("{reader}", pid) == cond for x in slice_)
    return 3 if item_id in SUBSET or (in_slice and item_id in POP) else 1


def b_conditions(pid: str) -> list[str]:
    conds = ["B0@full"] + (["M0@full"] if pid in S.READERS_B else [])
    conds += [f"{rep}@{b}" for rep in ("M2", "M4") for b in BUDGETS]
    conds += [f"{rep}@{b}/{B1_DIRECT}" for rep in ("M5", "M7") for b in BUDGETS]
    conds.append(f"M6@4000/{B1_DIRECT if pid in S.READERS_B else pid}")
    return conds


def rows_for(conditions: dict[str, list[str]], slice_: list[str], build_tokens: dict, timeout: dict, drop=()):
    """conditions: reader -> its conditions; `timeout`: compressor -> item ids whose M6 compression timed out; `drop`:
    (reader, condition) pairs not read at all."""
    def rows_of(run_id):
        out = []
        for pid, conds in conditions.items():
            for cond in conds:
                if (pid, cond) in drop:
                    continue
                rep, _, rest = cond.partition("@")
                budget, _, maker = rest.partition("/")
                for it in ITEMS:
                    iid, doc = it["item_id"], it["doc"]
                    kw = {}
                    if cond == "M0@full":
                        kw["tokens"] = (3000, 50) if pid == S.READER_A else (80_000, 50)
                        if pid == "tierB_llama33_70b_hf_ovhcloud" and iid in ROUTE_CONTEXT:
                            kw["not_sent"] = "route_context"
                    if rep in ("M5", "M7"):
                        kw["memory_key"] = mem_key(rep, int(budget), maker, doc)
                        kw["build_tokens"] = build_tokens[kw["memory_key"]]
                    if rep == "M6":
                        kw["memory_key"] = mem_key("M6", 4000, maker, iid)
                        timed = iid in timeout.get(maker, set())
                        kw["query_tokens"] = 0 if timed else 2100
                        if timed:
                            kw["not_sent"] = "route_timeout"
                    for r in range(reads(pid, cond, iid, slice_)):
                        out.append(S.row(run_id, pid, cond, it, iid not in wrong(pid, cond), repeat=r, **kw))
        return out
    return rows_of


def m6_ledgers(panel, book, timeout: dict) -> tuple[dict, dict]:
    """(stem -> (compressor, records), request map entries) of the M6 compressions: B1 direct for the B readers, each C
    reader for itself; the timed-out ones have a request and no answered record."""
    ledgers, entries = {}, {}
    for maker in (B1_DIRECT, *C_READERS):
        recs = []
        for k, it in enumerate(ITEMS):
            h = hashlib.sha256(f"m6|{maker}|{it['item_id']}".encode()).hexdigest()
            key = mem_key("M6", 4000, maker, it["item_id"])
            entries[key] = {"representation": "M6", "budget_tokens": 4000, "profile": maker,
                            "ledger": f"query__{maker}", "revisions": [[it["doc"], "1"]],
                            "requests": [{"request_hash": h}]}
            if it["item_id"] not in timeout.get(maker, set()):
                recs.append(S.call_record(panel.get(maker), h, it["item_id"], f"query/M6@4000/{maker}",
                                          S.usage(2000, 100), book))
        ledgers[f"query__{maker}"] = (maker, recs)
    return ledgers, entries


def write_paper_runs(root: Path, panel, book, spec: dict, drop=(),
                     timeout: dict = TIMEOUT) -> tuple[list, RevisionIndex]:
    index = revision_index(root)
    slice_ = spec["repeats"]["k1_scope_conditions"]
    b_records, b_entries, tokens = built(panel, book, index, B1_DIRECT)
    q_ledgers, q_entries = m6_ledgers(panel, book, timeout)
    runs = [S.write_run(root, rows_for({p: b_conditions(p) for p in TIERED}, slice_, tokens, timeout, drop), ITEMS, panel,
                        book, label="p3a", requests={**b_entries, **q_entries},
                        other_ledgers={f"build__{B1_DIRECT}": (B1_DIRECT, b_records), **q_ledgers})]
    for label, builder in (("cb20", GPTOSS20B), ("cb32", QWEN32B)):
        records, entries, tokens = built(panel, book, index, builder)
        conds = {p: [f"M7@{b}/{builder}" for b in BUDGETS] for p in C_READERS}
        runs.append(S.write_run(root, rows_for(conds, slice_, tokens, timeout), ITEMS, panel, book, label=label,
                                requests=entries, other_ledgers={f"build__{builder}": (builder, records)}))
    aref = {S.READER_A: ["B0@full", "M0@full", "M2@4000"]}
    runs.append(S.write_run(root, rows_for(aref, slice_, {}, timeout), ITEMS, panel, book, label="aref"))
    return [load_run(r) for r in runs], index


def paper_spec(panel) -> dict:
    return make_paper_spec(ITEMS, "synthetic.jsonl", "0" * 64, panel, {})


# ---------------------------------------------------------------------- the tests


@pytest.fixture(scope="module")
def paper_runs(tmp_path_factory, panel, book, bands):
    root = tmp_path_factory.mktemp("paper")
    spec = paper_spec(panel)
    runs, index = write_paper_runs(root, panel, book, spec)
    readback = memory_readback(runs, index, source={"manifest": "synthetic"})
    report = analyse(runs, ITEMS, panel, book, bands, pool=ITEMS, spec=spec, readback=readback)
    return spec, runs, readback, report


def test_the_readback_counts_build_truncated_m5_and_m7_located_per_revision(paper_runs):
    _, _, readback, _ = paper_runs
    assert readback["unreadable"] == []
    truncated = readback["m5_build_truncated"]
    assert len(truncated) == 6 and [k for k, v in truncated.items() if v] == [mem_key("M5", 4000, B1_DIRECT, "M-01")]
    located = readback["m7_located"]
    assert set(located) == {B1_DIRECT, GPTOSS20B, QWEN32B}
    assert located[GPTOSS20B]["BPM-001@1"] == {"candidates": 5, "located": 4, "share": 0.8}
    assert all(v == {"candidates": 5, "located": 5, "share": 1.0}
               for b in (B1_DIRECT, QWEN32B) for v in located[b].values())


def test_every_primary_comparison_is_computed_on_the_full_matrix(paper_runs):
    spec, runs, _, report = paper_runs
    primary = {p["id"]: p for p in report["primary"]}
    assert [p["id"] for p in report["primary"]] == PAPER_IDS
    assert all(p["computable"] and p["estimate"] is not None for p in primary.values())
    assert report["meta"]["spec"]["population_cells"] == 4 and len(report["meta"]["runs"]) == 4
    assert not any("not computable" in r for r in report["status"]["reasons"])
    assert all(p["small_tiers"] == [] for p in primary.values())          # B has 3 models, C_load_bearing 4
    # the missing list names the absences a computed estimate rests without (C1 Llama's timed-out u2a is a cell, not
    # a condition: nothing is missing), and nothing else
    assert all(p["missing"] == [] for p in primary.values())
    # dGap-M5: every tiered reader misses u1a on M5@4000 -> -0.25; M6 (C) loses u2a (C1 Llama's compression timed out)
    assert primary["dGap-M5-B"]["estimate"]["est"] == -0.25 == primary[f"dGap-M5-{CONSTRAINED_TIER}"]["estimate"]["est"]
    assert primary[f"dGap-M6-{CONSTRAINED_TIER}"]["estimate"]["n"] == 3
    assert primary["dGap-M6-B"]["estimate"]["n"] == 4
    assert primary["CBUILT-gptoss20b"]["estimate"]["est"] == 0.25 == primary["CBUILT-qwen32b"]["estimate"]["est"]
    assert primary["CBUILT-gptoss20b"]["estimate"]["models"] == sorted((*S.READERS_C, QWEN32B))
    assert primary["H_M7"]["estimate"]["verdict"] == "fail" and primary["H_S"]["estimate"]["verdict"] in (
        "holds", "does not hold", "not evaluable")
    assert primary["P0-A"]["k1"] and primary["P0-A"]["estimate"]["est"] == 1.0


def test_ref_b_is_chosen_once_and_held_under_the_sensitivities(paper_runs):
    _, _, _, report = paper_runs
    ref = {p["id"]: p for p in report["primary"]}["REF-B"]
    assert ref["estimate"]["chosen"] == "M2@4000" and ref["estimate"]["est"] == 1.0      # M0@full is 0.75
    # without U2 (same_rule medium) B1's M0@full is 1.0 too: a re-chosen tie would take M0@full; the rule holds M2
    assert ref["sensitivities"]["drop:same_rule_medium"]["chosen"] == "M2@4000"
    assert {s["chosen"] for s in ref["sensitivities"].values() if s} == {"M2@4000"}


def test_outcomes_are_judged_against_ref_b_and_reported_against_the_a_arm(paper_runs):
    _, _, _, report = paper_runs
    got = {p["id"]: p for p in report["primary"]}
    verdict, reported = got[f"OUTCOME-{CONSTRAINED_TIER}"], got[f"OUTCOME-{CONSTRAINED_TIER}-A"]
    assert (verdict["role"], reported["role"]) == ("verdict", "reported")
    # against REF-B (B1 M2@4000) the token clause is B1's full text (1000 / 80000): M2 escapes
    assert verdict["estimate"]["reference"] == [S.READER_B, "M2@4000"] and verdict["estimate"]["outcome"] == "escapable"
    # against the A arm the token clause is A's full text (1000 / 3000 > 1/4): nothing escapes, nothing binds
    assert reported["estimate"]["reference"] == [S.READER_A, "M0@full"]
    assert reported["estimate"]["outcome"] == "indeterminate" and reported["k1"]
    assert report["reference_arm_outcomes"][S.READER_A]["k1"]
    md = render_markdown(report, "paper")
    assert "## Trilemma outcome against the A reference arm (k = 1, reported apart)" in md


def test_k1_is_withdrawn_on_the_4k_slice_alone(paper_runs):
    spec, _, _, report = paper_runs
    assert not any(r.startswith("k = 1") for r in report["status"]["reasons"])
    cov = report["status"]["repeat_coverage"]
    assert cov["cells"] == cov["repeated"] > 0 and cov["conditions"] == spec["repeats"]["k1_scope_conditions"]
    marked = {(c["profile_id"], c["condition"]): c["k1_marked"] for c in report["capability"]}
    assert marked[(S.READER_A, "M0@full")] and marked[(S.READER_B, "M2@16000")]      # read once on the population
    assert marked[(S.READER_B, f"M5@2000/{B1_DIRECT}")] and marked[(QWEN32B, f"M7@16000/{GPTOSS20B}")]
    assert not marked[(S.READER_B, "M2@4000")] and not marked[(QWEN32B, f"M6@4000/{QWEN32B}")]
    assert not marked[(S.READERS_C[0], f"M7@4000/{QWEN32B}")]


def test_absences_k6_and_callout_are_reported(paper_runs):
    _, _, _, report = paper_runs
    ab = report["absences"]
    # every read counts: Llama-70B s3x x3 (the 20% subset), s3y; Llama-8B u2a x3 (the 4k slice); Qwen3-32B u4a x3, s3y
    assert ab["by_reason"] == {"route_context": 3 + 1, "route_timeout": 3 + 3 + 1}
    assert {(r["profile_id"], r["condition"]) for r in ab["by_condition"]} == {
        ("tierB_llama33_70b_hf_ovhcloud", "M0@full"), (S.READERS_C[0], f"M6@4000/{S.READERS_C[0]}"),
        (QWEN32B, f"M6@4000/{QWEN32B}")}
    supply = report["status"]["supply"]
    assert supply["pool"] is not None and supply["population"]["L3"]["units"] == 2
    md = render_markdown(report, "paper")
    for text in ("## Cells not sent", "route_context 4, route_timeout 7.", "## Supply check", "| spec population |"):
        assert text in md, text
    call = {c["id"]: c["sensitivities"] for c in report["sensitivity_callout"]}
    assert "leave_out:PJM" in call["dGap-M5-B"]           # U1 (PJM) out: the M5 loss vanishes


def test_memory_sensitivities_drop_build_truncated_m5_and_unlocated_m7(paper_runs):
    spec, _, _, report = paper_runs
    ms = report["memory_sensitivities"]
    trunc = ms["m5_build_truncated"]
    assert trunc["computed"] and trunc["exploratory"]
    assert (trunc["memories"], trunc["build_truncated_memories"], trunc["build_truncated_memories_read"]) == (6, 1, 1)
    assert trunc["population_items_dropped"] == 2 and set(trunc["answered_cells_dropped"]) == {f"M5@4000/{B1_DIRECT}"}
    assert set(trunc["comparisons"]) == {"dGap-M5-B", f"dGap-M5-{CONSTRAINED_TIER}", "interaction-M5", "H_S"}
    # the M5@4000 loss sits on M-01 (u1a), whose memory is truncated: without those cells the dGap is 0 on U2
    assert trunc["comparisons"]["dGap-M5-B"]["est"] == 0.0 and trunc["comparisons"]["dGap-M5-B"]["n"] == 2
    loc = ms["m7_located"]
    assert loc["computed"] and loc["threshold"] == 0.90
    assert (loc["revisions"], loc["revisions_kept"], loc["population_cells_kept"]) == (2, 1, 2)
    assert loc["below_threshold"] == {B1_DIRECT: 0, GPTOSS20B: 1, QWEN32B: 0}
    # only M-01 (U1) is kept: the gpt-oss-20b-built loss (u2b) leaves, the Qwen3-32B-built one (u1a) doubles
    assert loc["comparisons"]["CBUILT-gptoss20b"]["est"] == 0.0 and loc["comparisons"]["CBUILT-qwen32b"]["est"] == 0.5
    # exploratory: the spec's primary list and its unit-tag sensitivities are untouched
    primary = {p["id"]: p for p in report["primary"]}
    assert len(primary) == 17 and list(primary["dGap-M5-B"]["sensitivities"]) == [x["id"] for x in spec["sensitivities"]]
    md = render_markdown(report, "paper")
    assert "## Memory-level sensitivities (exploratory; not primary" in md


def test_the_readback_refuses_released_build_ledgers(paper_runs):
    """The released (slimmed) build ledgers replace each builder output by its length and sha256: the read-back,
    which reads the outputs, stops with that message instead of reading blank answers."""
    from dataclasses import replace
    _, runs, _, _ = paper_runs
    name = f"build__{B1_DIRECT}"
    slimmed = {h: {**{k: v for k, v in r.items() if k != "text"}, "text_chars": len(r.get("text") or ""),
                   "text_sha256": "0" * 64} for h, r in runs[0].ledgers[name].items()}
    released = replace(runs[0], ledgers={**runs[0].ledgers, name: slimmed})
    with pytest.raises(ValueError, match=r"released \(slimmed\) build ledger"):
        memory_readback([released, *runs[1:]], None)


def test_without_read_back_the_memory_sensitivities_are_listed_not_computed(paper_runs, panel, book, bands):
    spec, runs, _, _ = paper_runs
    report = analyse(runs, ITEMS, panel, book, bands, spec=spec)
    ms = report["memory_sensitivities"]
    assert not ms["m5_build_truncated"]["computed"] and not ms["m7_located"]["computed"]
    assert "--manifest" in ms["m5_build_truncated"]["reason"]


def test_a_thin_tier_is_amber_and_absent_conditions_are_not_computable(tmp_path, panel, book, bands):
    """B2 and Llama-70B do not read the B1-built M7@4000 (dGap-M7-B and interaction-M7 rest on 1 B model: AMBER);
    Qwen3-32B's M6 self-compressions all time out (a condition with no answered cell: the C_load_bearing M6 estimate
    rests on the other 3 readers and names it); without the A arm run, P0-A and the A outcome are not computable."""
    spec = paper_spec(panel)
    m7_b = f"M7@4000/{B1_DIRECT}"
    drop = {("B2_qwen3_235b_2507_hf_deepinfra", m7_b), ("tierB_llama33_70b_hf_ovhcloud", m7_b)}
    runs, _ = write_paper_runs(tmp_path, panel, book, spec, drop=drop,
                      timeout=dict(TIMEOUT, **{QWEN32B: {it["item_id"] for it in ITEMS}}))
    report = analyse(runs[:3], ITEMS, panel, book, bands, spec=spec)          # no A reference arm run
    got = {p["id"]: p for p in report["primary"]}
    assert got["dGap-M7-B"]["small_tiers"] == ["B"] and got["interaction-M7"]["small_tiers"] == ["B"]
    assert got["dGap-M7-B"]["estimate"]["models"] == [S.READER_B]
    assert any(r.startswith("tier estimate on too few models (AMBER): dGap-M7-B: tier B on 1 < 3") for r in
               report["status"]["reasons"])
    assert "2 of 3 readers lack it" in "; ".join(got["dGap-M7-B"]["missing"])
    m6 = got[f"dGap-M6-{CONSTRAINED_TIER}"]
    assert m6["computable"] and m6["estimate"]["models"] == sorted(S.READERS_C)
    assert any(f"{QWEN32B} (not sent (no answered cell))" in m for m in m6["missing"])
    for cid in ("P0-A", f"OUTCOME-{CONSTRAINED_TIER}-A"):
        assert not got[cid]["computable"] and got[cid]["estimate"] is None
        assert any(f"{S.READER_A} M0@full: not read" in m for m in got[cid]["missing"])
    assert any(r.startswith("2 primary comparisons not computable") and "P0-A" in r for r in report["status"]["reasons"])
    md = render_markdown(report, "thin")
    assert "not computable" in md and "### Inputs the analysed runs lack" in md

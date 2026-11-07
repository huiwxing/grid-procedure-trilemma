"""The reader matrix with M6 (query-focused compression) and M7 (typed records) end to end on the real
procedures.memory: the compressor defaults to each reader, per-question cost goes to the query call ledger and the
rows, M7 builds go to the build call ledger, compressions over the compressor's context ceiling are not sent, resume
sends nothing, build_only stops before any compression; condition parsing and the pending-compression count.
Synthetic corpus from tests/memory/test_memory.py; readers answered by the gold-echo mock of procedures.matrix.sanity;
no network."""
from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from items_fixture import make_items
from test_memory import write_corpus
from test_memory_m6_m7 import Endpoint, record
from procedures import ROOT
from procedures.matrix.sanity import MockTransport, offline
from procedures.readers import load_panel
from procedures.readers.ledger import read_ledger
from procedures.memory import BuilderLedger, RevisionIndex, build_memory
from procedures.matrix.run_matrix import Condition, parse_condition, pending_queries, run_matrix

C1, C2 = "C1_llama31_8b_hf_deepinfra", "C2_qwen3_8b_hf_nscale_nothink"
BUILDER = "B1_dsv41flash_hf_deepinfra"
MODAL = re.compile(r"[A-Z][^.\n]*?\b(?:shall|must|will|should|may|required)\b[^.\n]*\.")


def modal_records(part: str) -> list[dict]:
    return [record(subject="the unit", source_span=m[0]) for m in MODAL.finditer(part)]


class Grid:
    """Reader requests to the gold-echo mock; M7 record and M6 compression requests to the test Endpoint. A
    compression opens with the compressor's model name, so each compressor's memory (and reader prompt) differs."""

    def __init__(self, items, readers):
        panel = load_panel()
        self.reader = MockTransport(items, tuple(panel.get(p).user_message_suffix or "" for p in readers))
        self.memory = Endpoint(records_of=modal_records, compression=self.compressed)
        self.kinds: Counter = Counter()
        self.model = ""

    def compressed(self, content: str) -> str:
        passages = content.split("<passages>\n", 1)[1].rsplit("\n</passages>", 1)[0]
        return f"Notes by {self.model}. " + " ".join(x for x in passages.splitlines() if x and not x.startswith("["))

    def __call__(self, url, headers, payload, timeout):
        content = payload["messages"][-1]["content"]
        self.model = payload["model"]
        if "<passages>" in content or "JSON array of at most" in content:
            self.kinds["compress" if "<passages>" in content else "records"] += 1
            return self.memory(url, headers, payload, timeout)
        self.kinds["read"] += 1
        return self.reader(url, headers, payload, timeout)


@pytest.fixture
def index(tmp_path) -> RevisionIndex:
    return RevisionIndex.from_manifest(write_corpus(tmp_path))


def run(tmp_path, index, conditions, readers=(C1, C2), builder=BUILDER, transport=None, **kw):
    with offline():
        return run_matrix(make_items(), [parse_condition(c) for c in conditions], list(readers), index,
                          build_memory=build_memory, panel=load_panel(), label="m67", cap_usd=10.0,
                          corpus_index_sha256="0" * 64, builder_id=builder, out_root=tmp_path / "runs",
                          transport=transport, sleep=lambda s: None, concurrency=1,
                          text_of=lambda p: (tmp_path / p).read_text(encoding="utf-8"), **kw)


def load(report, name):
    text = (Path(report["run_dir"]) / name).read_text(encoding="utf-8")
    return [json.loads(x) for x in text.splitlines()] if name.endswith(".jsonl") else json.loads(text)


CONDS = ["M6@400", f"M6@400/{BUILDER}", "M7@400"]


def test_dry_run_plans_self_and_named_compressors_and_the_m7_builds(tmp_path, index):
    items = make_items()
    report = run(tmp_path, index, CONDS, dry_run=True)
    est = load(report, report["estimate"])
    assert est["queries"]["pending"] == 3 * len(items) == report["queries_pending"]
    assert set(est["queries"]["by_compressor"]) == {C1, C2, BUILDER} and report["query_usd_high"] > 0
    assert est["queries"]["over_ceiling"] == 0
    assert est["builds"]["pending"] == 3 and est["builds"]["escalations_possible"] == 3   # r2, r3, op2: one block each
    manifest = load(report, "manifest.json")
    assert manifest["conditions"][0] == {"representation": "M6", "budget_tokens": 400, "readers": None}
    assert manifest["conditions"][1]["compressor"] == BUILDER and set(manifest["compressors"]) == {C1, C2, BUILDER}
    slots = {k.split("|", 1)[1] for k in manifest["memory_cache_keys"]}
    assert slots == {f"M6@400/{C1}", f"M6@400/{C2}", f"M6@400/{BUILDER}", "M7@400"}
    assert not any(k.endswith(("M6@400/" + C1, "M7@400")) for k in manifest["memory_text_sha256"])
    for pid in (C1, C2):
        assert set(est["readers"][pid]["by_stage_high"]) == {f"read/M6@400/{pid}", f"read/M6@400/{BUILDER}",
                                                             f"read/M7@400/{BUILDER}"}
    assert not list((tmp_path / "runs" / report["run_id"]).glob("*__*.jsonl"))


def test_live_run_compresses_into_query_ledgers_and_rows_carry_the_per_question_cost(tmp_path, index):
    items = make_items()
    transport = Grid(items, (C1, C2))
    report = run(tmp_path, index, CONDS, transport=transport, dry_run=False)
    assert transport.kinds == {"compress": 9, "records": 3, "read": 3 * 3 * 2}
    run_dir = tmp_path / "runs" / report["run_id"]
    for compressor in (C1, C2, BUILDER):
        header, records = read_ledger(run_dir / f"query__{compressor}.jsonl")
        assert header["profile_id"] == compressor and len(records) == 3
        assert {r["stage"] for r in records} == {"query/M6@400"}
    _, builds = read_ledger(run_dir / f"build__{BUILDER}.jsonl")
    assert {r["stage"].split("@")[0] for r in builds} == {"build/M7"} and len(builds) == 3
    rows = load(report, report["outputs"]["rows"])
    assert len(rows) == 3 * 3 * 2 and all(r["capability"]["correct"] for r in rows)
    for r in rows:
        if r["representation"] == "M6":
            assert r["builder_profile_id"] in (r["profile_id"], BUILDER) and r["stage"].endswith(r["builder_profile_id"])
            assert r["sustainability"]["query_memory_tokens"] > 0 and r["access"]["query_memory_usd"] > 0
            assert r["sustainability"]["build_tokens"] == 0
        else:
            assert r["sustainability"]["build_tokens"] > 0 and r["sustainability"]["query_memory_tokens"] == 0
            assert r["access"]["query_memory_usd"] is None
    summary = load(report, report["outputs"]["summary"])["conditions"]
    assert summary[f"{C1}|read/M6@400/{C1}"]["query_memory_usd"] > 0
    assert summary[f"{C1}|read/M7@400/{BUILDER}"]["query_memory_tokens"] == 0
    # the reader saw the memory the query ledger holds (rebuilt from it: nothing is sent)
    item = next(it for it in items if it["item_id"] == "p1:value:new")
    with offline():
        mem = build_memory(item, "M6", 400, index, compressor_profile=C1,
                           ledger=BuilderLedger(run_dir / f"query__{C1}.jsonl", cap_usd=10.0, transport=transport))
        m7 = build_memory(item, "M7", 400, index, builder_profile=BUILDER,
                          ledger=BuilderLedger(run_dir / f"build__{BUILDER}.jsonl", cap_usd=10.0, transport=transport))
    assert mem.build["mode"] == "live" and "ramp limit" in mem.text
    assert mem.text.splitlines()[1].startswith("Notes by meta-llama/")
    assert m7.text.endswith("- subject: the unit | source_span: The ramp limit for the Test unit shall be 30 MW per "
                            "minute.")
    assert transport.kinds["compress"] == 9 and transport.kinds["records"] == 3
    _, reads = read_ledger(run_dir / f"read__{C1}.jsonl")
    by_stage = {(r["item_id"], r["stage"]): r for r in reads}
    assert mem.text in by_stage[(item["item_id"], f"read/M6@400/{C1}")]["request"]["messages"][-1]["content"]
    assert m7.text in by_stage[(item["item_id"], f"read/M7@400/{BUILDER}")]["request"]["messages"][-1]["content"]
    # resume: nothing is sent again
    again = Grid(items, (C1, C2))
    resumed = run(tmp_path, index, CONDS, transport=again, dry_run=False)
    assert resumed["run_id"] == report["run_id"] and again.kinds == Counter()


def test_a_compression_over_the_compressor_ceiling_is_not_sent_and_its_cells_are_infeasible(tmp_path, index):
    items = make_items()
    transport = Grid(items, (C1,))
    dry = run(tmp_path, index, ["M6@22000"], readers=(C1,), builder=None, dry_run=True)
    est = load(dry, dry["estimate"])
    assert est["queries"] == {"pending": 0, "usd_high": 0, "by_compressor": {}, "over_ceiling": 3}
    assert est["readers"][C1]["infeasible_by_stage"] == {f"read/M6@22000/{C1}": 3}
    report = run(tmp_path, index, ["M6@22000", "M6@400"], readers=(C1,), builder=None, transport=transport,
                       dry_run=False)
    assert transport.kinds == {"compress": 3, "read": 3}                      # the M6@400 ones only
    rows = load(report, report["outputs"]["rows"])
    over = [r for r in rows if r["stage"] == f"read/M6@22000/{C1}"]
    assert len(over) == 3 and all(r["outcome"] == "not_sent_infeasible" and r["access"]["feasible_at_B"] is False
                                  for r in over)
    _, records = read_ledger(tmp_path / "runs" / report["run_id"] / f"query__{C1}.jsonl")
    assert {r["stage"] for r in records} == {"query/M6@400"}


def test_build_only_stops_before_any_compression(tmp_path, index):
    items = make_items()
    transport = Grid(items, (C1, C2))
    report = run(tmp_path, index, CONDS, transport=transport, dry_run=False, build_only=True)
    assert report["mode"] == "build_only" and transport.kinds == {"records": 3}
    assert not list((tmp_path / "runs" / report["run_id"]).glob("query__*.jsonl"))


def test_conditions_name_their_compressor_and_builders_stay_with_m5_m7(tmp_path, index):
    assert parse_condition(f"M6@4000/{BUILDER}:{C1},{C2}") == Condition("M6", 4000, (C1, C2), BUILDER)
    c = parse_condition("M6@4000")
    assert (c.key, c.memory_slot(C1), c.identity()) == (
        "M6@4000", f"M6@4000/{C1}", {"representation": "M6", "budget_tokens": 4000, "readers": None})
    named = Condition("M6", 4000, compressor=BUILDER)
    assert (named.key, named.memory_slot(C1), named.identity()["compressor"]) == (
        f"M6@4000/{BUILDER}", f"M6@4000/{BUILDER}", BUILDER)
    assert Condition("M2", 4000).identity() == {"representation": "M2", "budget_tokens": 4000, "readers": None}
    with pytest.raises(ValueError, match="take a compressor"):
        Condition("M2", 4000, compressor=BUILDER)
    with pytest.raises(ValueError, match="same compressor"):
        run(tmp_path, index, ["M6@400", f"M6@400/{C1}"], readers=(C1,), builder=None, dry_run=True)
    with pytest.raises(ValueError, match="builder profile"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=BUILDER, dry_run=True)
    with pytest.raises(ValueError, match="build_only needs"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, dry_run=False, build_only=True)


def test_pending_queries_count_each_distinct_unanswered_request_once():
    def m6(key, compressor, mode="dry_run", usd=0.01):
        return SimpleNamespace(representation="M6", build={"mode": mode, "compressor_profile": compressor,
                                                           "request_hash": key, "usd": usd})
    memories = {("a", "s1"): m6("h1", C1), ("b", "s1"): m6("h1", C1), ("c", "s1"): m6("h2", C1),
                ("d", "s2"): m6("h3", C2), ("e", "s2"): m6("h4", C2, "no_input"), ("f", "s2"): m6("h5", C2, "live"),
                ("g", "s3"): SimpleNamespace(representation="M5", build={"mode": "dry_run"})}
    assert pending_queries(memories, {}) == {C1: {"h1": 0.01, "h2": 0.01}, C2: {"h3": 0.01}}
    assert pending_queries(memories, {C1: {"h2"}}, skip={("d", "s2")}) == {C1: {"h1": 0.01}}


class TimedGrid(Grid):
    """Grid that also records (wire model, request kind, read timeout) of every request."""

    def __init__(self, items, readers):
        super().__init__(items, readers)
        self.seen = []

    def __call__(self, url, headers, payload, timeout):
        content = payload["messages"][-1]["content"]
        kind = "compress" if "<passages>" in content else "records" if "JSON array of at most" in content else "read"
        self.seen.append((payload["model"], kind, timeout))
        return super().__call__(url, headers, payload, timeout)


def test_timeout_overrides_reach_reads_compressions_and_builds_and_leave_the_run_identity(tmp_path, index):
    """Transport only: every live stage of an overridden sender reads with the override, the others with the panel's
    120 s; run id, manifest bytes and ledger headers equal those of the same run without overrides."""
    items, panel = make_items(), load_panel()
    wire = {pid: panel.get(pid).client_profile().request_model for pid in (C1, C2, BUILDER)}
    runs = {}
    for name, overrides in (("plain", None), ("over", {C1: 360, BUILDER: 400})):
        transport = TimedGrid(items, (C1, C2))
        with offline():
            report = run_matrix(items, [parse_condition(c) for c in CONDS], [C1, C2], index,
                                build_memory=build_memory, panel=panel, label="m67", cap_usd=10.0,
                                corpus_index_sha256="0" * 64, builder_id=BUILDER, out_root=tmp_path / name,
                                transport=transport, sleep=lambda s: None, concurrency=1, dry_run=False,
                                text_of=lambda p: (tmp_path / p).read_text(encoding="utf-8"),
                                timeout_overrides=overrides)
        runs[name] = (report, transport, tmp_path / name / report["run_id"])
    (plain, _, plain_dir), (over, transport, over_dir) = runs["plain"], runs["over"]
    expected = {wire[C1]: 360.0, wire[BUILDER]: 400.0, wire[C2]: 120}
    assert {(k, t) for _, k, t in transport.seen} >= {("read", 360.0), ("compress", 360.0), ("records", 400.0),
                                                       ("compress", 400.0), ("read", 120), ("compress", 120)}
    assert all(t == expected[m] for m, _, t in transport.seen)
    assert over["run_id"] == plain["run_id"]
    assert (over_dir / "manifest.json").read_bytes() == (plain_dir / "manifest.json").read_bytes()
    ledgers = sorted(p.name for p in plain_dir.glob("*__*.jsonl") if "sends" not in p.name)
    assert ledgers == sorted(p.name for p in over_dir.glob("*__*.jsonl") if "sends" not in p.name) and ledgers
    for name in ledgers:
        heads = [json.loads((d / name).read_bytes().split(b"\n", 1)[0]) for d in (plain_dir, over_dir)]
        for head in heads:
            head.pop("created_at")
        assert heads[0] == heads[1], name
    invocations = [json.loads(x) for x in (over_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]
    assert invocations[-1]["outcome"] == "live"
    assert invocations[-1]["timeout_overrides"] == {BUILDER: 400.0, C1: 360.0}
    assert "timeout_overrides" not in json.loads((plain_dir / "invocations.jsonl").read_text().splitlines()[-1])


def test_timeout_overrides_must_name_a_sender_of_the_run_and_raise_its_timeout(tmp_path, index):
    with pytest.raises(ValueError, match="does not send"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, timeout_overrides={C2: 360})
    with pytest.raises(ValueError, match="only raises"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, timeout_overrides={C1: 120})
    assert not (tmp_path / "runs").exists()      # refused before the run directory exists


def test_cli_timeout_override_is_repeatable_and_checked(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_matrix_cli", ROOT / "scripts" / "run_matrix.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = {}

    def fake_run_matrix(*args, **kw):
        seen.update(kw)
        return {"run_id": "x"}

    monkeypatch.setattr(cli, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(cli.RevisionIndex, "from_manifest", staticmethod(lambda *a, **k: None))
    items, manifest, external = (tmp_path / n for n in ("items.jsonl", "manifest.csv", "external.csv"))
    items.write_text(json.dumps({"item_id": "i1", "doc": "d1"}) + "\n", encoding="utf-8")
    manifest.write_text("doc\n", encoding="utf-8")
    external.write_text("doc\n", encoding="utf-8")
    base = ["--label", "t", "--grid", "main", "--items", str(items), "--manifest", str(manifest),
            "--external", str(external)]
    assert cli.main(base + ["--timeout-override", f"{C1}=360", "--timeout-override", f"{C2}=240.5"]) == 0
    assert seen["timeout_overrides"] == {C1: 360.0, C2: 240.5}
    assert cli.main(base) == 0 and seen["timeout_overrides"] == {}
    for bad in ([f"{C1}=120"], [f"{C1}=nope"], [C1], [f"{C1}=360", "--timeout-override", f"{C1}=400"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(base + ["--timeout-override", *bad])
        assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "only raises" in err and "not a number" in err and "names a profile twice" in err


LLAMA = "tierB_llama33_70b_hf_ovhcloud"     # panel max_workers 1 (its route takes one request at a time)


def test_workers_overrides_reach_the_in_flight_limit_and_leave_the_run_identity(tmp_path, index, monkeypatch):
    """Transport only: the overridden sender's reads and compressions go out with the override as their in-flight
    limit, the other sender with its panel max_workers; run id, manifest bytes and ledger headers equal those of the
    same run without overrides, and only the overridden invocation records it."""
    import procedures.readers.executor as infer_run
    items, panel = make_items(), load_panel()
    wire = {pid: panel.get(pid).client_profile().request_model for pid in (LLAMA, C1)}
    limits = []
    real_workers = infer_run.workers

    def spy(concurrency, client_profile):
        n = real_workers(concurrency, client_profile)
        limits.append((client_profile.request_model, n))
        return n

    monkeypatch.setattr(infer_run, "workers", spy)
    runs = {}
    for name, overrides in (("plain", None), ("over", {LLAMA: 4})):
        limits.clear()
        with offline():
            report = run_matrix(items, [parse_condition("M6@400")], [LLAMA, C1], index, build_memory=build_memory,
                                panel=panel, label="m67", cap_usd=10.0, corpus_index_sha256="0" * 64,
                                out_root=tmp_path / name, transport=Grid(items, (LLAMA, C1)), sleep=lambda s: None,
                                dry_run=False, text_of=lambda p: (tmp_path / p).read_text(encoding="utf-8"),
                                workers_overrides=overrides)
        runs[name] = (report, set(limits), tmp_path / name / report["run_id"])
    (plain, plain_limits, plain_dir), (over, over_limits, over_dir) = runs["plain"], runs["over"]
    assert plain_limits == {(wire[LLAMA], 1), (wire[C1], 4)}
    assert over_limits == {(wire[LLAMA], 4), (wire[C1], 4)}
    assert over["run_id"] == plain["run_id"]
    assert (over_dir / "manifest.json").read_bytes() == (plain_dir / "manifest.json").read_bytes()
    ledgers = sorted(p.name for p in plain_dir.glob("*__*.jsonl") if "sends" not in p.name)
    assert ledgers == sorted(p.name for p in over_dir.glob("*__*.jsonl") if "sends" not in p.name) and ledgers
    for name in ledgers:
        heads = [json.loads((d / name).read_bytes().split(b"\n", 1)[0]) for d in (plain_dir, over_dir)]
        for head in heads:
            head.pop("created_at")
        assert heads[0] == heads[1], name
    last = json.loads((over_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert last["outcome"] == "live" and last["workers_overrides"] == {LLAMA: 4}
    assert "workers_overrides" not in json.loads((plain_dir / "invocations.jsonl").read_text().splitlines()[-1])


def test_workers_overrides_must_name_a_sender_of_the_run_and_raise_its_limit(tmp_path, index):
    with pytest.raises(ValueError, match="does not send"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, workers_overrides={LLAMA: 4})
    with pytest.raises(ValueError, match="only raises"):      # C1's panel max_workers is 4
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, workers_overrides={C1: 2})
    with pytest.raises(ValueError, match="only raises"):
        run(tmp_path, index, ["M6@400"], readers=(LLAMA,), builder=None, workers_overrides={LLAMA: 9})
    assert not (tmp_path / "runs").exists()      # refused before the run directory exists


def test_cli_workers_override_is_repeatable_and_checked(tmp_path, monkeypatch, capsys):
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_matrix_cli", ROOT / "scripts" / "run_matrix.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    seen = {}

    def fake_run_matrix(*args, **kw):
        seen.update(kw)
        return {"run_id": "x"}

    monkeypatch.setattr(cli, "run_matrix", fake_run_matrix)
    monkeypatch.setattr(cli.RevisionIndex, "from_manifest", staticmethod(lambda *a, **k: None))
    items, manifest, external = (tmp_path / n for n in ("items.jsonl", "manifest.csv", "external.csv"))
    items.write_text(json.dumps({"item_id": "i1", "doc": "d1"}) + "\n", encoding="utf-8")
    manifest.write_text("doc\n", encoding="utf-8")
    external.write_text("doc\n", encoding="utf-8")
    base = ["--label", "t", "--grid", "main", "--items", str(items), "--manifest", str(manifest),
            "--external", str(external)]
    assert cli.main(base + ["--workers-override", f"{LLAMA}=4", "--workers-override", f"{C1}=8"]) == 0
    assert seen["workers_overrides"] == {LLAMA: 4, C1: 8}
    assert cli.main(base) == 0 and seen["workers_overrides"] == {}
    for bad in ([f"{LLAMA}=9"], [f"{LLAMA}=0"], [f"{LLAMA}=two"], [LLAMA], [f"{LLAMA}=2", "--workers-override",
                                                                            f"{LLAMA}=4"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(base + ["--workers-override", *bad])
        assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "only raises" in err and "not an int" in err and "names a profile twice" in err


class GatewayCut(Grid):
    """Grid whose compressions by the wire model `model` of the questions in `cut` get a 504 on every attempt (the
    HF router cutting a non-streamed answer at ~121 s), or `status` instead; `cut_calls` counts them."""

    def __init__(self, items, readers, model, cut, status=504):
        super().__init__(items, readers)
        self.cut_model, self.cut, self.cut_calls, self.status = model, set(cut), 0, status

    def __call__(self, url, headers, payload, timeout):
        content = payload["messages"][-1]["content"]
        if "<passages>" in content and payload["model"] == self.cut_model and any(q in content for q in self.cut):
            self.cut_calls += 1
            return self.status, {}, b'{"error": "Gateway Time-out"}'
        return super().__call__(url, headers, payload, timeout)


def test_a_compression_that_times_out_is_route_timeout_and_its_reader_cells_are_not_sent(tmp_path, index):
    """A compression whose every attempt fails at unknown cost is a route_timeout memory; its reader cell is recorded,
    not sent, reason route_timeout, counted in the estimate, the summary and the access row (and the analysis layer's
    absences); a rerun does not send it again."""
    from procedures.analysis.costs import not_sent_by_reason
    from procedures.analysis.data import load_run, observations
    from procedures.matrix.sanity import check_run
    items, panel = make_items(), load_panel()
    cut = next(it for it in items if it["item_id"] == "p1:value:new")
    transport = GatewayCut(items, (C1,), panel.get(C1).client_profile().request_model, {cut["question"]})
    report = run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, transport=transport, dry_run=False)
    run_dir, stage = Path(report["run_dir"]), f"read/M6@400/{C1}"
    assert transport.cut_calls == 3 and transport.kinds == {"compress": 2, "read": 2}
    timed = [r for r in read_ledger(run_dir / f"query__{C1}.jsonl")[1] if r["outcome"] != "observed"]
    assert [(r["item_id"], [a["http_status"] for a in r["attempts"]]) for r in timed] == [(cut["item_id"], [504] * 3)]
    assert cut["item_id"] not in {r["item_id"] for r in read_ledger(run_dir / f"read__{C1}.jsonl")[1]}
    # 1. the estimate: the invocation names the one re-planned after the compressions
    est = load(report, report["estimate"])
    assert est["readers"][C1]["infeasible_by_reason"] == {"route_timeout": 1} and est["readers"][C1]["pending"] == 2
    assert est["queries"]["route_timeout"] == {C1: 1}
    planned = load(report, est["after_compressions"]["planned_estimate"])
    assert planned["readers"][C1]["infeasible_by_reason"] == {} and "route_timeout" not in planned["queries"]
    last = load(report, "invocations.jsonl")[-1]
    assert (last["outcome"], last["estimate"], last["route_timeout"]) == ("live", report["estimate"], {C1: 1})
    # 2. the summary and 3. the access row
    summary = load(report, report["outputs"]["summary"])["conditions"][f"{C1}|{stage}"]
    assert summary["infeasible_by_reason"] == {"route_timeout": 1} and summary["memory_empty"] == 0
    assert summary["outcomes"] == {"not_sent_infeasible": 1, "observed": 2}
    rows = load(report, report["outputs"]["rows"])
    (row,) = [r for r in rows if r["item_id"] == cut["item_id"]]
    assert (row["outcome"], row["access"]["infeasible_reason"], row["access"]["feasible_at_B"]) == (
        "not_sent_infeasible", "route_timeout", False) and row["provenance"]["request_hash"] is None
    assert sum(r["access"]["infeasible_reason"] == "route_timeout" for r in rows) == 1
    # the analysis layer lists it as an absence; the mechanism sanity checks hold (the estimate the invocation names
    # agrees)
    assert not_sent_by_reason(observations([load_run(run_dir)], items, panel)) == {"route_timeout": 1}
    with offline():
        assert check_run(run_dir, items, index, build_memory, report["outputs"]["summary"], mock=False)[
            "failures"] == []
    # the released form of the ledgers (the prompt messages dropped) is refused with a clear message
    released = tmp_path / "released" / run_dir.name
    shutil.copytree(run_dir, released)
    ledger = released / f"read__{C1}.jsonl"
    header, *calls = [json.loads(x) for x in ledger.read_text(encoding="utf-8").splitlines()]
    calls = [{**r, "request": {k: v for k, v in r["request"].items() if k != "messages"}} for r in calls]
    ledger.write_bytes("".join(json.dumps(x) + "\n" for x in (header, *calls)).encode("utf-8"))
    with pytest.raises(ValueError, match=r"released \(slimmed\) call ledgers without prompts: read__"):
        check_run(released, items, index, build_memory, None, mock=False)
    # a rerun: the timed-out compression is settled (not sent again), the planning estimate counts it
    again = Grid(items, (C1,))
    resumed = run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, transport=again, dry_run=False)
    assert again.kinds == Counter() and resumed["route_timeout"] == {C1: 1}
    est = load(resumed, resumed["estimate"])
    assert est["readers"][C1]["infeasible_by_reason"] == {"route_timeout": 1} and "after_compressions" not in est
    assert load(resumed, resumed["outputs"]["rows"]) == rows


def test_a_compression_that_fails_without_timing_out_is_not_route_timeout_and_is_sent_again(tmp_path, index):
    """Narrowed rule: a compression whose three attempts get HTTP 500 (unknown cost, but not a timeout) stays
    no_response, not route_timeout: the invocation stops, as every unanswered memory did before the route-timeout
    rule (no cell is read on a memory that did not execute: the mechanism sanity check), planning prices it as
    pending and plans its reader cell as usual, and a rerun sends it again."""
    from procedures.readers.executor import route_timed_out
    items, panel = make_items(), load_panel()
    cut = next(it for it in items if it["item_id"] == "p1:value:new")
    transport = GatewayCut(items, (C1,), panel.get(C1).client_profile().request_model, {cut["question"]}, status=500)
    with pytest.raises(RuntimeError, match=r"got no answer \(no_response\)"):
        run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, transport=transport, dry_run=False)
    assert transport.cut_calls == 3
    (run_dir,) = (tmp_path / "runs").iterdir()
    failed = [r for r in read_ledger(run_dir / f"query__{C1}.jsonl")[1] if r["outcome"] != "observed"]
    assert [(r["item_id"], r["outcome"], [a["http_status"] for a in r["attempts"]]) for r in failed] == [
        (cut["item_id"], "no_response", [500] * 3)]
    assert not route_timed_out(failed[0])
    # planning: the compression is pending again and its reader cell planned as usual (nothing route_timeout)
    planned = run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, dry_run=True)
    est = load(planned, planned["estimate"])
    assert est["readers"][C1]["infeasible_by_reason"] == {} and est["readers"][C1]["pending"] == 3
    assert "route_timeout" not in est["queries"] and "route_timeout" not in planned
    assert planned["queries_pending"] == est["queries"]["pending"] >= 1
    # a rerun sends it again, and every reader cell is observed
    again = Grid(items, (C1,))
    report = run(tmp_path, index, ["M6@400"], readers=(C1,), builder=None, transport=again, dry_run=False)
    assert again.kinds["compress"] == planned["queries_pending"] and again.kinds["read"] == 3
    assert "route_timeout" not in report
    rows = load(report, report["outputs"]["rows"])
    assert [(r["outcome"], r["access"]["infeasible_reason"]) for r in rows] == [("observed", None)] * 3
    latest = [r for r in read_ledger(run_dir / f"query__{C1}.jsonl")[1] if r["item_id"] == cut["item_id"]]
    assert [r["outcome"] for r in latest] == ["no_response", "observed"]


def test_a_closed_compressor_route_does_not_stop_the_run(tmp_path, index, monkeypatch):
    """Three compressions in a row time out, the compressor's route closes (RouteTripped): its unsent compressions
    become route_timeout for this invocation, the other compressor and the readers go on, and the trip is reported at
    the end. A rerun sends the unsent ones; the timed-out ones stay route_timeout."""
    import procedures.readers.executor as infer_run
    from procedures.readers.executor import RouteTripped
    monkeypatch.setattr(infer_run, "EARLY_SENDS", 0)      # three items only: keep the first-20 window out of the way
    monkeypatch.setattr(infer_run, "ROUTE_TIMEOUT_STREAK", 3)   # the scenario: three cut requests close the route
    # (504s are timeout-class, with a streak of six; this test is about compress_live's handling of a closed
    # compressor route, not the threshold value)
    items, panel = make_items(), load_panel()
    conds = ["M6@400", "M6@800"]
    wire = panel.get(C1).client_profile().request_model
    transport = GatewayCut(items, (C1, C2), wire, {it["question"] for it in items})
    with pytest.raises(RouteTripped, match=f"compressions by {C1}"):
        run(tmp_path, index, conds, builder=None, transport=transport, dry_run=False)
    assert transport.cut_calls == 9                         # three compressions x three attempts, then closed
    assert transport.kinds == {"compress": 6, "read": 3}     # C2 compressed and read everything (400 = 800 here)
    (run_dir,) = (tmp_path / "runs").iterdir()
    _, queries = read_ledger(run_dir / f"query__{C1}.jsonl")
    assert [(r["stage"], r["outcome"]) for r in queries] == [("query/M6@400", "no_response")] * 3
    stopped = json.loads((run_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert stopped["outcome"] == "stopped" and stopped["route_timeout"] == {C1: 6}
    assert stopped["sent"][C1]["sent"] == 0 and stopped["sent"][C2]["sent"] == 3
    assert list(stopped["breaker"]["tripped"]) == [f"huggingface:{wire}"]
    after = json.loads((run_dir / stopped["estimate"]).read_text(encoding="utf-8"))
    assert after["readers"][C1]["infeasible_by_reason"] == {"route_timeout": 6}
    again = Grid(items, (C1, C2))
    report = run(tmp_path, index, conds, builder=None, transport=again, dry_run=False)
    assert again.kinds == {"compress": 3, "read": 3} and report["route_timeout"] == {C1: 3}
    rows = load(report, report["outputs"]["rows"])
    got = Counter((r["profile_id"], r["stage"].split("/")[1], r["outcome"], r["access"]["infeasible_reason"])
                  for r in rows)
    assert got == {(C1, "M6@400", "not_sent_infeasible", "route_timeout"): 3, (C1, "M6@800", "observed", None): 3,
                   (C2, "M6@400", "observed", None): 3, (C2, "M6@800", "observed", None): 3}

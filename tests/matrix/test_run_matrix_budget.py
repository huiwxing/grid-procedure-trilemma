"""The matrix runner's budget options: the build-stage cap of a --build-only run, the approval (the spend grant for
one batch of paid runs) every paid run needs from the CLI (arms, grant, binding, cross-run tally), and the circuit
breaker closing one reader's route while the others go on. Offline: mocked transport, fake memory builder."""
import importlib.util
import json
from pathlib import Path

import pytest

from test_run_matrix import CONDITIONS, READERS, run
from procedures.readers import SpendCapExceeded
from procedures.readers.budget import ApprovalError, ApprovalExceeded, ApprovalGuard, load_approval, write_approval
from procedures.readers.ledger import Ledger
from procedures.readers.executor import RouteTripped

ROOT = Path(__file__).resolve().parents[2]
HEADER = {"profile_id": "p", "profile_fingerprint": "f" * 64, "price_book_sha256": "b" * 64}


def invocations(run_dir):
    return [json.loads(x) for x in (run_dir / "invocations.jsonl").read_text(encoding="utf-8").splitlines()]


def approval(tmp_path, granted, arms, approval_id="t1"):
    write_approval(approval_id, granted, arms, "test grant", root=tmp_path / "budget")
    return load_approval(approval_id, root=tmp_path / "budget")


def test_stage_cap_checks_and_caps_the_build_stage_only(items, panel, book, tmp_path, fake_builder, fake_reader,
                                                        gold_answer):
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True)
    builds = dry["build_usd_high"] + dry["build_usd_high_possible"]
    assert builds == pytest.approx(0.04) and dry["pending_usd_high"] > 0.01
    run_dir = tmp_path / dry["run_id"]
    with pytest.raises(ValueError, match="stage_cap_usd"):
        run(items, panel, book, tmp_path, fake_builder(), None, dry_run=False, stage_cap_usd=1.0)
    builder = fake_builder()
    with pytest.raises(SpendCapExceeded, match="exceeds the stage cap .*nothing was built or sent"):
        run(items, panel, book, tmp_path, builder, None, dry_run=False, build_only=True, stage_cap_usd=builds - 0.001)
    assert all(c["ledger"] is None for c in builder.calls)
    # a cap below the full run's pre-flight: the build stage alone is checked, and its ledger capped at the stage cap
    cap = builds + 0.001
    assert dry["pending_usd_high"] + builds > cap
    builder = fake_builder()
    built = run(items, panel, book, tmp_path, builder, None, dry_run=False, build_only=True, cap_usd=cap,
                stage_cap_usd=builds + 0.0005)
    assert built["mode"] == "build_only"
    caps = {c["ledger"].cap_usd for c in builder.calls if c["ledger"] is not None}
    assert len(caps) == 1 and caps.pop() == pytest.approx(builds + 0.0005)
    reader = fake_reader(items, gold_answer)
    with pytest.raises(SpendCapExceeded, match="pending reads"):     # the reader stage re-checks the whole run
        run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, cap_usd=cap)
    assert reader.calls == []
    log = [(x["outcome"], x.get("stage_cap_usd")) for x in invocations(run_dir)]
    assert log == [("dry_run", None), ("refused", builds - 0.001), ("built", builds + 0.0005), ("refused", None)]


def test_an_approved_run_binds_to_its_approval_and_counts_against_it(items, panel, book, tmp_path, fake_builder,
                                                                     fake_reader, gold_answer):
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True)
    need = dry["pending_usd_high"] + dry["build_usd_high"] + dry["build_usd_high_possible"]
    ap = approval(tmp_path, granted=2 * need, arms={"t": need + 0.01, "other": need - 0.02})
    with pytest.raises(ApprovalError, match="no arm for run label 'u'"):
        run(items, panel, book, tmp_path, fake_builder(), None, label="u", approval=ap)
    with pytest.raises(ValueError, match="exceeds arm 't'"):
        run(items, panel, book, tmp_path, fake_builder(), None, cap_usd=need + 0.02, approval=ap)
    priced = run(items, panel, book, tmp_path, fake_builder(), None, cap_usd=need + 0.01, approval=ap)
    assert priced["approval"]["preflight_usd"] == pytest.approx(need) and priced["approval"]["stage"] == "full"
    assert priced["approval"]["committed_usd"] == 0 and not ap.bindings_path.exists()     # a dry run binds nothing
    reader = fake_reader(items, gold_answer)
    live = run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, cap_usd=need + 0.01, approval=ap)
    run_dir = tmp_path / live["run_id"]
    assert json.loads((run_dir / "approvals.jsonl").read_text(encoding="utf-8"))["approval_id"] == "t1"
    assert json.loads(ap.bindings_path.read_text(encoding="utf-8"))["run_label"] == "t"
    assert live["approval"]["committed_usd"] == pytest.approx(live["spent_usd"]) and live["spent_usd"] > 0
    assert live["approval"]["in_flight_usd"] == 0
    assert [x.get("approval_id") for x in invocations(run_dir)] == [None, "t1", "t1"]   # dry, dry with it, live


def test_spend_of_another_run_under_the_approval_refuses_the_preflight(items, panel, book, tmp_path, fake_builder,
                                                                       fake_reader, gold_answer):
    dry = run(items, panel, book, tmp_path, fake_builder(), None, dry_run=True)
    need = dry["pending_usd_high"] + dry["build_usd_high"] + dry["build_usd_high_possible"]
    ap = approval(tmp_path, granted=need + 0.01, arms={"t": need + 0.01})
    other = tmp_path / "t_000000000000"          # an earlier run of the same arm, bound and with spend
    other.mkdir()
    ApprovalGuard(ap, other, "t").bind()
    Ledger(other / "read__x.jsonl", HEADER, cap_usd=1.0).append(
        {"kind": "call", "request_hash": "h", "outcome": "no_response", "usd_counted": 0.02,
         "usd_pending_unknown": 0.02})
    reader = fake_reader(items, gold_answer)
    with pytest.raises(ApprovalExceeded, match="pending unknown \\$0.0200|arm t"):
        run(items, panel, book, tmp_path, fake_builder(), reader, dry_run=False, cap_usd=need + 0.01, approval=ap)
    assert reader.calls == []
    run_dir = tmp_path / dry["run_id"]
    assert [x["outcome"] for x in invocations(run_dir)] == ["dry_run", "refused"]
    assert not (run_dir / "approvals.jsonl").exists()


def test_a_tripped_reader_route_stops_only_that_reader_and_then_the_run(items, panel, book, tmp_path, fake_builder,
                                                                        fake_reader, gold_answer):
    good = fake_reader(items, gold_answer)
    calls = []

    def transport(url, headers, payload, timeout):
        calls.append(payload["model"])
        c1 = [m for m in calls if m.startswith("meta-llama")]
        if payload["model"].startswith("meta-llama") and len(c1) > 20:   # C1's route: 20 answers, then 503s
            return 503, {}, b'{"error": "busy"}'
        return good(url, headers, payload, timeout)

    conditions = CONDITIONS[:2]
    with pytest.raises(RouteTripped, match="routes closed by the circuit breaker in this run: .*C1_llama31"):
        run(items, panel, book, tmp_path, fake_builder(), transport, dry_run=False, conditions=conditions,
            builder_id=None)
    assert len(items) * len(conditions) > 23
    # three requests of three 503s each (the breaker counts requests), then nothing more there
    assert len([m for m in calls if m.startswith("meta-llama")]) == 20 + 3 * 3
    assert len([m for m in calls if not m.startswith("meta-llama")]) == len(items) * len(conditions)   # C2 went on
    run_dir = next(d for d in tmp_path.iterdir() if d.name.startswith("t_"))
    stopped = invocations(run_dir)[-1]
    assert stopped["outcome"] == "stopped" and list(stopped["sent"]) == [READERS[1]]
    assert list(stopped["breaker"]["tripped"]) == ["huggingface:meta-llama/Llama-3.1-8B-Instruct:deepinfra"]


def test_cli_execute_requires_an_approval_and_an_arm(tmp_path, capsys):
    spec = importlib.util.spec_from_file_location("run_matrix_cli", ROOT / "scripts" / "run_matrix.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    manifest = ["--manifest", str(ROOT / "data" / "corpus" / "manifest.csv")]
    write_approval("t1", 1.0, {"records": 1.0}, "test", root=tmp_path)
    for argv, message in ((["--execute", *manifest], "--execute needs --approval"),
                          (["--stage-cap", "5"], "--stage-cap caps the build stage"),
                          (["--approval", "t1", "--approvals-root", str(tmp_path)], "no arm for label 'main'")):
        with pytest.raises(SystemExit) as exc:
            cli.main(["--label", "main", "--out-root", str(tmp_path / "runs"), *argv])
        assert exc.value.code == 2 and message in capsys.readouterr().err
    assert not (tmp_path / "runs").exists()

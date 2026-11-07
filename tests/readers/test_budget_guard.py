"""Budget safety of the paid executor: every attempt reserved up front, unknown-cost failures counted as
pending_unknown, the circuit breaker, the cross-process call-ledger and approval locks, and recovery of what a dead
executor left in flight; all offline (conftest blocks the network)."""
import json
import threading

import pytest

from test_executor import msg, openai_body
from procedures.readers import Cell, SpendCapExceeded, run_cells, upper_bound_usd
from procedures.readers.budget import ApprovalError, ApprovalExceeded, ApprovalGuard, SEMANTICS, load_approval, write_approval
from procedures.readers.ledger import FileLock, Ledger, LockBusy, read_ledger, sends_path
from procedures.readers.executor import RouteTripped, RunPaused, SpendGuard, attempt_cost, ledger_header, record_costs, request_cost, route_timed_out, timeout_class

PID = "C1_llama31_8b_hf_deepinfra"
ROUTE = "huggingface:meta-llama/Llama-3.1-8B-Instruct:deepinfra"
HEADER = {"profile_id": "p", "profile_fingerprint": "f" * 64, "price_book_sha256": "b" * 64}


def cells(n, prefix="q"):
    return [Cell(f"{prefix}{k}", "s", msg(f"{prefix}{k}")) for k in range(n)]


def sends(path):
    return [json.loads(x) for x in sends_path(path).read_text(encoding="utf-8").splitlines()]


def test_attempt_cost_classes():
    assert [attempt_cost(s) for s in (200, 204, 408, 500, 504, 599, None, 429, 400, 401, 425)] == \
        ["answered"] * 2 + ["unknown"] * 5 + ["rate_limited", "refused", "refused", "refused"]


def test_narrow_cap_and_three_timeouts_never_exceed_the_cap(panel, book, tmp_path, fake_transport):
    """Every attempt the client may make is reserved before the first is sent: a cap that covers one attempt of a
    three-attempt client sends nothing, and a three-attempt cap holds whatever happens."""
    pp = panel.get(PID)
    (cell,) = cells(1)
    bound = upper_bound_usd(cell, pp, book)
    narrow = fake_transport([TimeoutError("t")] * 3)
    with pytest.raises(SpendCapExceeded, match="stopped after 0 of 1"):
        run_cells([cell], pp, cap_usd=bound * 1.01, ledger_path=tmp_path / "narrow.jsonl", book=book,
                  transport=narrow, sleep=lambda s: None, concurrency=1)
    assert narrow.calls == [] and read_ledger(tmp_path / "narrow.jsonl")[1] == []
    cap = 3 * bound * 1.01
    timeouts = fake_transport([TimeoutError("t")] * 3)
    report = run_cells([cell], pp, cap_usd=cap, ledger_path=tmp_path / "three.jsonl", book=book, transport=timeouts,
                       sleep=lambda s: None, concurrency=1)     # one failed request: no trip, no pause
    assert report["breaker"]["early_failures"] == {ROUTE: 1} and report["breaker"]["tripped"] == {}
    ledger = Ledger(tmp_path / "three.jsonl", ledger_header(pp, book), cap_usd=cap)
    (rec,) = ledger.records()
    assert len(timeouts.calls) == 3 and rec["outcome"] == "no_response"
    assert rec["usd_counted"] == rec["usd_pending_unknown"] == pytest.approx(3 * bound)
    assert ledger.committed_usd() <= cap and ledger.in_flight_usd() == 0


def test_a_504_storm_stops_after_three_failed_requests(panel, book, tmp_path, fake_transport):
    """Persistent 5xx failures are counted as pending_unknown and stop the route: every such attempt is
    pending_unknown, and three failed requests (each after its three attempts) close the route and pause the run:
    the fourth is not sent, the ledger resumes. Quick 503s here: a 504 storm is timeout-class and has its
    own thresholds (test_timeout_class_requests_have_their_own_thresholds)."""
    pp = panel.get(PID)
    storm = fake_transport([(503, {"error": "busy"})] * 12)
    with pytest.raises(RunPaused, match="stopped after 3 of 4"):
        run_cells(cells(4), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book, transport=storm,
                  sleep=lambda s: None, concurrency=1)
    recs = read_ledger(tmp_path / "c1.jsonl")[1]
    assert len(storm.calls) == 9 and len(recs) == 3
    for rec in recs:
        assert rec["outcome"] == "no_response" and [a["http_status"] for a in rec["attempts"]] == [503] * 3
        assert rec["usd_counted"] == rec["usd_pending_unknown"] == pytest.approx(3 * rec["usd_upper_bound"])
    resumed = fake_transport([(200, openai_body())] * 4)
    report = run_cells(cells(4), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book, transport=resumed,
                       sleep=lambda s: None, concurrency=1)
    assert report["sent"] == 4
    assert report["pending_unknown_usd"] == pytest.approx(sum(r["usd_pending_unknown"] for r in recs))


def test_a_route_closes_after_three_consecutive_unknown_failures_for_the_rest_of_the_run(panel, book, tmp_path,
                                                                                        fake_transport):
    pp = panel.get(PID)
    guard = SpendGuard()
    ok = fake_transport([(200, openai_body())] * 20)
    run_cells(cells(20, "ok"), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book, transport=ok,
              sleep=lambda s: None, concurrency=1, guard=guard)      # past the first-20 window, no failure
    storm = fake_transport([(502, {"error": "bad gateway"})] * 9 + [(200, openai_body())] * 3)
    with pytest.raises(RouteTripped, match="3 consecutive unknown-cost failures"):
        run_cells(cells(4, "late"), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book, transport=storm,
                  sleep=lambda s: None, concurrency=1, guard=guard)
    assert len(storm.calls) == 9 and guard.breaker.state()["paused"] is None     # three requests x three attempts
    later = fake_transport([(200, openai_body())])
    with pytest.raises(RouteTripped):           # the same run: no new sends to that route
        run_cells(cells(1, "after"), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                  transport=later, sleep=lambda s: None, concurrency=1, guard=guard)
    assert later.calls == []


def test_rate_limits_are_not_failures_of_the_early_window(panel, book, tmp_path, fake_transport):
    pp = panel.get(PID)
    transport = fake_transport([(429, {"error": "rate"}), (429, {"error": "rate"}), (200, openai_body())] * 3)
    report = run_cells(cells(3), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                       transport=transport, sleep=lambda s: None, concurrency=1)
    assert report["sent"] == 3 and report["breaker"]["early_failures"] == {}
    assert all(r["usd_pending_unknown"] == 0 for r in report["records"])


def test_the_breaker_counts_requests_not_attempts(panel, book, tmp_path, fake_transport):
    """One request whose three attempts all time out is one failed request, recorded no_response with every
    attempt's bound pending_unknown; the route closes only after three different requests in a row fail that way, and
    an answered request (even after timed-out attempts) resets the count."""
    pp = panel.get(PID)
    path = tmp_path / "c1.jsonl"
    guard = SpendGuard()
    run_cells(cells(20, "ok"), pp, cap_usd=1.0, ledger_path=path, book=book, sleep=lambda s: None, concurrency=1,
              transport=fake_transport([(200, openai_body())] * 20), guard=guard)    # past the first-20 window

    def send(prefix, responses, n=1):
        transport = fake_transport(responses)
        report = run_cells(cells(n, prefix), pp, cap_usd=1.0, ledger_path=path, book=book, transport=transport,
                           sleep=lambda s: None, concurrency=1, guard=guard)
        return report, transport

    one, t1 = send("a", [TimeoutError("t")] * 3)                                   # one request, three timeouts
    (rec,) = one["records"]
    assert len(t1.calls) == 3 and rec["outcome"] == "no_response" and len(rec["attempts"]) == 3
    assert rec["usd_counted"] == rec["usd_pending_unknown"] == pytest.approx(3 * rec["usd_upper_bound"])
    assert guard.breaker.state()["tripped"] == {}
    two, _ = send("b", [(504, {"error": "gateway"})] * 3)                          # a second, different request
    assert two["records"][0]["outcome"] == "no_response" and guard.breaker.state()["tripped"] == {}
    three, _ = send("c", [TimeoutError("t"), TimeoutError("t"), (200, openai_body())])   # answered: count reset
    assert three["records"][0]["outcome"] == "observed" and guard.breaker.state()["tripped"] == {}
    four, _ = send("d", [TimeoutError("t")] * 3 + [(503, {"error": "busy"})] * 3, n=2)  # two more failed requests
    assert [r["outcome"] for r in four["records"]] == ["no_response"] * 2 and guard.breaker.state()["tripped"] == {}
    with pytest.raises(RouteTripped, match="3 consecutive unknown-cost failures .*stopped after 1 of 2"):
        send("e", [(502, {"error": "bad gateway"})] * 6, n=2)                      # the third in a row closes it
    state = guard.breaker.state()
    assert list(state["tripped"]) == [ROUTE] and state["paused"] is None and state["sends"] == {ROUTE: 26}
    assert state["early_failures"] == {}


def test_the_early_window_counts_requests(panel, book, tmp_path, fake_transport):
    """More than 10% failed requests among a route's first 20 requests pause the run: two failed requests (six
    unknown-cost attempts) do not, a third does; a request answered after timed-out attempts is not a failure."""
    pp = panel.get(PID)
    transport = fake_transport([TimeoutError("t")] * 3                              # request 0: failed
                               + [TimeoutError("t"), TimeoutError("t"), (200, openai_body())]   # 1: answered
                               + [(502, {"error": "bad gateway"})] * 3             # 2: failed (quick faults)
                               + [(200, openai_body())]                             # 3: answered
                               + [(400, {"error": "bad request"})]                  # 4: refused (a failure)
                               + [(200, openai_body())] * 2)
    with pytest.raises(RunPaused, match="3 failed requests among its first 5 requests .*stopped after 5 of 7"):
        run_cells(cells(7), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book, transport=transport,
                  sleep=lambda s: None, concurrency=1)
    assert len(transport.calls) == 11
    outcomes = [r["outcome"] for r in read_ledger(tmp_path / "c1.jsonl")[1]]
    assert outcomes == ["no_response", "observed", "no_response", "observed", "no_response"]


def test_timeout_class_requests_have_their_own_thresholds(panel, book, tmp_path, fake_transport):
    """A request that timed out by majority (route_timed_out) is an understood, disclosed absence, not a route-health
    signal at the quick-fault thresholds. It closes a route only after six in a row (not three), and pauses the run
    only when more than half of the first 20 requests time out (not 10%)."""
    import socket
    pp = panel.get(PID)
    timed_out = socket.timeout("The read operation timed out")
    # past the first-20 window: five timeout-class requests in a row do not close the route, the sixth does
    guard = SpendGuard()
    path = tmp_path / "streak.jsonl"
    run_cells(cells(20, "ok"), pp, cap_usd=1.0, ledger_path=path, book=book, sleep=lambda s: None, concurrency=1,
              transport=fake_transport([(200, openai_body())] * 20), guard=guard)
    five = run_cells(cells(5, "t"), pp, cap_usd=1.0, ledger_path=path, book=book, sleep=lambda s: None,
                     concurrency=1, transport=fake_transport([timed_out] * 15), guard=guard)
    assert [r["outcome"] for r in five["records"]] == ["no_response"] * 5 and guard.breaker.state()["tripped"] == {}
    with pytest.raises(RouteTripped, match="6 consecutive timeout-class requests"):
        run_cells(cells(2, "u"), pp, cap_usd=1.0, ledger_path=path, book=book, sleep=lambda s: None, concurrency=1,
                  transport=fake_transport([timed_out] * 6), guard=guard)
    # a quick fault after the answered reset keeps its own, tight count: three quick failures still close a route
    guard2 = SpendGuard()
    path2 = tmp_path / "quick.jsonl"
    run_cells(cells(20, "ok"), pp, cap_usd=1.0, ledger_path=path2, book=book, sleep=lambda s: None, concurrency=1,
              transport=fake_transport([(200, openai_body())] * 20), guard=guard2)
    with pytest.raises(RouteTripped, match="3 consecutive unknown-cost failures"):
        run_cells(cells(3, "q"), pp, cap_usd=1.0, ledger_path=path2, book=book, sleep=lambda s: None, concurrency=1,
                  transport=fake_transport([(503, {"error": "busy"})] * 9), guard=guard2)
    # the early window: 3 timeouts among the first 14 requests do not pause; more than 10 of the first 20 do
    script = []
    for i in range(14):
        script += [timed_out] * 3 if i in (0, 5, 9) else [(200, openai_body())]
    guard3 = SpendGuard()
    report = run_cells(cells(14, "e"), pp, cap_usd=1.0, ledger_path=tmp_path / "early.jsonl", book=book,
                       sleep=lambda s: None, concurrency=1, transport=fake_transport(script), guard=guard3)
    assert sum(r["outcome"] == "no_response" for r in report["records"]) == 3 and guard3.breaker.state()["paused"] is None
    assert guard3.breaker.state()["early_timeouts"] == {ROUTE: 3} and guard3.breaker.state()["early_failures"] == {}
    script = []
    for i in range(20):
        script += [timed_out] * 3 if i % 2 == 0 else [(200, openai_body())]      # 10 timeouts among 20: not paused
    script += [timed_out] * 3                                                    # ... the 11th within 20? no: 21st
    guard4 = SpendGuard()
    report = run_cells(cells(21, "w"), pp, cap_usd=1.0, ledger_path=tmp_path / "win.jsonl", book=book,
                       sleep=lambda s: None, concurrency=1, transport=fake_transport(script), guard=guard4)
    assert guard4.breaker.state()["paused"] is None and guard4.breaker.state()["early_timeouts"] == {ROUTE: 10}
    script = []
    for i in range(20):
        script += [timed_out] * 3 if i % 2 == 0 or i == 1 else [(200, openai_body())]   # 11 of the first 20
    guard5 = SpendGuard()
    with pytest.raises(RunPaused, match="11 timeout-class requests among its first 19 requests"):
        run_cells(cells(20, "p"), pp, cap_usd=1.0, ledger_path=tmp_path / "pause.jsonl", book=book,
                  sleep=lambda s: None, concurrency=1, transport=fake_transport(script), guard=guard5)


def test_request_cost_and_route_timed_out_classes():
    assert [request_cost(c) for c in ([], ["answered"], ["unknown", "answered"], ["unknown"] * 3,
                                      ["unknown", "rate_limited", "unknown"], ["rate_limited"] * 3,
                                      ["unknown", "refused"])] == \
        ["none", "answered", "answered", "unknown", "unknown", "rate_limited", "refused"]
    read_timeout = {"error": {"kind": "transport", "message": "The read operation timed out"}}
    timed = {"outcome": "no_response", "attempts": [read_timeout, {"http_status": 504}]}
    assert route_timed_out(timed)
    assert not route_timed_out({**timed, "error": {"kind": "circuit_open"}})       # cut by the breaker: not tried out
    assert not route_timed_out({**timed, "attempts": [{"http_status": 504}, {"http_status": 400}]})
    assert not route_timed_out({**timed, "outcome": "observed"}) and not route_timed_out(None)
    assert not route_timed_out({"outcome": "no_response", "attempts": []})


def test_route_timed_out_only_when_every_failed_attempt_timed_out():
    """Narrowed rule: a request is route_timeout only when every unknown-cost attempt is a timeout (client read
    timeout, 408, 504; a 429 in between is neutral). Other unknown-cost failures stay no_response."""
    def rec(*attempts):
        return {"outcome": "no_response", "attempts": list(attempts)}

    def transport(message):
        return {"error": {"kind": "transport", "message": message}}

    read_timeout = transport("The read operation timed out")
    assert [timeout_class(a) for a in (read_timeout, transport("<urlopen error timed out>"),
                                       transport("TimeoutError('timed out')"), {"http_status": 408},
                                       {"http_status": 504})] == [True] * 5
    assert [timeout_class(a) for a in (transport("[WinError 10061] No connection could be made"),
                                       transport("<urlopen error [Errno 11001] getaddrinfo failed>"),
                                       transport("IncompleteRead(12 bytes read, 30 more expected)"),
                                       {"error": {"kind": "transport"}}, {"http_status": 500},
                                       {"http_status": 502}, {"http_status": 503}, {"http_status": 429})] == [False] * 8
    assert route_timed_out(rec(read_timeout, read_timeout, read_timeout))                         # (a)
    assert route_timed_out(rec(*[{"http_status": 504}] * 3))                                       # (b)
    assert route_timed_out(rec(*[{"http_status": 408}] * 3))
    assert route_timed_out(rec(read_timeout, {"http_status": 429}, {"http_status": 504}))          # 429 neutral
    three_500 = rec(*[{"http_status": 500}] * 3)                                                    # (c)
    assert request_cost(record_costs(three_500)) == "unknown" and not route_timed_out(three_500)   # breaker: unknown
    # a 500 that took the provider's cutoff (nscale: ~61 s) is a timeout; a quick 500 is not
    assert timeout_class({"http_status": 500, "elapsed_seconds": 61.2})
    assert not timeout_class({"http_status": 500, "elapsed_seconds": 12.0})
    assert not timeout_class({"http_status": 500})
    slow_500 = rec(*[{"http_status": 500, "elapsed_seconds": 61.0}] * 3)
    assert request_cost(record_costs(slow_500)) == "unknown" and route_timed_out(slow_500)
    assert route_timed_out(rec({"http_status": 500, "elapsed_seconds": 61.0}, {"http_status": 500, "elapsed_seconds": 3.0},
                               {"http_status": 500, "elapsed_seconds": 61.0}))                      # majority slow
    assert not route_timed_out(rec({"http_status": 500, "elapsed_seconds": 61.0}, {"http_status": 500, "elapsed_seconds": 3.0},
                                   {"http_status": 500, "elapsed_seconds": 3.0}))                   # majority quick
    # (d) mixed, majority rule: two timeouts around one quick fault are a timeout; one timeout among two quick
    # faults is not
    assert route_timed_out(rec(read_timeout, {"http_status": 500}, read_timeout))
    assert route_timed_out(rec({"http_status": 500, "elapsed_seconds": 61.3}, {"http_status": 502, "elapsed_seconds": 10.8},
                               {"http_status": 504, "elapsed_seconds": 180.3}))                    # the nscale case
    assert not route_timed_out(rec(read_timeout, {"http_status": 500}, {"http_status": 502}))
    assert route_timed_out(rec(read_timeout, {"http_status": 500}))                                # 1 of 2: majority
    assert not route_timed_out(rec(*[{"http_status": 502}] * 3))
    assert not route_timed_out(rec(*[transport("IncompleteRead(0 bytes read)")] * 3))
    assert not route_timed_out(rec({"http_status": 429}, {"http_status": 429}, {"http_status": 429}))


def test_route_timed_out_on_the_records_run_cells_writes(panel, book, tmp_path, fake_transport):
    """The narrowed route_timed_out on real ledger records: three socket read timeouts (the message the C1 record
    carries) are a route timeout, and so is a timeout / quick 500 / timeout mix (majority rule); three refused
    connections, and a timeout / 500 / 500 mix, are not (all four requests are unknown-cost failures for the
    breaker)."""
    import socket
    pp = panel.get(PID)
    read_timeout = socket.timeout("The read operation timed out")
    scripts = {"t": [read_timeout] * 3, "c": [ConnectionRefusedError(10061, "refused")] * 3,
               "m": [read_timeout, (500, {}), read_timeout], "q": [read_timeout, (500, {}), (500, {})]}
    got = {}
    for name, script in scripts.items():
        report = run_cells(cells(1, name), pp, cap_usd=1.0, ledger_path=tmp_path / f"{name}.jsonl", book=book,
                           transport=fake_transport(script), sleep=lambda s: None)
        (rec,) = report["records"]
        got[name] = (rec["outcome"], request_cost(record_costs(rec)), route_timed_out(rec))
    assert got == {"t": ("no_response", "unknown", True), "c": ("no_response", "unknown", False),
                   "m": ("no_response", "unknown", True), "q": ("no_response", "unknown", False)}


def test_timeout_then_success_settles_each_attempt(panel, book, tmp_path, fake_transport):
    pp = panel.get(PID)
    report = run_cells(cells(1), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                       transport=fake_transport([TimeoutError("t"), (200, openai_body())]), sleep=lambda s: None)
    (rec,) = report["records"]
    bound = rec["usd_upper_bound"]
    assert rec["outcome"] == "observed" and rec["usd_reserved"] == pytest.approx(3 * bound)
    assert rec["usd_pending_unknown"] == pytest.approx(bound)
    assert rec["usd_counted"] == pytest.approx(rec["usd"] + bound)
    events = sends(tmp_path / "c1.jsonl")
    assert [e["event"] for e in events] == ["reserve", "settle"] and events[0]["send_id"] == rec["send_id"]
    assert events[0]["max_attempts"] == 3 and events[1]["usd_counted"] == pytest.approx(rec["usd_counted"])
    assert not (tmp_path / "c1.jsonl.executor.lock").exists() and not (tmp_path / "c1.jsonl.lock").exists()


def test_incomplete_read_is_a_transport_error_and_is_retried(panel, book, tmp_path, fake_transport):
    """A DeepSeek chunked response once ended early; http.client.IncompleteRead is an HTTPException, not an OSError,
    and escaped the vendored client's retry loop, stopping the whole build run (the main run's B-built arm, 2,085
    records in). It must cost one retry, like a timeout, and settle each attempt."""
    import http.client
    pp = panel.get(PID)
    report = run_cells(cells(1), pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                       transport=fake_transport([http.client.IncompleteRead(b""), (200, openai_body())]),
                       sleep=lambda s: None)
    (rec,) = report["records"]
    assert rec["outcome"] == "observed" and len(rec["attempts"]) == 2
    assert rec["attempts"][0]["error"]["kind"] == "transport"
    assert rec["usd_pending_unknown"] == pytest.approx(rec["usd_upper_bound"])
    events = sends(tmp_path / "c1.jsonl")
    assert [e["event"] for e in events] == ["reserve", "settle"]


def test_two_holders_of_one_ledger_cannot_both_reserve(tmp_path):
    """Two Ledger objects on one path share one reservation record: two $0.75 reservations cannot both pass a $1 cap
    (the reservation lives in the send log, read by every holder under the ledger's lock file)."""
    path = tmp_path / "two.jsonl"
    first, second = Ledger(path, HEADER, cap_usd=1), Ledger(path, HEADER, cap_usd=1)
    held = first.reserve(0.75, request_hash="h0")
    with pytest.raises(SpendCapExceeded, match="in flight \\$0.7500"):
        second.reserve(0.75, request_hash="h1")
    first.append({"kind": "call", "request_hash": "h0", "outcome": "observed", "usd_counted": 0.75}, held)
    assert Ledger(path, HEADER, cap_usd=1).committed_usd() == pytest.approx(0.75)
    with pytest.raises(SpendCapExceeded):
        second.reserve(0.75, request_hash="h1")     # the settled 0.75 is spent, seen by the other holder


def test_one_executor_per_ledger(panel, book, tmp_path, fake_transport):
    pp = panel.get(PID)
    path = tmp_path / "c1.jsonl"
    transport = fake_transport([(200, openai_body())])
    with Ledger(path, ledger_header(pp, book), cap_usd=1.0).executor():
        with pytest.raises(LockBusy, match="one executor per ledger"):
            run_cells(cells(1), pp, cap_usd=1.0, ledger_path=path, book=book, transport=transport)
    assert transport.calls == []
    assert run_cells(cells(1), pp, cap_usd=1.0, ledger_path=path, book=book, transport=transport)["sent"] == 1


def test_a_stale_short_lock_is_broken_but_a_live_one_waits(tmp_path):
    path = tmp_path / "x.lock"
    path.write_text("pid 1 since long ago\n", encoding="utf-8")
    with pytest.raises(LockBusy, match="still held"):
        with FileLock(path, stale_seconds=60, wait_seconds=0.05):
            pass
    with FileLock(path, stale_seconds=0.0, wait_seconds=1.0):      # older than 0 s: broken, then taken
        assert "pid" in path.read_text(encoding="utf-8")
    assert not path.exists() and list(tmp_path.iterdir()) == []


def test_a_refused_settlement_is_kept_and_recovered_once_the_cap_allows(tmp_path):
    path = tmp_path / "r.jsonl"
    ledger = Ledger(path, HEADER, cap_usd=1.0)
    held = ledger.reserve(0.5, request_hash="h")
    answer = {"kind": "call", "request_hash": "h", "outcome": "observed", "usd_counted": 1.2}
    with pytest.raises(SpendCapExceeded, match="kept in r.jsonl.sends"):
        ledger.append(answer, held)
    assert read_ledger(path)[1] == [] and ledger.in_flight_usd() == pytest.approx(1.2)   # counted at the larger
    with pytest.raises(SpendCapExceeded):
        Ledger(path, HEADER, cap_usd=1.0).recover_unsettled()
    raised = Ledger(path, HEADER, cap_usd=2.0)
    (line,) = raised.recover_unsettled()
    assert line["resolution"] == "kept_record_appended" and raised.done("h")["usd_counted"] == 1.2
    assert raised.in_flight_usd() == 0 and raised.spent_usd() == pytest.approx(1.2)


def test_resume_after_a_crash_lists_and_settles_the_unsettled_sends(panel, book, tmp_path, fake_transport):
    """A reservation written to the send log whose executor died before settling: the next run lists it, records it
    as a no_response whose whole reservation is pending_unknown, and sends the request again."""
    pp = panel.get(PID)
    path = tmp_path / "c1.jsonl"
    (cell,) = cells(1)
    crashed = Ledger(path, ledger_header(pp, book), cap_usd=1.0)
    lost = crashed.reserve(0.01, request_hash="deadbeef", stub={"item_id": "q0", "stage": "s", "repeat_idx": 0},
                           detail={"attempt_bound_usd": 0.01 / 3, "max_attempts": 3})
    assert crashed.in_flight_usd() == pytest.approx(0.01)   # counted while nobody settles it
    transport = fake_transport([(200, openai_body())])
    report = run_cells([cell], pp, cap_usd=1.0, ledger_path=path, book=book, transport=transport)
    (line,) = report["recovered"]
    assert (line["send_id"], line["request_hash"], line["resolution"]) == (lost.send_id, "deadbeef", "pending_unknown")
    recovered, answered = read_ledger(path)[1]
    assert recovered["error"]["kind"] == "unsettled_send" and recovered["item_id"] == "q0"
    assert recovered["usd_counted"] == recovered["usd_pending_unknown"] == pytest.approx(0.01)
    assert answered["outcome"] == "observed" and report["sent"] == 1
    assert report["pending_unknown_usd"] == pytest.approx(0.01)
    assert Ledger(path, ledger_header(pp, book), cap_usd=1.0).in_flight_usd() == 0


# ---- approval ledger -----------------------------------------------------------------------------------------

def approval(tmp_path, granted=1.0, arms=None, approval_id="t1"):
    write_approval(approval_id, granted, arms or {"a": 0.6, "b": 0.4}, "test grant", root=tmp_path)
    return load_approval(approval_id, root=tmp_path)


def test_an_approval_file_is_checked_and_write_once(tmp_path):
    ap = approval(tmp_path)
    doc = json.loads(ap.path.read_text(encoding="utf-8"))
    assert doc["semantics"] == SEMANTICS and doc["arms"] == {"a": 0.6, "b": 0.4} and doc["granted_usd"] == 1.0
    with pytest.raises(FileExistsError):
        write_approval("t1", 1.0, {"a": 0.5}, "again", root=tmp_path)
    with pytest.raises(ApprovalError, match="sum to"):
        write_approval("t2", 1.0, {"a": 0.6, "b": 0.5}, "over", root=tmp_path)
    with pytest.raises(ApprovalError, match="no approval"):
        load_approval("t3", root=tmp_path)
    with pytest.raises(ApprovalError, match="no arm"):
        ApprovalGuard(ap, tmp_path / "run", "c")


def test_pending_unknown_counts_toward_the_approval(panel, book, tmp_path, fake_transport):
    pp = panel.get(PID)
    ap = approval(tmp_path)
    run_dir = tmp_path / "runs" / "a_000000000001"
    run_dir.mkdir(parents=True)
    guard = ApprovalGuard(ap, run_dir, "a")
    guard.bind()
    run_cells(cells(1), pp, cap_usd=0.6, ledger_path=run_dir / f"read__{PID}.jsonl", book=book,
              transport=fake_transport([(504, {"error": "gateway"})] * 3), sleep=lambda s: None,
              guard=SpendGuard(approval=guard))
    (rec,) = read_ledger(run_dir / f"read__{PID}.jsonl")[1]
    pending = rec["usd_pending_unknown"]
    tally = guard.tally()
    assert pending > 0 and tally["pending_unknown_usd"] == pytest.approx(pending)
    assert tally["committed_usd"] == tally["arm_committed_usd"] == pytest.approx(pending)
    guard.check(0.6 - pending - 1e-9, "room left")
    with pytest.raises(ApprovalExceeded, match="arm a"):
        guard.check(0.6 - pending + 1e-6, "one more")
    assert json.loads((run_dir / "approvals.jsonl").read_text(encoding="utf-8"))["approval_id"] == "t1"


def test_two_executors_on_one_approval_cannot_double_reserve(tmp_path):
    """Two run directories under one arm, one executor thread each, reserving $0.10 at a time: the approval's lock
    and tally let exactly ten reservations through a $1.00 arm, whichever executor makes them."""
    ap = approval(tmp_path, granted=1.0, arms={"a": 1.0})
    ledgers, guards = [], []
    for n in range(2):
        run_dir = tmp_path / "runs" / f"a_00000000000{n}"
        run_dir.mkdir(parents=True)
        guards.append(ApprovalGuard(ap, run_dir, "a"))
        guards[-1].bind()
        ledgers.append(Ledger(run_dir / "read__p.jsonl", HEADER, cap_usd=1.0))
    taken = [0, 0]

    def executor(n):
        for k in range(10):
            try:
                guards[n].reserve(ledgers[n], 0.1, request_hash=f"{n}-{k}")
            except ApprovalExceeded:
                return
            taken[n] += 1

    threads = [threading.Thread(target=executor, args=(n,)) for n in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sum(taken) == 10 and guards[0].tally()["committed_usd"] == pytest.approx(1.0)
    with pytest.raises(ApprovalExceeded, match="granted|arm"):
        guards[1].reserve(ledgers[1], 0.01)
    assert len(ap.bindings_path.read_text(encoding="utf-8").splitlines()) == 2


def test_the_approval_counts_every_arm_against_the_grant(tmp_path):
    ap = approval(tmp_path, granted=1.0, arms={"a": 0.6, "b": 0.4})
    dirs = {label: tmp_path / "runs" / f"{label}_000000000000" for label in ("a", "b")}
    ledgers = {}
    for label, run_dir in dirs.items():
        run_dir.mkdir(parents=True)
        ApprovalGuard(ap, run_dir, label).bind()
        ledgers[label] = Ledger(run_dir / "build__p.jsonl", HEADER, cap_usd=1.0)
    guard_a, guard_b = ApprovalGuard(ap, dirs["a"], "a"), ApprovalGuard(ap, dirs["b"], "b")
    guard_a.reserve(ledgers["a"], 0.6)
    with pytest.raises(ApprovalExceeded, match="arm a"):
        guard_a.reserve(ledgers["a"], 0.01)
    guard_b.reserve(ledgers["b"], 0.4)
    tally = guard_b.tally()
    assert tally["in_flight_usd"] == pytest.approx(1.0) and tally["committed_usd"] == pytest.approx(1.0)
    with pytest.raises(ApprovalError, match="bound .* as 'a'"):
        ApprovalGuard(ap, dirs["a"], "b").bind()

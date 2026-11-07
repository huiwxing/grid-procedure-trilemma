"""run_cells with a mocked transport: ledger records, retries, resume, spend cap, dry-run cost."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import urllib.error

import pytest

from procedures.readers import Cell, LedgerError, SpendCapExceeded, request_hash, run_cells, scenario, upper_bound_usd
from procedures.readers.client import ConfigurationError
from procedures.readers.ledger import Ledger, read_ledger
from procedures.readers.tokens import estimate_tokens

ROOT = Path(__file__).resolve().parents[2]


def msg(text):
    return ({"role": "user", "content": text},)


def openai_body(prompt=1_000, completion=1_000, reasoning=None, text='{"value": "15", "unit": "min"}',
                model="served-model-x", reasoning_content=None):
    usage = {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion}
    if reasoning is not None:
        usage["completion_tokens_details"] = {"reasoning_tokens": reasoning}
    message = {"role": "assistant", "content": text}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    return {"id": "cmpl-1", "model": model, "system_fingerprint": "fp_test",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}], "usage": usage}


def test_live_run_writes_complete_records_and_prices_them(panel, book, tmp_path, fake_transport):
    pp = panel.get("A1_gpt6sol_reason_low")
    transport = fake_transport([(200, openai_body(prompt=10_000, completion=1_500, reasoning=1_000,
                                                  model="gpt-6-sol-2026-08-01"))])
    cells = [Cell("item-1", "read/M2@4000", msg("Under PJM Manual 3 ...\r\nwhat is the limit?"))]
    report = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "a1.jsonl", book=book,
                       transport=transport, sleep=lambda s: None)
    rec = report["records"][0]
    assert report["sent"] == 1 and len(transport.calls) == 1
    sent = transport.calls[0]["payload"]
    assert sent["max_completion_tokens"] == 4096 and "\r" not in sent["messages"][0]["content"]
    assert rec["request_hash"] == request_hash(pp.profile_id, pp.fingerprint, cells[0].messages, {}, 0)
    assert rec["served_model"] == "gpt-6-sol-2026-08-01" and rec["system_fingerprint"] == "fp_test"
    assert rec["finish_reason"] == "stop" and rec["status"] == "ok" and rec["outcome"] == "observed"
    assert rec["usage"]["reasoning"] == 1_000 and rec["usage"]["output_visible"] == 500
    assert rec["usd"] == pytest.approx((10_000 * 2.0 + 1_500 * 10.0) / 1e6)
    assert rec["usd_counted"] == rec["usd"] and rec["price"]["price_book_sha256"] == book.sha256
    assert rec["raw_response"]["usage"]["prompt_tokens"] == 10_000 and rec["raw_text"] is None
    assert rec["attempts"][0]["http_status"] == 200 and rec["started_at"] and rec["finished_at"]
    assert rec["reasoning_leak"] is False and rec["harness_digest"]
    lines = (tmp_path / "a1.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["kind"] for line in lines] == ["header", "call"]


def test_retry_backoff_goes_through_the_vendored_client(panel, book, tmp_path, fake_transport):
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    transport = fake_transport([(429, {"error": "rate"}), (503, {"error": "busy"}), (200, openai_body())])
    slept = []
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl",
                       book=book, transport=transport, sleep=slept.append)
    rec = report["records"][0]
    assert [a.get("http_status") for a in rec["attempts"]] == [429, 503, 200]
    assert slept == [1, 2] and rec["outcome"] == "observed"
    # the 429 is a definite refusal (not billed); the 503 may have been billed upstream: its bound, pending unknown
    bound = rec["usd_upper_bound"]
    assert rec["usd_pending_unknown"] == pytest.approx(bound)
    assert rec["usd_counted"] == pytest.approx((1_000 * 0.02 + 1_000 * 0.05) / 1e6 + bound)
    assert rec["usd_reserved"] == pytest.approx(3 * bound)   # every allowed attempt reserved before sending


def test_resume_skips_answered_requests_and_resends_unanswered(panel, book, tmp_path, fake_transport):
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    ledger = tmp_path / "c1.jsonl"
    cells = [Cell("i1", "s", msg("q1")), Cell("i2", "s", msg("q2"))]
    down = urllib.error.URLError("connection reset")
    first = fake_transport([(200, openai_body()), down, down, down])
    report = run_cells(cells, pp, cap_usd=1.0, ledger_path=ledger, book=book, transport=first, sleep=lambda s: None,
                       concurrency=1)      # one failed request (three attempts) neither trips nor pauses the route
    assert report["sent"] == 2 and report["breaker"]["tripped"] == {} and report["breaker"]["paused"] is None
    failed = read_ledger(ledger)[1][1]
    assert failed["outcome"] == "no_response" and failed["usd"] is None
    assert failed["usd_counted"] == pytest.approx(3 * failed["usd_upper_bound"])  # timeouts may have billed
    assert failed["usd_pending_unknown"] == failed["usd_counted"]
    resumed = fake_transport([(200, openai_body()), (200, openai_body())])
    report = run_cells(cells + [Cell("i3", "s", msg("q3"))], pp, cap_usd=1.0, ledger_path=ledger, book=book,
                       transport=resumed, sleep=lambda s: None, concurrency=1)
    assert report["skipped_done"] == 1 and report["sent"] == 2 and len(resumed.calls) == 2
    assert [r["outcome"] for r in report["records"]] == ["observed"] * 3
    call_lines = ledger.read_text(encoding="utf-8").splitlines()[1:]
    assert len(call_lines) == 4  # 2 first run + 2 resumed; nothing rewritten


def test_spend_cap_refuses_before_sending_anything(panel, book, tmp_path, fake_transport):
    pp = panel.get("A1_gpt6sol_reason_low")
    transport = fake_transport([(200, openai_body())])
    cells = [Cell(f"i{k}", "s", msg(f"q{k}")) for k in range(3)]
    with pytest.raises(SpendCapExceeded, match="nothing was sent"):
        run_cells(cells, pp, cap_usd=0.001, ledger_path=tmp_path / "a1.jsonl", book=book, transport=transport)
    assert transport.calls == []
    assert len((tmp_path / "a1.jsonl").read_text(encoding="utf-8").splitlines()) == 1  # header only


def test_spend_cap_stops_mid_run_before_the_request_that_could_exceed_it(panel, book, tmp_path, fake_transport):
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    cells = [Cell(f"i{k}", "s", msg(f"q{k}")) for k in range(3)]
    actual = (1_000 * 0.02 + 1_000 * 0.05) / 1e6
    bound = upper_bound_usd(cells[2], pp, book)
    transport = fake_transport([(200, openai_body())] * 3)
    # each request reserves all 3 attempts' bounds: the third needs 2 answers + 3 bounds, more than the cap
    with pytest.raises(SpendCapExceeded, match="stopped after 2 of 3"):
        run_cells(cells, pp, cap_usd=3 * bound + 1.5 * actual, ledger_path=tmp_path / "c1.jsonl", book=book,
                  transport=transport, concurrency=1)
    assert len(transport.calls) == 2


def test_missing_credential_and_duplicate_cells_fail_before_sending(panel, book, tmp_path, fake_transport,
                                                                    monkeypatch):
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    transport = fake_transport([])
    with pytest.raises(ValueError, match="duplicate"):
        run_cells([Cell("i", "s", msg("q")), Cell("j", "t", msg("q"))], pp, cap_usd=1.0,
                  ledger_path=tmp_path / "x.jsonl", book=book, transport=transport)
    monkeypatch.delenv("HF_TOKEN")
    with pytest.raises(ConfigurationError, match="HF_TOKEN"):
        run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "y.jsonl", book=book,
                  transport=transport)
    assert transport.calls == [] and not (tmp_path / "y.jsonl").exists()


def test_reasoning_leak_is_flagged_when_thinking_was_disabled(panel, book, tmp_path, fake_transport):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    body = openai_body(text="<think>let me see</think>{\"value\": \"15\"}", prompt=50, completion=40)
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "c2.jsonl",
                       book=book, transport=fake_transport([(200, body)]))
    assert report["records"][0]["reasoning_leak"] is True
    assert report["records"][0]["request"]["chat_template_kwargs"] == {"enable_thinking": False}


def nscale_body(reasoning_content, completion, prompt=22, text="\n\nREADY"):
    """Qwen/Qwen3-8B:nscale shape seen in the connectivity smoke run: reasoning_content, no
    completion_tokens_details."""
    body = openai_body(prompt=prompt, completion=completion, text=text, model="Qwen/Qwen3-8B",
                       reasoning_content=reasoning_content)
    body["usage"]["prompt_tokens_details"] = None
    return body


def test_user_message_suffix_is_sent_recorded_and_hashed(panel, book, tmp_path, fake_transport):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    transport = fake_transport([(200, nscale_body("\n\n", 6))])
    cells = [Cell("i", "s", ({"role": "system", "content": "Answer tersely."}, {"role": "user", "content": "q"}))]
    report = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "c2.jsonl", book=book, transport=transport)
    rec = report["records"][0]
    sent = [{"role": "system", "content": "Answer tersely."}, {"role": "user", "content": "q /no_think"}]
    assert transport.calls[0]["payload"]["messages"] == sent == rec["request"]["messages"]
    assert rec["user_message_suffix"] == " /no_think"
    assert rec["request_hash"] == request_hash(pp.profile_id, pp.fingerprint, sent, {}, 0,
                                               user_message_suffix=" /no_think")
    assert rec["request_hash"] not in {request_hash(pp.profile_id, pp.fingerprint, sent, {}, 0),
                                       request_hash(pp.profile_id, pp.fingerprint, cells[0].messages, {}, 0),
                                       request_hash(pp.profile_id, pp.fingerprint, sent, {}, 0,
                                                    user_message_suffix=" /think")}
    header = json.loads((tmp_path / "c2.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert header["user_message_suffix"] == " /no_think"
    assert rec["reasoning_leak"] is False and rec["usage"]["reasoning"] == 0  # whitespace-only reasoning_content


def test_changing_the_suffix_is_a_new_run_identity(panel, book, tmp_path, fake_transport):
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    ledger, cells = tmp_path / "c1.jsonl", [Cell("i", "s", msg("q"))]
    run_cells(cells, pp, cap_usd=1.0, ledger_path=ledger, book=book, transport=fake_transport([(200, openai_body())]))
    lines = ledger.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["user_message_suffix"] is None
    with pytest.raises(LedgerError, match="user_message_suffix"):
        run_cells(cells, replace(pp, user_message_suffix=" /x"), cap_usd=1.0, ledger_path=ledger, book=book,
                  transport=fake_transport([]))
    legacy = json.loads(lines[0])
    del legacy["user_message_suffix"]  # a header written before the field existed still resumes
    ledger.write_text("\n".join([json.dumps(legacy)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
    report = run_cells(cells, pp, cap_usd=1.0, ledger_path=ledger, book=book, transport=fake_transport([]))
    assert report["skipped_done"] == 1 and report["sent"] == 0


@pytest.mark.parametrize("reasoning_content, text, leak", [
    ("\n\n", "\n\nREADY", False),                          # /no_think in the connectivity smoke: empty think block
    (" \t\n", "READY", False),
    (None, "<think>\n\n</think>\n\nREADY", False),         # the same empty block inline
    ("Okay, the user wants READY.", "\n\nREADY", True),    # the leak shape of the connectivity smoke
    (None, "<think>let me see</think>READY", True),
    (None, "READY <think>cut off mid-thought", True),
])
def test_reasoning_leak_ignores_whitespace_only_reasoning(panel, book, tmp_path, fake_transport,
                                                         reasoning_content, text, leak):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "c2.jsonl", book=book,
                       transport=fake_transport([(200, nscale_body(reasoning_content, 6, text=text))]))
    assert report["records"][0]["reasoning_leak"] is leak


def test_nscale_leak_records_a_reasoning_estimate_without_changing_the_price(panel, book, tmp_path, fake_transport):
    pp = panel.get("C2_qwen3_8b_hf_nscale_nothink")
    thought = "Okay, the user wants me to return only the word READY. " * 8
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "c2.jsonl", book=book,
                       transport=fake_transport([(200, nscale_body(thought, 104, prompt=18))]))
    rec = report["records"][0]
    usage = rec["usage"]
    assert rec["reasoning_leak"] is True and rec["reasoning_text"] == thought
    assert usage["reasoning_tokens_estimated"] is True and usage["reasoning_content_chars"] == len(thought)
    assert usage["reasoning_tokens_estimate"] == estimate_tokens(thought)
    assert usage["reasoning"] is None and usage["output_total"] == 104
    assert rec["usd"] == pytest.approx((18 * 0.07 + 104 * 0.18) / 1e6)


def test_anthropic_record_normalises_native_usage(panel, book, tmp_path, fake_transport):
    pp = panel.get("A2_sonnet5_nothink")
    body = {"id": "msg_1", "type": "message", "role": "assistant", "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": "{\"value\": \"15\"}"}], "stop_reason": "end_turn",
            "usage": {"input_tokens": 1_000, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0,
                      "output_tokens": 20}}
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "a2.jsonl",
                       book=book, transport=fake_transport([(200, body)]))
    rec = report["records"][0]
    assert rec["request"]["thinking"] == {"type": "disabled"} and "temperature" not in rec["request"]
    assert rec["usage"]["reasoning"] == 0 and rec["usage"]["reasoning_basis"] == "absent_no_reasoning_text"
    assert rec["usd"] == pytest.approx((1_000 * 2.0 + 20 * 10.0) / 1e6) and rec["reasoning_leak"] is False


def test_harness_failure_after_a_paid_answer_keeps_the_raw_result(panel, book, tmp_path, fake_transport):
    from procedures.readers import PricingError
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    absurd = openai_body(prompt=2_000_000_000, completion=1)  # $40 > the $25 per-call unit assertion
    with pytest.raises(PricingError):
        run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                  transport=fake_transport([(200, absurd)]))
    rec = json.loads((tmp_path / "c1.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert rec["outcome"] == "harness_error" and "PricingError" in rec["error"]["message"]
    assert rec["result"]["raw_response"]["usage"]["prompt_tokens"] == 2_000_000_000
    assert rec["usd_counted"] == rec["usd_upper_bound"] > 0


def test_cell_labels_go_into_the_record_not_the_request(panel, book, tmp_path, fake_transport):
    from procedures.readers.executor import RECORD_FIELDS, ledger_header
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    plain, labelled = Cell("i", "s", msg("q")), Cell("i", "s", msg("q"), labels={"cap_level": 2})
    transport = fake_transport([(200, openai_body())])
    report = run_cells([labelled], pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                       transport=transport, sleep=lambda s: None)
    rec = report["records"][0]
    assert rec["cap_level"] == 2 and "cap_level" not in json.dumps(transport.calls[0]["payload"])
    assert rec["request_hash"] == request_hash(pp.profile_id, pp.fingerprint, plain.messages, {}, 0)
    assert set(rec) - {"cap_level"} <= RECORD_FIELDS                        # the reserved names cover every field
    again = run_cells([plain], pp, cap_usd=1.0, ledger_path=tmp_path / "c1.jsonl", book=book,
                      transport=fake_transport([]), sleep=lambda s: None)
    assert again["sent"] == 0 and again["records"][0]["cap_level"] == 2       # one request, labelled or not
    header, _ = read_ledger(tmp_path / "c1.jsonl")
    assert {k: header[k] for k in ledger_header(pp, book)} == ledger_header(pp, book)
    for name in ("usd_counted", "imported_from", "request_hash"):
        with pytest.raises(ValueError, match="ledger record fields"):
            Cell("i", "s", msg("q"), labels={name: 1})


# ---- dry run: assumed output clipped at the request's cap -------

def question_cells(q):
    """100 questions x {B0 and three 4k memories}, with q prompt tokens per call."""
    cells = []
    for i in range(100):
        item = f"q-{i:03d}"
        cells.append(Cell(item, "B0", msg(f"B0 {item}"), est_input_tokens=q))
        for rep in ("M2", "M4", "M5@4k"):
            cells.append(Cell(item, rep, msg(f"{rep} {item}"), est_input_tokens=4_000 + q))
    return cells


def dry(pp, book, cells, name):
    return run_cells(cells, pp, cap_usd=100.0, dry_run=True, book=book, assumptions=scenario(pp, name))


def test_dry_run_clips_assumed_output_at_the_request_cap(panel, book):
    """A 2,048-token output cap against the HIGH scenario's assumed output of gpt-oss-20b (300 visible + 3,000
    reasoning): every call is clipped. Hand-computed ($/1M rates x 1.10): 100x(500x0.03 + 2048x0.14)/1e6 +
    300x(4500x0.03 + 2048x0.14)/1e6 = 0.030172 + 0.126516 -> x1.1 = 0.1723568."""
    pp = panel.get("C3_gptoss20b_hf_deepinfra_low")
    cells = [replace(c, generation={"max_output_tokens": 2_048}) for c in question_cells(500)]
    report = dry(pp, book, cells, "high")
    assert report["clipped_calls"] == 400 and report["usd"] == pytest.approx(0.1723568, abs=5e-7)


def test_dry_run_never_builds_a_request_sender(panel, book):
    pp = panel.get("A1_gpt6sol_reason_low")
    report = run_cells([Cell("i", "s", msg("q"))], pp, cap_usd=0.0, dry_run=True, book=book)
    assert report["mode"] == "dry_run" and report["exceeds_cap"] is True
    assert report["usd_upper_bound"] >= report["usd"] > 0


# ---- connectivity smoke plan CLI (plan only; --execute is never exercised) ----------------

def load_smoke_plan():
    spec = importlib.util.spec_from_file_location("smoke_plan", ROOT / "scripts" / "smoke_plan.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_plan_prints_every_profile_without_sending(capsys):
    smoke = load_smoke_plan()
    assert smoke.main(["--json"]) == 0
    rows = json.loads(capsys.readouterr().out)
    # 17 profiles, 15 sendable
    assert len(rows) == 17 and sum(r["role"] not in ("negative", "candidate") for r in rows) == 15
    candidate = next(r for r in rows if r["role"] == "candidate")
    assert candidate["profile_id"] == "tierB_llama33_70b_hf_deepinfra" and candidate["cap_usd"] == 0.0
    assert all(r["cap_usd"] >= r["bound_usd"] for r in rows)
    assert 0 < sum(r["cap_usd"] for r in rows) < 0.10     # every attempt of every call reserved
    negative = next(r for r in rows if r["role"] == "negative")
    assert "fixed provider" in negative["refusal"]
    assert smoke.main([]) == 0
    text = capsys.readouterr().out
    assert "Nothing was sent" in text and "dummy-" not in text


def dispatching_transport(calls):
    """Answer each provider in its own wire format (offline stand-in for the paid smoke)."""
    def send(url, headers, payload, timeout):
        calls.append(url)
        if "api.anthropic.com" in url:
            body = {"model": "claude-sonnet-5", "content": [{"type": "text", "text": "READY"}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": 20, "cache_read_input_tokens": 0,
                                                         "cache_creation_input_tokens": 0, "output_tokens": 2}}
        elif "generativelanguage" in url:
            body = {"modelVersion": "gemini-3.8-flash", "usageMetadata": {
                        "promptTokenCount": 18, "candidatesTokenCount": 2, "thoughtsTokenCount": 30,
                        "totalTokenCount": 50},
                    "candidates": [{"content": {"parts": [{"text": "READY"}]}, "finishReason": "STOP"}]}
        elif "api.deepseek.com" in url:
            body = openai_body(prompt=18, completion=2, reasoning=0, text="READY", model="deepseek-flash")
            body["usage"].update(prompt_cache_hit_tokens=0, prompt_cache_miss_tokens=18)
        else:
            body = openai_body(prompt=18, completion=2, reasoning=0, text="READY", model=payload["model"])
        return 200, {}, json.dumps(body).encode("utf-8")
    return send


def test_smoke_plan_only_and_ledger_tag(panel, book, tmp_path, monkeypatch, capsys):
    import procedures.readers.client.client as client_module
    smoke = load_smoke_plan()
    assert smoke.main(["--json", "--only", "C2_qwen3_8b_hf_nscale_nothink"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert [r["profile_id"] for r in rows] == ["C2_qwen3_8b_hf_nscale_nothink"]
    assert rows[0]["user_message_suffix"] == " /no_think"
    for argv in (["--only", "no_such_profile"], ["--ledger-tag", "../escape"]):
        with pytest.raises(SystemExit):
            smoke.main(argv)
    calls = []
    monkeypatch.setattr(client_module, "http_transport", dispatching_transport(calls))
    monkeypatch.setattr(smoke, "LEDGER_DIR", tmp_path)
    assert smoke.execute(rows, panel, book, cap_usd=0.05, ledger_tag="rerun") == 0
    assert len(calls) == 1
    assert [p.name for p in tmp_path.glob("*.jsonl")] == ["C2_qwen3_8b_hf_nscale_nothink__rerun.jsonl"]
    _, records = read_ledger(tmp_path / "C2_qwen3_8b_hf_nscale_nothink__rerun.jsonl")
    assert records[0]["request"]["messages"][-1]["content"].endswith("READY /no_think")


def test_smoke_execute_function_offline_with_dispatching_transport(panel, book, tmp_path, monkeypatch, capsys):
    """Exercises the paid path's code with a fake transport and dummy keys; the CLI flag is never passed."""
    import procedures.readers.client.client as client_module
    smoke = load_smoke_plan()
    calls = []
    monkeypatch.setattr(client_module, "http_transport", dispatching_transport(calls))
    monkeypatch.setattr(smoke, "LEDGER_DIR", tmp_path)
    rows = smoke.build_plan(panel, book)
    assert smoke.execute(rows, panel, book, cap_usd=0.10) == 0
    assert len(calls) == 15 and len(list(tmp_path.glob("*.jsonl"))) == 15   # the candidate is never sent
    assert not list(tmp_path.glob("tierB_llama33_70b_hf_deepinfra*"))
    printed = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert all(p["outcome"] == "observed" and p["usd"] > 0 for p in printed)
    assert smoke.execute(rows, panel, book, cap_usd=0.10) == 0 and len(calls) == 15  # resume sends nothing
    with pytest.raises(SystemExit, match="exceeds --cap-usd"):
        smoke.execute(rows, panel, book, cap_usd=0.001)


class InFlight:
    """A transport that answers every request and records the most requests it held at once; each request waits
    (up to `hold` s) for a second one to arrive, so a profile allowed two or more in flight shows it."""

    def __init__(self, hold: float = 2.0):
        import threading
        self.lock, self.second = threading.Lock(), threading.Event()
        self.now = self.peak = self.calls = 0
        self.hold = hold

    def __call__(self, url, headers, payload, timeout):
        with self.lock:
            self.now += 1
            self.calls += 1
            self.peak = max(self.peak, self.now)
            if self.now >= 2:
                self.second.set()
        self.second.wait(self.hold)
        with self.lock:
            self.now -= 1
        return 200, {}, json.dumps(openai_body(prompt=50, completion=5)).encode("utf-8")


def test_workers_is_the_profile_limit_lowered_by_the_callers_concurrency(panel):
    from procedures.readers.executor import workers
    llama, b2 = (panel.get(p).client_profile() for p in ("tierB_llama33_70b_hf_ovhcloud",
                                                          "B2_qwen3_235b_2507_hf_deepinfra"))
    assert (workers(None, llama), workers(4, llama), workers(1, llama)) == (1, 1, 1)
    assert (workers(None, b2), workers(2, b2), workers(8, b2)) == (4, 2, 4)
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="concurrency"):
            workers(bad, b2)


@pytest.mark.parametrize("pid, concurrency, serial", [
    ("tierB_llama33_70b_hf_ovhcloud", None, True),       # its route takes one request at a time
    ("tierB_llama33_70b_hf_ovhcloud", 4, True),          # a caller's concurrency never raises the route's limit
    ("B2_qwen3_235b_2507_hf_deepinfra", None, False),    # the default 4: the transport sees two or more at once
])
def test_live_run_keeps_at_most_the_profiles_max_workers_in_flight(panel, book, tmp_path, pid, concurrency, serial):
    pp = panel.get(pid)
    transport = InFlight(hold=0.3 if serial else 5.0)
    cells = [Cell(f"i{n}", "s", msg(f"question {n}")) for n in range(4)]
    report = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "l.jsonl", book=book, transport=transport,
                       sleep=lambda s: None, concurrency=concurrency)
    assert report["sent"] == transport.calls == 4
    assert transport.peak == 1 if serial else 2 <= transport.peak <= 4
    assert [r["request_hash"] for r in report["records"]] == \
        [request_hash(pp.profile_id, pp.fingerprint, c.messages, {}, 0) for c in cells]


class Timeouts:
    """Answers every request and records the read timeout the vendored client passed down."""

    def __init__(self):
        self.seen = []

    def __call__(self, url, headers, payload, timeout):
        self.seen.append((payload["model"], timeout))
        return 200, {}, json.dumps(openai_body(prompt=50, completion=5)).encode("utf-8")


def test_a_timeout_override_reaches_the_transport_of_its_profile_only(panel, book, tmp_path):
    from procedures.readers.executor import timeout_overrides
    c1, c2 = panel.get("C1_llama31_8b_hf_deepinfra"), panel.get("C2_qwen3_8b_hf_nscale_nothink")
    transport = Timeouts()
    cells = [Cell(f"i{n}", "s", msg(f"question {n}")) for n in range(3)]
    with timeout_overrides({c1.profile_id: 360}):
        for pp in (c1, c2):
            run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / f"{pp.profile_id}.jsonl", book=book,
                      transport=transport, sleep=lambda s: None)
    by_model = {}
    for model, timeout in transport.seen:
        by_model.setdefault(model, set()).add(timeout)
    assert by_model == {c1.client_profile().request_model: {360.0}, c2.client_profile().request_model: {120}}
    # outside the block the panel timeout is back
    run_cells([Cell("j", "s", msg("again"))], c1, cap_usd=1.0, ledger_path=tmp_path / "after.jsonl", book=book,
              transport=transport, sleep=lambda s: None)
    assert transport.seen[-1] == (c1.client_profile().request_model, 120)


def test_a_timeout_override_keeps_the_fingerprint_hashes_and_ledger_header_byte_identical(panel, book, tmp_path):
    from procedures.readers.executor import ledger_header, send_profile, timeout_overrides
    pp = panel.get("C1_llama31_8b_hf_deepinfra")
    base = pp.client_profile()
    with timeout_overrides({pp.profile_id: 360}):
        raised = send_profile(pp, base)
        header_inside = ledger_header(pp, book)
    assert raised.timeout == 360.0 and base.timeout == 120 and pp.client_profile().timeout == 120
    assert raised.fingerprint == base.fingerprint and raised.semantic_identity() == base.semantic_identity()
    assert header_inside == ledger_header(pp, book)
    cells = [Cell(f"i{n}", "s", msg(f"question {n}")) for n in range(2)]
    plain = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "plain.jsonl", book=book,
                      transport=Timeouts(), sleep=lambda s: None)
    with timeout_overrides({pp.profile_id: 360}):
        over = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "over.jsonl", book=book,
                         transport=Timeouts(), sleep=lambda s: None)
    first = [json.loads((tmp_path / f"{n}.jsonl").read_bytes().split(b"\n", 1)[0]) for n in ("plain", "over")]
    assert first[0].pop("created_at") and first[1].pop("created_at")    # the only field a second ledger may differ in
    assert json.dumps(first[0], sort_keys=True) == json.dumps(first[1], sort_keys=True)
    assert first[1]["profile_spec"]["timeout"] == 120
    assert [(r["request_hash"], r["profile_fingerprint"]) for r in plain["records"]] == \
        [(r["request_hash"], r["profile_fingerprint"]) for r in over["records"]]


@pytest.mark.parametrize("seconds", [120, 119.5, 60, 0, -1, float("inf"), float("nan"), True, "360"])
def test_a_timeout_override_only_raises(panel, seconds):
    from procedures.readers.executor import send_profile, timeout_overrides
    with pytest.raises(ValueError, match="only raises"):
        with timeout_overrides({"C1_llama31_8b_hf_deepinfra": seconds}):
            pass
    pp = panel.get("C1_llama31_8b_hf_deepinfra")   # nor above the floor but not above a profile's own timeout
    with timeout_overrides({pp.profile_id: 200}):
        with pytest.raises(ValueError, match="> 300"):
            send_profile(pp, replace(pp.client_profile(), timeout=300))


def test_parse_timeout_override():
    from procedures.readers.executor import parse_timeout_override
    assert parse_timeout_override("C1_llama31_8b_hf_deepinfra=360") == ("C1_llama31_8b_hf_deepinfra", 360.0)
    assert parse_timeout_override(" X =120.5") == ("X", 120.5)
    for bad, match in (("C1", "PROFILE_ID=SECONDS"), ("=360", "PROFILE_ID=SECONDS"), ("C1=abc", "not a number"),
                       ("C1=120", "only raises"), ("C1=90", "only raises")):
        with pytest.raises(ValueError, match=match):
            parse_timeout_override(bad)


LLAMA = "tierB_llama33_70b_hf_ovhcloud"


def test_a_workers_override_raises_the_in_flight_limit_of_its_profile_only(panel, book, tmp_path):
    """The override is the profile's limit inside the block (the caller's concurrency still lowers it); another
    profile, and the same profile outside the block, keep the panel's max_workers."""
    from procedures.readers.executor import send_profile, workers, workers_overrides
    llama, b2 = panel.get(LLAMA), panel.get("B2_qwen3_235b_2507_hf_deepinfra")
    with workers_overrides({LLAMA: 4}):
        raised = send_profile(llama, llama.client_profile())
        assert send_profile(b2, b2.client_profile()) == b2.client_profile()      # not overridden: unchanged
        assert (workers(None, raised), workers(2, raised), workers(8, raised)) == (4, 2, 4)
    assert llama.client_profile().max_workers == 1 and send_profile(llama, llama.client_profile()).max_workers == 1
    cells = [Cell(f"i{n}", "s", msg(f"question {n}")) for n in range(4)]
    peaks = {}
    for name, overrides, concurrency in (("over", {LLAMA: 4}, None), ("lowered", {LLAMA: 4}, 1),
                                         ("plain", None, None)):
        transport = InFlight(hold=5.0 if name == "over" else 0.3)
        with workers_overrides(overrides):
            report = run_cells(cells, llama, cap_usd=1.0, ledger_path=tmp_path / f"{name}.jsonl", book=book,
                               transport=transport, sleep=lambda s: None, concurrency=concurrency)
        assert report["sent"] == transport.calls == 4
        peaks[name] = transport.peak
    assert 2 <= peaks["over"] <= 4 and peaks["lowered"] == 1 and peaks["plain"] == 1


def test_a_workers_override_keeps_the_fingerprint_hashes_and_ledger_header_byte_identical(panel, book, tmp_path):
    from procedures.readers.executor import ledger_header, send_profile, workers_overrides
    pp = panel.get(LLAMA)
    base = pp.client_profile()
    with workers_overrides({LLAMA: 4}):
        raised = send_profile(pp, base)
        header_inside = ledger_header(pp, book)
    assert raised.max_workers == 4 and base.max_workers == 1
    assert raised.fingerprint == base.fingerprint and raised.semantic_identity() == base.semantic_identity()
    assert header_inside == ledger_header(pp, book)
    cells = [Cell(f"i{n}", "s", msg(f"question {n}")) for n in range(2)]
    plain = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "plain.jsonl", book=book,
                      transport=Timeouts(), sleep=lambda s: None)
    with workers_overrides({LLAMA: 4}):
        over = run_cells(cells, pp, cap_usd=1.0, ledger_path=tmp_path / "over.jsonl", book=book,
                         transport=Timeouts(), sleep=lambda s: None)
    first = [json.loads((tmp_path / f"{n}.jsonl").read_bytes().split(b"\n", 1)[0]) for n in ("plain", "over")]
    assert first[0].pop("created_at") and first[1].pop("created_at")
    assert json.dumps(first[0], sort_keys=True) == json.dumps(first[1], sort_keys=True)
    assert first[1]["profile_spec"]["max_workers"] == 1
    assert [(r["request_hash"], r["profile_fingerprint"]) for r in plain["records"]] == \
        [(r["request_hash"], r["profile_fingerprint"]) for r in over["records"]]


@pytest.mark.parametrize("n", [0, -1, 9, 100, True, 1.5, "4", None])
def test_a_workers_override_is_an_int_up_to_the_ceiling(n):
    from procedures.readers.executor import workers_overrides
    with pytest.raises(ValueError, match="only raises"):
        with workers_overrides({LLAMA: n}):
            pass


def test_a_workers_override_never_lowers_a_profiles_limit(panel):
    from procedures.readers.executor import send_profile, workers_overrides
    b2 = panel.get("B2_qwen3_235b_2507_hf_deepinfra")       # panel max_workers 4
    for n in (1, 3):
        with workers_overrides({b2.profile_id: n}):
            with pytest.raises(ValueError, match=r"int in \[4, 8\]"):
                send_profile(b2, b2.client_profile())
    with workers_overrides({b2.profile_id: 4}):                # equal to the panel value: allowed, no change
        assert send_profile(b2, b2.client_profile()).max_workers == 4


def test_parse_workers_override():
    from procedures.readers.executor import parse_workers_override
    assert parse_workers_override(f"{LLAMA}=4") == (LLAMA, 4)
    assert parse_workers_override(" X = 8") == ("X", 8)
    for bad, match in (("X", "PROFILE_ID=N"), ("=4", "PROFILE_ID=N"), ("X=abc", "not an int"), ("X=2.5", "not an int"),
                       ("X=0", "only raises"), ("X=9", "only raises")):
        with pytest.raises(ValueError, match=match):
            parse_workers_override(bad)

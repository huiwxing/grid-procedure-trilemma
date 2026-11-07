"""Shared pytest setup: the `procedures` package is imported from the repository root. Shared fixtures: `offline`
(no network, dummy credentials; the readers, matrix and scoring directories apply it to every test), the panel and
the price book, and the reader harness of the readers, matrix and scoring tests: synthetic items over a fake
two-revision document, a fake memory builder and a fake reader endpoint that answers per question."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys

import pytest

_ROOT = str(Path(__file__).resolve().parents[1])
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from procedures.scoring import render_lettered_options  # noqa: E402

KEY_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "HF_TOKEN")
DOC_NEW, DOC_OLD = "data/corpus/test/m99_rev2.txt", "data/corpus/test/m99_rev1.txt"
DOC_TEXT = ("Revision text. When the feed fails, notify the operator within the hour. Otherwise file a report by day "
            "end. The limit is 15 minutes or 2 hours. Frequency threshold 59.0 Hz, later 59.5 Hz, and 59.6 Hz.")
OPTS = ["notify the operator within the hour", "file a report by day end", "none of these"]
OPTS2 = ["open the tie breaker", "hold the schedule", "none of these"]
NIE_TAIL = " If no such requirement applies on that date, answer not_in_effect."
BUILD_USD = 0.01  # the fake builder's HIGH price of one dry-run summary


def _refuse_network(*args, **kwargs):
    raise AssertionError("an offline test tried to open a network connection")


@pytest.fixture
def offline(monkeypatch):
    """Every transport the inference layer could use refuses, and each key variable holds a dummy value."""
    import urllib.request
    import procedures.readers.client.client as client_module
    monkeypatch.setattr(client_module, "http_transport", _refuse_network)
    monkeypatch.setattr(urllib.request, "urlopen", _refuse_network)
    for name in KEY_ENVS:
        monkeypatch.setenv(name, f"dummy-{name.lower()}")


@pytest.fixture(scope="session")
def panel():
    from procedures.readers import load_panel
    return load_panel()


@pytest.fixture(scope="session")
def book():
    from procedures.readers import load_price_book
    return load_price_book()


def side_question(stem: str, options: list[str], as_of: bool) -> str:
    return f"{stem}, which of the following applies? {render_lettered_options(options)}." + (NIE_TAIL if as_of else "")


def make_items() -> list[dict]:
    s2 = "Under PJM Manual 99: Test Operations (Revision 2, effective 2026-01-01)"
    asof = "Under PJM Manual 99: Test Operations as in effect on {}"
    common = {"family": "PJM", "doc": "M-99", "evidence_path": DOC_NEW, "m0_paths": [DOC_NEW]}
    both = {"m0_paths": [DOC_OLD, DOC_NEW]}
    return [
        {**common, "item_id": "s2p:side:a", "type": "S2-side", "frame": "S2", "answer_type": "closed_set",
         "question": side_question(f"{s2}, when the feed fails", OPTS, False), "options": OPTS,
         "gold": {"value": OPTS[0], "unit": None}, "gold_label": "A", "other_side": {"value": OPTS[1], "unit": None},
         "pair_id": "s2p", "side": "a"},
        {**common, "item_id": "s2p:side:b", "type": "S2-side", "frame": "S2", "answer_type": "closed_set",
         "question": side_question(f"{s2}, when the feed is restored", OPTS, False), "options": OPTS,
         "gold": {"value": OPTS[1], "unit": None}, "gold_label": "B", "other_side": {"value": OPTS[0], "unit": None},
         "pair_id": "s2p", "side": "b"},
        {**common, "item_id": "s2p:value:a", "type": "S2-value", "frame": "S2", "answer_type": "numeric",
         "question": f"{s2}, when the feed fails, what is the time limit for the notice?",
         "gold": {"value": "15", "unit": "min"}, "other_side": {"value": "2", "unit": "h"},
         "pair_id": "s2p", "side": "a"},
        {**common, "item_id": "s2p:value:b", "type": "S2-value", "frame": "S2", "answer_type": "numeric",
         "question": f"{s2}, when the feed is restored, what is the time limit for the notice?",
         "gold": {"value": "2", "unit": "h"}, "other_side": {"value": "15", "unit": "min"},
         "pair_id": "s2p", "side": "b"},
        {**common, "item_id": "s3x:deadline", "type": "S3-deadline", "frame": "S3", "answer_type": "numeric",
         "question": f"{asof.format('2026-03-01')}, The operator shall confirm the outage within how many business "
                     f"days?{NIE_TAIL}",
         "gold": {"value": "30", "unit": "business day"}, "other_side": None, "pair_id": None, "side": None},
        {**common, **both, "item_id": "s1q:value:new", "type": "S1-value", "frame": "S1", "answer_type": "numeric",
         "question": f"{asof.format('2026-03-01')}, what is the threshold for frequency?{NIE_TAIL}",
         "gold": {"value": "59.5", "unit": "Hz"}, "other_side": {"value": "59.0", "unit": "Hz"},
         "pair_id": "s1q", "side": "new"},
        {**common, **both, "evidence_path": DOC_OLD, "item_id": "s1q:value:old", "type": "S1-value", "frame": "S1",
         "answer_type": "numeric", "question": f"{asof.format('2025-03-01')}, what is the threshold for frequency?"
                                               f"{NIE_TAIL}",
         "gold": {"value": "59.0", "unit": "Hz"}, "other_side": {"value": "59.5", "unit": "Hz"},
         "pair_id": "s1q", "side": "old"},
        {**common, "item_id": "l3u:a@cur", "type": "L3-side", "frame": "L3", "answer_type": "closed_set",
         "unit_id": "l3u", "cell": "a@cur", "branch": "a", "branch_role": "rule", "cell_kind": "rule",
         "question": side_question(f"{asof.format('2026-03-01')}, when the telemetry link to the control room drops",
                                   OPTS, True),
         "options": OPTS, "gold": {"value": OPTS[0], "unit": None}, "gold_label": "A",
         "other_side": {"value": OPTS[1], "unit": None}, "other_branch_outcome": OPTS[1], "pair_id": "s2p",
         "side": "a"},
        {**common, "item_id": "l3u:b@cur", "type": "L3-side", "frame": "L3", "answer_type": "closed_set",
         "unit_id": "l3u", "cell": "b@cur", "branch": "b", "branch_role": "exception", "cell_kind": "exception",
         "question": side_question(f"{asof.format('2026-03-01')}, when the link is back but the log is missing",
                                   OPTS, True),
         "options": OPTS, "gold": {"value": OPTS[1], "unit": None}, "gold_label": "B",
         "other_side": {"value": OPTS[0], "unit": None}, "other_branch_outcome": OPTS[0], "pair_id": "s2p",
         "side": "b"},
        {**common, **both, "evidence_path": DOC_OLD, "item_id": "l3v:a@old", "type": "L3-side", "frame": "L3",
         "answer_type": "closed_set", "unit_id": "l3v", "cell": "a@old", "branch": "a", "branch_role": "rule",
         "cell_kind": "not_in_effect",
         "question": side_question(f"{asof.format('2025-03-01')}, when a neighbour asks for relief", OPTS2, True),
         "options": OPTS2, "gold": {"value": "not_in_effect", "unit": None}, "gold_label": None,
         "other_side": {"value": OPTS2[0], "unit": None}, "other_branch_outcome": OPTS2[1], "pair_id": "l2v",
         "side": "a"},
        {**common, **both, "item_id": "l3v:a@new", "type": "L3-side", "frame": "L3", "answer_type": "closed_set",
         "unit_id": "l3v", "cell": "a@new", "branch": "a", "branch_role": "rule", "cell_kind": "rule",
         "question": side_question(f"{asof.format('2026-03-01')}, when a neighbour asks for relief", OPTS2, True),
         "options": OPTS2, "gold": {"value": OPTS2[0], "unit": None}, "gold_label": "A",
         "other_side": {"value": "not_in_effect", "unit": None}, "other_branch_outcome": OPTS2[1],
         "pair_id": "l2v", "side": "a"},
        {**common, **both, "evidence_path": DOC_OLD, "item_id": "l3w:b@old", "type": "L3-value", "frame": "L3",
         "answer_type": "numeric", "unit_id": "l3w", "cell": "b@old", "branch": "b", "branch_role": "parallel",
         "cell_kind": "parallel",
         "question": f"{asof.format('2025-03-01')}, when a multiple outage is screened, what is the threshold for "
                     f"frequency?{NIE_TAIL}",
         "gold": {"value": "59.0", "unit": "Hz", "surface": "59.0 Hz"},
         "other_side": {"value": "not_in_effect", "unit": None}, "other_branch_outcome": "59.6 Hz",
         "pair_id": "s1w:b", "side": "b"},
        {**common, **both, "item_id": "l3w:b@new", "type": "L3-value", "frame": "L3", "answer_type": "numeric",
         "unit_id": "l3w", "cell": "b@new", "branch": "b", "branch_role": "parallel", "cell_kind": "not_in_effect",
         "question": f"{asof.format('2026-03-01')}, when a multiple outage is screened, what is the threshold for "
                     f"frequency?{NIE_TAIL}",
         "gold": {"value": "not_in_effect", "unit": None}, "other_side": {"value": "59.0", "unit": "Hz"},
         "other_branch_outcome": "not_in_effect", "pair_id": "s1w:b", "side": "b"},
    ]


@pytest.fixture
def items():
    return make_items()


def _text_of(path: str) -> str:
    return DOC_TEXT


@pytest.fixture
def text_of():
    return _text_of


def _gold_answer(item: dict, letters: bool = True) -> dict:
    gold = item["gold"]
    if gold["value"] == "not_in_effect":
        return {"value": "not_in_effect"}
    if item.get("options"):
        return {"value": item["gold_label"] if letters else gold["value"], "unit": None, "evidence_span": ""}
    return {"value": gold["value"], "unit": gold.get("unit"), "evidence_span": "quoted"}


@pytest.fixture
def gold_answer():
    return _gold_answer


@dataclass(frozen=True)
class FakeMemory:
    """The procedures.memory.MemoryResult fields the harness reads."""
    text: str
    representation: str
    budget_tokens: int | None
    revisions_used: list
    provenance: list
    build_tokens: int
    cache_key: str
    notes: str
    build: dict | None = None


class FakeBuilder:
    """Stands in for procedures.memory.build_memory (same signature); records every call. Like procedures.memory, the
    summary memory (M5) is built live only when given a call ledger (unless `live_m5` is False, as with an unset
    builder credential); otherwise it returns a dry-run placeholder under the same cache_key, naming its budget as
    procedures.memory's stand-ins do. Closed book (B0) is the empty text."""

    def __init__(self, sizes: dict | None = None, live_m5: bool = True):
        self.calls = []
        self.sizes = sizes or {}
        self.live_m5 = live_m5

    def __call__(self, item, representation, budget_tokens, manifest, *, builder_profile=None, ledger=None):
        self.calls.append({"item_id": item["item_id"], "representation": representation, "budget": budget_tokens,
                           "builder_profile": builder_profile, "ledger": ledger})
        key = hashlib.sha256(json.dumps([item["m0_paths"], representation, budget_tokens, builder_profile])
                             .encode()).hexdigest()
        live = representation == "M5" and ledger is not None and self.live_m5
        if representation == "B0":
            text = ""
        elif representation in self.sizes:
            text = "grid " * self.sizes[representation]
        elif representation == "M5":
            text = (f"[{item['doc']}] summary written by {builder_profile}" if live
                    else f"[dry-run placeholder: no summary of {item['doc']} was built for a {budget_tokens}-token "
                         "memory]")
        else:
            text = f"[{item['doc']} {representation}] {DOC_TEXT}"
        build = None
        if representation == "M5":  # one sub-block summary per revision file, priced when dry (as MemoryResult.build),
            mode = "live" if live else "dry_run"  # with two cap levels (level 2: key "sum:<path>@2", "req:<path>@2")
            build = {"mode": mode, "builder_profile": builder_profile, "blocks": [
                {"block_key": f"sum:{p}", "request_hash": f"req:{p}", "mode": mode, "usd": BUILD_USD,
                 "levels": [{"cap_level": n, "block_key": f"sum:{p}" + ("" if n == 1 else f"@{n}"),
                             "request_hash": f"req:{p}" + ("" if n == 1 else f"@{n}"), "mode": mode, "usd": BUILD_USD}
                            for n in (1, 2)]}
                for p in item["m0_paths"]]}
        return FakeMemory(text=text, representation=representation, budget_tokens=budget_tokens,
                          revisions_used=[[p, "r"] for p in item["m0_paths"]], provenance=[],
                          build_tokens=4000 if representation == "M5" else 0, cache_key=key,
                          notes="dry_run" if build and not live else "fake", build=build)


@pytest.fixture
def fake_builder():
    return FakeBuilder


class FakeReader:
    """A reader endpoint: finds the item by the question closing the last user message and answers with
    `answer(item)` (a dict, sent as JSON, or a raw string)."""

    def __init__(self, items: list[dict], answer):
        self.by_question = {it["question"]: it for it in items}
        self.answer = answer
        self.calls = []

    def __call__(self, url, headers, payload, timeout):
        self.calls.append(payload)
        content = payload["messages"][-1]["content"].removesuffix(" /no_think")
        item = self.by_question[content.rsplit("Question: ", 1)[1]]
        answer = self.answer(item)
        text = answer if isinstance(answer, str) else json.dumps(answer)
        body = {"id": f"cmpl-{len(self.calls)}", "model": payload["model"].split(":")[0],
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 400, "completion_tokens": 20, "total_tokens": 420}}
        return 200, {"x-request-id": f"req-{len(self.calls)}"}, json.dumps(body).encode("utf-8")


@pytest.fixture
def fake_reader():
    return FakeReader

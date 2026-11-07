"""Mechanism checks of a reader-matrix run: the call ledgers show that every representation really executed and
reached the reader. `MockTransport` and `offline()` run the real reader matrix (procedures.matrix.run_matrix) with
every call answered offline (tests/memory/test_run_matrix_m6_m7.py): nothing leaves the process and no real
key is read (the key variables are set to a dummy inside `offline()` and restored after).
`check_run(mock=False)` runs the same checks on a paid run directory before any result is read
(scripts/sanity_check.py; its report is the run's sanity_report.json). The checks read the prompts, so they
need the raw call ledgers: the released ledgers are slimmed (no prompt messages) and `check_run` refuses them.

Representations: B0 closed book, M0 full text, M2 BM25 retrieval, M4 constraint sentences, M5 summary and M7 records
(built once per revision, build ledger), M6 query-focused compression (one call per question, query ledger).

* `select_items`: the items named by id prefixes; each prefix must name exactly one item.
* `MockTransport`: a procedures.readers.client transport. A reader request (the last user message ends with
  "Question: {question}", after the profile's suffix is removed) gets the item's gold as the answer schema's JSON
  (the gold letter on a lettered item, not_in_effect on a not_in_effect cell, value + unit otherwise); an M5
  sub-block build request (procedures.memory.summary.SUBBLOCK_PROMPT) gets the one-bullet placeholder summary
  `mock_summary(header, section, part)`; an M7 sub-block build request (procedures.memory.records.RECORD_PROMPT) gets
  `mock_records(section, part, text)`, one typed record whose source span is the start of the part's first non-blank
  line (so it is located verbatim); an M6 compression request (procedures.memory.compressor.COMPRESS_PROMPT) gets
  `mock_compression(passages)`, one line naming the hash of the passages it was given. Usage is the reference token
  count of the messages and of the reply, so the ledgers price mock usage; every response names its provider and
  system fingerprint "offline-mock". A request of the anthropic_messages adapter (its anthropic-version header) is
  answered in that adapter's response shape; a build or compression request to a model of `reasoning_field_models`
  is answered with a blank content and the answer in reasoning_content (the path
  procedures.memory.summary.answer_text maps for a profile with answer_from_reasoning_field, which covers build and
  compression answers as well as reader answers).
* `check_run`: reads the run directory back and checks
  - builds (build ledger, M5 and M7 records told apart by stage): every request carried its sub-block's text
    (procedures.memory.summary.sections / subblocks tile the revision), every sub-block of every as-of revision was
    built for each built-once representation of the run, and (mock) every answer is the mock one (build and
    compression answers are read as the memory reads them, procedures.memory.summary.answer_text; the observed
    records answered from the reasoning field are counted per ledger, `text_from_reasoning_field`);
  - compressions (query ledgers, M6): every request carried the item's question and the M2 retrieval at
    procedures.memory.compressor.INPUT_FACTOR x the budget verbatim, and (mock) every answer is the mock one;
  - per reader call: the representation reached the prompt (B0: no memory block; M0 / M2 / M4: the memory text,
    rebuilt independently with procedures.memory.build_memory and bound by the run manifest's cache key, occurs
    verbatim in the user message; M5 / M7: the memory assembled from the build ledger's answers
    (procedures.memory.summary.assemble / procedures.memory.records.assemble over
    procedures.memory.summary.answers_from_records, the plan bound by the manifest's cache key) occurs verbatim; M6:
    the memory assembled from the query ledger (procedures.memory.compressor.from_records) occurs verbatim; for
    M5 / M6 / M7 no other revision's header does; an empty memory passes when its block holds only
    procedures.readers.prompts.EMPTY_MEMORY, `memory_in_prompt`); the revisions used are the item's
    (M2 / M4 / M5 / M6 / M7: [(doc, revision)], also the revision of the gold's evidence file unless the gold is
    not_in_effect; M0: the revisions of `m0_paths`);
  - M7 records (mock: every mechanical check of procedures.memory.records is zero and every record's source span is
    located; paid: the counts are reported);
  - repeats (run manifest `repeats`): for a repeated item every answered (reader, stage) has the k repeats 0..k-1,
    with k distinct request hashes over identical messages; an item outside the subset has repeat 0 only; a stage's
    repeated items are procedures.matrix.run_matrix.repeated_for (a run with the 4k-slice re-read, k1_slice: the 4k
    slice re-reads the subset and the population); a repeat 0 sent once for two stages (identical messages, e.g. an
    M7 memory the same at 2k and 4k) is recorded under the first and counts for both;
  - feasibility (with the summary: the rows and estimate of the invocation that wrote it): a cell over its ceiling
    has no answer and is counted `not_sent_infeasible`, as many per reader as the invocation's estimate planned; in
    the mock run every feasible cell is answered, and the gold echo scores 1.0 over the answered cells of every
    reader x condition.
  With `mock=False` the provider, mock-answer and gold-echo checks are skipped (the mechanism check of a paid run,
  made before its results are read; it needs the raw call ledgers, with prompts).
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import socket
import urllib.request

from procedures import stored
from procedures.readers.ledger import read_ledger
from procedures.readers.tokens import estimate_message_tokens, estimate_tokens
from procedures.memory import summary as m5, compressor as m6, records as m7
from procedures.memory.summary import SUBBLOCK_PROMPT, parse_call_id, sections, subblocks
from procedures.corpus.revisions import asof_revisions, in_ladder_order
from procedures.readers.cells import BUILT_ONCE, LLM_BUILT, QUERY_BUILT, parse_stage
from procedures.matrix.run_matrix import repeated_for
from procedures.readers.prompts import EMPTY_MEMORY, MEMORY_CLOSE, MEMORY_OPEN, is_empty_memory

KEY_ENVS = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "HF_TOKEN")
MOCK = "offline-mock"
QUESTION = "Question: "
BUILD_OPEN = SUBBLOCK_PROMPT.split("{", 1)[0]
RECORD_OPEN = m7.RECORD_PROMPT.split("{", 1)[0]
COMPRESS_OPEN = m6.COMPRESS_PROMPT.split("{", 1)[0]
DOC_HEADER = "Document: "
SECTION_HEADER = "\nSection: "
PART_HEADER = "\nPart: "
PART_OPEN, PART_CLOSE = "\n<part>\n", "\n</part>"
PASSAGES_OPEN, PASSAGES_CLOSE = "\n<passages>\n", "\n</passages>"
SPAN_CHARS = 120
READBACK = {"M5": "m5_build_readback", "M7": "m7_build_readback", "M6": "m6_query_readback"}
IN_PROMPT = {"M5": "m5_summary_in_prompt", "M7": "m7_records_in_prompt", "M6": "m6_compression_in_prompt"}


def _refuse(*args, **kwargs):
    raise ConnectionError("the sanity run is offline: a network connection was attempted")


@contextmanager
def offline():
    """Block every network path the inference layer could take and set each key variable to a dummy value; both are
    restored on exit."""
    import procedures.readers.client.client as client_module
    patches = [(client_module, "http_transport"), (urllib.request, "urlopen"), (socket, "create_connection"),
               (socket.socket, "connect")]
    saved = [(obj, name, getattr(obj, name)) for obj, name in patches]
    env = {name: os.environ.get(name) for name in KEY_ENVS}
    try:
        for obj, name in patches:
            setattr(obj, name, _refuse)
        for name in KEY_ENVS:
            os.environ[name] = f"{MOCK}-{name.lower()}"
        yield
    finally:
        for obj, name, value in saved:
            setattr(obj, name, value)
        for name, value in env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def select_items(items: list[dict], prefixes: list[str]) -> list[dict]:
    """The items named by `prefixes` (item id prefixes, each matching exactly one item), ordered by item id."""
    chosen: dict[str, dict] = {}
    for prefix in prefixes:
        hits = [it for it in items if it["item_id"].startswith(prefix)]
        if len(hits) != 1:
            raise ValueError(f"item id prefix {prefix!r} matches {len(hits)} items, not exactly one")
        chosen[hits[0]["item_id"]] = hits[0]
    return [chosen[k] for k in sorted(chosen)]


def gold_answer(item: dict) -> dict:
    gold = item["gold"]
    if gold["value"] == "not_in_effect":
        return {"value": "not_in_effect", "unit": None, "evidence_span": ""}
    if item.get("options"):
        return {"value": item["gold_label"], "unit": None, "evidence_span": ""}
    return {"value": gold["value"], "unit": gold.get("unit"), "evidence_span": ""}


def mock_summary(header: str, section: str, part: str) -> str:
    return f"- [{MOCK} summary of {header}, {section}, {part}]"


def mock_records(section: str, part: str, text: str) -> str:
    """One typed record per sub-block: its subject names the sub-block, its source span is the first SPAN_CHARS
    characters of the part's first non-blank line (verbatim, so it is located); [] for a blank part."""
    line = next((x.strip() for x in text.splitlines() if x.strip()), "")
    if not line:
        return "[]"
    record = {name: None for name in m7.RECORD_FIELDS}
    record.update(subject=f"{MOCK} record of {section}, {part}", source_span=line[:SPAN_CHARS].strip())
    return json.dumps([record], ensure_ascii=False)


def mock_compression(passages: str) -> str:
    return f"[{MOCK} compression of passages {hashlib.sha256(passages.encode('utf-8')).hexdigest()[:12]}]"


def _between(content: str, head: str, tail: str) -> str:
    return content.split(head, 1)[1].rsplit(tail, 1)[0]


def _labels(content: str) -> tuple[str, str, str]:
    return (content.split(DOC_HEADER, 1)[1].split("\n", 1)[0], content.split(SECTION_HEADER, 1)[1].split("\n", 1)[0],
            content.split(PART_HEADER, 1)[1].split("\n", 1)[0])


class MockTransport:
    """Gold echo for reader requests; placeholder answers for M5 / M7 build and M6 compression requests (module
    docstring)."""

    def __init__(self, items: list[dict], suffixes: tuple[str, ...], reasoning_field_models=()):
        by_question: dict[str, dict] = {}
        for it in items:
            if it["question"] in by_question:
                raise ValueError(f"two items share a question: {by_question[it['question']]['item_id']}, "
                                 f"{it['item_id']}")
            by_question[it["question"]] = it
        self.by_question = by_question
        self.suffixes = tuple(s for s in suffixes if s)
        self.reasoning_field_models = frozenset(reasoning_field_models)
        self.calls = 0

    def reply(self, content: str) -> tuple[str, bool]:
        """(the mock answer, whether the request is a build or compression request)."""
        if content.startswith(BUILD_OPEN):
            return mock_summary(*_labels(content)), True
        if content.startswith(RECORD_OPEN):
            _, section, part = _labels(content)
            return mock_records(section, part, _between(content, PART_OPEN, PART_CLOSE)), True
        if content.startswith(COMPRESS_OPEN):
            return mock_compression(_between(content, PASSAGES_OPEN, PASSAGES_CLOSE)), True
        for suffix in self.suffixes:
            content = content.removesuffix(suffix)
        question = content.rsplit(QUESTION, 1)[1]
        return json.dumps(gold_answer(self.by_question[question])), False

    def __call__(self, url, headers, payload, timeout):
        self.calls += 1
        text, made = self.reply(payload["messages"][-1]["content"])
        prompt, completion = estimate_message_tokens(payload["messages"]), estimate_tokens(text)
        model = payload["model"].split(":")[0]
        if "anthropic-version" in headers:              # the anthropic_messages adapter's response shape
            body = {"id": f"{MOCK}-{self.calls}", "type": "message", "role": "assistant", "model": model,
                    "provider": MOCK, "system_fingerprint": MOCK, "content": [{"type": "text", "text": text}],
                    "stop_reason": "end_turn", "usage": {"input_tokens": prompt, "output_tokens": completion}}
        else:
            message = {"role": "assistant", "content": text}
            if made and model in self.reasoning_field_models:     # a reasoning-field profile: answer in reasoning
                message = {"role": "assistant", "content": "", "reasoning_content": text}
            body = {"id": f"{MOCK}-{self.calls}", "model": model, "provider": MOCK, "system_fingerprint": MOCK,
                    "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                              "total_tokens": prompt + completion}}
        return 200, {"x-request-id": f"{MOCK}-{self.calls}"}, json.dumps(body).encode("utf-8")


def unbuilt(revision, built: set[tuple[int, int]]) -> list[tuple[int, int]]:
    """(section, sub-block) indices of the revision's sub-block build requests
    (procedures.memory.summary.sections / subblocks, shared by M5 and M7) missing from `built`."""
    return [(sec.index, blk.index) for sec in sections(revision) for blk in subblocks(revision, sec)
            if (sec.index, blk.index) not in built]


def memory_in_prompt(text: str, user: str) -> bool:
    """The memory reached the reader's user message: verbatim, or, for an empty memory, as the memory block holding
    only procedures.readers.prompts.EMPTY_MEMORY (a flagged empty memory)."""
    if is_empty_memory(text):
        return "\n".join((MEMORY_OPEN, EMPTY_MEMORY, MEMORY_CLOSE)) in user
    return text in user


def _user(record: dict) -> str:
    return record["request"]["messages"][-1]["content"]


def _latest(path: Path) -> dict[str, dict]:
    return {r["request_hash"]: r for r in read_ledger(path)[1]}


def ledger_paths(run_dir: Path, kind: str) -> list[Path]:
    """The call ledgers of one kind (read / build / query) in `run_dir`, by their plain names (a released ledger is
    stored as <name>.gz; procedures.stored reads either)."""
    names = {stored.plain_name(p) for pattern in (f"{kind}__*.jsonl", f"{kind}__*.jsonl.gz")
             for p in Path(run_dir).glob(pattern)}
    return [Path(run_dir) / name for name in sorted(names)]


def without_prompts(run_dir: Path) -> list[str]:
    """The call ledgers of `run_dir` that hold a call record whose request carries no prompt messages: the released
    (slimmed) form, from which no prompt can be checked."""
    out = []
    for kind in ("build", "query", "read"):
        for path in ledger_paths(run_dir, kind):
            records = _latest(path).values()
            if any(isinstance(r.get("request"), dict) and "messages" not in r["request"] for r in records):
                out.append(path.name)
    return out


def _invocation(run_dir: Path, summary_name: str) -> dict | None:
    """The latest invocations.jsonl line whose outputs name `summary_name`, else the latest line with outputs (a
    summary rewritten by hand keeps the rows and estimate of the run's last scoring)."""
    path = run_dir / "invocations.jsonl"
    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
    scored = [x for x in lines if x.get("outputs")]
    return next((x for x in reversed(scored) if x["outputs"]["summary"] == summary_name), scored[-1] if scored else None)


def check_run(run_dir: Path, items: list[dict], index, build_memory, summary_name: str | None = None, *,
              mock: bool = True) -> dict:
    """Mechanism checks over a run directory (see the module docstring). Returns a JSON-able report whose `failures`
    list is empty when every check holds. `mock` False (a paid run, checked before its results are read): the
    provider, mock-answer and gold-echo checks are skipped, calls without a recorded request are counted, not
    checked; `summary_name` None skips the feasibility and score checks. Refuses a run whose ledgers are the released
    (slimmed) ones, which carry no prompts."""
    run_dir = Path(run_dir)
    slimmed = without_prompts(run_dir)
    if slimmed:
        raise ValueError(f"{run_dir.name}: released (slimmed) call ledgers without prompts: {', '.join(slimmed)}; "
                         "the mechanism check needs the raw ledgers (the run's recorded check is its "
                         "sanity_report.json)")
    by_id = {it["item_id"]: it for it in items}
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    keys = manifest["memory_cache_keys"]
    builder = next(iter(manifest["builder"] or {}), None)
    built_kinds = sorted({c["representation"] for c in manifest["conditions"]} & BUILT_ONCE)
    repeats = manifest.get("repeats") or {"k": 1, "item_ids": []}
    failures: list[dict] = []
    memo: dict[tuple, object] = {}
    llm_text: dict[tuple, str | None] = {}
    ledgers: dict[Path, dict] = {}
    m7_stats = {"memories": 0, "candidates": 0, "located": 0, "packed": 0,
                "checks": {k: 0 for k in ("subblocks_without_array", "subblocks_not_stop", "subblocks_over_quota",
                                          "malformed_records", "unlocated_records", "over_budget")}}

    def fail(check: str, **info) -> None:
        failures.append({"check": check, **info})

    def records(path: Path) -> dict:
        if path not in ledgers:
            ledgers[path] = _latest(path) if stored.stored(path).exists() else {}
        return ledgers[path]

    def memory(item: dict, rep: str, budget, maker):
        key = (item["item_id"], rep, budget, maker)
        if key not in memo:
            kw = ({"builder_profile": maker} if rep in BUILT_ONCE else
                  {"compressor_profile": maker} if rep in QUERY_BUILT else {})
            memo[key] = build_memory(item, rep, budget, index, **kw)
        return memo[key]

    def built_text(item: dict, rep: str, budget: int, maker: str) -> str | None:
        """The live LLM-made memory assembled from the run's ledgers (None: an answer is missing or unusable)."""
        asof = in_ladder_order(asof_revisions(item, index).revisions)
        key = (tuple(asof) if rep == "M5" else item["item_id"], rep, budget, maker)   # M6 / M7 depend on the question
        if key in llm_text:
            return llm_text[key]
        try:
            if rep == "M6":
                path = run_dir / f"query__{maker}.jsonl"
                mem = m6.from_records(asof, item["item_id"], item["question"], budget, maker, records(path), path)
            else:
                path = run_dir / f"build__{maker}.jsonl"
                calls = m5.plan(asof, budget, maker) if rep == "M5" else m7.plan(asof, maker)
                builds = m5.answers_from_records(calls, m5.panel().get(maker), records(path), path)
                if rep == "M5":
                    mem = m5.assemble(calls, builds, budget)
                else:
                    mem = m7.assemble(calls, builds, item["question"], budget)
                    m7_stats["memories"] += 1
                    for name in ("candidates", "located", "packed"):
                        m7_stats[name] += getattr(mem, name)
                    for name, n in mem.checks().items():
                        m7_stats["checks"][name] += n
                    if mock and (any(mem.checks().values()) or not mem.candidates):
                        fail("m7_mock_records", item_id=item["item_id"], budget=budget, checks=mem.checks(),
                             candidates=mem.candidates)
            llm_text[key] = mem.text
        except (RuntimeError, OSError, ValueError) as exc:
            fail(READBACK[rep], item_id=item["item_id"], budget=budget, error=repr(exc)[:300])
            llm_text[key] = None
        return llm_text[key]

    revisions = {(x.doc, x.revision): x for x in (index.by_path(p) for it in items for p in it["m0_paths"])}
    built: dict[tuple, set] = {}
    build_calls = {rep: 0 for rep in built_kinds}
    from_reasoning: dict[str, int] = {}     # observed build / query records answered from the reasoning field
    for path in ledger_paths(run_dir, "build"):
        for r in records(path).values():
            rep = r["stage"].split("@", 1)[0].removeprefix("build/")
            try:
                doc, rev, k, j = parse_call_id(r["item_id"])
            except ValueError:
                fail("build_call_id", call=r["item_id"])
                continue
            revision = revisions[(doc, rev)]
            secs = sections(revision)
            blocks = subblocks(revision, secs[k]) if k < len(secs) else ()
            block = blocks[j] if j < len(blocks) else None
            if block is None or revision.text()[block.char_start:block.char_end] not in _user(r):
                fail("build_prompt_block_text", revision=f"{doc}@{rev}", block=f"{k}.{j}", representation=rep)
            if r["outcome"] != "observed":
                continue
            build_calls[rep] = build_calls.get(rep, 0) + 1
            built.setdefault((rep, doc, rev), set()).add((k, j))
            answer, mapped = m5.answer_text(r, m5.panel().get(path.stem.split("__", 1)[1]))
            from_reasoning[path.stem] = from_reasoning.get(path.stem, 0) + mapped
            if mock and block is not None:
                want = (mock_summary(revision.header(), secs[k].label(), block.label()) if rep == "M5" else
                        mock_records(secs[k].label(), block.label(), revision.text()[block.char_start:block.char_end]))
                if answer != want:
                    fail("mock_build_answer", revision=f"{doc}@{rev}", block=f"{k}.{j}", representation=rep)
    want_builds = sorted({f"{it['doc']}@{it['revision']}" for it in items})
    complete = {rep: sorted(f"{doc}@{rev}" for (kind, doc, rev), got in built.items()
                            if kind == rep and not unbuilt(revisions[(doc, rev)], got)) for rep in built_kinds}
    for rep in built_kinds:
        if complete[rep] != want_builds:
            fail("build_revisions", representation=rep, got=complete[rep], want=want_builds)
    query_calls: dict[str, int] = {}
    for path in ledger_paths(run_dir, "query"):
        compressor = path.stem.split("__", 1)[1]
        for r in records(path).values():
            item = by_id[r["item_id"]]
            budget = int(r["stage"].rsplit("@", 1)[1])
            user = _user(r)
            passages = memory(item, "M2", m6.INPUT_FACTOR * budget, None).text
            if f"{QUESTION}{item['question']}\n" not in user or not passages or\
                    f"{PASSAGES_OPEN}{passages}{PASSAGES_CLOSE}" not in user:
                fail("m6_prompt_passages", item_id=item["item_id"], budget=budget, compressor=compressor)
            if r["outcome"] != "observed":
                continue
            query_calls[compressor] = query_calls.get(compressor, 0) + 1
            answer, mapped = m5.answer_text(r, m5.panel().get(compressor))
            from_reasoning[path.stem] = from_reasoning.get(path.stem, 0) + mapped
            if mock and answer != mock_compression(passages):
                fail("mock_query_answer", item_id=item["item_id"], budget=budget, compressor=compressor)
    counts: dict[str, int] = {}
    unrecorded = repeat_calls = 0
    sent: dict[tuple, dict[int, tuple]] = {}
    for path in ledger_paths(run_dir, "read"):
        for r in records(path).values():
            item = by_id[r["item_id"]]
            rep, budget, maker = parse_stage(r["stage"])
            cond = f"{rep}@{'full' if budget is None else budget}"
            counts[cond] = counts.get(cond, 0) + 1
            repeat_calls += r["repeat_idx"] > 0
            if mock and (r["outcome"] != "observed" or r["observed_provider"] != MOCK):
                fail("mock_answer", item_id=item["item_id"], stage=r["stage"], outcome=r["outcome"])
            if "messages" not in (r.get("request") or {}):
                unrecorded += 1
                continue
            sent.setdefault((path.stem, item["item_id"], r["stage"]), {})[r["repeat_idx"]] = (
                r["request_hash"], json.dumps(r["request"]["messages"], sort_keys=True))
            user = _user(r)
            suffix = r.get("user_message_suffix") or ""
            if not user.endswith(item["question"] + suffix):
                fail("question_last", item_id=item["item_id"], stage=r["stage"])
            want_rev = [[item["doc"], item["revision"]]]
            evidence = index.by_path(item["evidence_path"])   # not_in_effect S1 (revision pair) gold: other date's file
            if item["gold"]["value"] != "not_in_effect" and [evidence.doc, evidence.revision] != want_rev[0]:
                fail("evidence_revision", item_id=item["item_id"], evidence=[evidence.doc, evidence.revision])
            if rep == "B0":
                if MEMORY_OPEN in user or user != QUESTION + item["question"] + suffix:
                    fail("b0_no_memory", item_id=item["item_id"], profile=path.stem)
                continue
            mem = memory(item, rep, budget, maker)
            slot = f"{cond}/{maker}" if rep in QUERY_BUILT else cond
            if mem.cache_key != keys[f"{item['item_id']}|{slot}"]:
                fail("memory_rebuilt_differs", item_id=item["item_id"], condition=slot)
            used = [list(x) for x in mem.revisions_used]
            if rep in LLM_BUILT:
                asof = revisions[(item["doc"], item["revision"])]
                text = built_text(item, rep, budget, maker)
                if text is None or not memory_in_prompt(text, user):
                    fail(IN_PROMPT[rep], item_id=item["item_id"], profile=path.stem, budget=budget)
                if any(f"[{x.header()}]" in user for x in revisions.values() if x is not asof):
                    fail(f"{rep.lower()}_other_revision_in_prompt", item_id=item["item_id"], profile=path.stem)
            elif not memory_in_prompt(mem.text, user):
                fail("memory_text_in_prompt", item_id=item["item_id"], condition=cond, profile=path.stem)
            if rep == "M0":
                want = sorted([index.by_path(p).doc, index.by_path(p).revision] for p in item["m0_paths"])
                if sorted(used) != want:
                    fail("m0_revisions", item_id=item["item_id"], got=used, want=want)
            elif used != want_rev:
                fail("asof_revisions", item_id=item["item_id"], condition=cond, got=used, want=want_rev)
    first_reads = {}          # (ledger, item, messages) -> the repeat-0 request recorded under some stage
    for (profile, iid, _), got in sent.items():
        if 0 in got:
            first_reads.setdefault((profile, iid, got[0][1]), got[0])
    for (profile, iid, stage), got in sorted(sent.items()):
        rep, budget, _ = parse_stage(stage)
        want = list(range(repeats["k"])) if iid in repeated_for(repeats, rep, budget) else [0]
        if 0 not in got:      # its repeat 0 shared the request of another stage's (the same memory text)
            shared = first_reads.get((profile, iid, next(iter(got.values()))[1]))
            got = {0: shared, **got} if shared else got
        if sorted(got) != want:
            fail("repeat_set", profile=profile, item_id=iid, stage=stage, got=sorted(got), want=want)
        elif len({h for h, _ in got.values()}) != len(got) or len({m for _, m in got.values()}) != 1:
            fail("repeat_requests", profile=profile, item_id=iid, stage=stage)
    scores, infeasible = {}, {}
    if summary_name is not None:
        summary = json.loads((run_dir / summary_name).read_text(encoding="utf-8"))
        invocation = _invocation(run_dir, summary_name)
        if invocation is None:
            fail("summary_invocation", summary=summary_name)
        else:
            rows = stored.read_jsonl(run_dir / invocation["outputs"]["rows"])
            estimate = json.loads((run_dir / invocation["estimate"]).read_text(encoding="utf-8"))
            for row in rows:
                pid = row["profile_id"]
                if not row["access"]["feasible_at_B"]:
                    infeasible[pid] = infeasible.get(pid, 0) + 1
                    if row["outcome"] != "not_sent_infeasible" or row["provenance"]["request_hash"] is not None:
                        fail("infeasible_sent", profile=pid, item_id=row["item_id"], stage=row["stage"])
                elif mock and row["outcome"] != "observed":
                    fail("feasible_unanswered", profile=pid, item_id=row["item_id"], stage=row["stage"],
                         outcome=row["outcome"])
            planned = {pid: e["infeasible"] for pid, e in estimate["readers"].items() if e["infeasible"]}
            if infeasible != planned:
                fail("infeasible_count", got=infeasible, planned=planned)
        for key, s in sorted(summary["conditions"].items()):
            cap = s.get("capability", {})
            n, correct = cap.get("n") or 0, cap.get("correct") or 0
            skipped = s["outcomes"].get("not_sent_infeasible", 0)
            scores[key] = {"n": cap.get("n"), "correct": cap.get("correct"), "outcomes": s["outcomes"]}
            if mock and (n + skipped != len(items) or correct != n):
                fail("gold_echo_score", condition=key, n=cap.get("n"), correct=cap.get("correct"))
    return {"reader_calls_by_condition": dict(sorted(counts.items())), "repeat_calls": repeat_calls,
            "calls_without_request": unrecorded, "builds": complete, "build_calls": build_calls,
            "query_calls": dict(sorted(query_calls.items())), "m7_records": m7_stats,
            "text_from_reasoning_field": dict(sorted(from_reasoning.items())),
            "infeasible_by_reader": dict(sorted(infeasible.items())), "scores": scores, "failures": failures}

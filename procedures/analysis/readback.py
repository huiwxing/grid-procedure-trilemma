"""Memory read-back for the two exploratory memory sensitivities of the paper's analysis: summary memories (M5)
whose build was truncated, and the share of typed constraint records (M7) whose source span is found in the revision
text. Zero network: it reads the runs' memory request maps, their build call ledgers (already loaded by
procedures.analysis.data.load_run, latest record per request hash) and the revision texts.

The read-back recomputes, from the recorded build calls, the quantities the memory build check counted, so the
analysis sees the memories that check counted:
* M5 (`m5_build_truncated`): per M5 memory of a run's request map (cache_key -> revisions, budget, builder, ledger),
  the plan `procedures.memory.summary.plan(revisions, budget, builder)` read from the build ledger by
  `procedures.memory.summary.answers_from_records`; the memory is `build_truncated` when a
  sub-block's answer used finished truncated (the flag `procedures.memory.summary.assemble` carries: a truncated
  answer is sent again at a larger output cap, cap level 2, and this one was still cut there).
* M7 (`m7_located`): per builder and revision of the maps' M7 memories, `m7.plan([revision], builder)` read by
  `m5.answers_from_records`, then `m7.record_units`: candidates (records after the per-revision de-duplication) and
  located (source span found in the revision text); share = located / candidates. An M7 build depends on neither the
  budget nor the question, so a revision's count is the same for every memory of the budget ladder.
Each memory's request map must agree with the plan: every block key the map lists is a block key of the plan; a
memory whose map was written from another plan (planned by a different code version) is listed in `unreadable` with
the reason, and the report computes no sensitivity over a representation with an unreadable memory. A missing ledger
record or an unanswered one still raises (as in the mechanism sanity check: a memory that was not really built is an
error, never a gap to skip). Revision texts come from a RevisionIndex (the frozen corpus manifest and the external
dates the runs were planned with).

Needs the raw build ledgers: the released (slimmed) ledgers replace the builders' outputs by their length and sha256,
and `memory_readback` refuses them with that message.
"""
from __future__ import annotations

from procedures.memory import summary as m5, records as m7
from procedures.corpus.revisions import RevisionIndex, RevisionText

from procedures.analysis.data import Run
from procedures.analysis.stats import r4

BUILD_TRUNCATED = "build_truncated"


def revision_key(doc: str, revision: str) -> str:
    return f"{doc}@{revision}"


def _revision(index: RevisionIndex, doc: str, revision: str) -> RevisionText:
    found = [r for r in index.ladder(doc) if r.revision == revision]
    if len(found) != 1:
        raise KeyError(f"revision {revision!r} of {doc!r} is not on the index's ladder")
    return found[0]


class PlanMismatch(ValueError):
    """A memory's request map lists requests the current plan does not make (it was planned by a different code
    version): it cannot be read back and is listed as such."""


def _check_plan(run: Run, key: str, entry: dict, calls) -> None:
    planned = {k for c in calls for k in c.keys}
    listed = {q["block_key"] for q in entry["requests"]}
    if not listed <= planned:
        raise PlanMismatch(f"{run.run_id}: memory {key[:12]} lists {len(listed - planned)} block keys that the plan of "
                           f"{entry['revisions']} by {entry['profile']} does not make: the map and the plan disagree")


def _records(run: Run, entry: dict) -> dict:
    records = run.ledgers.get(entry["ledger"])
    if records is None:
        raise ValueError(f"{run.run_id}: no ledger {entry['ledger']} for the memories its request map lists")
    return records


def _entries(runs: list[Run], representation: str):
    for run in runs:
        for key, entry in sorted(((run.memory_requests or {}).get("memories") or {}).items()):
            if entry["representation"] == representation:
                yield run, key, entry


def _unreadable(run: Run, key: str, entry: dict, error: PlanMismatch) -> dict:
    return {"run_id": run.run_id, "memory": key, "representation": entry["representation"],
            "profile": entry["profile"], "reason": str(error)}


def m5_build_truncated(runs: list[Run], index: RevisionIndex) -> tuple[dict[str, bool], list[dict]]:
    """(M5 memory cache_key -> whether the memory is build_truncated, the memories that cannot be read back), over
    every run's request map."""
    out: dict[str, bool] = {}
    unreadable = []
    for run, key, entry in _entries(runs, "M5"):
        revisions = [_revision(index, doc, rev) for doc, rev in entry["revisions"]]
        calls = m5.plan(revisions, entry["budget_tokens"], entry["profile"])
        try:
            _check_plan(run, key, entry, calls)
        except PlanMismatch as error:
            unreadable.append(_unreadable(run, key, entry, error))
            continue
        builds = m5.answers_from_records(calls, m5.panel().get(entry["profile"]), _records(run, entry),
                                         f"{run.run_id}/{entry['ledger']}")
        truncated = any(BUILD_TRUNCATED in b.flags for b in builds)
        if out.setdefault(key, truncated) != truncated:
            raise ValueError(f"memory {key[:12]} reads back differently in two runs")
    return out, unreadable


def m7_located(runs: list[Run], index: RevisionIndex) -> tuple[dict[str, dict[str, dict]], list[dict]]:
    """(builder -> revision key -> {candidates, located, share}, the memories that cannot be read back), over the M7
    memories of every run's request map."""
    todo: dict[tuple[str, str], tuple[Run, dict, RevisionText]] = {}
    unreadable = []
    for run, key, entry in _entries(runs, "M7"):
        revisions = [_revision(index, doc, rev) for doc, rev in entry["revisions"]]
        try:
            _check_plan(run, key, entry, m7.plan(revisions, entry["profile"]))
        except PlanMismatch as error:
            unreadable.append(_unreadable(run, key, entry, error))
            continue
        for r in revisions:
            todo.setdefault((entry["profile"], revision_key(r.doc, r.revision)), (run, entry, r))
    out: dict[str, dict[str, dict]] = {}
    for (builder, rkey), (run, entry, revision) in sorted(todo.items(), key=lambda kv: kv[0]):
        calls = m7.plan([revision], builder)            # a revision's calls are the same in any revision set
        builds = m5.answers_from_records(calls, m5.panel().get(builder), _records(run, entry),
                                         f"{run.run_id}/{entry['ledger']}")
        units, _, _ = m7.record_units(calls, builds)
        located = sum(u.provenance["located"] for u in units)
        out.setdefault(builder, {})[rkey] = {"candidates": len(units), "located": located,
                                             "share": r4(located / len(units)) if units else None}
    return out, unreadable


def check_build_outputs(runs: list[Run]) -> None:
    """Raises when a build ledger that the runs' M5 / M7 memories name holds the released (slimmed) form of its
    records (`text_sha256` in place of the builder's output `text`): the read-back reads the outputs."""
    for run in runs:
        names = sorted({e["ledger"] for e in ((run.memory_requests or {}).get("memories") or {}).values()
                        if e["representation"] in ("M5", "M7")})
        for name in names:
            if any("text" not in r and "text_sha256" in r for r in (run.ledgers.get(name) or {}).values()):
                raise ValueError(f"{run.run_id}: {name} is a released (slimmed) build ledger: the builders' outputs "
                                 "are replaced by their length and sha256, so the memory read-back cannot run (it "
                                 "needs the raw build ledgers)")


def memory_readback(runs: list[Run], index: RevisionIndex, source: dict | None = None) -> dict:
    """{"m5_build_truncated", "m7_located", "unreadable", "source"}: what
    procedures.analysis.report.analyse(readback=...) reads; `unreadable` lists the memories whose map disagrees with
    the current plan (never silently skipped: the report does not compute a sensitivity over a representation with
    unreadable memories). Raises on released (slimmed) build ledgers (`check_build_outputs`)."""
    check_build_outputs(runs)
    truncated, bad5 = m5_build_truncated(runs, index)
    located, bad7 = m7_located(runs, index)
    return {"m5_build_truncated": truncated, "m7_located": located, "unreadable": bad5 + bad7,
            "source": source or {}}

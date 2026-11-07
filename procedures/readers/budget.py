"""Spend approval: a written grant of new spend for one batch of paid runs, with a cap per run label (the budget
mechanism, not a result).

``data/budget/approval_<id>.json`` (write-once, ``write_approval``): {approval_id, granted_usd, semantics = "new spend
for this batch", arms: {run label: cap}, created, notes}; the arms' caps sum to at most granted_usd. Spend already
made before the approval (other run directories) is not part of it; a run directory bound to it counts whole.

A paid matrix run binds its run directory to the approval (``ApprovalGuard.bind``: one line in
``approval_<id>.bindings.jsonl`` beside the approval, and one in the run directory's ``approvals.jsonl``); its run
label must be one of the arms. Every reservation of the run (procedures.readers.executor through SpendGuard) then
goes through ``ApprovalGuard.reserve``: under the approval's cross-process lock file (``approval_<id>.lock``,
procedures.readers.ledger.FileLock) it re-reads every bound run directory — each call ledger's usd_counted
(pending_unknown included) and each send log's open reservations, whichever process made them — and refuses
(``ApprovalExceeded``, a SpendCapExceeded) when that total + the new reservation would pass granted_usd, or the total
of the runs bound under this run's label (its arm) + it would pass the arm's cap; only then does the call ledger
reserve (and write the reservation to its send log) inside the same lock. Two executors on two arms of one approval
therefore cannot both take the same room, and a crashed executor's open reservations stay counted until its next run
settles them.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import threading

from procedures.readers.ledger import SETTLE_TOLERANCE_USD, FileLock, JsonlTail, LedgerError, SendLog, SpendCapExceeded, utc_now
from procedures import ROOT

APPROVALS_ROOT = ROOT / "data" / "budget"
SEMANTICS = "new spend for this batch"
FIELDS = ("approval_id", "granted_usd", "semantics", "arms", "created", "notes")
APPROVAL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


class ApprovalError(ValueError):
    """An approval file is missing, malformed, or does not cover the run."""


class ApprovalExceeded(SpendCapExceeded):
    """A reservation or pre-flight would pass the approval's grant or the run's arm."""


def _usd(value, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value < float("inf"):
        raise ApprovalError(f"{what} must be a non-negative finite number of USD, got {value!r}")
    return float(value)


@dataclass(frozen=True)
class Approval:
    approval_id: str
    granted_usd: float
    arms: dict
    created: str
    notes: str
    path: Path

    @property
    def bindings_path(self) -> Path:
        return self.path.with_name(f"approval_{self.approval_id}.bindings.jsonl")

    @property
    def lock_path(self) -> Path:
        return self.path.with_name(f"approval_{self.approval_id}.lock")


def approval_file(approval_id: str, root: Path = APPROVALS_ROOT) -> Path:
    if APPROVAL_ID.fullmatch(approval_id or "") is None:
        raise ApprovalError(f"approval id {approval_id!r} must match {APPROVAL_ID.pattern}")
    return Path(root) / f"approval_{approval_id}.json"


def _checked(doc: dict, path: Path) -> Approval:
    if not isinstance(doc, dict) or sorted(doc) != sorted(FIELDS):
        raise ApprovalError(f"{path}: an approval has exactly the fields {list(FIELDS)}")
    if doc["semantics"] != SEMANTICS:
        raise ApprovalError(f"{path}: semantics must be {SEMANTICS!r}, got {doc['semantics']!r}")
    if path.name != f"approval_{doc['approval_id']}.json":
        raise ApprovalError(f"{path}: file name does not match approval_id {doc['approval_id']!r}")
    granted = _usd(doc["granted_usd"], f"{path}: granted_usd")
    arms = doc["arms"]
    if not isinstance(arms, dict) or not arms:
        raise ApprovalError(f"{path}: arms must map at least one run label to its cap")
    arms = {label: _usd(cap, f"{path}: arm {label!r}") for label, cap in arms.items()}
    if sum(arms.values()) > granted:
        raise ApprovalError(f"{path}: the arms' caps sum to ${sum(arms.values()):.4f} > granted ${granted:.4f}")
    if not isinstance(doc["created"], str) or not isinstance(doc["notes"], str):
        raise ApprovalError(f"{path}: created and notes are strings")
    return Approval(doc["approval_id"], granted, arms, doc["created"], doc["notes"], path)


def load_approval(approval_id: str, root: Path = APPROVALS_ROOT) -> Approval:
    path = approval_file(approval_id, root)
    if not path.is_file():
        raise ApprovalError(f"no approval {approval_id!r}: {path} does not exist (scripts/approval.py new writes "
                            "the file once the spend is granted)")
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApprovalError(f"{path}: malformed JSON ({exc})") from exc
    return _checked(doc, path)


def write_approval(approval_id: str, granted_usd: float, arms: dict, notes: str,
                   root: Path = APPROVALS_ROOT) -> Path:
    """Write approval_<id>.json once (refuses an existing file: an approval is never edited; a larger grant is a
    new approval whose arms name the same run labels)."""
    path = approval_file(approval_id, root)
    doc = {"approval_id": approval_id, "granted_usd": granted_usd, "semantics": SEMANTICS, "arms": dict(arms),
           "created": utc_now(), "notes": notes}
    _checked(doc, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True) + "\n")
    return path


class _RunTally:
    """What one run directory has committed: every ledger's usd_counted (and its pending_unknown part) plus every
    send log's open reservations, read incrementally (send logs before ledgers, as procedures.readers.ledger does)."""

    def __init__(self, run_dir: Path):
        self.run_dir = Path(run_dir)
        self._ledgers: dict[str, JsonlTail] = {}
        self._sends: dict[str, SendLog] = {}
        self.counted = 0.0
        self.pending_unknown = 0.0

    def refresh(self) -> None:
        for path in sorted(self.run_dir.glob("*.jsonl.sends")):
            self._sends.setdefault(path.name, SendLog(path)).refresh()
        for path in sorted(self.run_dir.glob("*__*.jsonl")):
            for line in self._ledgers.setdefault(path.name, JsonlTail(path)).read():
                if line.get("kind") == "header":
                    continue
                if line.get("kind") != "call" or "usd_counted" not in line:
                    raise LedgerError(f"{path}: not a call record: {line!r:.200}")
                self.counted += float(line["usd_counted"])
                self.pending_unknown += float(line.get("usd_pending_unknown") or 0.0)

    @property
    def in_flight(self) -> float:
        return sum(log.in_flight_usd() for log in self._sends.values())

    def summary(self) -> dict:
        return {"counted_usd": self.counted, "pending_unknown_usd": self.pending_unknown,
                "in_flight_usd": self.in_flight, "committed_usd": self.counted + self.in_flight}


class ApprovalGuard:
    """One executor's handle on an approval for one run directory (module docstring)."""

    def __init__(self, approval: Approval, run_dir: Path, run_label: str):
        if run_label not in approval.arms:
            raise ApprovalError(f"approval {approval.approval_id!r} has no arm for run label {run_label!r} "
                                f"(arms: {sorted(approval.arms)})")
        self.approval, self.run_dir, self.run_label = approval, Path(run_dir), run_label
        self.arm_cap_usd = approval.arms[run_label]
        self._lock = threading.Lock()
        self._runs: dict[str, _RunTally] = {}

    def _file_lock(self) -> FileLock:
        return FileLock(self.approval.lock_path)

    def _bound(self) -> dict[Path, str]:
        """run directory -> run label of every binding (its path relative to the approval file's directory)."""
        out = {}
        if self.approval.bindings_path.exists():
            for line in self.approval.bindings_path.read_text(encoding="utf-8").splitlines():
                entry = json.loads(line)
                out[(self.approval.path.parent / entry["run_dir"]).resolve()] = entry["run_label"]
        return out

    def bind(self) -> None:
        """Bind this run directory to the approval (idempotent; a directory bound under another label refuses)."""
        with self._lock, self._file_lock():
            here = self.run_dir.resolve()
            bound = self._bound()
            if here in bound:
                if bound[here] != self.run_label:
                    raise ApprovalError(f"{here} is bound to approval {self.approval.approval_id!r} as "
                                        f"{bound[here]!r}, not {self.run_label!r}")
                return
            entry = {"run_dir": Path(os.path.relpath(here, self.approval.path.parent.resolve())).as_posix(),
                     "run_label": self.run_label, "bound_at": utc_now()}
            for path, line in ((self.approval.bindings_path, entry),
                               (self.run_dir / "approvals.jsonl",
                                {"approval_id": self.approval.approval_id, "run_label": self.run_label,
                                 "bound_at": entry["bound_at"]})):
                with open(path, "a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n")
                    stream.flush()
                    os.fsync(stream.fileno())

    def _tally(self) -> dict:
        """(call under both locks) The approval's committed spend over its bound runs, this run counted even before
        it is bound (a dry run's view)."""
        here = self.run_dir.resolve()
        runs = {**self._bound(), here: self.run_label}
        for path in runs:
            self._runs.setdefault(str(path), _RunTally(path)).refresh()
        per = {str(p): self._runs[str(p)].summary() for p in sorted(runs)}
        return {"approval_id": self.approval.approval_id, "granted_usd": self.approval.granted_usd,
                "run_label": self.run_label, "arm_cap_usd": self.arm_cap_usd,
                "committed_usd": sum(v["committed_usd"] for v in per.values()),
                "pending_unknown_usd": sum(v["pending_unknown_usd"] for v in per.values()),
                "in_flight_usd": sum(v["in_flight_usd"] for v in per.values()),
                "run_committed_usd": per[str(here)]["committed_usd"],
                "arm_committed_usd": sum(per[str(p)]["committed_usd"] for p, label in runs.items()
                                         if label == self.run_label),
                "runs": per}

    def tally(self) -> dict:
        with self._lock, self._file_lock():
            return self._tally()

    def _refuse(self, tally: dict, usd: float, what: str) -> None:
        if tally["committed_usd"] + usd > self.approval.granted_usd + SETTLE_TOLERANCE_USD:
            raise ApprovalExceeded(f"approval {self.approval.approval_id}: committed ${tally['committed_usd']:.4f} "
                                   f"(pending unknown ${tally['pending_unknown_usd']:.4f}, in flight "
                                   f"${tally['in_flight_usd']:.4f}) + {what} ${usd:.4f} exceeds granted "
                                   f"${self.approval.granted_usd:.4f}")
        if tally["arm_committed_usd"] + usd > self.arm_cap_usd + SETTLE_TOLERANCE_USD:
            raise ApprovalExceeded(f"approval {self.approval.approval_id}, arm {self.run_label}: committed "
                                   f"${tally['arm_committed_usd']:.4f} + {what} ${usd:.4f} exceeds the arm's cap "
                                   f"${self.arm_cap_usd:.4f}")

    def check(self, usd: float, what: str) -> dict:
        """Pre-flight: refuse if `usd` more would pass the grant or the arm; returns the tally it read."""
        with self._lock, self._file_lock():
            tally = self._tally()
            self._refuse(tally, usd, what)
            return tally

    def reserve(self, ledger, usd: float, **kw):
        """ledger.reserve(usd, **kw) if the approval and the arm have room, all under the approval lock."""
        with self._lock, self._file_lock():
            self._refuse(self._tally(), usd, "reservation")
            return ledger.reserve(usd, **kw)


def approval_status(approval: Approval) -> dict:
    """Read-only tally of an approval over the runs bound to it (no lock: a snapshot for a person to read)."""
    runs = {}
    if approval.bindings_path.exists():
        for line in approval.bindings_path.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            tally = _RunTally(approval.path.parent / entry["run_dir"])
            tally.refresh()
            runs[entry["run_dir"]] = {"run_label": entry["run_label"], **tally.summary()}
    committed = sum(r["committed_usd"] for r in runs.values())
    return {"approval_id": approval.approval_id, "granted_usd": approval.granted_usd, "semantics": SEMANTICS,
            "arms": {label: {"cap_usd": cap, "committed_usd": sum(r["committed_usd"] for r in runs.values()
                                                                   if r["run_label"] == label)}
                     for label, cap in approval.arms.items()},
            "committed_usd": committed, "left_usd": approval.granted_usd - committed,
            "pending_unknown_usd": sum(r["pending_unknown_usd"] for r in runs.values()),
            "in_flight_usd": sum(r["in_flight_usd"] for r in runs.values()), "runs": runs}

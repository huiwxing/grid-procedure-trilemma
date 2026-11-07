"""Record a spend approval for a batch of paid runs, or read its status.

    py -3.12 scripts/approval.py new --id batch --granted-usd 150 --arm main=100 --arm records=15 --arm cloud=35 \
        --notes "who approved what, when"
    py -3.12 scripts/approval.py status --id batch

The arms are run labels (scripts/run_matrix.py --label; the paper's three run labels had one arm each). `new` writes
data/budget/approval_<id>.json once (procedures.readers.budget.write_approval): the amount of new spend
granted for this batch and a cap per run label (arm); the caps sum to at most the grant. A paid run
(scripts/run_matrix.py --execute --approval <id>) binds its run directory to the approval, and every reservation of
every run bound to it is checked against the grant and the arm's cap under one cross-process lock. `status` prints
what the bound runs have committed, hold pending and have in flight, per arm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT  # noqa: E402
from procedures.items.publish import display  # noqa: E402
from procedures.readers.budget import APPROVALS_ROOT, approval_status, load_approval, write_approval  # noqa: E402


def cmd_new(a) -> int:
    arms = {}
    for arm in a.arms:
        label, sep, usd = arm.partition("=")
        if not sep or label in arms:
            raise SystemExit(f"--arm takes LABEL=USD once per label, got {arm!r}")
        arms[label] = float(usd)
    path = write_approval(a.id, a.granted_usd, arms, a.notes, root=a.root)
    print(json.dumps({"written": display(path, ROOT), **json.loads(path.read_text(encoding="utf-8"))}, indent=1,
                     ensure_ascii=False))
    return 0


def cmd_status(a) -> int:
    print(json.dumps(approval_status(load_approval(a.id, root=a.root)), indent=1, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("new", help="record a new approval (write-once)")
    n.add_argument("--id", required=True, help="approval id, named by --approval on paid runs")
    n.add_argument("--granted-usd", type=float, required=True, help="new spend granted for this batch")
    n.add_argument("--arm", action="append", dest="arms", required=True, metavar="LABEL=USD",
                   help="cap of one run label (repeatable); the caps sum to at most --granted-usd")
    n.add_argument("--notes", required=True, help="who approved what, when")
    t = sub.add_parser("status", help="committed / pending / in-flight spend per arm")
    t.add_argument("--id", required=True)
    for sp in (n, t):
        sp.add_argument("--root", type=Path, default=APPROVALS_ROOT, help="approvals directory (default data/budget)")
    a = ap.parse_args(argv)
    return {"new": cmd_new, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())

"""Write the executable analysis spec of the paper: the population, tiers, comparisons and sensitivities to compute.

The spec (procedures.analysis.spec.make_paper_spec) is made from the main run's item file, the panel and the constants
of procedures.analysis.spec / procedures.matrix.grids; scripts/report.py --spec executes it. Zero network, zero spend;
written as <out-dir>/spec.json (default out/spec/). The paper's report executed data/analysis/spec.json, made from
data/items/questions_with_earlier_cells.jsonl (the default --items): the spec this script writes equals it except
panel_sha256 (the hash of the panel file as it is now). The supply / power check of the L3 items (enough
units, cells and owner families, and the share of units with a literal exception or not-in-effect cell) is computed
on the item file and on the population, recorded in the spec and printed.

  py -3.12 scripts/make_spec.py --verify-items data/items/questions.jsonl

The spec records the path of its item file, relative to the repository root.

--verify-items (repeatable) checks the written spec against another item file (procedures.analysis.spec.check_spec:
e.g. the 244-cell item file the main run read) and prints its population cells present / missing. Then:
scripts/report.py --spec <file>.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from procedures import ROOT
from procedures.analysis.capability import supply
from procedures.analysis.spec import check_spec, in_population, make_paper_spec, write_spec
from procedures.readers import load_panel

DEFAULT_ITEMS = ROOT / "data" / "items" / "questions_with_earlier_cells.jsonl"


def read_items(path: Path) -> tuple[list[dict], str]:
    data = Path(path).read_bytes().replace(b"\r\n", b"\n")
    return [json.loads(x) for x in data.decode("utf-8").splitlines() if x.strip()], hashlib.sha256(data).hexdigest()


def shown(path: Path) -> str:
    path = Path(path).resolve()
    return (path.relative_to(ROOT) if path.is_relative_to(ROOT) else path).as_posix()


def main(argv: list[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", type=Path, default=DEFAULT_ITEMS, help="item file the spec is made from")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "out" / "spec", help="output directory (default out/spec)")
    ap.add_argument("--verify-items", type=Path, action="append", default=[],
                    help="another item file to check the written spec against (repeatable)")
    a = ap.parse_args(argv)
    items, sha = read_items(a.items)
    panel = load_panel()
    population = [it for it in items if in_population(it)]
    supply_check = {"item_file": supply(items)["L3"], "population": supply(population)["L3"]}
    spec = make_paper_spec(items, shown(a.items), sha, panel, supply_check)
    path = write_spec(spec, a.out_dir)
    pop = spec["population"]
    print(f"spec: {shown(path)}")
    print(f"population: {pop['cells']} cells / {pop['units']} units; by family {pop['units_by_family']}")
    print(f"tiers: {spec['tiers']}")
    print(f"reference arms: {spec['reference_arms']}; builders: {spec['builders']}")
    print(f"k = 1 scope: {spec['repeats']['k1_scope_conditions']}")
    for part, got in supply_check.items():
        print(f"supply / power check (L3 units) on the {part}: {got['literal_units']} / {got['units']} units literal = "
              f"{got['literal_share']} (cell_kind diagnostic {got['cell_kind_share']}); met {got['met']}")
        print("  sensitivities: " + "; ".join(f"{k} {v['literal_units']}/{v['units']} = {v['literal_share']}"
                                              for k, v in got["sensitivity"].items()))
    print("comparisons: " + ", ".join(c["id"] for c in spec["comparisons"]))
    print("sensitivities: " + ", ".join(s["id"] for s in spec["sensitivities"]))
    for other in a.verify_items:
        rows, _ = read_items(other)
        got = check_spec(spec, rows, panel)
        units = {it["unit_id"] for it in rows if it["item_id"] in set(got["present"])}
        print(f"verified on {shown(other)}: {len(got['present'])} population cells / {len(units)} units present, "
              f"{len(got['missing'])} missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

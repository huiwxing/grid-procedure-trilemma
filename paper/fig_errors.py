"""Fig. 3: what the constrained panel's wrong answers are, per path at the 4k budget (shares from exploratory.py).

  py -3.12 paper/fig_errors.py [paper/exploratory.json] [--out-dir <dir>]

Stacked bars of wrong-answer shares for the four constrained readers at the 4k context budget, per answered read
(exploratory), from exploratory.json -> error_mix_C_tier_primary: wrong branch, "no branch applies" (none
of these or not in effect), abstention and other. Knowledge paths first, in the order of Sec. III, then rule sentences
and the no-document control. Compression covers completed questions only. Nothing is recomputed. Writes
fig_errors.pdf into --out-dir (default out/paper/; the figure of the paper is paper/fig_errors.pdf).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "paper"
B1 = "B1_dsv41flash_direct_nothink"
ROWS = [
    ("read/M2@4000", "Retrieval"),
    (f"read/M5@4000/{B1}", "Summary"),
    (f"read/M7@4000/{B1}", "Records"),
    ("read/M6@4000", r"Compression$^{\ddagger}$"),
    (None, None),
    ("read/M4@4000", "Rule sentences"),
    ("read/B0@full", "No document"),
]
SEGMENTS = [
    ("branch", "Wrong branch", "#4a4a4a"),
    ("none_or_not_in_effect", "“No branch applies”", "#d62728"),
    ("abstain", "Abstains", "#f4a261"),
    ("other_wrong", "Other", "#c9c9c9"),
]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("exploratory", type=Path, nargs="?", default=HERE / "exploratory.json",
                    help="the file exploratory.py wrote (default: paper/exploratory.json)")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where fig_errors.pdf goes (default out/paper)")
    args = ap.parse_args(argv)
    mix = json.loads(args.exploratory.read_text(encoding="utf-8"))["error_mix_C_tier_primary"]
    plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Times New Roman"],
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, ax = plt.subplots(figsize=(3.5, 1.72))
    fig.subplots_adjust(left=0.235, right=0.975, bottom=0.22, top=0.843)
    ticks, labels = [], []
    y = 0.0
    for key, label in ROWS:
        if key is None:
            ax.axhline(y - 0.2, color="#bbbbbb", lw=0.5)
            y += 0.4
            continue
        left = 0.0
        for seg, _, colour in SEGMENTS:
            width = mix[key][seg]["est"]
            ax.barh(y, width, left=left, color=colour, height=0.62, edgecolor="white", lw=0.4)
            left += width
        if abs(left - mix[key]["wrong"]) >= 0.001:
            print(f"warning: the wrong-answer shares of {key} sum to {left:.4f}, not its wrong share "
                  f"{mix[key]['wrong']:.4f}", file=sys.stderr)
        ax.text(left + 0.008, y, f"{mix[key]['wrong']:.2f}", va="center", fontsize=6.8)
        ticks.append(y)
        labels.append(label)
        y += 1.0
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y - 0.45, -0.55)
    ax.set_xlim(0, 0.68)
    ax.set_xticks([0, 0.2, 0.4, 0.6], ["0", "0.2", "0.4", "0.6"])
    ax.set_xlabel("Share of all answers that are wrong", fontsize=7, labelpad=2)
    ax.tick_params(axis="y", length=0, labelsize=7)
    ax.tick_params(axis="x", labelsize=6.8, length=2)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.legend(handles=[Patch(color=c, label=lab) for _, lab, c in SEGMENTS], loc="upper center",
               bbox_to_anchor=(0.55, 0.999), ncol=4, frameon=False, fontsize=6.8,
               handlelength=0.9, handletextpad=0.35, columnspacing=0.8)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "fig_errors.pdf"
    fig.savefig(target, metadata={"CreationDate": None})
    print(f"{target} ({sum(1 for k, _ in ROWS if k)} rows; 3.5 x 1.72 in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

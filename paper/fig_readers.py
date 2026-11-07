"""Fig. 2: accuracy of every server and constrained reader on the four 4k paths, from the Table I values.

  py -3.12 paper/fig_readers.py [paper/numbers.json paper/printed_values.json] [--out-dir <dir>]

One row per reader of the server and constrained panels (the deployment choice), one marker per 4k knowledge path:
retrieval, the DeepSeek-built summary and records, and per-question compression. x is question accuracy exactly as
printed in Table I (printed_values.json -> main_table_direct_3dp, accuracy_3dp). Compression for readers that did not
finish every question is drawn hollow at its accuracy on completed questions, with a cross at task success when
unfinished tasks count as failures (printed_values.json -> completion_adjusted_readers, success_with_incomplete_zero).
Vertical lines mark the open-weight reference (REF-B) and the cloud comparator on the full text (P0-A), from
numbers.json. The legend's token costs are literals copied from Table I's token row, rounded. Nothing is
recomputed. Writes fig_readers.pdf into --out-dir (default out/paper/; the figure of the paper is
paper/fig_readers.pdf).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "out" / "paper"
PATHS = [  # legend label with Table I token cost, Table I key, colour, marker, size, vertical dodge (row units)
    ("Retrieval, original text: ≈4k tokens", "M2", "#1f77b4", "o", 4.2, -0.2),
    ("Summary, built once: ≈4k", "M5_B", "#d62728", "s", 3.7, -0.067),
    ("Records, built once: ≈4k", "M7_B", "#8c2d5e", "D", 3.4, 0.067),
    ("Compression, per question: 10–15k", "M6", "#ff7f0e", "^", 4.2, 0.2),
]
GROUPS = [
    ("Server panel",
     [("DeepSeek-V4.1-Flash", ""), ("Qwen3-235B-A22B", ""), ("Llama-3.3-70B", "")]),
    ("Constrained panel",
     [("gpt-oss-20b", "24 GB"), ("Qwen3-8B", "24 GB"), ("Qwen3-32B", "48 GB"),
      ("Llama-3.1-8B", "24 GB")]),
]
W, H = 3.5, 2.82   # inches: one IEEE column
QUESTIONS = 158   # the test set; a reader that completed fewer compression questions is drawn hollow


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("numbers", type=Path, nargs="?", default=HERE / "numbers.json",
                    help="the file extract_numbers.py wrote (default: paper/numbers.json)")
    ap.add_argument("printed_values", type=Path, nargs="?", default=HERE / "printed_values.json",
                    help="the file printed_values.py wrote (default: paper/printed_values.json)")
    ap.add_argument("--out-dir", type=Path, default=OUT, help="where fig_readers.pdf goes (default out/paper)")
    args = ap.parse_args(argv)
    numbers = json.loads(args.numbers.read_text(encoding="utf-8"))
    printed = json.loads(args.printed_values.read_text(encoding="utf-8"))
    table = {row["name"]: {k: (float(v["accuracy_3dp"]) if v else None) for k, v in row["methods"].items()}
             for row in printed["main_table_direct_3dp"]}
    unfinished = {c["name"]: c["arms"]["M6"]["success_with_incomplete_zero"]
                  for c in printed["completion_adjusted_readers"] if c["arms"]["M6"]["observed_questions"] < QUESTIONS}
    ref, cloud = numbers["primary"]["REF-B"]["est"], numbers["primary"]["P0-A"]["est"]

    plt.rcParams.update({"font.size": 7, "font.family": "serif", "font.serif": ["Times New Roman"],
                         "axes.linewidth": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([1.0 / W, 0.32 / H, 2.44 / W, 2.07 / H])

    y, shade, plotted, crosses = 0.0, False, 0, 0
    for header, readers in GROUPS:
        ax.text(0.402, y, header, ha="left", va="center", fontsize=7, weight="bold", color="#222222")
        y += 0.85
        for name, gpu in readers:
            if shade:
                ax.axhspan(y - 0.44, y + 0.44, color="#f3f3f3", lw=0, zorder=0)
            shade = not shade
            xs = [table[name][key] for _, key, *_ in PATHS]
            ax.plot([min(xs), max(xs)], [y, y], color="#bdbdbd", lw=0.6, zorder=1)
            for (_, key, colour, marker, ms, dodge), x in zip(PATHS, xs):
                hollow = key == "M6" and name in unfinished
                ax.plot(x, y + dodge, marker=marker, ms=ms, ls="", mfc="white" if hollow else colour,
                        mec=colour if hollow else "white", mew=0.8 if hollow else 0.3, zorder=3)
                plotted += 1
                if hollow:
                    ax.plot(unfinished[name], y + dodge, marker="x", ms=3.6, ls="", color="#444444",
                            mew=0.8, zorder=3)
                    crosses += 1
            ax.text(-0.02, y, name, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    fontsize=7)
            if gpu:
                ax.text(-0.02, y + 0.36, gpu, transform=ax.get_yaxis_transform(), ha="right",
                        va="center", fontsize=6.6, color="#555555")
            y += 0.95
        y += 0.15

    ax.axvline(ref, color="#333333", ls=(0, (3, 2)), lw=0.7, zorder=2)
    ax.axvline(cloud, color="#333333", ls=(0, (1, 1.4)), lw=0.8, zorder=2)
    ax.text(ref - 0.006, -0.62, "Open-weight ref.", ha="right", va="bottom", fontsize=6.8, color="#333333")
    ax.text(cloud - 0.006, -0.62, "Cloud,\nfull text", ha="right", va="bottom", fontsize=6.8, color="#333333",
            linespacing=1.0)

    ax.set_ylim(y - 0.55, -0.62)
    ax.set_yticks([])
    ax.set_xlim(0.40, 0.95)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9], ["0.4", "0.5", "0.6", "0.7", "0.8", "0.9"])
    ax.set_xlabel("Question accuracy (4k context budget)", fontsize=7, labelpad=2)
    ax.tick_params(axis="x", labelsize=6.8, length=2.2, pad=1.5)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e3e3e3", lw=0.4)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    handles = [Line2D([], [], marker=m, ls="", color=c, markersize=ms, label=lab) for lab, _, c, m, ms, _ in PATHS]
    handles.append(Line2D([], [], marker="x", ls="", color="#444444", markersize=3.6, mew=0.8,
                          label="Unfinished counted as failed"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
               fontsize=6.8, handlelength=0.8, handletextpad=0.3, columnspacing=1.0, borderaxespad=0.1)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / "fig_readers.pdf"
    fig.savefig(target, metadata={"CreationDate": None})
    print(f"{target} ({plotted} Table I values, {crosses} completion crosses; {W} x {H} in)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

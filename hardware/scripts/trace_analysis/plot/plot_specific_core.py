#!/usr/bin/env python3
"""Per-core stall detail report.  Generates one PNG + PDF per requested core.

Trace-backed subplots are deprecated in the official CSV-only workflow.
The old trace-derived code is intentionally kept in commented form below for
reference, but the active plotting path uses only stall_timeseries CSV data.

Usage:
    python plot_specific_core.py <csv> <core_id ...> [options]

Positional arguments:
    csv              Path to stall_timeseries_benchmark.csv
                     (produced by _gen_stall_timeseries_batch.py)
    core             One or more core IDs to plot (e.g. 0 1 2 3)

Options:
    --output-dir DIR   Output directory for PNGs [default: <csv-dir>/plots]
    --prefix STR       Filename prefix [default: core_detail]
                       Output files: <prefix>_core<id>.png, .pdf
    --traces-dir DIR   Deprecated compatibility flag. Ignored in the
                       official CSV-only workflow.
    --section N        Keep only rows from section N (repeatable: --section 0 --section 1)
    --group N          Keep only rows from group N   (repeatable)
    --tile N           Keep only rows from tile N    (repeatable)
    --show             Display figures interactively instead of just saving

Examples:
    # Single core, section 1 only
    python plot_specific_core.py results/stall_timeseries_benchmark.csv 0 --section 1

    # All 4 cores in tile 0
    python plot_specific_core.py results/stall_timeseries_benchmark.csv 0 1 2 3 --section 1

    # Custom output directory
    python plot_specific_core.py results/stall_timeseries_benchmark.csv 0 \\
        --section 1 --output-dir my_plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plot_output_paths import pdf_path_for_png

from _stall_plot_common import (
    STALL_CATEGORIES, STALL_COLORS, stall_label,
    load_rows, filter_rows, split_stall_kind,
    set_cycle_ticks,
    plot_categories_cumulative,
)


# ── Series builder ────────────────────────────────────────────────────────────

def build_core_series(rows, core_id):
    core_rows = sorted((r for r in rows if r["core"] == core_id), key=lambda r: r["cycle"])
    if not core_rows:
        raise ValueError(f"No rows for core {core_id}")

    cycles = np.array([r["cycle"] for r in core_rows], dtype=int)
    issue = np.array([1.0 if r["state"] == "issue" else 0.0 for r in core_rows])
    stall = 1.0 - issue

    cat_current = {}
    for cat in STALL_CATEGORIES:
        cat_current[cat] = np.array([
            1.0 if r["state"] == "stall" and cat in split_stall_kind(r["stall_kind"]) else 0.0
            for r in core_rows
        ])

    present = [c for c in STALL_CATEGORIES if np.any(cat_current[c] > 0)]

    return {
        "core_id": core_id, "cycles": cycles,
        "issue_current": issue, "stall_current": stall,
        "issue_cumulative": np.cumsum(issue), "stall_cumulative": np.cumsum(stall),
        "category_current": cat_current,
        "category_cumulative": {c: np.cumsum(v) for c, v in cat_current.items()},
        "present_categories": present,
    }


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_strip(series):
    """Build a 1-D array mapping each cycle to 0 (issue) or 1..N (stall cat)."""
    cat_indices = {c: i for i, c in enumerate(STALL_CATEGORIES)}
    strip = np.zeros(len(series["cycles"]))
    for ci, cyc_val in enumerate(series["issue_current"]):
        if cyc_val > 0:
            strip[ci] = 0
        else:
            for cat in STALL_CATEGORIES:
                if series["category_current"][cat][ci] > 0:
                    strip[ci] = cat_indices[cat] + 1
                    break
    return strip


# DEPRECATED: trace-backed subplot helpers are disabled in the official
# CSV-only workflow. Keep the old implementation below for reference only.
# def _find_trace(csv_path, cid, fallback_dir, traces_dir=None):
#     trace_path = locate_trace_file(csv_path, cid, traces_dir=traces_dir)
#     if trace_path is None:
#         for cand in fallback_dir.glob("trace_hart_*.trace"):
#             if cand.name == f"trace_hart_0x{cid:08x}.trace":
#                 trace_path = cand
#                 break
#     return trace_path


# ── Single-figure renderer ────────────────────────────────────────────────────

def write_core_detail(png_path, csv_path, series, traces_dir=None):
    """All-in-one core detail figure.

        Subplot order (top to bottom):
            1. Colour strip — execution state over time (stall subtypes)
            2. Breakdown bar — total cycle summary
            3. Cumulative stall causes
    """
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    cid = series["core_id"]
    cycles = series["cycles"]
    cats = series["present_categories"]

    # DEPRECATED: trace-backed instruction-type and outstanding-load panels are
    # disabled in the official CSV-only workflow. Keep the old implementation
    # below for reference only.
    # trace_path = _find_trace(csv_path, cid, png_path.parent.parent, traces_dir=traces_dir)
    # mem = build_memory_request_series(trace_path, cycles) if trace_path else None

    nrows = 3
    ratios = [1, 1.2, 2.5]
    fig_h = 14

    fig, axes = plt.subplots(nrows, 1, figsize=(20, fig_h),
                             gridspec_kw={"height_ratios": ratios},
                             constrained_layout=True)
    row = 0

    # ── 1. colour strip ──────────────────────────────────────────────────
    strip = _build_strip(series)
    strip_colors = [STALL_COLORS["issue"]] + [STALL_COLORS[c] for c in STALL_CATEGORIES]
    strip_cmap = ListedColormap(strip_colors)
    axes[row].imshow(strip[np.newaxis, :], aspect="auto", interpolation="nearest",
                     cmap=strip_cmap, vmin=0, vmax=len(STALL_CATEGORIES), origin="lower")
    axes[row].set_yticks([])
    axes[row].set_title("Execution State by Stall Cause")
    set_cycle_ticks(axes[row], np.arange(len(cycles)), cycles)
    legend_handles = [
        Patch(facecolor=STALL_COLORS["issue"], edgecolor="black", linewidth=0.5,
              label="Issuing"),
    ] + [Patch(facecolor=STALL_COLORS[c], edgecolor="black", linewidth=0.5,
              label=stall_label(c)) for c in cats]
    axes[row].legend(handles=legend_handles, loc="lower left",
                     bbox_to_anchor=(0, 1.02), frameon=True, fancybox=True,
                     edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                     ncol=len(legend_handles),
                     handleheight=1.15, handlelength=2.5)
    axes[row].set_xlabel("Cycle")
    row += 1

    # DEPRECATED: trace-backed instruction-type strip is disabled in the
    # official CSV-only workflow. Keep the old implementation below for
    # reference only.
    # itype_strip = np.zeros(len(cycles))
    # for ci in range(len(cycles)):
    #     if series["issue_current"][ci] > 0:
    #         itype_strip[ci] = mem["itype_current"][ci]
    # itype_colors = [ITYPE_COLORS[k] for k in ITYPE_ORDER]
    # itype_cmap = ListedColormap(itype_colors)
    # axes[row].imshow(itype_strip[np.newaxis, :], aspect="auto", interpolation="nearest",
    #                  cmap=itype_cmap, vmin=0, vmax=len(ITYPE_ORDER) - 1, origin="lower")
    # axes[row].set_yticks([])
    # axes[row].set_title("Instruction Type")
    # set_cycle_ticks(axes[row], np.arange(len(cycles)), cycles)
    # itype_handles = [Patch(facecolor=ITYPE_COLORS[k], edgecolor="black",
    #                        linewidth=0.5, label=ITYPE_LABELS[k])
    #                  for k in ITYPE_ORDER]
    # axes[row].legend(handles=itype_handles, loc="lower left",
    #                  bbox_to_anchor=(0, 1.02), frameon=True, fancybox=True,
    #                  edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
    #                  ncol=len(ITYPE_ORDER),
    #                  handleheight=1.15, handlelength=2.5)
    # axes[row].set_xlabel("Cycle")
    # row += 1

    # ── 3. stacked horizontal bar — total cycle breakdown ────────────────
    total = float(len(cycles))
    issue_total = float(series["issue_cumulative"][-1])
    bar_labels = ["Issuing"] + [stall_label(c) for c in cats]
    bar_values = [issue_total] + [float(series["category_cumulative"][c][-1]) for c in cats]
    bar_colors = [STALL_COLORS["issue"]] + [STALL_COLORS[c] for c in cats]

    ax_bar = axes[row]
    left = 0.0
    for lbl, val, col in zip(bar_labels, bar_values, bar_colors):
        pct = val / total * 100 if total else 0
        display_lbl = lbl
        count_text = f"{int(val)} cyc."
        ax_bar.barh("Cycles", val, left=left, color=col, edgecolor="white", linewidth=0.5,
                     label=f"{lbl} ({pct:.1f}%)")
        if val / total > 0.10:
            ax_bar.text(left + val / 2, 0, f"{display_lbl}\n{pct:.1f}%\n{count_text}", ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        elif val / total > 0.04:
            medium_text = f"{pct:.1f}%\n{count_text}"
            if display_lbl in {"RAW", "LSU"}:
                medium_text = f"{display_lbl}\n{pct:.1f}%\n{count_text}"
            ax_bar.text(left + val / 2, 0, medium_text, ha="center", va="center",
                        fontsize=9, fontweight="bold", color="white")
        left += val
    ax_bar.set_xlim(0, total)
    ax_bar.set_xlabel("Cycle amount")
    ax_bar.set_title(f"Cycle Breakdown  ({int(total)} total cycles)")
    ax_bar.legend(loc="upper left", frameon=True,
                  fancybox=True, edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                  ncol=1, handleheight=1.15, handlelength=2.5)
    row += 1

    # ── 4. cumulative stall causes ───────────────────────────────────────
    plot_categories_cumulative(axes[row], cycles, series["category_cumulative"], cats,
                               "Cumulative Stall Causes")
    axes[row].set_xlabel("Cycle")
    set_cycle_ticks(axes[row], cycles)
    row += 1

    # DEPRECATED: trace-backed outstanding-loads subplot is disabled in the
    # official CSV-only workflow. Keep the old implementation below for
    # reference only.
    # plot_outstanding_loads(
    #     axes[row], cycles, mem["outstanding_loads"],
    #     "Outstanding Loads"
    # )
    # axes[row].set_xlabel("Cycle")
    # set_cycle_ticks(axes[row], cycles)
    # row += 1

    fig.suptitle(f"Core {cid} Detail Report", fontsize=17, fontweight="bold")

    fig.savefig(png_path, dpi=96, bbox_inches="tight")
    pdf_path = pdf_path_for_png(png_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, png_path, pdf_path


# ── CLI & main ────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-core stall detail report.")
    p.add_argument("csv", help="CSV from _gen_stall_timeseries_batch.py")
    p.add_argument("core", type=int, nargs="+", help="Core ID(s) to plot")
    p.add_argument("--output-dir", default=None, help="Defaults to <csv-dir>/plots")
    p.add_argument("--traces-dir", default=None, help="Deprecated compatibility flag; ignored")
    p.add_argument("--prefix", default="core_detail")
    p.add_argument("--section", type=int, action="append", help="Filter by section")
    p.add_argument("--group", type=int, action="append", help="Filter by group")
    p.add_argument("--tile", type=int, action="append", help="Filter by tile")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path}")

    out = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)

    rows = filter_rows(load_rows(csv_path),
                       section=args.section, group=args.group, tile=args.tile)
    if not rows:
        raise SystemExit("No rows after filtering")

    figs = []
    for cid in args.core:
        series = build_core_series(rows, cid)
        png = out / f"{args.prefix}_core{cid}.png"
        fig, png_out, pdf_out = write_core_detail(png, csv_path, series, traces_dir=args.traces_dir)
        figs.append(fig)
        print(f"Wrote {png_out}")
        print(f"Wrote {pdf_out}")

    if args.show:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Per-tile stall detail report with optional overview page.

When requested, produces an overview PNG + PDF. When tile IDs are given,
also produces one detail PNG + PDF per tile.

Usage:
    python _plot_specific_tile.py <csv> [tile_id ...] [options]

Positional arguments:
    csv              Path to stall_timeseries_benchmark.csv
                     (produced by _gen_stall_timeseries_batch.py)
    tile             Zero or more tile IDs for detail pages (optional).

Options:
    --output-dir DIR   Output directory for PNGs [default: <csv-dir>/plots]
    --prefix STR       Filename prefix [default: tile_detail]
                       Output files: <prefix>_overview.png/.pdf,
                                     <prefix>_tile<id>.png/.pdf
    --window N         Cycle window for windowed aggregation [default: 64]
    --overview         Also generate the cluster overview PNG + PDF
    --traces-dir DIR   Directory containing .trace files for the
                       outstanding-loads subplot.  If omitted, the script
                       looks next to the CSV.
    --section N        Keep only rows from section N (repeatable: --section 0 --section 1)
    --group N          Keep only rows from group N   (repeatable)
    --show             Display figures interactively instead of just saving

Examples:
    # Overview only (no tile detail)
    python _plot_specific_tile.py results/stall_timeseries_benchmark.csv --section 1 --overview

    # Overview + detail for tiles 0 and 1
    python _plot_specific_tile.py results/stall_timeseries_benchmark.csv 0 1 --section 1 --overview

    # Wider aggregation window, custom output dir
    python _plot_specific_tile.py results/stall_timeseries_benchmark.csv 0 \\
        --section 1 --window 128 --output-dir my_plots/
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _stall_plot_common import (
    STALL_CATEGORIES, STALL_COLORS, stall_label,
    ITYPE_COLORS, ITYPE_LABELS, ITYPE_ORDER,
    load_rows, filter_rows, split_stall_kind,
    locate_trace_file, build_memory_request_series,
    set_cycle_ticks, plot_outstanding_loads,
    plot_categories_current, plot_categories_cumulative,
)


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_rows(rows, window, context_field="tile"):
    min_cycle = min(r["cycle"] for r in rows)
    max_cycle = max(r["cycle"] for r in rows)
    nw = (max_cycle - min_cycle) // window + 1

    issue = np.zeros(nw)
    stall = np.zeros(nw)
    mix = {c: np.zeros(nw) for c in STALL_CATEGORIES}
    ctxs = {}

    for r in rows:
        wi = (r["cycle"] - min_cycle) // window
        cv = r[context_field]
        if cv not in ctxs:
            ctxs[cv] = {
                "issue": np.zeros(nw), "stall": np.zeros(nw),
                "mix": {c: np.zeros(nw) for c in STALL_CATEGORIES},
            }
        if r["state"] == "issue":
            issue[wi] += 1; ctxs[cv]["issue"][wi] += 1
        else:
            stall[wi] += 1; ctxs[cv]["stall"][wi] += 1
            cats = split_stall_kind(r["stall_kind"])
            w = 1.0 / len(cats)
            for c in cats:
                mix[c][wi] += w; ctxs[cv]["mix"][c][wi] += w

    x_centers = np.array([min_cycle + i * window + window / 2.0 for i in range(nw)])
    wf = float(window)
    cvs = sorted(ctxs)
    ctx_issue = np.array([ctxs[v]["issue"] / wf for v in cvs])
    ctx_stall = np.array([ctxs[v]["stall"] / wf for v in cvs])

    return {
        "window": window, "min_cycle": min_cycle, "max_cycle": max_cycle,
        "num_windows": nw, "x_centers": x_centers,
        "overall": {
            "issue_count": issue / wf,
            "stall_count": stall / wf,
            "stall_reason_count": {c: v / wf for c, v in mix.items()},
        },
        "context_values": cvs,
        "context_issue_count": ctx_issue,
        "context_stall_count": ctx_stall,
    }


# ── Tile series builder ──────────────────────────────────────────────────────

def build_tile_series(rows, csv_path, tile_id, traces_dir=None):
    tile_rows = [r for r in rows if r["tile"] == tile_id]
    if not tile_rows:
        raise ValueError(f"No rows for tile {tile_id}")

    cycles = np.array(sorted({r["cycle"] for r in tile_rows}), dtype=int)
    c2i = {int(c): i for i, c in enumerate(cycles)}
    n = len(cycles)
    issue_cur = np.zeros(n); stall_cur = np.zeros(n)
    cat_cur = {c: np.zeros(n) for c in STALL_CATEGORIES}

    cores = sorted({r["core"] for r in tile_rows})
    core2i = {cid: i for i, cid in enumerate(cores)}
    per_core_state = np.zeros((len(cores), n))

    cat_index = {c: i for i, c in enumerate(STALL_CATEGORIES)}  # ins=0..other=5
    for r in tile_rows:
        ci = c2i[r["cycle"]]
        if r["state"] == "issue":
            issue_cur[ci] += 1; per_core_state[core2i[r["core"]], ci] = 1.0
        else:
            stall_cur[ci] += 1
            cats_hit = split_stall_kind(r["stall_kind"])
            for cat in cats_hit:
                cat_cur[cat][ci] += 1
            # Pick first stall category for heatmap colour (2..7)
            if cats_hit:
                per_core_state[core2i[r["core"]], ci] = 2.0 + cat_index[cats_hit[0]]
            else:
                per_core_state[core2i[r["core"]], ci] = 2.0  # fallback

    out_cur = np.zeros(n); store_cur = np.zeros(n); cum_li = np.zeros(n); cum_amo_li = np.zeros(n); cum_lr = np.zeros(n); cum_si = np.zeros(n)
    # per-core instruction type: 0=stalled, 1=load, 2=store, 3=mac, 4=other
    per_core_itype = np.zeros((len(cores), n))
    for cid in cores:
        tp = locate_trace_file(csv_path, cid, traces_dir=traces_dir)
        if tp is None:
            continue
        cc = np.array(sorted({r["cycle"] for r in tile_rows if r["core"] == cid}), dtype=int)
        if len(cc) == 0:
            continue
        ms = build_memory_request_series(tp, cc)
        cc2i = {int(c): i for i, c in enumerate(cc)}
        ci_core = core2i[cid]
        for c in cc:
            ti = c2i[int(c)]; ci = cc2i[int(c)]
            out_cur[ti] += ms["outstanding_loads"][ci]
            store_cur[ti] += ms["store_issue_current"][ci]
            cum_li[ti] += ms["load_issue_cumulative"][ci]
            cum_amo_li[ti] += ms["amo_load_issue_cumulative"][ci]
            cum_lr[ti] += ms["load_return_cumulative"][ci]
            cum_si[ti] += ms["store_issue_cumulative"][ci]
            # Instruction type for this core at this cycle
            if per_core_state[ci_core, ti] == 1.0:  # issuing
                per_core_itype[ci_core, ti] = ms["itype_current"][ci]

    present = [c for c in STALL_CATEGORIES if np.any(cat_cur[c] > 0)]
    return {
        "tile_id": tile_id, "cores": cores, "per_core_state": per_core_state,
        "per_core_itype": per_core_itype,
        "cycles": cycles,
        "issue_current": issue_cur, "stall_current": stall_cur,
        "issue_cumulative": np.cumsum(issue_cur), "stall_cumulative": np.cumsum(stall_cur),
        "category_current": cat_cur,
        "category_cumulative": {c: np.cumsum(v) for c, v in cat_cur.items()},
        "present_categories": present,
        "outstanding_current": out_cur,
        "store_issue_current": store_cur,
        "load_issue_cumulative": cum_li,
        "amo_load_issue_cumulative": cum_amo_li,
        "load_return_cumulative": cum_lr,
        "store_issue_cumulative": cum_si,
    }


# ── Overview page ─────────────────────────────────────────────────────────────

def _agg_ticks(ax, agg, max_ticks=12):
    set_cycle_ticks(ax, np.arange(agg["num_windows"]), agg["x_centers"], max_ticks)


def write_overview_page(path, agg, filter_desc, window):
    fig = plt.figure(figsize=(16, 12), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3])
    ax_prog = fig.add_subplot(gs[0, 0])
    ax_mix = fig.add_subplot(gs[0, 1])
    ax_hm = fig.add_subplot(gs[1, :])

    # 1 — progress
    x = np.arange(agg["num_windows"])
    ic = agg["overall"]["issue_count"]
    sc = agg["overall"]["stall_count"]
    ax_prog.plot(x, ic, color=STALL_COLORS["issue"], lw=2.5, label="Issuing")
    ax_prog.fill_between(x, 0, ic, color=STALL_COLORS["issue"], alpha=0.18)
    ax_prog.plot(x, sc, color="#4d4d4d", lw=2.0, label="Stalled")
    ax_prog.fill_between(x, 0, sc, color="#4d4d4d", alpha=0.10)
    ax_prog.set_ylim(0, max(1.0, float(np.nanmax([ic.max(), sc.max()])) * 1.08))
    ax_prog.set_title("1. Progress: issuing vs stalled cores")
    ax_prog.set_ylabel("Cores")
    ax_prog.grid(True, axis="y", alpha=0.25)
    _agg_ticks(ax_prog, agg)
    ax_prog.legend(loc="upper right", frameon=True, fancybox=True,
                   edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                   ncol=2, handleheight=1.7, handlelength=2.5)

    # 2 — stall composition
    arrays = [agg["overall"]["stall_reason_count"][c] for c in STALL_CATEGORIES]
    colors = [STALL_COLORS[c] for c in STALL_CATEGORIES]
    ax_mix.stackplot(x, arrays, labels=[stall_label(c) for c in STALL_CATEGORIES],
                     colors=colors, alpha=0.95)
    total = np.sum(np.vstack(arrays), axis=0)
    ax_mix.set_ylim(0, max(1.0, float(np.nanmax(total)) * 1.08))
    ax_mix.set_title("2. Stall composition by cause")
    ax_mix.set_ylabel("Stalled cores")
    ax_mix.grid(True, axis="y", alpha=0.25)
    _agg_ticks(ax_mix, agg)
    ax_mix.legend(loc="upper right", ncol=3, frameon=True, fancybox=True,
                  edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                  handleheight=1.7, handlelength=2.5)

    # 3 — merged heatmap: stall fraction per tile (auto-crop to tiles with any activity)
    all_activity = agg["context_issue_count"] + agg["context_stall_count"]
    active_mask = np.any(all_activity > 0, axis=1)
    active_indices = np.where(active_mask)[0]
    if len(active_indices) == 0:
        active_indices = np.arange(len(agg["context_values"]))
    active_labels = [str(agg["context_values"][i]) for i in active_indices]

    issue_mat = agg["context_issue_count"][active_indices, :]
    stall_mat = agg["context_stall_count"][active_indices, :]
    total_mat = issue_mat + stall_mat
    with np.errstate(divide="ignore", invalid="ignore"):
        stall_frac = np.where(total_mat > 0, stall_mat / total_mat, np.nan)

    img = ax_hm.imshow(stall_frac, aspect="auto", interpolation="nearest",
                       cmap="RdYlGn_r", vmin=0, vmax=1, origin="lower")
    ax_hm.set_facecolor("#E0E0E0")
    ax_hm.set_title("3. Per-tile stall fraction (green = issuing, red = stalled)")
    ax_hm.set_ylabel("Tile")
    ax_hm.set_xlabel("Cycle")
    _agg_ticks(ax_hm, agg)
    n_active = len(active_indices)
    if n_active <= 32:
        ax_hm.set_yticks(np.arange(n_active))
        ax_hm.set_yticklabels(active_labels)
    else:
        step = max(1, n_active // 16)
        shown = list(range(0, n_active, step))
        if (n_active - 1) not in shown:
            shown.append(n_active - 1)
        ax_hm.set_yticks(shown)
        ax_hm.set_yticklabels([active_labels[i] for i in shown])
    cbar = fig.colorbar(img, ax=ax_hm, fraction=0.028, pad=0.02)
    cbar.set_label("Stall fraction")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%\n(all issuing)", "25%", "50%", "75%", "100%\n(all stalled)"])

    fig.suptitle("Cluster Stall Overview", fontsize=17, fontweight="bold")
    fig.text(0.01, -0.005,
             f"Reconstructed from annotated traces, not a true cycle logger. "
             f"Filters: {filter_desc}. Window={window} cycles.",
             fontsize=9, alpha=0.82, va="top")
    fig.savefig(path, dpi=96, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, pdf_path


# ── Tile detail — single merged figure ────────────────────────────────────────

def write_tile_detail(png_path, ts):
    """All-in-one tile detail figure.

    Subplot order (top to bottom):
      1. Per-core state heatmap (stall subtypes)
      2. Per-core instruction-type heatmap (load/store/other/stalled)
      3. Cycle breakdown bar
      4. Current stall causes
      5. Cumulative stall causes
    6. Outstanding loads
    """
    tid = ts["tile_id"]
    cycles = ts["cycles"]
    cats = ts["present_categories"]

    nrows = 6
    ratios = [1.2, 1.2, 0.8, 1.5, 1.5, 1.5]
    fig, axes = plt.subplots(nrows, 1, figsize=(20, 29),
                             gridspec_kw={"height_ratios": ratios},
                             constrained_layout=True)
    row = 0

    # ── 1. per-core state heatmap (stall subtype colours) ────────────────
    n_cats = len(STALL_CATEGORIES)  # 6
    hm_colors = ["#FFFFFF", STALL_COLORS["issue"]] + [STALL_COLORS[c] for c in STALL_CATEGORIES]
    hm_cmap = plt.matplotlib.colors.ListedColormap(hm_colors)
    img = axes[row].imshow(ts["per_core_state"], aspect="auto", interpolation="nearest",
                           cmap=hm_cmap, vmin=0, vmax=1 + n_cats, origin="lower")
    axes[row].set_title("Per-Core Execution State by Stall Cause")
    axes[row].set_ylabel("Core in tile")
    axes[row].set_yticks(np.arange(len(ts["cores"])))
    axes[row].set_yticklabels([str(c) for c in ts["cores"]])
    axes[row].set_yticks(np.arange(-0.5, len(ts["cores"]), 1.0), minor=True)
    axes[row].grid(which="minor", axis="y", color="#F2F2F2", linewidth=2.0)
    axes[row].tick_params(which="minor", left=False)
    set_cycle_ticks(axes[row], np.arange(len(cycles)), cycles)
    from matplotlib.patches import Patch as _Patch
    hm_handles = [_Patch(facecolor=STALL_COLORS["issue"], edgecolor="black",
                         linewidth=0.5, label="Issuing")] + \
                 [_Patch(facecolor=STALL_COLORS[c], edgecolor="black", linewidth=0.5,
                         label=stall_label(c)) for c in cats]
    axes[row].legend(handles=hm_handles, loc="lower left",
                     bbox_to_anchor=(0, 1.02), frameon=True, fancybox=True,
                     edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                     ncol=len(hm_handles),
                     handleheight=1.15, handlelength=2.5)
    axes[row].set_xlabel("Cycle")
    row += 1

    # ── 2. per-core instruction-type heatmap ─────────────────────────────
    # 0=stalled, 1=load, 2=store, 3=mac, 4=other
    itype_colors = [ITYPE_COLORS[k] for k in ITYPE_ORDER]
    itype_cmap = plt.matplotlib.colors.ListedColormap(itype_colors)
    axes[row].imshow(ts["per_core_itype"], aspect="auto", interpolation="nearest",
                     cmap=itype_cmap, vmin=0, vmax=len(ITYPE_ORDER) - 1, origin="lower")
    axes[row].set_title("Per-Core Instruction Type")
    axes[row].set_ylabel("Core in tile")
    axes[row].set_yticks(np.arange(len(ts["cores"])))
    axes[row].set_yticklabels([str(c) for c in ts["cores"]])
    axes[row].set_yticks(np.arange(-0.5, len(ts["cores"]), 1.0), minor=True)
    axes[row].grid(which="minor", axis="y", color="#F2F2F2", linewidth=2.0)
    axes[row].tick_params(which="minor", left=False)
    set_cycle_ticks(axes[row], np.arange(len(cycles)), cycles)
    it_handles = [_Patch(facecolor=ITYPE_COLORS[k], edgecolor="black",
                         linewidth=0.5, label=ITYPE_LABELS[k])
                  for k in ITYPE_ORDER]
    axes[row].legend(handles=it_handles, loc="lower left",
                     bbox_to_anchor=(0, 1.02), frameon=True, fancybox=True,
                     edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                     ncol=len(ITYPE_ORDER),
                     handleheight=1.15, handlelength=2.5)
    axes[row].set_xlabel("Cycle")
    row += 1

    # ── 3. stacked horizontal bar: summed core-cycle breakdown ───────────
    total = float(ts["issue_cumulative"][-1] + ts["stall_cumulative"][-1])
    bar_labels = ["Issuing"] + [stall_label(c) for c in cats]
    bar_values = [float(ts["issue_cumulative"][-1])] + [float(ts["category_cumulative"][c][-1]) for c in cats]
    bar_colors = [STALL_COLORS["issue"]] + [STALL_COLORS[c] for c in cats]

    ax_bar = axes[row]
    left = 0.0
    for lbl, val, col in zip(bar_labels, bar_values, bar_colors):
        pct = val / total * 100 if total else 0
        display_lbl = lbl
        count_text = f"{int(val)} cyc."
        ax_bar.barh("Core-cycles", val, left=left, color=col, edgecolor="white", linewidth=0.5,
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
    ax_bar.set_xlabel("Core-cycle amount")
    ax_bar.set_title(f"Tile Summed Core-Cycle Breakdown  ({int(total)} total core-cycles across {len(ts['cores'])} cores)")
    ax_bar.legend(loc="upper left", frameon=True,
                  fancybox=True, edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
                  ncol=1,
                  handleheight=1.15, handlelength=2.5)
    row += 1

    # ── 4. current stall causes ──────────────────────────────────────────
    plot_categories_current(axes[row], cycles, ts["category_current"], cats,
                            "Current Stall Causes",
                            ylabel="Total stalled cores")
    axes[row].set_xlabel("Cycle")
    set_cycle_ticks(axes[row], cycles)
    row += 1

    # ── 5. cumulative stall causes ───────────────────────────────────────
    plot_categories_cumulative(axes[row], cycles, ts["category_cumulative"], cats,
                               "Cumulative Stall Causes")
    axes[row].set_xlabel("Cycle")
    set_cycle_ticks(axes[row], cycles)
    row += 1

    # ── 6. outstanding loads ─────────────────────────────────────────────
    plot_outstanding_loads(
        axes[row], cycles, ts["outstanding_current"],
        "Outstanding Loads"
    )
    axes[row].set_xlabel("Cycle")
    set_cycle_ticks(axes[row], cycles)
    row += 1

    fig.suptitle(f"Tile {tid} Detail Report", fontsize=17, fontweight="bold")

    fig.savefig(png_path, dpi=96, bbox_inches="tight")
    pdf_path = png_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    return fig, pdf_path


# ── CLI & main ────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Per-tile stall report with overview.")
    p.add_argument("csv", help="CSV from _gen_stall_timeseries_batch.py")
    p.add_argument("tile", type=int, nargs="*", help="Tile ID(s) for detail pages (optional)")
    p.add_argument("--output-dir", default=None, help="Defaults to <csv-dir>/plots")
    p.add_argument("--traces-dir", default=None, help="Directory containing trace files")
    p.add_argument("--prefix", default="tile_detail")
    p.add_argument("--window", type=int, default=64, help="Cycle window for aggregation")
    p.add_argument("--overview", action="store_true", help="Also generate the cluster overview page")
    p.add_argument("--section", type=int, action="append", help="Filter by section")
    p.add_argument("--group", type=int, action="append", help="Filter by group")
    p.add_argument("--show", action="store_true")
    return p.parse_args(argv)


def _filter_desc(args):
    parts = []
    for name in ("section", "group"):
        vals = getattr(args, name, None)
        if vals:
            parts.append(f"{name}={','.join(str(v) for v in sorted(set(vals)))}")
    if args.tile:
        parts.append(f"tile={','.join(str(v) for v in sorted(args.tile))}")
    return " | ".join(parts) if parts else "all rows"


def main(argv=None):
    args = parse_args(argv)
    if args.window <= 0:
        raise SystemExit("--window must be positive")

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        raise SystemExit(f"Missing CSV: {csv_path}")

    out = Path(args.output_dir) if args.output_dir else csv_path.parent / "plots"
    out.mkdir(parents=True, exist_ok=True)

    rows = filter_rows(load_rows(csv_path), section=args.section, group=args.group)
    if not rows:
        raise SystemExit("No rows after filtering")

    figs = []

    if args.overview:
        agg = aggregate_rows(rows, args.window, context_field="tile")
        ov_path = out / "overview_workload.png"
        fig_ov, pdf_ov = write_overview_page(ov_path, agg, _filter_desc(args), args.window)
        figs.append(fig_ov)
        print(f"Wrote {ov_path}")
        print(f"Wrote {pdf_ov}")

    # per-tile detail
    for tid in args.tile:
        ts = build_tile_series(rows, csv_path, tid, traces_dir=args.traces_dir)
        png = out / f"{args.prefix}_tile{tid}.png"
        fig, pdf = write_tile_detail(png, ts)
        figs.append(fig)
        print(f"Wrote {png}")
        print(f"Wrote {pdf}")

    if args.show:
        plt.show()
    else:
        for f in figs:
            plt.close(f)


if __name__ == "__main__":
    main()

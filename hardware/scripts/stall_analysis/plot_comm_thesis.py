#!/usr/bin/env python3
"""Thesis-quality communication analysis figures for MemPool.

Generates publication-ready figures including:
    1. Tile-to-tile traffic matrix
    2. Group-level traffic aggregate
    3. Locality breakdown & latency by network distance
    4. Communication-stall correlation (tile-level scatter + temporal overlay)
    5. Temporal communication profile (stacked area + incoming heatmap + latency)

Usage:
    python plot_comm_thesis.py <result_dir> [--section 1]
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, LogLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# ---------------------------------------------------------------------------
# Thesis style constants
# ---------------------------------------------------------------------------

FONT_TITLE    = 13
FONT_SUBTITLE = 11
FONT_LABEL    = 10.5
FONT_TICK     = 9
FONT_ANNOT    = 8.5
FONT_LEGEND   = 9

# Colorblind-safe palette (Okabe-Ito-inspired)
COL_LOCAL     = "#0072B2"   # blue  – local / intra-tile
COL_SAME_GRP  = "#56B4E9"  # sky blue – same group, different tile
COL_REMOTE    = "#D55E00"  # vermillion – remote / inter-group
COL_ACCENT    = "#E69F00"  # amber – highlights / p95
COL_NEUTRAL   = "#999999"
COL_LSU       = "#17BECF"  # teal – LSU stall (from stall palette)

GRP_COLORS       = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]
GRP_COLORS_LIGHT = ["#B3D9EF", "#F4C4A0", "#A3DFC9", "#E8C6DA"]

DPI = 300
FIG_TEXT_COLOR = "#222222"

LOCALITY_CLASSES = ["local", "same_group", "remote"]
LOCALITY_LABELS  = {
    "local":      "Local (intra-tile)",
    "same_group": "Same group",
    "remote":     "Remote (inter-group)",
}
LOCALITY_COLORS = {
    "local":      COL_LOCAL,
    "same_group": COL_SAME_GRP,
    "remote":     COL_REMOTE,
}


class PiecewiseNorm(Normalize):
    """Continuous normalization with explicit anchor remapping."""

    def __init__(self, x_points, y_points, vmin=None, vmax=None, clip=False):
        super().__init__(vmin=vmin, vmax=vmax, clip=clip)
        self.x_points = np.asarray(x_points, dtype=float)
        self.y_points = np.asarray(y_points, dtype=float)

    def __call__(self, value, clip=None):
        result, is_scalar = self.process_value(value)
        data = np.asarray(result.data, dtype=float)
        mapped = np.interp(data, self.x_points, self.y_points)
        masked = np.ma.array(mapped, mask=np.ma.getmask(result), copy=False)
        return masked[0] if is_scalar else masked

    def inverse(self, value):
        data = np.asarray(value, dtype=float)
        return np.interp(data, self.y_points, self.x_points)


def _apply_thesis_style():
    """Configure matplotlib for thesis-quality output."""
    plt.rcParams.update({
        "font.family":        "serif",
        "font.size":          FONT_TICK,
        "axes.titlesize":     FONT_SUBTITLE,
        "axes.labelsize":     FONT_LABEL,
        "axes.edgecolor":     "#444444",
        "axes.linewidth":     0.7,
        "axes.facecolor":     "white",
        "figure.facecolor":   "white",
        "xtick.labelsize":    FONT_TICK,
        "ytick.labelsize":    FONT_TICK,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "xtick.major.size":   3.5,
        "ytick.major.size":   3.5,
        "xtick.minor.size":   2.0,
        "ytick.minor.size":   2.0,
        "legend.fontsize":    FONT_LEGEND,
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#cccccc",
        "legend.fancybox":    True,
        "grid.alpha":         0.25,
        "grid.color":         "#888888",
        "grid.linewidth":     0.5,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.08,
        "text.color":         FIG_TEXT_COLOR,
        "axes.labelcolor":    FIG_TEXT_COLOR,
    })


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _pint(v):
    return 0 if v is None or v == "" else int(v)


def _pfloat(v):
    return 0.0 if v is None or v == "" else float(v)


def _load_csv(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _filter_section(rows, section):
    if section is None:
        return rows
    return [r for r in rows if _pint(r.get("section")) == section]


def _resolve_paths(input_path: str):
    """Return dict of data paths from a result directory."""
    p = Path(input_path).resolve()
    if p.is_dir() and (p / "data").is_dir():
        return {
            "result_dir":  p,
            "summary":    p / "data" / "comm_summary",
            "timeseries": p / "data" / "comm_timeseries",
            "events":     p / "data" / "comm_events_benchmark.csv",
            "stalls":     p / "data" / "stall_timeseries_benchmark.csv",
            "output":     p / "plots" / "communication",
        }
    return {
        "result_dir":  p.parent.parent if p.parent.name == "data" else p.parent,
        "summary":    p,
        "timeseries": p,
        "events":     p / "comm_events_benchmark.csv",
        "stalls":     p / "stall_timeseries_benchmark.csv",
        "output":     p / "plots",
    }


def _get_common_section_overlap(result_dir: Path, section: int):
    """Return [max(start_i), min(end_i)] across trace section markers, if available."""
    trace_dir = result_dir / "traces"
    if not trace_dir.is_dir():
        return None

    pattern = re.compile(rf"Performance metrics for section {section} @ \((\d+), (\d+)\):")
    starts = []
    ends = []
    for trace_path in sorted(trace_dir.glob("trace_hart_*.trace")):
        text = trace_path.read_text(errors="ignore")
        match = pattern.search(text)
        if match:
            starts.append(int(match.group(1)))
            ends.append(int(match.group(2)))

    if not starts or not ends:
        return None

    overlap_start = max(starts)
    overlap_end = min(ends)
    if overlap_end < overlap_start:
        return None
    return overlap_start, overlap_end


def _nice_count(v):
    """Format a count for annotation (e.g., 1234 → '1.2k')."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def _clean_spine(ax, top=False, right=False):
    """Remove top/right spines and enable y-grid."""
    ax.spines["top"].set_visible(top)
    ax.spines["right"].set_visible(right)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _derive_suffix(output_dir: Path) -> str:
    """Extract a short kernel tag from the output directory path.

    .../results/<app>/<kernel>/<variant>/plots/communication/
    Drops 'conflict_' and the variant (e.g. 'baseline') to produce:
      4x4 → 4x4, 4x4_conflict_opt → 4x4_opt,
      4x4_asm → 4x4_asm, 4x4_conflict_opt_asm → 4x4_asm_opt.
    """
    try:
        comm = output_dir.resolve()
        variant_dir = comm.parent.parent          # .../kernel/variant
        kernel = variant_dir.parent.name           # e.g. '4x4_conflict_opt_asm'
        if kernel:
            tag = kernel.replace("conflict_", "")
            # Reorder: 4x4_opt_asm → 4x4_asm_opt
            if tag.endswith("_opt_asm"):
                tag = tag.replace("_opt_asm", "_asm_opt")
            return tag
    except Exception:
        pass
    return ""


def _save(fig, output_dir, name, section):
    suffix = _derive_suffix(output_dir)
    stem = f"{name}_{suffix}" if suffix else name
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    legacy_pdf = output_dir / f"{stem}.pdf"
    if legacy_pdf.exists():
        legacy_pdf.unlink()

    fig.savefig(output_dir / f"{stem}.png", dpi=DPI)
    fig.savefig(pdf_dir / f"{stem}.pdf", dpi=DPI)
    plt.close(fig)
    print(f"  → {stem}.png  +  pdf/{stem}.pdf")


def _migrate_legacy_pdfs(output_dir: Path):
    """Move old flat-layout PDFs into pdf/ subdir and remove old thesis_* files."""
    pdf_dir = output_dir / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for legacy_pdf in output_dir.glob("*.pdf"):
        target = pdf_dir / legacy_pdf.name
        legacy_pdf.replace(target)
    # Remove old thesis_* naming convention files
    for legacy in output_dir.glob("thesis_*"):
        legacy.unlink()
    for legacy in pdf_dir.glob("thesis_*"):
        legacy.unlink()


# ===================================================================
# Figure 1 – Traffic Matrix with Group Structure
# ===================================================================

def _build_traffic_matrices(source_dest_rows, n_groups):
    issue_rows = [
        (_pint(r["source_tile"]), _pint(r["dest_tile"]), _pint(r["count"]))
        for r in source_dest_rows
        if (r.get("event_type") or "").strip() in ("load_issue", "store_issue")
    ]
    if not issue_rows:
        return None

    max_tile = max(max(s, d) for s, d, _ in issue_rows)
    n_tiles = max_tile + 1
    tpg = n_tiles // n_groups

    mat = np.zeros((n_tiles, n_tiles), dtype=float)
    for s, d, c in issue_rows:
        mat[s, d] += c

    gmat = np.zeros((n_groups, n_groups), dtype=float)
    for s, d, c in issue_rows:
        sg, dg = s // tpg, d // tpg
        if sg < n_groups and dg < n_groups:
            gmat[sg, dg] += c

    return {
        "issue_rows": issue_rows,
        "mat": mat,
        "gmat": gmat,
        "n_tiles": n_tiles,
        "tpg": tpg,
    }

def plot_traffic_matrix(source_dest_rows, n_groups, output_dir, section):
    """Emit the standalone group-level traffic aggregate figure."""
    matrices = _build_traffic_matrices(source_dest_rows, n_groups)
    if matrices is None:
        print("  [skip] No traffic data for matrix")
        return

    gmat = matrices["gmat"]
    sec_lbl = f"Section {section}" if section is not None else "All sections"

    # Group-level traffic aggregate
    fig2, ax2 = plt.subplots(figsize=(7.2, 5.8))
    gmax = gmat.max()
    im2 = ax2.imshow(gmat, origin="lower", cmap="YlOrRd", aspect="equal",
                     norm=Normalize(vmin=0, vmax=gmax), interpolation="nearest")
    for i in range(n_groups):
        for j in range(n_groups):
            val = gmat[i, j]
            txt_col = "white" if val > gmax * 0.6 else FIG_TEXT_COLOR
            if val > 0:
                ax2.text(j, i, _nice_count(val), ha="center", va="center",
                         fontsize=FONT_ANNOT + 1, fontweight="bold", color=txt_col)
    ax2.set_xticks(range(n_groups))
    ax2.set_yticks(range(n_groups))
    ax2.set_xticklabels([f"Group {g}" for g in range(n_groups)], fontsize=FONT_TICK)
    ax2.set_yticklabels([f"Group {g}" for g in range(n_groups)], fontsize=FONT_TICK)
    ax2.set_xlabel("Destination group", fontsize=FONT_LABEL)
    ax2.set_ylabel("Source group", fontsize=FONT_LABEL)
    ax2.set_title("Group-level traffic aggregate",
                  fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="3.5%", pad=1.00)
    cb2 = fig2.colorbar(im2, cax=cax2)
    cb2.set_label("Total events", fontsize=FONT_LABEL - 1)
    cb2.ax.tick_params(labelsize=FONT_TICK - 1)

    total = gmat.sum()
    group_nonzero = int(np.count_nonzero(gmat))
    fig2.text(0.5, -0.02,
             f"{sec_lbl}  ·  {n_groups} groups  ·  {_nice_count(total)} total events  ·  "
             f"{group_nonzero} active group pairs",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig2, output_dir, "traffic_matrix_groups", section)


# ===================================================================
# Figure 1b – Zoomed Traffic Matrix (active groups only)
# ===================================================================

def plot_traffic_matrix_zoom(source_dest_rows, n_groups, output_dir, section,
                             zoom_groups=None):
    """Rectangular zoomed heatmap: active source groups (y) × all dest tiles (x).
    Shows the full destination reach of active groups, not just within-group traffic."""
    issue_rows = [
        (_pint(r["source_tile"]), _pint(r["dest_tile"]), _pint(r["count"]))
        for r in source_dest_rows
        if (r.get("event_type") or "").strip() in ("load_issue", "store_issue")
    ]
    if not issue_rows:
        print("  [skip] No traffic data for zoom matrix")
        return

    max_tile = max(max(s, d) for s, d, _ in issue_rows)
    n_tiles = max_tile + 1
    tpg = n_tiles // n_groups

    mat = np.zeros((n_tiles, n_tiles), dtype=float)
    for s, d, c in issue_rows:
        mat[s, d] += c

    # Auto-detect active source groups: those with cross-tile traffic
    if zoom_groups is None:
        group_nonlocal = np.zeros(n_groups)
        for s, d, c in issue_rows:
            sg = s // tpg
            if s != d:
                group_nonlocal[sg] += c
        zoom_groups = [g for g in range(n_groups) if group_nonlocal[g] > 0]
        if not zoom_groups:
            zoom_groups = list(range(n_groups))

    # Source tiles = active source groups; dest tiles = all tiles with traffic
    src_tiles = []
    for g in zoom_groups:
        src_tiles.extend(range(g * tpg, (g + 1) * tpg))
    # Dest axis: all groups that receive traffic from source tiles
    dest_groups_set = set()
    for s, d, c in issue_rows:
        if s in set(src_tiles) and c > 0:
            dest_groups_set.add(d // tpg)
    dest_groups = sorted(dest_groups_set)
    dest_tiles = []
    for g in dest_groups:
        dest_tiles.extend(range(g * tpg, (g + 1) * tpg))

    sub = mat[np.ix_(src_tiles, dest_tiles)]
    n_src = len(src_tiles)
    n_dst = len(dest_tiles)

    cell_size = 0.34
    # Adaptive: maximize cell size while staying under 8000px at output DPI
    max_px = 7900  # leave margin below 8192
    max_w_inches = max_px / DPI - 3.5
    max_h_inches = max_px / DPI - 2.5
    cell_size = min(0.50, max_w_inches / max(n_dst, 1),
                          max_h_inches / max(n_src, 1))
    cell_size = max(cell_size, 0.20)  # floor for very large matrices
    fig_w = n_dst * cell_size + 3.5   # +colorbar + labels
    fig_h = n_src * cell_size + 2.5   # +title + xlabel
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = sub.max()
    vmin = max(1.0, sub[sub > 0].min()) if np.any(sub > 0) else 1.0
    sub_plot = sub.copy()
    sub_plot[sub_plot == 0] = np.nan
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#F5F5F5")
    im = ax.imshow(sub_plot, origin="lower", cmap=cmap, aspect="equal",
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")

    # Annotate cells (only feasible when matrix is small enough)
    if n_src * n_dst <= 4500:
        annot_fs = max(6.0, FONT_ANNOT - 0.5 * (max(n_src, n_dst) > 32)
                                        - 0.5 * (max(n_src, n_dst) > 48))
        for i in range(n_src):
            for j in range(n_dst):
                val = sub[i, j]
                if val > 0:
                    txt_col = "white" if val > vmax * 0.3 else FIG_TEXT_COLOR
                    ax.text(j, i, _nice_count(val), ha="center", va="center",
                            fontsize=annot_fs, fontweight="bold", color=txt_col)

    # Group boundaries — source (y-axis)
    src_offsets = {}  # group -> y-offset in sub-matrix
    offset = 0
    for g in zoom_groups:
        if offset > 0:
            ax.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(-0.04, mid, f"G{g}", ha="right", va="center",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[g % len(GRP_COLORS)],
                transform=ax.get_yaxis_transform(), clip_on=False)
        src_offsets[g] = offset
        offset += tpg

    # Group boundaries — dest (x-axis)
    dst_offsets = {}  # group -> x-offset in sub-matrix
    offset = 0
    for gi, g in enumerate(dest_groups):
        if gi > 0:
            ax.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(mid, -0.06, f"G{g}", ha="center", va="top",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[g % len(GRP_COLORS)],
                transform=ax.get_xaxis_transform(), clip_on=False)
        dst_offsets[g] = offset
        offset += tpg

    # Highlight self-group blocks (src_group == dest_group)
    for g in zoom_groups:
        if g in dst_offsets:
            rect_xy = (dst_offsets[g] - 0.5, src_offsets[g] - 0.5)
            gcol = GRP_COLORS[g % len(GRP_COLORS)]
            # Dark backing outline for contrast
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=8.0, edgecolor="#000000",
                facecolor="none", alpha=0.25, zorder=4, clip_on=False,
            ))
            # Colored glow
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=5.0, edgecolor=gcol,
                facecolor="none", alpha=0.5, zorder=5, clip_on=False,
            ))
            # Crisp colored border
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=3.0, edgecolor=gcol,
                facecolor="none", zorder=6, clip_on=False,
            ))

    tick_fs = max(6.5, FONT_TICK - 1.0 * (n_dst > 48))
    ax.set_xticks(range(n_dst))
    ax.set_yticks(range(n_src))
    ax.set_xticklabels(dest_tiles, fontsize=tick_fs, rotation=90)
    ax.set_yticklabels(src_tiles, fontsize=tick_fs)
    ax.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
    ax.set_ylabel("Source tile", fontsize=FONT_LABEL)

    src_label = ", ".join(f"G{g}" for g in zoom_groups)
    dst_label = ", ".join(f"G{g}" for g in dest_groups)
    ax.set_title("Tile-to-tile traffic volume (load + store events)",
                 fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=1.00)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Load + store events", fontsize=FONT_LABEL - 1)
    # Explicit clean tick values across the log range
    import math as _m
    _lo, _hi = _m.log10(max(vmin, 1)), _m.log10(max(vmax, 1))
    _ticks = []
    for exp in range(int(_m.floor(_lo)), int(_m.ceil(_hi)) + 1):
        base = 10 ** exp
        for mult in [1, 2.5, 5]:
            v = base * mult
            if vmin <= v <= vmax:
                _ticks.append(v)
    if not _ticks:
        _ticks = [vmin, vmax]
    cb.set_ticks(_ticks)
    cb.set_ticklabels([_nice_count(t) for t in _ticks])
    cb.ax.tick_params(labelsize=FONT_TICK)

    total = sub.sum()
    nonzero = int(np.count_nonzero(sub))
    sec_lbl = f"Section {section}" if section is not None else "All sections"
    fig.text(0.5, -0.005,
             f"{sec_lbl}  ·  {n_src} source tiles ({src_label}) → "
             f"{n_dst} dest tiles ({dst_label})  ·  "
             f"{_nice_count(total)} events  ·  {nonzero} active pairs",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "traffic_matrix", section)
    # Clean up legacy file names
    for pattern in ("thesis_traffic_matrix_zoom_*", "thesis_traffic_matrix_*"):
        for legacy in output_dir.glob(pattern):
            legacy.unlink()
        pdf_dir = output_dir / "pdf"
        if pdf_dir.is_dir():
            for legacy in pdf_dir.glob(pattern):
                legacy.unlink()


# ===================================================================
# Figure 2 – Locality Breakdown & Latency by Network Distance
# ===================================================================

def plot_locality_latency(source_tile_rows, raw_events_path, n_groups,
                          output_dir, section):
    """Three panels:
      (a) Remote-traffic fraction heatmap strip per tile
      (b) Per-group traffic volume: local / same-group / remote stacked bars
      (c) Load-return latency by locality class (bar = mean, tri = p95)
    """
    # (a)/(b) data from source_tile_locality summary
    sorted_tiles = sorted(source_tile_rows, key=lambda r: _pint(r["source_tile"]))
    tile_ids = [_pint(r["source_tile"]) for r in sorted_tiles]
    groups   = [_pint(r["source_group"]) for r in sorted_tiles]
    total    = np.array([_pint(r["total_events"]) for r in sorted_tiles], dtype=float)
    local    = np.array([_pint(r["local_events"]) for r in sorted_tiles], dtype=float)
    remote   = np.array([_pint(r["remote_events"]) for r in sorted_tiles], dtype=float)
    remote_pct = np.where(total > 0, remote / total * 100, 0)

    n_tiles = len(tile_ids)
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    # (c) Latency by locality class from raw events
    lat_by_loc = {lc: [] for lc in LOCALITY_CLASSES}
    if raw_events_path.is_file():
        with raw_events_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if section is not None and _pint(r.get("section")) != section:
                    continue
                if r["event_type"] != "load_return" or not r.get("latency", ""):
                    continue
                if r.get("is_local") == "1":
                    lc = "local"
                elif r.get("is_same_group") == "1":
                    lc = "same_group"
                else:
                    lc = "remote"
                lat_by_loc[lc].append(float(r["latency"]))

    fig = plt.figure(figsize=(11, 10))
    gs_fig = gridspec.GridSpec(3, 1, height_ratios=[0.7, 1.1, 1.2], hspace=0.40)

    # (a) Remote traffic heatmap strip
    ax_a = fig.add_subplot(gs_fig[0])
    strip = remote_pct.reshape(1, -1)
    cmap_strip = plt.cm.RdYlGn_r.copy()
    im_a = ax_a.imshow(strip, aspect="auto", cmap=cmap_strip, vmin=0, vmax=100,
                       extent=[-0.5, n_tiles - 0.5, -0.5, 0.5],
                       interpolation="nearest")
    for g in range(1, n_groups):
        ax_a.axvline(g * tpg - 0.5, color="#333333", lw=1.2)
    for g in range(n_groups):
        mid = g * tpg + tpg / 2 - 0.5
        ax_a.text(mid, 0.75, f"Group {g}", ha="center", va="bottom",
                  fontsize=FONT_ANNOT, fontweight="bold", color=GRP_COLORS[g % len(GRP_COLORS)])
        g_mask = np.array(groups) == g
        if g_mask.any():
            avg = remote_pct[g_mask].mean()
            ax_a.text(mid, 0.0, f"{avg:.0f}%", ha="center", va="center",
                      fontsize=FONT_ANNOT + 1, fontweight="bold",
                      color="white" if avg > 50 else FIG_TEXT_COLOR)
    ax_a.set_yticks([])
    ax_a.set_xlabel("Tile", fontsize=FONT_LABEL)
    ax_a.set_xticks(np.arange(0, n_tiles, max(1, tpg // 2)))
    ax_a.set_title("(a)  Remote traffic fraction per tile",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=8, loc="left")
    divider_a = make_axes_locatable(ax_a)
    cax_a = divider_a.append_axes("right", size="3.5%", pad=1.00)
    cb_a = fig.colorbar(im_a, cax=cax_a)
    cb_a.set_label("Remote %", fontsize=FONT_LABEL - 1)
    cb_a.ax.tick_params(labelsize=FONT_TICK - 1)

    # (b) Per-group stacked traffic
    ax_b = fig.add_subplot(gs_fig[1])
    grp_counts = {lc: np.zeros(n_groups) for lc in LOCALITY_CLASSES}
    if raw_events_path.is_file():
        with raw_events_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if section is not None and _pint(r.get("section")) != section:
                    continue
                et = (r.get("event_type") or "").strip()
                if et not in ("load_issue", "store_issue"):
                    continue
                g = _pint(r.get("group", 0))
                if g >= n_groups:
                    continue
                if r.get("is_local") == "1":
                    grp_counts["local"][g] += 1
                elif r.get("is_same_group") == "1":
                    grp_counts["same_group"][g] += 1
                else:
                    grp_counts["remote"][g] += 1
    else:
        for i, g in enumerate(groups):
            if g < n_groups:
                grp_counts["local"][g] += local[i]
                grp_counts["remote"][g] += remote[i]

    x_b = np.arange(n_groups)
    bar_w = 0.55
    bottom = np.zeros(n_groups)
    for lc in LOCALITY_CLASSES:
        vals = grp_counts[lc]
        ax_b.bar(x_b, vals, bar_w, bottom=bottom,
                 color=LOCALITY_COLORS[lc], edgecolor="white", lw=0.5,
                 label=LOCALITY_LABELS[lc])
        for xi in range(n_groups):
            if vals[xi] > 0:
                mid_y = bottom[xi] + vals[xi] / 2
                bar_total = sum(grp_counts[c][xi] for c in LOCALITY_CLASSES)
                pct = vals[xi] / bar_total * 100 if bar_total > 0 else 0
                if pct > 8:
                    ax_b.text(xi, mid_y, f"{pct:.0f}%",
                              ha="center", va="center",
                              fontsize=FONT_ANNOT, fontweight="bold", color="white")
        bottom += vals

    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels([f"Group {g}" for g in range(n_groups)])
    ax_b.set_ylabel("Event count", fontsize=FONT_LABEL)
    ax_b.set_title("(b)  Traffic volume per group by network distance",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=8, loc="left")
    ax_b.legend(loc="upper right", fontsize=FONT_LEGEND)
    ax_b.yaxis.set_major_locator(MaxNLocator(integer=True))
    _clean_spine(ax_b)

    # (c) Latency by locality class
    ax_c = fig.add_subplot(gs_fig[2])
    x_c = np.arange(len(LOCALITY_CLASSES))
    bar_w_c = 0.50
    means, p50s, p95s, counts = [], [], [], []
    for lc in LOCALITY_CLASSES:
        vals = lat_by_loc[lc]
        if vals:
            a = np.array(vals)
            means.append(a.mean())
            p50s.append(np.median(a))
            p95s.append(np.percentile(a, 95))
            counts.append(len(vals))
        else:
            means.append(0); p50s.append(0); p95s.append(0); counts.append(0)

    colors_c = [LOCALITY_COLORS[lc] for lc in LOCALITY_CLASSES]
    ax_c.bar(x_c, means, bar_w_c, color=colors_c,
             edgecolor="white", lw=0.5, alpha=0.85, zorder=2)
    ax_c.scatter(x_c, p95s, marker="^", s=55, color=COL_ACCENT,
                 edgecolor="#333333", lw=0.6, zorder=5, label="p95 latency")
    ax_c.scatter(x_c, p50s, marker="D", s=35, color="#333333",
                 edgecolor="white", lw=0.5, zorder=5, label="Median latency")

    for xi, (m, p50, p95, n) in enumerate(zip(means, p50s, p95s, counts)):
        if m > 0:
            ax_c.text(xi, m / 2, f"avg {m:.1f}\nn={_nice_count(n)}",
                      ha="center", va="center", fontsize=FONT_ANNOT,
                      fontweight="bold", color="white")
            ax_c.text(xi + 0.12, p95 + 0.8, f"{p95:.0f}",
                      ha="left", va="bottom", fontsize=FONT_ANNOT,
                      color=COL_ACCENT, fontstyle="italic")

    ax_c.set_xticks(x_c)
    ax_c.set_xticklabels([LOCALITY_LABELS[lc] for lc in LOCALITY_CLASSES],
                         rotation=15, ha="right")
    ax_c.set_ylabel("Latency (cycles)", fontsize=FONT_LABEL)
    ax_c.set_title("(c)  Load-return latency by network distance",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=8, loc="left")
    ax_c.legend(loc="upper right", fontsize=FONT_LEGEND)
    _clean_spine(ax_c)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    total_lat = sum(counts)
    fig.text(0.5, -0.005,
             f"{sec_lbl}  ·  {n_tiles} tiles × {n_groups} groups  ·  "
             f"{_nice_count(total_lat)} load returns with latency  ·  "
             "▲ p95   ◆ median",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "locality_overview", section)


# ===================================================================
# Figure 3 – Communication–Stall Correlation
# ===================================================================

def plot_comm_stall_correlation(raw_events_path, stalls_path,
                                tile_ts_rows, n_groups,
                                output_dir, section):
    """Two panels:
      (a) Scatter: per-tile non-local traffic fraction vs LSU-stall fraction
      (b) Temporal overlay: aggregate non-local event rate + LSU stall rate
    """
    if not raw_events_path.is_file() or not stalls_path.is_file():
        print("  [skip] Missing events or stalls CSV for correlation figure")
        return

    # -- (a) Per-tile aggregates ------------------------------------------
    tile_comm = defaultdict(lambda: {"local": 0, "same_group": 0, "remote": 0})
    with raw_events_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if section is not None and _pint(r.get("section")) != section:
                continue
            et = (r.get("event_type") or "").strip()
            if et not in ("load_issue", "store_issue"):
                continue
            t = _pint(r.get("tile"))
            if r.get("is_local") == "1":
                tile_comm[t]["local"] += 1
            elif r.get("is_same_group") == "1":
                tile_comm[t]["same_group"] += 1
            else:
                tile_comm[t]["remote"] += 1

    tile_stall = defaultdict(lambda: {"total": 0, "stall": 0, "lsu": 0})
    with stalls_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if section is not None and _pint(r.get("section")) != section:
                continue
            t = _pint(r.get("tile"))
            tile_stall[t]["total"] += 1
            if r["state"] == "stall":
                tile_stall[t]["stall"] += 1
                if "lsu" in r.get("stall_kind", ""):
                    tile_stall[t]["lsu"] += 1

    tiles = sorted(set(tile_comm.keys()) & set(tile_stall.keys()))
    if not tiles:
        print("  [skip] No overlapping tiles between comm and stall data")
        return

    n_tiles = max(tiles) + 1
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    remote_fracs, lsu_fracs, tile_groups = [], [], []
    for t in tiles:
        c = tile_comm[t]
        s = tile_stall[t]
        total_comm = c["local"] + c["same_group"] + c["remote"]
        rf = (c["same_group"] + c["remote"]) / total_comm * 100 if total_comm > 0 else 0
        lf = s["lsu"] / s["total"] * 100 if s["total"] > 0 else 0
        remote_fracs.append(rf)
        lsu_fracs.append(lf)
        tile_groups.append(t // tpg if tpg > 0 else 0)

    remote_fracs = np.array(remote_fracs)
    lsu_fracs    = np.array(lsu_fracs)
    tile_groups  = np.array(tile_groups)

    # -- (b) Temporal data ------------------------------------------------
    has_temporal = bool(tile_ts_rows)
    if has_temporal:
        windows = sorted(set(int(r["window_index"]) for r in tile_ts_rows))
        n_windows = len(windows)
        by_wt = {}
        for r in tile_ts_rows:
            by_wt[(int(r["window_index"]), int(r["tile"]))] = r

        ts_tiles = sorted(set(int(r["tile"]) for r in tile_ts_rows))
        cycle_centers = []
        agg_remote     = np.zeros(n_windows)
        agg_total_comm = np.zeros(n_windows)

        for wi, w in enumerate(windows):
            cc = None
            for t in ts_tiles:
                r = by_wt.get((w, t))
                if r is None:
                    continue
                if cc is None:
                    cc = float(r["window_center_cycle"])
                same = int(r.get("same_group_events", 0))
                rem  = int(r.get("remote_group_events", 0))
                loc  = int(r.get("local_events", 0))
                agg_remote[wi]     += same + rem
                agg_total_comm[wi] += loc + same + rem
            cycle_centers.append(cc if cc is not None else 0)
        cycle_centers = np.array(cycle_centers)
        remote_rate = np.where(agg_total_comm > 0,
                               agg_remote / agg_total_comm * 100, 0)

        # Per-window LSU stall rate from stall timeseries
        win_starts = np.array([
            float(by_wt.get((w, ts_tiles[0]), {}).get("window_start_cycle", 0))
            for w in windows])
        win_ends = np.array([
            float(by_wt.get((w, ts_tiles[0]), {}).get("window_end_cycle", 0))
            for w in windows])

        lsu_per_window   = np.zeros(n_windows)
        total_per_window = np.zeros(n_windows)
        with stalls_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if section is not None and _pint(r.get("section")) != section:
                    continue
                cyc = _pint(r.get("cycle"))
                for wi in range(n_windows):
                    if win_starts[wi] <= cyc < win_ends[wi]:
                        total_per_window[wi] += 1
                        if r["state"] == "stall" and "lsu" in r.get("stall_kind", ""):
                            lsu_per_window[wi] += 1
                        break

        with np.errstate(invalid="ignore", divide="ignore"):
            lsu_rate = np.where(total_per_window > 0,
                                lsu_per_window / total_per_window * 100, 0)

    # --- Figure ----------------------------------------------------------
    n_rows = 2 if has_temporal else 1
    fig_h  = 9.5 if has_temporal else 5.5
    fig = plt.figure(figsize=(11, fig_h))
    if has_temporal:
        gs_fig = gridspec.GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.35)
    else:
        gs_fig = gridspec.GridSpec(1, 1)

    # (a) Scatter
    ax_a = fig.add_subplot(gs_fig[0])
    for g in range(n_groups):
        mask = tile_groups == g
        if not mask.any():
            continue
        ax_a.scatter(remote_fracs[mask], lsu_fracs[mask],
                     s=60, color=GRP_COLORS[g % len(GRP_COLORS)], edgecolor="#333333", lw=0.5,
                     alpha=0.8, label=f"Group {g}", zorder=3)

    finite_mask = np.isfinite(remote_fracs) & np.isfinite(lsu_fracs)
    fit_x = remote_fracs[finite_mask]
    fit_y = lsu_fracs[finite_mask]
    if len(fit_x) > 2 and np.ptp(fit_x) > 0 and np.ptp(fit_y) > 0:
        coeffs = np.polyfit(fit_x, fit_y, 1)
        x_fit = np.linspace(fit_x.min(), fit_x.max(), 100)
        y_fit = np.polyval(coeffs, x_fit)
        # Clip regression line to non-negative y
        y_fit = np.clip(y_fit, 0, None)
        r_val = np.corrcoef(fit_x, fit_y)[0, 1]
        ax_a.plot(x_fit, y_fit, color=COL_NEUTRAL, lw=1.8, ls="--",
                  alpha=0.7, zorder=2)
        ax_a.text(0.97, 0.05, f"r = {r_val:.2f}",
                  transform=ax_a.transAxes, ha="right", va="bottom",
                  fontsize=FONT_ANNOT + 1, color=COL_NEUTRAL,
                  fontstyle="italic",
                  bbox=dict(facecolor="white", edgecolor="#cccccc",
                            boxstyle="round,pad=0.3", alpha=0.85))

    ax_a.set_xlim(-3, max(remote_fracs.max() * 1.08, 5))
    ax_a.set_ylim(0, max(lsu_fracs.max() * 1.25, 1))
    ax_a.set_xlabel("Non-local traffic fraction (% of issued loads+stores)",
                    fontsize=FONT_LABEL)
    ax_a.set_ylabel("LSU stall fraction (% of core-cycles)", fontsize=FONT_LABEL)
    ax_a.set_title("(a)  Per-tile: non-local traffic vs. LSU stall rate",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=8, loc="left")
    ax_a.legend(loc="upper left", fontsize=FONT_LEGEND, ncol=2,
                frameon=True, fancybox=True, edgecolor="#cccccc",
                facecolor="white", framealpha=0.9)
    _clean_spine(ax_a)

    # (b) Temporal overlay
    if has_temporal:
        ax_b = fig.add_subplot(gs_fig[1])
        ln1 = ax_b.plot(cycle_centers, remote_rate, color=COL_REMOTE, lw=2.2,
                        marker="o", markersize=4, markerfacecolor="white",
                        markeredgecolor=COL_REMOTE, markeredgewidth=1.0,
                        label="Non-local traffic %", zorder=3)
        ax_b.fill_between(cycle_centers, 0, remote_rate,
                          color=COL_REMOTE, alpha=0.10)
        ax_b.set_ylabel("Non-local traffic (%)", fontsize=FONT_LABEL,
                        color=COL_REMOTE)
        ax_b.tick_params(axis="y", labelcolor=COL_REMOTE)

        ax_b2 = ax_b.twinx()
        ln2 = ax_b2.plot(cycle_centers, lsu_rate, color=COL_LSU, lw=2.2,
                         marker="s", markersize=4, markerfacecolor="white",
                         markeredgecolor=COL_LSU, markeredgewidth=1.0,
                         label="LSU stall rate %", zorder=3)
        ax_b2.fill_between(cycle_centers, 0, lsu_rate,
                           color=COL_LSU, alpha=0.10)
        ax_b2.set_ylabel("LSU stall rate (%)", fontsize=FONT_LABEL,
                         color=COL_LSU)
        ax_b2.tick_params(axis="y", labelcolor=COL_LSU)

        lns = ln1 + ln2
        labs = [l.get_label() for l in lns]
        ax_b.legend(lns, labs, loc="upper right", fontsize=FONT_LEGEND,
                    frameon=True, fancybox=True, edgecolor="#cccccc",
                    facecolor="white", framealpha=0.9)

        ax_b.set_xlabel("Cycle", fontsize=FONT_LABEL)
        ax_b.set_title("(b)  Temporal: non-local traffic and LSU stall rate",
                       fontsize=FONT_SUBTITLE, fontweight="bold", pad=8, loc="left")
        ax_b.set_xlim(cycle_centers.min() - 20, cycle_centers.max() + 20)
        ax_b.grid(axis="y", alpha=0.25)
        ax_b.set_axisbelow(True)
        ax_b.spines["top"].set_visible(False)
        ax_b2.spines["top"].set_visible(False)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    fig.text(0.5, -0.005,
             f"{sec_lbl}  ·  {len(tiles)} tiles  ·  "
             f"Dashed line = linear fit",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "comm_vs_stall", section)


# ===================================================================
# Figure 4 – Temporal Communication Profile
# ===================================================================

def plot_temporal_profile(tile_ts_rows, raw_events_path, result_dir, n_groups, output_dir, section):
    """Three panels (shared x-axis = cycle):
      (a) Stacked area: local / same-group / remote traffic over time
      (b) Per-tile incoming communication intensity heatmap
      (c) Average load-return latency over time, including locality breakdown
    """
    windows = sorted(set(int(r["window_index"]) for r in tile_ts_rows))
    n_windows = len(windows)
    window_pos = {window_index: pos for pos, window_index in enumerate(windows)}
    tiles = sorted(set(int(r["tile"]) for r in tile_ts_rows))
    n_tiles = len(tiles)
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    window_bounds = {}
    for r in tile_ts_rows:
        wi = int(r["window_index"])
        if wi not in window_bounds:
            window_bounds[wi] = (
                int(r["window_start_cycle"]),
                int(r["window_end_cycle"]),
            )

    by_wt = {}
    for r in tile_ts_rows:
        by_wt[(int(r["window_index"]), int(r["tile"]))] = r

    cycle_centers  = []
    agg_local      = np.zeros(n_windows)
    agg_same       = np.zeros(n_windows)
    agg_remote     = np.zeros(n_windows)
    heatmap_in     = np.zeros((n_tiles, n_windows))
    agg_lat        = np.zeros(n_windows)
    agg_lat_weight = np.zeros(n_windows)
    loc_lat_sum    = np.zeros(n_windows)
    loc_lat_n      = np.zeros(n_windows)
    same_lat_sum   = np.zeros(n_windows)
    same_lat_n     = np.zeros(n_windows)
    remote_lat_sum = np.zeros(n_windows)
    remote_lat_n   = np.zeros(n_windows)

    for wi, w in enumerate(windows):
        cc = None
        for ti, t in enumerate(tiles):
            r = by_wt.get((w, t))
            if r is None:
                continue
            if cc is None:
                cc = float(r["window_center_cycle"])
            loc  = int(r.get("local_events", 0))
            same = int(r.get("same_group_events", 0))
            rem  = int(r.get("remote_group_events", 0))
            agg_local[wi]  += loc
            agg_same[wi]   += same
            agg_remote[wi] += rem
            heatmap_in[ti, wi] = int(r.get("incoming_events", 0))
            lat   = _pfloat(r.get("outgoing_avg_latency", 0))
            lat_n = int(r.get("outgoing_latency_samples", 0))
            if lat_n > 0:
                agg_lat[wi] += lat * lat_n
                agg_lat_weight[wi] += lat_n
        cycle_centers.append(cc if cc is not None else 0)

    cycle_centers = np.array(cycle_centers)
    section_start_cycle = min(start for start, _ in window_bounds.values())
    section_end_cycle = max(end for _, end in window_bounds.values())
    rel_cycle_centers = cycle_centers - section_start_cycle
    avg_lat = np.divide(
        agg_lat,
        agg_lat_weight,
        out=np.full(n_windows, np.nan),
        where=agg_lat_weight > 0,
    )

    if raw_events_path.is_file() and tile_ts_rows:
        first_window_start = min(start for start, _ in window_bounds.values())
        window_size = int(tile_ts_rows[0].get("window_size", 64)) if tile_ts_rows else 64
        with raw_events_path.open(newline="") as f:
            for r in csv.DictReader(f):
                if section is not None and _pint(r.get("section")) != section:
                    continue
                if (r.get("event_type") or "").strip() != "load_return":
                    continue
                lat = r.get("latency", "")
                cyc = r.get("cycle", "")
                if not lat or not cyc:
                    continue
                cycle = int(cyc)
                wi_guess = (cycle - first_window_start) // window_size
                bounds = window_bounds.get(wi_guess)
                if bounds is None or not (bounds[0] <= cycle <= bounds[1]):
                    wi_guess = None
                    for wi, (start, end) in window_bounds.items():
                        if start <= cycle <= end:
                            wi_guess = wi
                            break
                if wi_guess is None:
                    continue
                wi_pos = window_pos.get(wi_guess)
                if wi_pos is None:
                    continue
                latency = float(lat)
                if r.get("is_local") == "1":
                    loc_lat_sum[wi_pos] += latency
                    loc_lat_n[wi_pos] += 1
                elif r.get("is_same_group") == "1":
                    same_lat_sum[wi_pos] += latency
                    same_lat_n[wi_pos] += 1
                else:
                    remote_lat_sum[wi_pos] += latency
                    remote_lat_n[wi_pos] += 1

    avg_local_lat = np.divide(loc_lat_sum, loc_lat_n, out=np.full(n_windows, np.nan), where=loc_lat_n > 0)
    avg_same_lat = np.divide(same_lat_sum, same_lat_n, out=np.full(n_windows, np.nan), where=same_lat_n > 0)
    avg_remote_lat = np.divide(remote_lat_sum, remote_lat_n, out=np.full(n_windows, np.nan), where=remote_lat_n > 0)

    fig = plt.figure(figsize=(12.8, 10.8))
    gs_fig = gridspec.GridSpec(
        3, 2,
        width_ratios=[40, 1.5],
        height_ratios=[1.2, 1.5, 1.0],
        hspace=0.34,
        wspace=0.24,
    )
    x_min = 0
    x_max = section_end_cycle - section_start_cycle
    x_label = f"Cycles (Section {section})" if section is not None else "Cycles"

    # (a) Stacked area
    ax_a = fig.add_subplot(gs_fig[0, 0])
    ax_a.fill_between(rel_cycle_centers, 0, agg_local,
                      step="mid", color=COL_LOCAL, alpha=0.85,
                      label=LOCALITY_LABELS["local"])
    ax_a.fill_between(rel_cycle_centers, agg_local, agg_local + agg_same,
                      step="mid", color=COL_SAME_GRP, alpha=0.75,
                      label=LOCALITY_LABELS["same_group"])
    ax_a.fill_between(rel_cycle_centers, agg_local + agg_same,
                      agg_local + agg_same + agg_remote,
                      step="mid", color=COL_REMOTE, alpha=0.75,
                      label=LOCALITY_LABELS["remote"])
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylabel("Events per window", fontsize=FONT_LABEL)
    ax_a.set_title("(a)  Aggregate traffic by locality class",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    ax_a.legend(loc="upper right", fontsize=FONT_LEGEND, ncol=3)
    _clean_spine(ax_a)
    ax_a.tick_params(labelbottom=True)
    ax_a.set_xlabel(x_label, fontsize=FONT_LABEL, labelpad=2)

    # (b) Per-tile incoming heatmap
    ax_b = fig.add_subplot(gs_fig[1, 0], sharex=ax_a)
    if n_windows > 1:
        half_w = (rel_cycle_centers[1] - rel_cycle_centers[0]) / 2
    else:
        half_w = 32
    extent = [0, x_max,
              -0.5, n_tiles - 0.5]
    vmax_heat = (np.percentile(heatmap_in[heatmap_in > 0], 95)
                 if np.any(heatmap_in > 0) else 1)
    cmap_heat = plt.cm.YlOrRd.copy()
    cmap_heat.set_bad(color="#F5F5F5")
    heat_plot = heatmap_in.copy()
    heat_plot[heat_plot == 0] = np.nan
    im_b = ax_b.imshow(heat_plot, aspect="auto", cmap=cmap_heat,
                       vmin=0, vmax=vmax_heat, extent=extent,
                       origin="lower", interpolation="nearest")
    for g in range(1, n_groups):
        ax_b.axhline(g * tpg - 0.5, color="#333333", lw=0.8, ls="--", alpha=0.5)
    for g in range(n_groups):
        mid = g * tpg + tpg / 2 - 0.5
        ax_b.text(1.003, mid, f"G{g}", ha="left", va="center",
                  transform=ax_b.get_yaxis_transform(),
                  fontsize=FONT_ANNOT, fontweight="bold",
                  color=GRP_COLORS[g % len(GRP_COLORS)], clip_on=False)
    ax_b.set_ylabel("Destination tile", fontsize=FONT_LABEL)
    ax_b.set_title("(b)  Memory requests received per tile over time",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    cax_b = fig.add_subplot(gs_fig[1, 1])
    cb_b = fig.colorbar(im_b, cax=cax_b)
    cb_b.set_label("Incoming events / window", fontsize=FONT_LABEL - 1)
    cb_b.ax.tick_params(labelsize=FONT_TICK - 1)
    ax_b.tick_params(labelbottom=True)
    ax_b.set_xlabel(x_label, fontsize=FONT_LABEL, labelpad=2)
    ax_b.set_yticks(np.arange(0, n_tiles, max(1, tpg // 2)))

    # (c) Load latency over time
    ax_c = fig.add_subplot(gs_fig[2, 0], sharex=ax_a)
    ax_c.plot(rel_cycle_centers, avg_lat, color=COL_REMOTE, lw=2.2,
              zorder=3, label="Overall Avg.")
    ax_c.plot(rel_cycle_centers, avg_local_lat, color=COL_LOCAL, lw=1.8,
              zorder=4, label="Tile Avg.")
    ax_c.plot(rel_cycle_centers, avg_same_lat, color=COL_SAME_GRP, lw=1.8,
              zorder=4, label="Group Avg.")
    ax_c.plot(rel_cycle_centers, avg_remote_lat, color=COL_ACCENT, lw=1.8,
              zorder=4, label="Cluster Avg.")
    ax_c.fill_between(rel_cycle_centers, 0, avg_lat, color=COL_REMOTE, alpha=0.12)
    ax_c.set_xlim(x_min, x_max)
    ax_c.set_xlabel(x_label, fontsize=FONT_LABEL)
    ax_c.set_ylabel("Latency (cycles)", fontsize=FONT_LABEL)
    ax_c.set_title("(c)  Load-return latency over time by locality",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    ax_c.legend(loc="upper right", fontsize=FONT_LEGEND, ncol=2)
    _clean_spine(ax_c)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    window_size = int(tile_ts_rows[0].get("window_size", 64)) if tile_ts_rows else 64
    fig.text(0.5, -0.005,
             f"{sec_lbl}  ·  {n_tiles} tiles  ·  {n_windows} windows × "
             f"{window_size} cycles  ·  Relative range: 0–{x_max:.0f}  ·  "
             f"Absolute cycle range: {cycle_centers[0]:.0f}–{cycle_centers[-1]:.0f}",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "temporal_overview", section)


# ===================================================================
# Figure 5 – Load-Return Latency Over Time (standalone, multi-granularity)
# ===================================================================

def plot_latency_over_time(tile_ts_rows, n_groups, output_dir, section):
    """Standalone latency figure for cross-design comparison.
    Two panels:
      (a) System-wide average latency + min/max envelope
      (b) Per-group average latency (one line per group)
    """
    windows = sorted(set(int(r["window_index"]) for r in tile_ts_rows))
    n_windows = len(windows)
    tiles = sorted(set(int(r["tile"]) for r in tile_ts_rows))
    n_tiles = len(tiles)
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    by_wt = {}
    for r in tile_ts_rows:
        by_wt[(int(r["window_index"]), int(r["tile"]))] = r

    cycle_centers = []
    sys_lat = np.zeros(n_windows)
    sys_weight = np.zeros(n_windows)
    grp_lat = np.zeros((n_groups, n_windows))
    grp_weight = np.zeros((n_groups, n_windows))
    tile_avgs = np.full((n_tiles, n_windows), np.nan)

    for wi, w in enumerate(windows):
        cc = None
        for ti, t in enumerate(tiles):
            r = by_wt.get((w, t))
            if r is None:
                continue
            if cc is None:
                cc = float(r["window_center_cycle"])
            lat = _pfloat(r.get("outgoing_avg_latency", 0))
            n_samples = int(r.get("outgoing_latency_samples", 0))
            if n_samples > 0:
                sys_lat[wi] += lat * n_samples
                sys_weight[wi] += n_samples
                g = t // tpg if tpg > 0 else 0
                if g < n_groups:
                    grp_lat[g, wi] += lat * n_samples
                    grp_weight[g, wi] += n_samples
                tile_avgs[ti, wi] = lat
        cycle_centers.append(cc if cc is not None else 0)

    cycle_centers = np.array(cycle_centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        sys_avg = np.where(sys_weight > 0, sys_lat / sys_weight, np.nan)
        grp_avg = np.where(grp_weight > 0, grp_lat / grp_weight, np.nan)
    tile_has_data = np.isfinite(tile_avgs).any(axis=0)
    tile_min = np.full(n_windows, np.nan)
    tile_max = np.full(n_windows, np.nan)
    if tile_has_data.any():
        valid_tile_avgs = tile_avgs[:, tile_has_data]
        tile_min[tile_has_data] = np.nanmin(valid_tile_avgs, axis=0)
        tile_max[tile_has_data] = np.nanmax(valid_tile_avgs, axis=0)

    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                      gridspec_kw={"hspace": 0.18})

    # (a) System-wide
    ax_a.fill_between(cycle_centers, tile_min, tile_max,
                      color=COL_REMOTE, alpha=0.12, label="Tile min\u2013max range")
    ax_a.plot(cycle_centers, sys_avg, color=COL_REMOTE, lw=2.5,
              marker="o", markersize=4, markerfacecolor="white",
              markeredgecolor=COL_REMOTE, markeredgewidth=1.2,
              label="System average", zorder=3)
    valid = ~np.isnan(sys_avg)
    if valid.any():
        mean_val = np.nanmean(sys_avg)
        ax_a.axhline(mean_val, color=COL_NEUTRAL, lw=1.2, ls=":",
                     alpha=0.6, zorder=1)
        ax_a.text(cycle_centers[valid][-1], mean_val,
                  f"  {mean_val:.1f} cyc", va="bottom", ha="left",
                  fontsize=FONT_ANNOT, color=COL_NEUTRAL, fontstyle="italic")
    ax_a.set_ylabel("Load-return latency (cycles)", fontsize=FONT_LABEL)
    ax_a.set_title("(a)  System-wide average load-return latency",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    ax_a.legend(loc="upper right", fontsize=FONT_LEGEND)
    _clean_spine(ax_a)

    # (b) Per-group
    for g in range(n_groups):
        valid_g = ~np.isnan(grp_avg[g])
        if not valid_g.any():
            continue
        ax_b.plot(cycle_centers, grp_avg[g],
                  color=GRP_COLORS[g % len(GRP_COLORS)], lw=2.0,
                  marker="o", markersize=3.5, markerfacecolor="white",
                  markeredgecolor=GRP_COLORS[g % len(GRP_COLORS)],
                  markeredgewidth=1.0,
                  label=f"Group {g}", zorder=3)
    ax_b.set_xlabel("Cycle", fontsize=FONT_LABEL)
    ax_b.set_ylabel("Load-return latency (cycles)", fontsize=FONT_LABEL)
    ax_b.set_title("(b)  Per-group average load-return latency",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    ax_b.legend(loc="upper right", fontsize=FONT_LEGEND, ncol=2)
    _clean_spine(ax_b)

    x_min = cycle_centers.min() - 20
    x_max = cycle_centers.max() + 20
    ax_a.set_xlim(x_min, x_max)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    window_size = int(tile_ts_rows[0].get("window_size", 64)) if tile_ts_rows else 64
    fig.text(0.5, -0.005,
             f"{sec_lbl}  \u00b7  {n_groups} groups \u00d7 {tpg} tiles/group ({n_tiles} tiles)  \u00b7  "
             f"{n_windows} windows \u00d7 {window_size} cycles",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "latency_timeseries", section)


# ===================================================================
# Figure 6b – Per-tile latency within a group
# ===================================================================

def plot_per_tile_group_latency(tile_ts_rows, n_groups, output_dir, section,
                                target_group=0):
    """Per-tile average latency over time for tiles in one group,
    with the group average as a reference line."""
    windows = sorted(set(int(r["window_index"]) for r in tile_ts_rows))
    n_windows = len(windows)
    tiles = sorted(set(int(r["tile"]) for r in tile_ts_rows))
    n_tiles = len(tiles)
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    grp_tiles = [t for t in tiles if t // tpg == target_group]
    if not grp_tiles:
        print(f"  [skip] No tiles in group {target_group}")
        return

    by_wt = {}
    for r in tile_ts_rows:
        by_wt[(int(r["window_index"]), int(r["tile"]))] = r

    cycle_centers = []
    tile_avgs = np.full((len(grp_tiles), n_windows), np.nan)
    grp_lat = np.zeros(n_windows)
    grp_weight = np.zeros(n_windows)

    for wi, w in enumerate(windows):
        cc = None
        for ti, t in enumerate(grp_tiles):
            r = by_wt.get((w, t))
            if r is None:
                continue
            if cc is None:
                cc = float(r["window_center_cycle"])
            lat = _pfloat(r.get("outgoing_avg_latency", 0))
            n_samples = int(r.get("outgoing_latency_samples", 0))
            if n_samples > 0:
                tile_avgs[ti, wi] = lat
                grp_lat[wi] += lat * n_samples
                grp_weight[wi] += n_samples
        cycle_centers.append(cc if cc is not None else 0)

    cycle_centers = np.array(cycle_centers)
    with np.errstate(invalid="ignore", divide="ignore"):
        grp_avg = np.where(grp_weight > 0, grp_lat / grp_weight, np.nan)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Individual tile lines (thin, distinct colors)
    cmap_tiles = plt.cm.tab20(np.linspace(0, 1, len(grp_tiles)))
    for ti, t in enumerate(grp_tiles):
        valid = ~np.isnan(tile_avgs[ti])
        if not valid.any():
            continue
        ax.plot(cycle_centers, tile_avgs[ti],
                color=cmap_tiles[ti], lw=1.2, alpha=0.7,
                label=f"Tile {t}", zorder=2)

    # Group average (bold)
    gcol = GRP_COLORS[target_group % len(GRP_COLORS)]
    ax.plot(cycle_centers, grp_avg, color=gcol, lw=3.0,
            marker="o", markersize=5, markerfacecolor="white",
            markeredgecolor=gcol, markeredgewidth=1.5,
            label=f"Group {target_group} avg", zorder=4)

    # Overall mean reference
    valid_avg = ~np.isnan(grp_avg)
    if valid_avg.any():
        mean_val = np.nanmean(grp_avg)
        ax.axhline(mean_val, color=COL_NEUTRAL, lw=1.2, ls=":",
                   alpha=0.6, zorder=1)
        ax.text(cycle_centers[valid_avg][-1], mean_val,
                f"  {mean_val:.1f} cyc", va="bottom", ha="left",
                fontsize=FONT_ANNOT, color=COL_NEUTRAL, fontstyle="italic")

    ax.set_xlabel("Cycle", fontsize=FONT_LABEL)
    ax.set_ylabel("Load-return latency (cycles)", fontsize=FONT_LABEL)
    ax.set_title(f"Per-tile load-return latency — Group {target_group} "
                 f"(tiles {grp_tiles[0]}\u2013{grp_tiles[-1]})",
                 fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)
    leg = ax.legend(loc="upper right", fontsize=FONT_LEGEND - 1, ncol=4,
                    framealpha=0.9, handlelength=2.5)
    for line in leg.get_lines():
        line.set_linewidth(3.0)
    _clean_spine(ax)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    window_size = int(tile_ts_rows[0].get("window_size", 64)) if tile_ts_rows else 64
    fig.text(0.5, -0.005,
             f"{sec_lbl}  \u00b7  Group {target_group}: {len(grp_tiles)} tiles  \u00b7  "
             f"{n_windows} windows \u00d7 {window_size} cycles",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, f"latency_tile_g{target_group}", section)


# ===================================================================
# Figure 7 – Traffic Volume vs. Actual Latency
# ===================================================================

def plot_traffic_vs_latency(raw_events_path, n_groups, output_dir, section):
    """Two panels showing where loads go and how long they actually take:
      (a) Tile×tile latency heatmap (avg measured latency per pair)
      (b) Scatter: load count vs avg latency per (src,dest) pair, by locality
    """
    if not raw_events_path.is_file():
        print("  [skip] Missing raw events CSV")
        return

    # Aggregate per (source_tile, dest_tile): count, total_latency, locality
    pair_data = defaultdict(lambda: {"count": 0, "lat_sum": 0.0, "lat_n": 0,
                                      "locality": "remote"})
    with raw_events_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if section is not None and _pint(r.get("section")) != section:
                continue
            if (r.get("event_type") or "").strip() != "load_return":
                continue
            st = _pint(r.get("tile"))
            dt = _pint(r.get("dest_tile", -1))
            lat = r.get("latency", "")
            if dt < 0 or not lat:
                continue
            lat = float(lat)
            key = (st, dt)
            d = pair_data[key]
            d["count"] += 1
            d["lat_sum"] += lat
            d["lat_n"] += 1
            if r.get("is_local") == "1":
                d["locality"] = "local"
            elif r.get("is_same_group") == "1":
                d["locality"] = "same_group"

    if not pair_data:
        print("  [skip] No load_return events with latency")
        return

    max_tile = max(max(s, d) for s, d in pair_data)
    n_tiles = max_tile + 1
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    # Build latency matrix
    lat_mat = np.full((n_tiles, n_tiles), np.nan)
    for (s, d), v in pair_data.items():
        if v["lat_n"] > 0:
            lat_mat[s, d] = v["lat_sum"] / v["lat_n"]

    # Auto-detect active source groups (those with cross-tile traffic)
    group_nonlocal = np.zeros(n_groups)
    for (s, d), v in pair_data.items():
        sg = s // tpg
        if s != d and sg < n_groups:
            group_nonlocal[sg] += v["count"]
    zoom_groups = [g for g in range(n_groups) if group_nonlocal[g] > 0]
    if not zoom_groups:
        zoom_groups = list(range(n_groups))

    # Source tiles = active source groups
    src_tiles = []
    for g in zoom_groups:
        src_tiles.extend(range(g * tpg, (g + 1) * tpg))
    src_set = set(src_tiles)

    # Dest tiles = all groups receiving traffic from source tiles
    dest_groups_set = set()
    for (s, d), v in pair_data.items():
        if s in src_set and v["count"] > 0:
            dest_groups_set.add(d // tpg)
    dest_groups = sorted(dest_groups_set)
    dest_tiles = []
    for g in dest_groups:
        dest_tiles.extend(range(g * tpg, (g + 1) * tpg))

    sub = lat_mat[np.ix_(src_tiles, dest_tiles)]
    n_src = len(src_tiles)
    n_dst = len(dest_tiles)

    cell_size = 0.34
    # Adaptive: maximize cell size while staying under 8000px at output DPI
    max_px = 7900  # leave margin below 8192
    max_w_inches = max_px / DPI - 3.5
    max_h_inches = max_px / DPI - 2.5
    cell_size = min(0.50, max_w_inches / max(n_dst, 1),
                          max_h_inches / max(n_src, 1))
    cell_size = max(cell_size, 0.20)  # floor for very large matrices
    fig_w = n_dst * cell_size + 3.5
    fig_h = n_src * cell_size + 2.5

    # ---- Figure (a): full-width latency heatmap ----
    fig_a, ax_a = plt.subplots(figsize=(fig_w, fig_h))

    # Rectangular latency heatmap (src groups → all dest tiles)
    # Green → yellow → red sequential (low=fast/green, high=slow/red)
    _cmap_colors = ["#1a9641", "#55b748", "#91cf60", "#d0ec8a",
                    "#f0f4a4", "#fee08b", "#fdae61", "#f46d43",
                    "#d73027", "#a50026"]
    cmap_lat = LinearSegmentedColormap.from_list("lat_GnYlRd", _cmap_colors, N=256)
    cmap_lat.set_bad(color="#FFFFFF")
    valid_lats = sub[~np.isnan(sub)]
    vmin_lat = np.floor(valid_lats.min()) if len(valid_lats) else 1
    vmax_lat = np.ceil(valid_lats.max()) if len(valid_lats) else 10
    lat_norm = PowerNorm(gamma=0.65, vmin=vmin_lat, vmax=vmax_lat)
    im_a = ax_a.imshow(sub, origin="lower", cmap=cmap_lat, aspect="equal",
                       norm=lat_norm, interpolation="nearest")

    # Cell annotations with latency values
    if n_src * n_dst <= 4500:
        annot_fs = max(6.0, FONT_ANNOT - 0.5 * (max(n_src, n_dst) > 32)
                                        - 0.5 * (max(n_src, n_dst) > 48))
        for i in range(n_src):
            for j in range(n_dst):
                val = sub[i, j]
                if not np.isnan(val):
                    ax_a.text(j, i, f"{val:.0f}", ha="center", va="center",
                              fontsize=annot_fs, fontweight="bold", color="black")

    # Group boundaries — source (y-axis)
    src_offsets = {}
    offset = 0
    for g in zoom_groups:
        if offset > 0:
            ax_a.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax_a.text(-0.04, mid, f"G{g}", ha="right", va="center",
                  fontsize=FONT_ANNOT + 3, fontweight="bold",
                  color=GRP_COLORS[g % len(GRP_COLORS)],
                  transform=ax_a.get_yaxis_transform(), clip_on=False)
        src_offsets[g] = offset
        offset += tpg

    # Group boundaries — dest (x-axis)
    dst_offsets = {}
    offset = 0
    for gi, g in enumerate(dest_groups):
        if gi > 0:
            ax_a.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax_a.text(mid, -0.06, f"G{g}", ha="center", va="top",
                  fontsize=FONT_ANNOT + 3, fontweight="bold",
                  color=GRP_COLORS[g % len(GRP_COLORS)],
                  transform=ax_a.get_xaxis_transform(), clip_on=False)
        dst_offsets[g] = offset
        offset += tpg

    # Highlight self-group blocks (src_group == dest_group)
    for g in zoom_groups:
        if g in dst_offsets:
            rect_xy = (dst_offsets[g] - 0.5, src_offsets[g] - 0.5)
            gcol = GRP_COLORS[g % len(GRP_COLORS)]
            # Dark backing outline for contrast against any background
            ax_a.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=8.0, edgecolor="#000000",
                facecolor="none", alpha=0.25, zorder=4, clip_on=False,
            ))
            # Colored glow
            ax_a.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=5.0, edgecolor=gcol,
                facecolor="none", alpha=0.5, zorder=5, clip_on=False,
            ))
            # Crisp colored border
            ax_a.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=3.0, edgecolor=gcol,
                facecolor="none", zorder=6, clip_on=False,
            ))

    tick_fs = max(6.5, FONT_TICK - 1.5 * (n_dst > 48))
    ax_a.set_xticks(range(n_dst))
    ax_a.set_yticks(range(n_src))
    ax_a.set_xticklabels(dest_tiles, fontsize=tick_fs, rotation=90)
    ax_a.set_yticklabels(src_tiles, fontsize=tick_fs)
    ax_a.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
    ax_a.set_ylabel("Source tile", fontsize=FONT_LABEL)

    src_label = ", ".join(f"G{g}" for g in zoom_groups)
    dst_label = ", ".join(f"G{g}" for g in dest_groups)
    ax_a.set_title("Average load-return latency per tile (cycles)",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)
    divider_a = make_axes_locatable(ax_a)
    cax_a = divider_a.append_axes("right", size="3.5%", pad=1.00)
    cb_a = fig_a.colorbar(im_a, cax=cax_a)
    cb_a.set_label("Avg latency (cycles)", fontsize=FONT_LABEL - 1)
    # Clean round-number ticks
    _lo_i, _hi_i = int(np.floor(vmin_lat)), int(np.ceil(vmax_lat))
    _round_ticks = [_lo_i] + [v for v in range(5, _hi_i + 1, 5) if v > _lo_i and v < _hi_i] + [_hi_i]
    cb_a.set_ticks(_round_ticks)
    cb_a.ax.tick_params(labelsize=FONT_TICK)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    n_pairs = len(pair_data)
    total_loads = sum(v["count"] for v in pair_data.values())
    global_avg = (sum(v["lat_sum"] for v in pair_data.values()) /
                  max(1, sum(v["lat_n"] for v in pair_data.values())))
    fig_a.text(0.5, -0.005,
               f"{sec_lbl}  ·  {n_src} source tiles ({src_label}) → "
               f"{n_dst} dest tiles ({dst_label})  ·  "
               f"{n_pairs} active pairs  ·  "
               f"{_nice_count(total_loads)} load returns  ·  "
               f"Global avg: {global_avg:.1f} cycles",
               ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig_a, output_dir, "latency_matrix", section)

    # ---- Figure (b): contention scatter (separate page) ----
    fig_b, ax_b = plt.subplots(figsize=(7, 5.5))
    for lc in LOCALITY_CLASSES:
        xs, ys = [], []
        for (s, d), v in pair_data.items():
            if v["locality"] == lc and v["lat_n"] > 0:
                xs.append(v["count"])
                ys.append(v["lat_sum"] / v["lat_n"])
        if xs:
            ax_b.scatter(xs, ys, s=40, alpha=0.7,
                         color=LOCALITY_COLORS[lc], edgecolor="#333333", lw=0.4,
                         label=LOCALITY_LABELS[lc], zorder=3)

    ax_b.set_xlabel("Load returns (count per src→dest pair)", fontsize=FONT_LABEL)
    ax_b.set_ylabel("Avg load-return latency (cycles)", fontsize=FONT_LABEL)
    ax_b.set_title("Contention: traffic volume vs. actual latency",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)
    ax_b.legend(loc="upper right", fontsize=FONT_LEGEND)
    _clean_spine(ax_b)

    fig_b.text(0.5, -0.005,
               f"{sec_lbl}  ·  {n_pairs} src→dest pairs  ·  "
               f"{_nice_count(total_loads)} load returns  ·  "
               f"Global avg: {global_avg:.1f} cycles",
               ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig_b, output_dir, "latency_contention", section)


def plot_latency_over_minimum(raw_events_path, n_groups, output_dir, section):
    """Tile-pair latency heatmap normalized by hierarchy-aware ideal minima."""
    if not raw_events_path.is_file():
        print("  [skip] Missing raw events CSV")
        return

    pair_data = defaultdict(lambda: {"count": 0, "lat_sum": 0.0, "lat_n": 0,
                                      "locality": "remote"})
    with raw_events_path.open(newline="") as f:
        for r in csv.DictReader(f):
            if section is not None and _pint(r.get("section")) != section:
                continue
            if (r.get("event_type") or "").strip() != "load_return":
                continue
            st = _pint(r.get("tile"))
            dt = _pint(r.get("dest_tile", -1))
            lat = r.get("latency", "")
            if dt < 0 or not lat:
                continue
            lat = float(lat)
            key = (st, dt)
            data = pair_data[key]
            data["count"] += 1
            data["lat_sum"] += lat
            data["lat_n"] += 1
            if r.get("is_local") == "1":
                data["locality"] = "local"
            elif r.get("is_same_group") == "1":
                data["locality"] = "same_group"

    if not pair_data:
        print("  [skip] No load_return events with latency")
        return

    max_tile = max(max(s, d) for s, d in pair_data)
    n_tiles = max_tile + 1
    tpg = n_tiles // n_groups if n_groups > 0 else n_tiles

    lat_mat = np.full((n_tiles, n_tiles), np.nan)
    loc_mat = np.full((n_tiles, n_tiles), "", dtype=object)
    for (s, d), data in pair_data.items():
        if data["lat_n"] > 0:
            lat_mat[s, d] = data["lat_sum"] / data["lat_n"]
            loc_mat[s, d] = data["locality"]

    group_nonlocal = np.zeros(n_groups)
    for (s, d), data in pair_data.items():
        sg = s // tpg
        if s != d and sg < n_groups:
            group_nonlocal[sg] += data["count"]
    zoom_groups = [g for g in range(n_groups) if group_nonlocal[g] > 0]
    if not zoom_groups:
        zoom_groups = list(range(n_groups))

    src_tiles = []
    for group in zoom_groups:
        src_tiles.extend(range(group * tpg, (group + 1) * tpg))
    src_set = set(src_tiles)

    dest_groups_set = set()
    for (s, d), data in pair_data.items():
        if s in src_set and data["count"] > 0:
            dest_groups_set.add(d // tpg)
    dest_groups = sorted(dest_groups_set)
    dest_tiles = []
    for group in dest_groups:
        dest_tiles.extend(range(group * tpg, (group + 1) * tpg))

    sub_lat = lat_mat[np.ix_(src_tiles, dest_tiles)]
    sub_loc = loc_mat[np.ix_(src_tiles, dest_tiles)]
    n_src = len(src_tiles)
    n_dst = len(dest_tiles)

    cell_size = 0.34
    max_px = 7900
    max_w_inches = max_px / DPI - 3.5
    max_h_inches = max_px / DPI - 2.5
    cell_size = min(0.50, max_w_inches / max(n_dst, 1),
                          max_h_inches / max(n_src, 1))
    cell_size = max(cell_size, 0.20)
    fig_w = n_dst * cell_size + 3.5
    fig_h = n_src * cell_size + 2.5

    baseline_map = {"local": 1.0, "same_group": 3.0, "remote": 5.0}
    delta_mat = np.full_like(sub_lat, np.nan, dtype=float)
    for i in range(n_src):
        for j in range(n_dst):
            latency = sub_lat[i, j]
            if np.isnan(latency):
                continue
            locality = sub_loc[i, j] or "remote"
            delta_mat[i, j] = max(0.0, latency - baseline_map.get(locality, 5.0))

    valid_delta = delta_mat[~np.isnan(delta_mat)]
    vmax_delta = np.ceil(valid_delta.max()) if len(valid_delta) else 1.0
    vmax_delta = max(1.0, vmax_delta)
    rounded_delta_mat = np.where(
        np.isnan(delta_mat),
        np.nan,
        np.floor(delta_mat + 0.5),
    )

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    _cmap_colors = ["#1a9641", "#55b748", "#91cf60", "#d0ec8a",
                    "#f0f4a4", "#fee08b", "#fdae61", "#f46d43",
                    "#d73027", "#a50026"]
    cmap_delta = LinearSegmentedColormap.from_list(
        "lat_over_minimum",
        _cmap_colors,
        N=256,
    )
    cmap_delta.set_bad(color="#FFFFFF")
    base_norm = PowerNorm(gamma=0.38, vmin=0.0, vmax=vmax_delta)
    if vmax_delta > 2.0:
        x_points = [0.0, 1.0, min(2.0, vmax_delta)]
        y_points = [float(base_norm(0.0)),
                    float(base_norm(min(1.0, vmax_delta))) * 0.78,
                    float(base_norm(min(2.0, vmax_delta))) * 0.86]
        if vmax_delta > 10.0:
            x_points.extend([5.0, 10.0, vmax_delta])
            y_points.extend([
                float(base_norm(5.0)),
                min(0.97, float(base_norm(10.0)) + 0.10),
                1.0,
            ])
        elif vmax_delta > 5.0:
            x_points.extend([5.0, vmax_delta])
            y_points.extend([float(base_norm(5.0)), 1.0])
        else:
            x_points.append(vmax_delta)
            y_points.append(1.0)
        y_points = np.maximum.accumulate(y_points)
        delta_norm = PiecewiseNorm(x_points, y_points, vmin=0.0, vmax=vmax_delta)
    else:
        delta_norm = base_norm
    im = ax.imshow(rounded_delta_mat, origin="lower", cmap=cmap_delta, aspect="equal",
                   norm=delta_norm, interpolation="nearest")

    if n_src * n_dst <= 4500:
        annot_fs = max(6.0, FONT_ANNOT - 0.5 * (max(n_src, n_dst) > 32)
                                        - 0.5 * (max(n_src, n_dst) > 48))
        for i in range(n_src):
            for j in range(n_dst):
                value = delta_mat[i, j]
                if not np.isnan(value):
                    rounded_value = int(np.floor(value + 0.5))
                    ax.text(j, i, f"{rounded_value}", ha="center", va="center",
                            fontsize=annot_fs, fontweight="bold", color="black")

    src_offsets = {}
    offset = 0
    for group in zoom_groups:
        if offset > 0:
            ax.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(-0.04, mid, f"G{group}", ha="right", va="center",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[group % len(GRP_COLORS)],
                transform=ax.get_yaxis_transform(), clip_on=False)
        src_offsets[group] = offset
        offset += tpg

    dst_offsets = {}
    offset = 0
    for idx, group in enumerate(dest_groups):
        if idx > 0:
            ax.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(mid, -0.06, f"G{group}", ha="center", va="top",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[group % len(GRP_COLORS)],
                transform=ax.get_xaxis_transform(), clip_on=False)
        dst_offsets[group] = offset
        offset += tpg

    for group in zoom_groups:
        if group in dst_offsets:
            rect_xy = (dst_offsets[group] - 0.5, src_offsets[group] - 0.5)
            group_col = GRP_COLORS[group % len(GRP_COLORS)]
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=8.0, edgecolor="#000000",
                facecolor="none", alpha=0.25, zorder=4, clip_on=False,
            ))
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=5.0, edgecolor=group_col,
                facecolor="none", alpha=0.5, zorder=5, clip_on=False,
            ))
            ax.add_patch(Rectangle(
                rect_xy, tpg, tpg,
                linewidth=3.0, edgecolor=group_col,
                facecolor="none", zorder=6, clip_on=False,
            ))

    tick_fs = max(6.5, FONT_TICK - 1.5 * (n_dst > 48))
    ax.set_xticks(range(n_dst))
    ax.set_yticks(range(n_src))
    ax.set_xticklabels(dest_tiles, fontsize=tick_fs, rotation=90)
    ax.set_yticklabels(src_tiles, fontsize=tick_fs)
    ax.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
    ax.set_ylabel("Source tile", fontsize=FONT_LABEL)
    ax.set_title("Average load-return latency above ideal minimum",
                 fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=1.00)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Excess latency over ideal minimum (cycles)", fontsize=FONT_LABEL - 1)
    ticks = sorted(set([0, 1, 2, 5] + [v for v in range(10, int(vmax_delta) + 1, 5)] + [int(vmax_delta)]))
    ticks = [tick for tick in ticks if 0 <= tick <= vmax_delta]
    cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=FONT_TICK)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    src_label = ", ".join(f"G{group}" for group in zoom_groups)
    dst_label = ", ".join(f"G{group}" for group in dest_groups)
    n_pairs = len(pair_data)
    total_loads = sum(data["count"] for data in pair_data.values())
    global_avg_delta = float(np.nanmean(valid_delta)) if len(valid_delta) else 0.0
    fig.text(0.5, -0.005,
             f"{sec_lbl}  ·  {n_src} source tiles ({src_label}) → "
             f"{n_dst} dest tiles ({dst_label})  ·  "
             f"{n_pairs} active pairs  ·  "
             f"{_nice_count(total_loads)} load returns  ·  "
             f"Global avg excess: {global_avg_delta:.1f} cycles  ·  "
             f"Baselines: local=1, same-group=3, remote=5",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "latency_excess_matrix", section)


# ===================================================================
# CLI
# ===================================================================

def _parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Generate thesis-quality communication analysis figures.")
    p.add_argument("input_path",
                   help="Benchmark result directory (contains data/ and plots/)")
    p.add_argument("--section", type=int, default=None,
                   help="Section to plot (default: all)")
    p.add_argument("--n-groups", type=int, default=4,
                   help="Number of tile groups (default: 4)")
    p.add_argument("--figures", nargs="*", default=None,
                   help="Which figures: matrix zoom locality correlation "
                        "temporal latency tile_latency contention latency_over_minimum (default: all)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _apply_thesis_style()

    paths = _resolve_paths(args.input_path)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_pdfs(output_dir)

    section  = args.section
    n_groups = args.n_groups
    figs = (set(args.figures) if args.figures
            else {"matrix", "zoom", "locality", "correlation", "temporal",
                  "latency", "tile_latency", "contention"})

    print(f"Generating thesis figures → {output_dir}")

    sd_rows = _filter_section(
        _load_csv(paths["summary"] / "source_dest_counts.csv"), section)
    st_rows = _filter_section(
        _load_csv(paths["summary"] / "source_tile_locality.csv"), section)

    ts_path = paths["timeseries"] / "comm_timeseries_tiles.csv"
    ts_rows = (_filter_section(_load_csv(ts_path), section)
               if ts_path.is_file() else [])

    if "matrix" in figs:
        print("\n[1/5] Traffic matrix …")
        plot_traffic_matrix_zoom(sd_rows, n_groups, output_dir, section)
        plot_traffic_matrix(sd_rows, n_groups, output_dir, section)

    if "zoom" in figs:
        print("\n[2/5] Traffic matrix (zoomed alias) …")
        plot_traffic_matrix_zoom(sd_rows, n_groups, output_dir, section)

    if "locality" in figs:
        print("\n[3/5] Locality & latency …")
        plot_locality_latency(st_rows, paths["events"], n_groups,
                              output_dir, section)

    if "correlation" in figs:
        print("\n[4/5] Communication–stall correlation …")
        plot_comm_stall_correlation(paths["events"], paths["stalls"],
                                    ts_rows, n_groups, output_dir, section)

    if "temporal" in figs and ts_rows:
        print("\n[5/6] Temporal profile …")
        plot_temporal_profile(ts_rows, paths["events"], paths["result_dir"], n_groups, output_dir, section)

    if "latency" in figs and ts_rows:
        print("\n[6/7] Latency over time …")
        plot_latency_over_time(ts_rows, n_groups, output_dir, section)

    if "tile_latency" in figs and ts_rows:
        for tg in range(n_groups):
            print(f"\n[6b] Per-tile latency — Group {tg} …")
            plot_per_tile_group_latency(ts_rows, n_groups, output_dir,
                                        section, target_group=tg)

    if "contention" in figs:
        print("\n[7/7] Traffic volume vs. latency …")
        plot_traffic_vs_latency(paths["events"], n_groups, output_dir, section)

    if "latency_over_minimum" in figs:
        print("\n[7b] Latency above hierarchy minimum …")
        plot_latency_over_minimum(paths["events"], n_groups, output_dir, section)

    print("\nDone.")


if __name__ == "__main__":
    main()

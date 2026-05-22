#!/usr/bin/env python3
"""Thesis-quality communication analysis figures for MemPool.

Generates publication-ready figures including:
    1. Tile-to-tile traffic matrix
    2. Group-level traffic aggregate
    3. Source/destination request pressure by tile
    4. Locality breakdown & latency by network distance
    5. Temporal communication profile (stacked area + incoming heatmap + latency)

Usage:
    python plot_comm_thesis.py <result_dir> [--section 1]
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm, Normalize, PowerNorm, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

# ---------------------------------------------------------------------------
# Thesis style constants
# ---------------------------------------------------------------------------

FONT_SUBTITLE = 11
FONT_LABEL    = 10.5
FONT_TICK     = 9
FONT_ANNOT    = 8.5
FONT_LEGEND   = 9

# Colorblind-safe palette (Okabe-Ito-inspired)
COL_LOCAL     = "#0072B2"   # blue  – local / intra-tile
COL_SAME_SUB  = "#009E73"  # green – same subgroup, different tile
COL_SAME_GRP  = "#56B4E9"  # sky blue – same group, different tile
COL_REMOTE    = "#D55E00"  # vermillion – remote / inter-group
COL_ACCENT    = "#E69F00"  # amber – highlights / p95
COL_NEUTRAL   = "#999999"

GRP_COLORS       = ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]

DPI = 300
FIG_TEXT_COLOR = "#222222"

LOCALITY_LABELS  = {
    "local":         "Local (intra-tile)",
    "same_subgroup": "Same subgroup",
    "same_group":    "Same group, other subgroup",
    "remote":        "Remote (inter-group)",
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


def _load_topology(path: Path, default_n_groups: int):
    topology = {
        "n_groups": default_n_groups,
        "n_tiles": None,
        "tpg": None,
        "n_subgroups_per_group": None,
        "tiles_per_subgroup": None,
    }
    topo_path = path / "topology.env"
    if not topo_path.is_file():
        return topology

    values = {}
    with topo_path.open() as handle:
        for line in handle:
            line = line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()

    n_groups = int(values.get("NUM_GROUPS", topology["n_groups"]))
    num_cores = int(values.get("NUM_CORES", 0))
    cores_per_tile = int(values.get("NUM_CORES_PER_TILE", 0))
    n_tiles = num_cores // cores_per_tile if num_cores and cores_per_tile else None
    tpg = n_tiles // n_groups if n_tiles and n_groups else None
    n_sub = int(values.get("NUM_SUB_GROUPS_PER_GROUP", 0)) or None
    tps = tpg // n_sub if tpg and n_sub else None

    topology["n_groups"] = n_groups
    topology["n_tiles"] = n_tiles
    topology["tpg"] = tpg
    topology["n_subgroups_per_group"] = n_sub
    topology["tiles_per_subgroup"] = tps
    return topology


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
            "events":     p / "data" / "comm_events_benchmark.csv",
            "output":     p / "plots" / "communication",
        }
    return {
        "result_dir":  p.parent.parent if p.parent.name == "data" else p.parent,
        "events":     p / "comm_events_benchmark.csv",
        "output":     p / "plots",
    }


def _nice_count(v):
    """Format a count for annotation (e.g., 1234 → '1.2k')."""
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1000:
        return f"{v / 1000:.1f}k"
    return f"{v:.0f}"


def _classify_locality(row):
    if row.get("is_local") == "1" or _pint(row.get("is_local")) == 1:
        return "local"
    if row.get("is_same_subgroup") == "1" or _pint(row.get("is_same_subgroup")) == 1:
        return "same_subgroup"
    if row.get("is_same_group") == "1" or _pint(row.get("is_same_group")) == 1:
        return "same_group"
    return "remote"


def _latency_baseline_map(topology):
    """Ideal minima for the trace-measured load-return metric."""
    n_subgroups = topology.get("n_subgroups_per_group") or 1
    if n_subgroups <= 1:
        return {
            "local": 1.0,
            "same_subgroup": 3.0,
            "same_group": 3.0,
            "remote": 5.0,
        }
    return {
        "local": 1.0,
        "same_subgroup": 3.0,
        "same_group": 5.0,
        "remote": 7.0,
    }


def _build_pair_data(raw_events):
    """Build per-(source_tile, dest_tile) latency aggregates from pre-loaded raw events."""
    pair_data = defaultdict(lambda: {"count": 0, "lat_sum": 0.0, "lat_n": 0,
                                      "latencies": [],
                                      "locality": "remote", "source_group": -1,
                                      "dest_group": -1})
    for r in raw_events:
        if (r.get("event_type") or "").strip() != "load_return":
            continue
        st = _pint(r.get("tile"))
        dt = _pint(r.get("dest_tile", -1))
        lat = r.get("latency", "")
        cyc = r.get("cycle", "")
        if dt < 0 or not lat or not cyc:
            continue
        lat = float(lat)
        d = pair_data[(st, dt)]
        d["count"] += 1
        d["lat_sum"] += lat
        d["lat_n"] += 1
        d["latencies"].append((int(cyc), lat))
        d["locality"] = _classify_locality(r)
        d["source_group"] = _pint(r.get("group"))
        d["dest_group"] = _pint(r.get("dest_group"))
    return pair_data


def _add_subgroup_boxes(ax, topology, tpg, n_groups, zorder_base=7):
    """Draw subgroup diagonal boxes, boundary lines, and axis labels."""
    tps = topology.get("tiles_per_subgroup")
    n_sub = topology.get("n_subgroups_per_group")
    n_tiles = n_groups * tpg
    if not tps or not n_sub or tps >= tpg:
        return  # no subgroups or subgroup == group

    # Dashed boundary lines at each subgroup edge (skip group boundaries)
    for sg_boundary in range(tps, n_tiles, tps):
        if sg_boundary % tpg != 0:
            ax.axhline(sg_boundary - 0.5, color="#888888", lw=0.6,
                       ls="--", alpha=0.5, zorder=3)
            ax.axvline(sg_boundary - 0.5, color="#888888", lw=0.6,
                       ls="--", alpha=0.5, zorder=3)

    # Diagonal boxes for each subgroup
    for g in range(n_groups):
        gcol = GRP_COLORS[g % len(GRP_COLORS)]
        for s in range(n_sub):
            origin = g * tpg + s * tps - 0.5
            ax.add_patch(Rectangle(
                (origin, origin), tps, tps,
                linewidth=1.5, edgecolor=gcol,
                facecolor="none", ls="--", alpha=0.7,
                zorder=zorder_base, clip_on=False,
            ))

    # Axis labels — placed between group label and tick labels
    for g in range(n_groups):
        gcol = GRP_COLORS[g % len(GRP_COLORS)]
        for s in range(n_sub):
            sg_mid = g * tpg + s * tps + tps / 2
            ax.text(-0.02, sg_mid, f"s{s}", ha="right", va="center",
                    fontsize=FONT_ANNOT - 1, color=gcol, alpha=0.6,
                    transform=ax.get_yaxis_transform(), clip_on=False)
            ax.text(sg_mid, -0.03, f"s{s}", ha="center", va="top",
                    fontsize=FONT_ANNOT - 1, color=gcol, alpha=0.6,
                    transform=ax.get_xaxis_transform(), clip_on=False)


def _timeseries_locality_counts(row):
    local = _pint(row.get("local_events", 0))
    same_subgroup = _pint(row.get("same_subgroup_events", 0))
    same_group_total = _pint(row.get("same_group_events", 0))
    same_group_other = max(0, same_group_total - same_subgroup)
    remote = _pint(row.get("remote_group_events", 0))
    return {
        "local": local,
        "same_subgroup": same_subgroup,
        "same_group": same_group_other,
        "remote": remote,
    }


# ---------------------------------------------------------------------------
# Inline summarisation (replaces summarize_comm_events.py and
# summarize_comm_timeseries.py — all consumers are inside this file)
# ---------------------------------------------------------------------------

WINDOW_SIZE = 64


def _build_source_dest_rows(raw_events):
    """GROUP BY (section, tile, dest_tile, event_type, …) → count.

    Produces rows compatible with the former source_dest_counts.csv schema
    so that _build_traffic_matrices / plot_traffic_matrix_* work unchanged.
    """
    from collections import Counter
    counts = Counter()
    for r in raw_events:
        et = (r.get("event_type") or "").strip()
        counts[(
            _pint(r.get("section")),
            _pint(r.get("tile")),
            _pint(r.get("group")),
            _pint(r.get("subgroup")),
            _pint(r.get("dest_tile", -1)),
            _pint(r.get("dest_group", -1)),
            _pint(r.get("dest_subgroup", -1)),
            (r.get("region") or "").strip(),
            et,
        )] += 1

    out = []
    for key in sorted(counts):
        sec, tile, grp, sub, dt, dg, dsub, region, et = key
        out.append({
            "section":          sec,
            "source_tile":      tile,
            "source_group":     grp,
            "source_subgroup":  "" if sub < 0 else sub,
            "dest_tile":        "" if dt < 0 else dt,
            "dest_group":       "" if dg < 0 else dg,
            "dest_subgroup":    "" if dsub < 0 else dsub,
            "region":           region,
            "event_type":       et,
            "count":            counts[key],
        })
    return out


def _build_timeseries_rows(raw_events):
    """Window raw events into per-(window, tile) buckets.

    Only the 12 fields actually consumed by the plotting functions are
    produced.  Also computes per-locality-class latency sums so that
    plot_temporal_profile panel (c) does not need a second pass over
    raw_events.
    """
    cycles = [int(r["cycle"]) for r in raw_events
              if r.get("cycle") not in (None, "")]
    if not cycles:
        return []
    min_cycle = min(cycles)
    window = WINDOW_SIZE

    Bucket = lambda: {
        "incoming_events": 0,
        "local_events": 0,
        "same_subgroup_events": 0,
        "same_group_events": 0,
        "remote_group_events": 0,
        "lat_total": 0.0,
        "lat_n": 0,
        "loc_lat_total": 0.0,   "loc_lat_n": 0,
        "sub_lat_total": 0.0,   "sub_lat_n": 0,
        "grp_lat_total": 0.0,   "grp_lat_n": 0,
        "rem_lat_total": 0.0,   "rem_lat_n": 0,
    }
    grouped = defaultdict(Bucket)

    for r in raw_events:
        cyc_s = r.get("cycle", "")
        if not cyc_s:
            continue
        cycle = int(cyc_s)
        tile_s = r.get("tile", "")
        if not tile_s:
            continue
        source_tile = int(tile_s)
        wi = (cycle - min_cycle) // window
        et = (r.get("event_type") or "").strip()

        # Source-side locality counts
        src = grouped[(wi, source_tile)]
        is_local = _pint(r.get("is_local"))
        is_same_sub = _pint(r.get("is_same_subgroup"))
        is_same_grp = _pint(r.get("is_same_group"))
        if is_local == 1:
            src["local_events"] += 1
        elif is_same_sub == 1:
            src["same_subgroup_events"] += 1
            src["same_group_events"] += 1
        elif is_same_grp == 1:
            src["same_group_events"] += 1
        elif r.get("dest_tile") not in (None, ""):
            src["remote_group_events"] += 1

        # Outgoing latency (load_return only)
        lat_s = r.get("latency", "")
        if lat_s and et == "load_return":
            lat = float(lat_s)
            src["lat_total"] += lat
            src["lat_n"] += 1
            # Per-locality-class latency for panel (c)
            locality = _classify_locality(r)
            if locality == "local":
                src["loc_lat_total"] += lat; src["loc_lat_n"] += 1
            elif locality == "same_subgroup":
                src["sub_lat_total"] += lat; src["sub_lat_n"] += 1
            elif locality == "same_group":
                src["grp_lat_total"] += lat; src["grp_lat_n"] += 1
            else:
                src["rem_lat_total"] += lat; src["rem_lat_n"] += 1

        # Destination-side incoming count
        dt_s = r.get("dest_tile", "")
        if dt_s:
            grouped[(wi, int(dt_s))]["incoming_events"] += 1

    out = []
    for (wi, tile) in sorted(grouped):
        b = grouped[(wi, tile)]
        start = min_cycle + wi * window
        out.append({
            "window_index":           wi,
            "tile":                   tile,
            "window_start_cycle":     start,
            "window_end_cycle":       start + window - 1,
            "window_center_cycle":    start + window / 2.0,
            "window_size":            window,
            "incoming_events":        b["incoming_events"],
            "local_events":           b["local_events"],
            "same_subgroup_events":   b["same_subgroup_events"],
            "same_group_events":      b["same_group_events"],
            "remote_group_events":    b["remote_group_events"],
            "outgoing_avg_latency":   (b["lat_total"] / b["lat_n"]) if b["lat_n"] else 0.0,
            "outgoing_latency_samples": b["lat_n"],
            # Per-locality-class latency (used by plot_temporal_profile panel c)
            "_loc_lat_total":         b["loc_lat_total"],
            "_loc_lat_n":             b["loc_lat_n"],
            "_sub_lat_total":         b["sub_lat_total"],
            "_sub_lat_n":             b["sub_lat_n"],
            "_grp_lat_total":         b["grp_lat_total"],
            "_grp_lat_n":             b["grp_lat_n"],
            "_rem_lat_total":         b["rem_lat_total"],
            "_rem_lat_n":             b["rem_lat_n"],
        })
    return out


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

def _build_traffic_matrices(source_dest_rows, topology):
    n_groups = topology["n_groups"]
    issue_rows = [
        (
            _pint(r["source_tile"]),
            _pint(r["dest_tile"]),
            _pint(r["count"]),
            _pint(r.get("source_group")),
            _pint(r.get("dest_group")),
        )
        for r in source_dest_rows
        if (r.get("event_type") or "").strip() in ("load_issue", "store_issue")
    ]
    if not issue_rows:
        return None

    max_tile = max(max(s, d) for s, d, *_ in issue_rows)
    n_tiles = topology.get("n_tiles") or (max_tile + 1)
    tpg = topology.get("tpg") or (n_tiles // n_groups)

    mat = np.zeros((n_tiles, n_tiles), dtype=float)
    for s, d, c, _, _ in issue_rows:
        if 0 <= s < n_tiles and 0 <= d < n_tiles:
            mat[s, d] += c

    gmat = np.zeros((n_groups, n_groups), dtype=float)
    for s, d, c, sg, dg in issue_rows:
        if sg < 0:
            sg = s // tpg
        if dg < 0:
            dg = d // tpg
        if sg < n_groups and dg < n_groups:
            gmat[sg, dg] += c

    return {
        "issue_rows": issue_rows,
        "mat": mat,
        "gmat": gmat,
        "n_tiles": n_tiles,
        "tpg": tpg,
    }

def plot_traffic_matrix_group(source_dest_rows, topology, output_dir, section):
    """Emit the standalone group-level traffic aggregate figure."""
    n_groups = topology["n_groups"]
    matrices = _build_traffic_matrices(source_dest_rows, topology)
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
# Figure 1b – Full Tile-level Traffic Matrix
# ===================================================================

def plot_traffic_matrix_full(source_dest_rows, topology, output_dir, section):
    """Full-chip square tile-to-tile request heatmap (n_tiles × n_tiles).
    All groups are shown on both axes so the matrix is always square."""
    n_groups = topology["n_groups"]
    matrices = _build_traffic_matrices(source_dest_rows, topology)
    if matrices is None:
        print("  [skip] No traffic data for zoom matrix")
        return

    mat = matrices["mat"]
    n_tiles = matrices["n_tiles"]
    tpg = matrices["tpg"]

    # Always show ALL groups on both axes → full square matrix
    zoom_groups = list(range(n_groups))
    dest_groups = list(range(n_groups))

    src_tiles = list(range(n_tiles))
    dest_tiles = list(range(n_tiles))

    sub = mat  # full n_tiles × n_tiles
    n_src = n_tiles
    n_dst = n_tiles

    # Adaptive figure sizing — cell_size drives annotation legibility
    max_px = 8000 if n_tiles > 64 else 5400  # higher res for large matrices
    cell_size = max(0.20, min(0.50, (max_px / DPI - 2.5) / max(n_tiles, 1)))
    side_inches = max(6.0, n_tiles * cell_size)
    fig_w = side_inches + 2.5  # colorbar + labels
    fig_h = side_inches + 2.0  # title + xlabel
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    vmax = sub.max()
    vmin = max(1.0, sub[sub > 0].min()) if np.any(sub > 0) else 1.0
    sub_plot = sub.copy()
    sub_plot[sub_plot == 0] = np.nan
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#F5F5F5")
    im = ax.imshow(sub_plot, origin="lower", cmap=cmap, aspect="equal",
                   norm=LogNorm(vmin=vmin, vmax=vmax), interpolation="nearest")

    # Annotate cells — font scales with cell size for readability.
    cell_pt = cell_size * 72
    annot_fs = min(FONT_ANNOT, max(2.0, cell_pt * 0.28))
    fw = "bold" if max(n_src, n_dst) <= 32 else "normal"
    for i in range(n_src):
        for j in range(n_dst):
            val = sub[i, j]
            if val > 0:
                txt_col = "white" if val > vmax * 0.3 else FIG_TEXT_COLOR
                ax.text(j, i, _nice_count(val), ha="center", va="center",
                        fontsize=annot_fs, fontweight=fw, color=txt_col)

    # Group boundaries — source (y-axis)
    src_offsets = {}
    for g in zoom_groups:
        offset = g * tpg
        if g > 0:
            ax.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(-0.04, mid, f"G{g}", ha="right", va="center",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[g % len(GRP_COLORS)],
                transform=ax.get_yaxis_transform(), clip_on=False)
        src_offsets[g] = offset

    # Group boundaries — dest (x-axis)
    dst_offsets = {}
    for g in dest_groups:
        offset = g * tpg
        if g > 0:
            ax.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
        mid = offset + tpg / 2
        ax.text(mid, -0.06, f"G{g}", ha="center", va="top",
                fontsize=FONT_ANNOT + 3, fontweight="bold",
                color=GRP_COLORS[g % len(GRP_COLORS)],
                transform=ax.get_xaxis_transform(), clip_on=False)
        dst_offsets[g] = offset

    # Highlight self-group diagonal blocks
    for g in zoom_groups:
        rect_xy = (dst_offsets[g] - 0.5, src_offsets[g] - 0.5)
        gcol = GRP_COLORS[g % len(GRP_COLORS)]
        ax.add_patch(Rectangle(
            rect_xy, tpg, tpg,
            linewidth=4.0, edgecolor="#000000",
            facecolor="none", alpha=0.2, zorder=4, clip_on=False,
        ))
        ax.add_patch(Rectangle(
            rect_xy, tpg, tpg,
            linewidth=2.5, edgecolor=gcol,
            facecolor="none", zorder=5, clip_on=False,
        ))

    # Highlight self-subgroup diagonal blocks
    _add_subgroup_boxes(ax, topology, tpg, n_groups)

    # Tick labels — show ticks at subgroup boundaries for readability
    if n_tiles > 64:
        _tps = topology.get("tiles_per_subgroup")
        if _tps:
            tick_positions = list(range(0, n_tiles, _tps))
            if (n_tiles - 1) not in tick_positions:
                tick_positions.append(n_tiles - 1)
        else:
            tick_positions = list(range(0, n_tiles, tpg))
            tick_positions.append(n_tiles - 1)
        tick_labels = [str(t) for t in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=FONT_TICK - 1, rotation=90)
        ax.set_yticklabels(tick_labels, fontsize=FONT_TICK - 1)
    else:
        tick_fs = max(6.5, FONT_TICK - 1.0 * (n_dst > 48))
        ax.set_xticks(range(n_dst))
        ax.set_yticks(range(n_src))
        ax.set_xticklabels(dest_tiles, fontsize=tick_fs, rotation=90)
        ax.set_yticklabels(src_tiles, fontsize=tick_fs)
    ax.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
    ax.set_ylabel("Source tile", fontsize=FONT_LABEL)

    src_label = ", ".join(f"G{g}" for g in zoom_groups)
    dst_label = ", ".join(f"G{g}" for g in dest_groups)
    ax.set_title("Tile-to-tile request volume (load/store issues)",
                 fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3.5%", pad=1.00)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Load/store issues", fontsize=FONT_LABEL - 1)
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
             f"{sec_lbl}  ·  {n_tiles} tiles  ·  {n_groups} groups  ·  "
             f"{_nice_count(total)} requests  ·  {nonzero} active pairs",
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


def _latency_style_colormap(name):
    colors = ["#1a9641", "#55b748", "#91cf60", "#d0ec8a",
              "#f0f4a4", "#fee08b", "#fdae61", "#f46d43",
              "#d73027", "#a50026"]
    cmap = LinearSegmentedColormap.from_list(name, colors, N=256)
    cmap.set_bad(color="#F5F5F5")
    return cmap


def _pressure_norm(source_pressure, dest_pressure):
    combined = np.concatenate([source_pressure, dest_pressure])
    positive = combined[combined > 0]
    if len(positive) == 0:
        return Normalize(vmin=0, vmax=1), [0, 1]
    vmin = max(1.0, float(positive.min()))
    vmax = max(vmin * 1.01, float(positive.max()))
    import math as _m
    lo = _m.floor(_m.log10(vmin))
    hi = _m.ceil(_m.log10(vmax))
    ticks = []
    for exp in range(int(lo), int(hi) + 1):
        base = 10 ** exp
        for mult in (1, 2.5, 5):
            value = base * mult
            if vmin <= value <= vmax:
                ticks.append(value)
    if not ticks:
        ticks = [vmin, vmax]
    return LogNorm(vmin=vmin, vmax=vmax), ticks


def _pressure_grid(values, n_groups, tpg):
    grid = np.full((n_groups, tpg), np.nan)
    for tile, value in enumerate(values):
        group = tile // tpg
        local_tile = tile % tpg
        if group < n_groups:
            grid[group, local_tile] = value if value > 0 else np.nan
    return grid


def _format_pressure_grid_axis(ax, tpg, n_groups, topology):
    ax.set_yticks(range(n_groups))
    ax.set_yticklabels([f"G{group}" for group in range(n_groups)], fontsize=FONT_TICK)
    for label, color in zip(ax.get_yticklabels(), GRP_COLORS):
        label.set_color(color)
        label.set_fontweight("bold")

    tps = topology.get("tiles_per_subgroup")
    if tps and tps < tpg:
        ticks = list(range(0, tpg, tps))
        if (tpg - 1) not in ticks:
            ticks.append(tpg - 1)
        for subgroup_boundary in range(tps, tpg, tps):
            ax.axvline(subgroup_boundary - 0.5, color="#777777", lw=0.8,
                       ls="--", alpha=0.65)
    else:
        step = max(1, tpg // 8)
        ticks = list(range(0, tpg, step))
        if (tpg - 1) not in ticks:
            ticks.append(tpg - 1)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(tick) for tick in ticks], fontsize=FONT_TICK)
    ax.set_xlabel("Tile index inside group", fontsize=FONT_LABEL)

    for boundary in range(1, n_groups):
        ax.axhline(boundary - 0.5, color="#333333", lw=1.4, alpha=0.65)


def plot_request_pressure_by_tile(source_dest_rows, topology, output_dir, section):
    """Collapse the tile-to-tile request matrix into source and destination pressure."""
    matrices = _build_traffic_matrices(source_dest_rows, topology)
    if matrices is None:
        print("  [skip] No traffic data for request pressure")
        return

    mat = matrices["mat"]
    n_tiles = matrices["n_tiles"]
    tpg = matrices["tpg"]
    n_groups = topology["n_groups"]
    source_pressure = mat.sum(axis=1)
    dest_pressure = mat.sum(axis=0)

    cmap = _latency_style_colormap("request_pressure_GnYlRd")
    norm, ticks = _pressure_norm(source_pressure, dest_pressure)
    src_plot = _pressure_grid(source_pressure, n_groups, tpg)
    dst_plot = _pressure_grid(dest_pressure, n_groups, tpg)

    fig_w = max(10.5, tpg * 0.35 + 4.0)
    fig, axes = plt.subplots(2, 1, figsize=(fig_w, 6.8), sharex=True,
                             gridspec_kw={"hspace": 0.42})
    panel_data = [
        (axes[0], src_plot, source_pressure,
         "(a) Source pressure: each tile summed over all destinations"),
        (axes[1], dst_plot, dest_pressure,
         "(b) Destination pressure: each tile summed over all sources"),
    ]

    im = None
    for ax, image_data, values, title in panel_data:
        im = ax.imshow(image_data, origin="lower", cmap=cmap, aspect="auto",
                       norm=norm, interpolation="nearest")
        ax.set_title(title, fontsize=FONT_SUBTITLE, fontweight="bold", pad=8)
        _format_pressure_grid_axis(ax, tpg, n_groups, topology)

        if n_tiles <= 128:
            annot_fs = max(5.0, min(FONT_ANNOT, fig_w * 11.5 / max(tpg, 1)))
            for tile, value in enumerate(values):
                if value <= 0:
                    continue
                txt_col = "white" if norm(value) > 0.62 else FIG_TEXT_COLOR
                ax.text(tile % tpg, tile // tpg, _nice_count(value), ha="center", va="center",
                        fontsize=annot_fs, fontweight="bold", color=txt_col)

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="2.5%", pad=0.25)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Load/store issues", fontsize=FONT_LABEL - 1)
    cb.set_ticks(ticks)
    cb.set_ticklabels([_nice_count(tick) for tick in ticks])
    cb.ax.tick_params(labelsize=FONT_TICK)

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    fig.suptitle("Tile request pressure from collapsed source-destination matrix",
                 fontsize=FONT_SUBTITLE + 1, fontweight="bold", y=0.99)
    fig.text(0.5, 0.01,
             f"{sec_lbl}  ·  {n_tiles} tiles  ·  {n_groups} groups  ·  "
             f"{_nice_count(mat.sum())} total load/store issues",
             ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

    _save(fig, output_dir, "request_pressure_by_tile", section)


# ===================================================================
# Figure 2 – Temporal Communication Profile
# ===================================================================

def plot_temporal_profile(tile_ts_rows, raw_events, result_dir, n_groups, output_dir, section):
    """Three panels (shared x-axis = cycle):
      (a) Stacked area: local / same-group / remote traffic over time
      (b) Per-tile incoming communication intensity heatmap
      (c) Average load-return latency over time, including locality breakdown
    """
    windows = sorted(set(int(r["window_index"]) for r in tile_ts_rows))
    n_windows = len(windows)
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
    agg_same_sub   = np.zeros(n_windows)
    agg_same       = np.zeros(n_windows)
    agg_remote     = np.zeros(n_windows)
    heatmap_in     = np.zeros((n_tiles, n_windows))
    agg_lat        = np.zeros(n_windows)
    agg_lat_weight = np.zeros(n_windows)
    loc_lat_sum    = np.zeros(n_windows)
    loc_lat_n      = np.zeros(n_windows)
    same_sub_lat_sum = np.zeros(n_windows)
    same_sub_lat_n   = np.zeros(n_windows)
    same_lat_sum     = np.zeros(n_windows)
    same_lat_n       = np.zeros(n_windows)
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
            loc_counts = _timeseries_locality_counts(r)
            agg_local[wi]    += loc_counts["local"]
            agg_same_sub[wi] += loc_counts["same_subgroup"]
            agg_same[wi]     += loc_counts["same_group"]
            agg_remote[wi]   += loc_counts["remote"]
            heatmap_in[ti, wi] = int(r.get("incoming_events", 0))
            lat   = _pfloat(r.get("outgoing_avg_latency", 0))
            lat_n = int(r.get("outgoing_latency_samples", 0))
            if lat_n > 0:
                agg_lat[wi] += lat * lat_n
                agg_lat_weight[wi] += lat_n
            # Per-locality-class latency (pre-computed by _build_timeseries_rows)
            loc_lat_sum[wi]      += float(r.get("_loc_lat_total", 0))
            loc_lat_n[wi]        += int(r.get("_loc_lat_n", 0))
            same_sub_lat_sum[wi] += float(r.get("_sub_lat_total", 0))
            same_sub_lat_n[wi]   += int(r.get("_sub_lat_n", 0))
            same_lat_sum[wi]     += float(r.get("_grp_lat_total", 0))
            same_lat_n[wi]       += int(r.get("_grp_lat_n", 0))
            remote_lat_sum[wi]   += float(r.get("_rem_lat_total", 0))
            remote_lat_n[wi]     += int(r.get("_rem_lat_n", 0))
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

    avg_local_lat = np.divide(loc_lat_sum, loc_lat_n, out=np.full(n_windows, np.nan), where=loc_lat_n > 0)
    avg_same_sub_lat = np.divide(same_sub_lat_sum, same_sub_lat_n, out=np.full(n_windows, np.nan), where=same_sub_lat_n > 0)
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
    ax_a.fill_between(rel_cycle_centers, agg_local, agg_local + agg_same_sub,
                      step="mid", color=COL_SAME_SUB, alpha=0.75,
                      label=LOCALITY_LABELS["same_subgroup"])
    ax_a.fill_between(rel_cycle_centers, agg_local + agg_same_sub, agg_local + agg_same_sub + agg_same,
                      step="mid", color=COL_SAME_GRP, alpha=0.75,
                      label=LOCALITY_LABELS["same_group"])
    ax_a.fill_between(rel_cycle_centers, agg_local + agg_same_sub + agg_same,
                      agg_local + agg_same_sub + agg_same + agg_remote,
                      step="mid", color=COL_REMOTE, alpha=0.75,
                      label=LOCALITY_LABELS["remote"])
    ax_a.set_xlim(x_min, x_max)
    ax_a.set_ylabel("Events per window", fontsize=FONT_LABEL)
    ax_a.set_title("(a)  Aggregate traffic by locality class",
                   fontsize=FONT_SUBTITLE, fontweight="bold", pad=6, loc="left")
    ax_a.legend(loc="upper right", fontsize=FONT_LEGEND, ncol=2)
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
              zorder=4, label="Local Avg.")
    ax_c.plot(rel_cycle_centers, avg_same_sub_lat, color=COL_SAME_SUB, lw=1.8,
              zorder=4, label="Same-subgroup Avg.")
    ax_c.plot(rel_cycle_centers, avg_same_lat, color=COL_SAME_GRP, lw=1.8,
              zorder=4, label="Same-group Avg.")
    ax_c.plot(rel_cycle_centers, avg_remote_lat, color=COL_ACCENT, lw=1.8,
              zorder=4, label="Remote Avg.")
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

    overview_dir = output_dir.parent / "overview"
    overview_dir.mkdir(parents=True, exist_ok=True)
    _save(fig, overview_dir, "overview_temporal", section)


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
# Figure 7 – Tile-pair Latency Heatmap
# ===================================================================

def plot_traffic_vs_latency(pair_data, topology, output_dir, section):
    """Full-chip square tile×tile latency heatmap (avg measured latency per pair)."""
    if not pair_data:
        print("  [skip] No load_return events with latency")
        return

    TRIM_FRAC = 0.10  # drop first/last 10% of the cycle range (startup/teardown)

    n_groups = topology["n_groups"]
    max_tile = max(max(s, d) for s, d in pair_data)
    n_tiles = topology.get("n_tiles") or (max_tile + 1)
    tpg = topology.get("tpg") or (n_tiles // n_groups)

    # Determine stable computation window (drop startup/teardown transients)
    all_cycles = [c for v in pair_data.values() for c, _ in v["latencies"]]
    if all_cycles:
        c_min, c_max = min(all_cycles), max(all_cycles)
        c_range = c_max - c_min
        c_lo = c_min + c_range * TRIM_FRAC
        c_hi = c_max - c_range * TRIM_FRAC
    else:
        c_lo, c_hi = -1, float('inf')

    # Build two latency matrices from the same collected data:
    #   original — plain mean over all events in the section
    #   refined  — 80th percentile over the stable computation window
    REFINED_PCTL = 80
    lat_variants = {}
    for variant in ("original", "refined"):
        mat = np.full((n_tiles, n_tiles), np.nan)
        for (s, d), v in pair_data.items():
            if v["lat_n"] <= 0:
                continue
            if variant == "original":
                mat[s, d] = v["lat_sum"] / v["lat_n"]
            else:  # refined
                lats = np.array([lat for cyc, lat in v["latencies"]
                                 if c_lo <= cyc <= c_hi])
                if len(lats) == 0:
                    mat[s, d] = v["lat_sum"] / v["lat_n"]  # fallback
                    continue
                mat[s, d] = float(np.percentile(lats, REFINED_PCTL))
        lat_variants[variant] = mat

    # Adaptive figure sizing
    max_px = 8000 if n_tiles > 64 else 5400
    max_w_inches = max_px / DPI - 3.5
    max_h_inches = max_px / DPI - 2.5
    cell_size = min(0.50, max_w_inches / max(n_tiles, 1),
                          max_h_inches / max(n_tiles, 1))
    cell_size = max(cell_size, 0.20)
    fig_w = n_tiles * cell_size + 3.5
    fig_h = n_tiles * cell_size + 2.5

    # Shared colourmap setup (used by both variants)
    _cmap_colors = ["#1a9641", "#55b748", "#91cf60", "#d0ec8a",
                    "#f0f4a4", "#fee08b", "#fdae61", "#f46d43",
                    "#d73027", "#a50026"]
    cmap_lat = LinearSegmentedColormap.from_list("lat_GnYlRd", _cmap_colors, N=256)
    cmap_lat.set_bad(color="#FFFFFF")
    N_GREEN = 4
    N_CMAP  = len(_cmap_colors)
    REMOTE_CMAP_POS = (N_GREEN - 1) / (N_CMAP - 1)
    K_CONTENTION = 3.0
    LOG_CMAP_END = 0.95
    TAIL_MULT    = 2.0

    baseline_map = _latency_baseline_map(topology)
    b_local  = baseline_map["local"]
    b_sub    = baseline_map["same_subgroup"]
    b_remote = baseline_map["remote"]

    vmin_lat = b_local
    vmax_fixed = K_CONTENTION * b_remote
    vmax_tail  = TAIL_MULT * vmax_fixed

    x_pts = [vmin_lat]
    y_pts = [0.0]
    hierarchy_span = max(b_remote - vmin_lat, 1e-6)
    for bval in (b_local, b_sub):
        if bval > vmin_lat:
            frac = REMOTE_CMAP_POS * (bval - vmin_lat) / hierarchy_span
            x_pts.append(bval)
            y_pts.append(frac)
    x_pts.append(b_remote)
    y_pts.append(REMOTE_CMAP_POS)
    log_denom = np.log(vmax_fixed / b_remote)
    for i in range(1, 21):
        f = i / 20.0
        x_val = b_remote + f * (vmax_fixed - b_remote)
        y_val = REMOTE_CMAP_POS + (LOG_CMAP_END - REMOTE_CMAP_POS) * np.log(x_val / b_remote) / log_denom
        x_pts.append(x_val)
        y_pts.append(y_val)
    tail_span = vmax_tail - vmax_fixed
    for i in range(1, 11):
        f = i / 10.0
        x_pts.append(vmax_fixed + f * tail_span)
        y_pts.append(LOG_CMAP_END + (1.0 - LOG_CMAP_END) * np.sqrt(f))
    lat_norm = PiecewiseNorm(x_pts, y_pts, vmin=vmin_lat, vmax=vmax_tail)

    _tick_vals = sorted(set([int(vmin_lat), int(b_sub), int(b_remote)] +
                            list(range(int(b_remote) + 2, int(vmax_fixed), 2)) +
                            [int(vmax_fixed), int(vmax_tail)]))
    sec_lbl = f"Section {section}" if section is not None else "All sections"
    n_pairs = len(pair_data)
    total_loads = sum(v["count"] for v in pair_data.values())

    # ---- Render heatmap for each variant ----
    _variant_meta = {
        "original": {
            "title": "Load-return latency per tile pair (cycles, mean)",
            "save_name": "latency_matrix",
            "global_avg_fn": lambda: (sum(v["lat_sum"] for v in pair_data.values()) /
                                      max(1, sum(v["lat_n"] for v in pair_data.values()))),
        },
        "refined": {
            "title": "Load-return latency per tile pair (cycles, steady-state p80)",
            "save_name": "latency_matrix_refined",
            "global_avg_fn": lambda: (float(np.percentile([lat for v in pair_data.values()
                                                            for cyc, lat in v["latencies"]
                                                            if c_lo <= cyc <= c_hi], REFINED_PCTL))
                                      if all_cycles else 0.0),
        },
    }
    for variant, meta in _variant_meta.items():
        sub = lat_variants[variant]
        sub_rounded = np.round(sub)
        sub_int = np.clip(sub_rounded, vmin_lat, vmax_tail)

        fig_a, ax_a = plt.subplots(figsize=(fig_w, fig_h))
        im_a = ax_a.imshow(sub_int, origin="lower", cmap=cmap_lat, aspect="equal",
                           norm=lat_norm, interpolation="nearest")

        cell_pt = cell_size * 72
        annot_fs = min(FONT_ANNOT, max(2.0, cell_pt * 0.28))
        fw = "bold" if n_tiles <= 32 else "normal"
        for i in range(n_tiles):
            for j in range(n_tiles):
                val = sub_rounded[i, j]
                if not np.isnan(val):
                    ax_a.text(j, i, f"{val:.0f}", ha="center", va="center",
                              fontsize=annot_fs, fontweight=fw, color="black")

        src_offsets = {}
        for g in range(n_groups):
            offset = g * tpg
            if g > 0:
                ax_a.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
            mid = offset + tpg / 2
            ax_a.text(-0.04, mid, f"G{g}", ha="right", va="center",
                      fontsize=FONT_ANNOT + 3, fontweight="bold",
                      color=GRP_COLORS[g % len(GRP_COLORS)],
                      transform=ax_a.get_yaxis_transform(), clip_on=False)
            src_offsets[g] = offset

        dst_offsets = {}
        for g in range(n_groups):
            offset = g * tpg
            if g > 0:
                ax_a.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
            mid = offset + tpg / 2
            ax_a.text(mid, -0.06, f"G{g}", ha="center", va="top",
                      fontsize=FONT_ANNOT + 3, fontweight="bold",
                      color=GRP_COLORS[g % len(GRP_COLORS)],
                      transform=ax_a.get_xaxis_transform(), clip_on=False)
            dst_offsets[g] = offset

        for g in range(n_groups):
            if g in src_offsets and g in dst_offsets:
                rect_xy = (dst_offsets[g] - 0.5, src_offsets[g] - 0.5)
                gcol = GRP_COLORS[g % len(GRP_COLORS)]
                ax_a.add_patch(Rectangle(
                    rect_xy, tpg, tpg,
                    linewidth=8.0, edgecolor="#000000",
                    facecolor="none", alpha=0.25, zorder=4, clip_on=False,
                ))
                ax_a.add_patch(Rectangle(
                    rect_xy, tpg, tpg,
                    linewidth=5.0, edgecolor=gcol,
                    facecolor="none", alpha=0.5, zorder=5, clip_on=False,
                ))
                ax_a.add_patch(Rectangle(
                    rect_xy, tpg, tpg,
                    linewidth=3.0, edgecolor=gcol,
                    facecolor="none", zorder=6, clip_on=False,
                ))

        # Highlight self-subgroup diagonal blocks
        _add_subgroup_boxes(ax_a, topology, tpg, n_groups)

        tick_fs = max(6.5, FONT_TICK - 1.5 * (n_tiles > 48))
        ax_a.set_xticks(range(n_tiles))
        ax_a.set_yticks(range(n_tiles))
        ax_a.set_xticklabels(range(n_tiles), fontsize=tick_fs, rotation=90)
        ax_a.set_yticklabels(range(n_tiles), fontsize=tick_fs)
        ax_a.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
        ax_a.set_ylabel("Source tile", fontsize=FONT_LABEL)

        ax_a.set_title(meta["title"],
                       fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)
        divider_a = make_axes_locatable(ax_a)
        cax_a = divider_a.append_axes("right", size="3.5%", pad=1.00)
        cb_a = fig_a.colorbar(im_a, cax=cax_a)
        cb_a.set_label("Avg latency (cycles)", fontsize=FONT_LABEL - 1)
        cb_a.set_ticks(_tick_vals)
        cb_a.set_ticklabels([str(t) if t < vmax_tail else f"{t}+"
                             for t in _tick_vals])
        cb_a.ax.tick_params(labelsize=FONT_TICK)

        global_avg = meta["global_avg_fn"]()
        fig_a.text(0.5, -0.005,
                   f"{sec_lbl}  ·  {n_tiles} tiles  ·  {n_groups} groups  ·  "
                   f"{n_pairs} active pairs  ·  "
                   f"{_nice_count(total_loads)} load returns  ·  "
                   f"Global avg: {global_avg:.1f} cycles",
                   ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

        _save(fig_a, output_dir, meta["save_name"], section)


def plot_latency_over_minimum(pair_data, topology, output_dir, section):
    """Tile-pair latency heatmap normalized by hierarchy-aware ideal minima."""
    if not pair_data:
        print("  [skip] No load_return events with latency")
        return

    TRIM_FRAC = 0.10  # drop first/last 10% of the cycle range (startup/teardown)

    n_groups = topology["n_groups"]
    max_tile = max(max(s, d) for s, d in pair_data)
    n_tiles = topology.get("n_tiles") or (max_tile + 1)
    tpg = topology.get("tpg") or (n_tiles // n_groups)

    # Determine stable computation window (drop startup/teardown transients)
    all_cycles = [c for v in pair_data.values() for c, _ in v["latencies"]]
    if all_cycles:
        c_min, c_max = min(all_cycles), max(all_cycles)
        c_range = c_max - c_min
        c_lo = c_min + c_range * TRIM_FRAC
        c_hi = c_max - c_range * TRIM_FRAC
    else:
        c_lo, c_hi = -1, float('inf')

    # Build both original and refined latency matrices
    lat_variants = {}
    loc_mat = np.full((n_tiles, n_tiles), "", dtype=object)

    # Original: plain mean, all events
    mat_orig = np.full((n_tiles, n_tiles), np.nan)
    for (s, d), data in pair_data.items():
        if data["lat_n"] > 0:
            mat_orig[s, d] = data["lat_sum"] / data["lat_n"]
            loc_mat[s, d] = data["locality"]
    lat_variants["original"] = mat_orig

    # Refined: 80th percentile over trimmed window
    REFINED_PCTL = 80
    mat_ref = np.full((n_tiles, n_tiles), np.nan)
    for (s, d), data in pair_data.items():
        if data["lat_n"] > 0:
            trimmed = [lat for cyc, lat in data["latencies"] if c_lo <= cyc <= c_hi]
            if trimmed:
                mat_ref[s, d] = float(np.percentile(trimmed, REFINED_PCTL))
            else:
                mat_ref[s, d] = data["lat_sum"] / data["lat_n"]
            loc_mat[s, d] = data["locality"]
    lat_variants["refined"] = mat_ref

    # Adaptive figure sizing
    max_px = 8000 if n_tiles > 64 else 5400
    max_w_inches = max_px / DPI - 3.5
    max_h_inches = max_px / DPI - 2.5
    cell_size = min(0.50, max_w_inches / max(n_tiles, 1),
                          max_h_inches / max(n_tiles, 1))
    cell_size = max(cell_size, 0.20)
    fig_w = n_tiles * cell_size + 3.5
    fig_h = n_tiles * cell_size + 2.5

    baseline_map = _latency_baseline_map(topology)

    # Shared colormap
    _cmap_colors = ["#1a9641", "#55b748", "#91cf60", "#d0ec8a",
                    "#f0f4a4", "#fee08b", "#fdae61", "#f46d43",
                    "#d73027", "#a50026"]
    cmap_delta = LinearSegmentedColormap.from_list("lat_over_minimum", _cmap_colors, N=256)
    cmap_delta.set_bad(color="#FFFFFF")

    sec_lbl = f"Section {section}" if section is not None else "All sections"
    n_pairs = len(pair_data)
    total_loads = sum(data["count"] for data in pair_data.values())

    _variant_meta = {
        "original": {
            "title": "Excess load-return latency above ideal minimum (mean)",
            "save_name": "latency_excess_matrix",
        },
        "refined": {
            "title": "Excess load-return latency above ideal minimum (steady-state p80)",
            "save_name": "latency_excess_matrix_refined",
        },
    }

    for variant, meta in _variant_meta.items():
        sub_lat = lat_variants[variant]

        delta_mat = np.full_like(sub_lat, np.nan, dtype=float)
        for i in range(n_tiles):
            for j in range(n_tiles):
                latency = sub_lat[i, j]
                if np.isnan(latency):
                    continue
                locality = loc_mat[i, j] or "remote"
                delta_mat[i, j] = max(0.0, latency - baseline_map.get(locality, baseline_map["remote"]))

        delta_mat = np.round(delta_mat)

        valid_delta = delta_mat[~np.isnan(delta_mat)]
        vmax_delta = np.ceil(valid_delta.max()) if len(valid_delta) else 1.0
        vmax_delta = max(1.0, vmax_delta)

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

        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(delta_mat, origin="lower", cmap=cmap_delta, aspect="equal",
                       norm=delta_norm, interpolation="nearest")

        cell_pt = cell_size * 72
        annot_fs = min(FONT_ANNOT, max(2.0, cell_pt * 0.28))
        fw = "bold" if n_tiles <= 32 else "normal"
        for i in range(n_tiles):
            for j in range(n_tiles):
                value = delta_mat[i, j]
                if not np.isnan(value):
                    rounded_value = int(np.floor(value + 0.5))
                    ax.text(j, i, f"{rounded_value}", ha="center", va="center",
                            fontsize=annot_fs, fontweight=fw, color="black")

        src_offsets = {}
        for group in range(n_groups):
            offset = group * tpg
            if group > 0:
                ax.axhline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
            mid = offset + tpg / 2
            ax.text(-0.04, mid, f"G{group}", ha="right", va="center",
                    fontsize=FONT_ANNOT + 3, fontweight="bold",
                    color=GRP_COLORS[group % len(GRP_COLORS)],
                    transform=ax.get_yaxis_transform(), clip_on=False)
            src_offsets[group] = offset

        dst_offsets = {}
        for group in range(n_groups):
            offset = group * tpg
            if group > 0:
                ax.axvline(offset - 0.5, color="#333333", lw=2.0, alpha=0.7)
            mid = offset + tpg / 2
            ax.text(mid, -0.06, f"G{group}", ha="center", va="top",
                    fontsize=FONT_ANNOT + 3, fontweight="bold",
                    color=GRP_COLORS[group % len(GRP_COLORS)],
                    transform=ax.get_xaxis_transform(), clip_on=False)
            dst_offsets[group] = offset

        for group in range(n_groups):
            if group in src_offsets and group in dst_offsets:
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

        # Highlight self-subgroup diagonal blocks
        _add_subgroup_boxes(ax, topology, tpg, n_groups)

        tick_fs = max(6.5, FONT_TICK - 1.5 * (n_tiles > 48))
        ax.set_xticks(range(n_tiles))
        ax.set_yticks(range(n_tiles))
        ax.set_xticklabels(range(n_tiles), fontsize=tick_fs, rotation=90)
        ax.set_yticklabels(range(n_tiles), fontsize=tick_fs)
        ax.set_xlabel("Destination tile", fontsize=FONT_LABEL, labelpad=20)
        ax.set_ylabel("Source tile", fontsize=FONT_LABEL)
        ax.set_title(meta["title"],
                     fontsize=FONT_SUBTITLE, fontweight="bold", pad=10)

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3.5%", pad=1.00)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("Excess latency over ideal minimum (cycles)", fontsize=FONT_LABEL - 1)
        ticks = sorted(set([0, 1, 2, 5] + [v for v in range(10, int(vmax_delta) + 1, 5)] + [int(vmax_delta)]))
        ticks = [tick for tick in ticks if 0 <= tick <= vmax_delta]
        cb.set_ticks(ticks)
        cb.ax.tick_params(labelsize=FONT_TICK)

        global_avg_delta = float(np.nanmean(valid_delta)) if len(valid_delta) else 0.0
        fig.text(0.5, -0.005,
                 f"{sec_lbl}  ·  {n_tiles} tiles  ·  {n_groups} groups  ·  "
                 f"{n_pairs} active pairs  ·  "
                 f"{_nice_count(total_loads)} load returns  ·  "
                 f"Global avg excess: {global_avg_delta:.1f} cycles  ·  "
                 f"Baselines: local={baseline_map['local']:.0f}, "
                 f"same-subgroup={baseline_map['same_subgroup']:.0f}, "
                 f"same-group={baseline_map['same_group']:.0f}, "
                 f"remote={baseline_map['remote']:.0f}",
                 ha="center", fontsize=FONT_ANNOT, color="#666666", style="italic")

        _save(fig, output_dir, meta["save_name"], section)


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
                   help="Which figures: matrix "
                        "pressure temporal latency tile_latency latency_matrix latency_over_minimum (default: all)")
    return p.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    _apply_thesis_style()

    paths = _resolve_paths(args.input_path)
    output_dir = paths["output"]
    output_dir.mkdir(parents=True, exist_ok=True)
    _migrate_legacy_pdfs(output_dir)

    section  = args.section
    topology = _load_topology(paths["result_dir"], args.n_groups)
    n_groups = topology["n_groups"]
    figs = (set(args.figures) if args.figures
            else {"matrix", "temporal",
                  "latency", "tile_latency", "latency_matrix",
                  "latency_over_minimum"})

    print(f"Generating thesis figures → {output_dir}")

    # Load raw events once — all summary data is computed inline from this.
    raw_events = (_filter_section(_load_csv(paths["events"]), section)
                  if paths["events"].is_file() else [])

    sd_rows = _build_source_dest_rows(raw_events)
    ts_rows = _build_timeseries_rows(raw_events)

    if "matrix" in figs:
        print("\n[1/7] Traffic matrix …")
        plot_traffic_matrix_full(sd_rows, topology, output_dir, section)
        plot_traffic_matrix_group(sd_rows, topology, output_dir, section)

    if "matrix" in figs or "pressure" in figs:
        print("\n[2/7] Source/destination request pressure …")
        plot_request_pressure_by_tile(sd_rows, topology, output_dir, section)

    if "temporal" in figs and ts_rows:
        print("\n[3/7] Temporal profile …")
        plot_temporal_profile(ts_rows, raw_events, paths["result_dir"], n_groups, output_dir, section)

    if "latency" in figs and ts_rows:
        print("\n[4/7] Latency over time …")
        plot_latency_over_time(ts_rows, n_groups, output_dir, section)

    if "tile_latency" in figs and ts_rows:
        for tg in range(n_groups):
            print(f"\n[5/7] Per-tile latency — Group {tg} …")
            plot_per_tile_group_latency(ts_rows, n_groups, output_dir,
                                        section, target_group=tg)

    # Build pair_data once for the two latency heatmap figures.
    need_pair = {"latency_matrix", "latency_over_minimum"} & figs
    pair_data = _build_pair_data(raw_events) if need_pair else {}

    if "latency_matrix" in figs:
        print("\n[6/7] Tile-pair latency heatmap …")
        plot_traffic_vs_latency(pair_data, topology, output_dir, section)

    if "latency_over_minimum" in figs:
        print("\n[7/7] Latency above hierarchy minimum …")
        plot_latency_over_minimum(pair_data, topology, output_dir, section)

    print("\nDone.")


if __name__ == "__main__":
    main()

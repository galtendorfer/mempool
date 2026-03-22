"""Shared constants, data loading, and plot primitives for stall analysis scripts."""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ── Constants ─────────────────────────────────────────────────────────────────

STALL_CATEGORIES = ["ins", "raw", "lsu", "acc", "wfi", "other"]
STALL_COLORS = {
    "issue": "#78C679", "stall": "#D9D9D9",
    "ins": "#4C78A8", "raw": "#F58518", "lsu": "#17BECF",
    "acc": "#E45756", "wfi": "#B279A2", "other": "#9D755D",
}
STALL_LABELS = {
    "issue": "Issuing", "stall": "Stalled",
    "ins": "Instruction fetch", "raw": "RAW", "lsu": "LSU",
    "acc": "Accelerator", "wfi": "WFI", "other": "Other / mixed",
}

# Instruction-type colour palette — high contrast for thin strips
ITYPE_COLORS = {
    "stalled": "#FFFFFF",  # white — clearly inactive
    "load":    "#1F77B4",  # saturated blue
    "store":   "#FF7F0E",  # bright orange
    "mac":     "#D62728",  # bold red
    "other":   "#2CA02C",  # saturated green
}
ITYPE_LABELS = {
    "stalled": "Stalled",
    "load":    "Load",
    "store":   "Store",
    "mac":     "MAC / Multiply",
    "other":   "Other",
}
# Ordered list for building colourmaps (index 0=stalled, 1=load, …)
ITYPE_ORDER = ["stalled", "load", "store", "mac", "other"]

_MAC_MNEMONICS = frozenset({
    "mul", "mulh", "mulhu", "mulhsu",
    "div", "divu", "rem", "remu",
    "p.mac",
})

_AMO_MNEMONICS = frozenset({
    "amoadd.w", "amoadd.d", "amoxor.w", "amoxor.d",
    "amoor.w", "amoor.d", "amoand.w", "amoand.d",
    "amomin.w", "amomin.d", "amomax.w", "amomax.d",
    "amominu.w", "amominu.d", "amomaxu.w", "amomaxu.d",
    "amoswap.w", "amoswap.d", "lr.w", "lr.d", "sc.w", "sc.d",
})


def stall_label(cat):
    return STALL_LABELS.get(cat, cat)


# ── Data loading & filtering ──────────────────────────────────────────────────

def _int(v):
    return None if v is None or v == "" else int(v)


def load_rows(csv_path):
    rows = []
    with Path(csv_path).open(newline="") as f:
        for r in csv.DictReader(f):
            rows.append({
                "core": _int(r.get("core")),
                "group": _int(r.get("group")),
                "tile": _int(r.get("tile")),
                "section": _int(r.get("section")),
                "cycle": _int(r.get("cycle")),
                "state": (r.get("state") or "").strip(),
                "stall_kind": (r.get("stall_kind") or "").strip(),
            })
    return rows


def filter_rows(rows, *, section=None, group=None, tile=None, core=None):
    """Filter rows by optional sets of section/group/tile/core IDs."""
    filters = {}
    if section:
        filters["section"] = set(section)
    if group:
        filters["group"] = set(group)
    if tile:
        filters["tile"] = set(tile)
    if core:
        filters["core"] = set(core)
    if not filters:
        return rows
    return [r for r in rows if all(r[k] in v for k, v in filters.items())]


def split_stall_kind(kind):
    if not kind or kind == "none":
        return ["other"]
    cats = [p.strip() if p.strip() in STALL_CATEGORIES else "other"
            for p in kind.split("+") if p.strip()]
    return cats or ["other"]


# ── Trace file helpers ────────────────────────────────────────────────────────

def locate_trace_file(csv_path, core_id, traces_dir=None):
    search_roots = []
    if traces_dir is not None:
        search_roots.append(Path(traces_dir))
    for root in (csv_path.parent, csv_path.parent.parent):
        search_roots.append(root)
        traces_sub = root / "traces"
        if traces_sub.is_dir():
            search_roots.append(traces_sub)
    for root in search_roots:
        for name in (f"trace_hart_0x{core_id:08x}.trace", f"trace_hart_{core_id}.trace"):
            p = root / name
            if p.is_file():
                return p
    return None


def build_memory_request_series(trace_path, cycles):
    cycle_set = set(int(c) for c in cycles)
    loads = {c: 0 for c in cycle_set}
    amo_loads = {c: 0 for c in cycle_set}
    stores = {c: 0 for c in cycle_set}
    returns = {c: 0 for c in cycle_set}
    mnemonics = {c: "" for c in cycle_set}

    for line in trace_path.read_text().splitlines():
        s = line.strip()
        if not s or not s[0].isdigit():
            continue
        before, ann = (line.split("#;", 1) + [""])[:2]
        parts = before.split()
        if len(parts) < 2:
            continue
        try:
            cyc = int(parts[1])
        except ValueError:
            continue
        if cyc not in cycle_set:
            continue
        loads[cyc] += ann.count("<~~")
        if len(parts) >= 4 and parts[3] in _AMO_MNEMONICS:
            amo_loads[cyc] += ann.count("<~~")
        stores[cyc] += ann.count("~~>")
        returns[cyc] += len(re.findall(r"\(lsu\)\s+[^,]+<--", ann))
        if len(parts) >= 4:
            mnemonics[cyc] = parts[3]

    outstanding, cur_loads, cur_amo_loads, cur_stores, itypes = [], [], [], [], []
    cum_loads, cum_amo_loads, cum_returns, cum_stores = [], [], [], []
    run_out = run_l = run_a = run_r = run_s = 0
    for c in cycles:
        c = int(c)
        run_l += loads[c]; run_a += amo_loads[c]; run_s += stores[c]; run_r += returns[c]
        run_out = max(0, run_out + loads[c] - returns[c])
        outstanding.append(float(run_out))
        cur_loads.append(float(loads[c]))
        cur_amo_loads.append(float(amo_loads[c]))
        cur_stores.append(float(stores[c]))
        cum_loads.append(float(run_l))
        cum_amo_loads.append(float(run_a))
        cum_returns.append(float(run_r))
        cum_stores.append(float(run_s))
        # Instruction type: 1=load, 2=store, 3=mac, 4=other
        # (caller assigns 0=stalled based on issue/stall state)
        if loads[c] > 0:
            itypes.append(1)
        elif stores[c] > 0:
            itypes.append(2)
        elif mnemonics[c] in _MAC_MNEMONICS:
            itypes.append(3)
        else:
            itypes.append(4)

    return {
        "outstanding_loads": np.array(outstanding),
        "load_issue_current": np.array(cur_loads),
        "amo_load_issue_current": np.array(cur_amo_loads),
        "store_issue_current": np.array(cur_stores),
        "load_issue_cumulative": np.array(cum_loads),
        "amo_load_issue_cumulative": np.array(cum_amo_loads),
        "load_return_cumulative": np.array(cum_returns),
        "store_issue_cumulative": np.array(cum_stores),
        "itype_current": np.array(itypes, dtype=float),
    }


# ── Plot primitives ───────────────────────────────────────────────────────────

def set_cycle_ticks(ax, positions, labels=None, max_ticks=25):
    """Place x ticks at nice round cycle values.

    *positions* are the tick coordinates on the axis.
    *labels* (if given) are the cycle values to display.
    When labels are provided the function picks nice round values from
    the label range and maps them back to the nearest position."""
    n = len(positions)
    if n <= 1:
        return
    lbl = labels if labels is not None else positions
    lo, hi = float(lbl[0]), float(lbl[-1])
    span = hi - lo
    if span <= 0:
        return
    # pick a nice step: 1,2,5 × 10^k that gives <= max_ticks intervals
    raw_step = span / max_ticks
    mag = 10 ** int(np.floor(np.log10(max(raw_step, 1))))
    for mult in (1, 2, 5, 10, 20, 50, 100):
        step = mag * mult
        if span / step <= max_ticks:
            break
    first = int(np.ceil(lo / step)) * step
    nice_vals = np.arange(first, hi + 1, step)
    # always include the very first and last cycle
    nice_vals = np.unique(np.concatenate(([lo], nice_vals, [hi])))
    # map each nice value to the nearest position index
    lbl_arr = np.array(lbl, dtype=float)
    tick_pos = []
    tick_lbl = []
    for v in nice_vals:
        idx = int(np.argmin(np.abs(lbl_arr - v)))
        tick_pos.append(positions[idx])
        tick_lbl.append(str(int(lbl_arr[idx])))
    # deduplicate ticks that are too close in label-space after mapping
    min_gap = step * 0.45
    filtered_pos, filtered_lbl = [tick_pos[0]], [tick_lbl[0]]
    for p, l in zip(tick_pos[1:], tick_lbl[1:]):
        if abs(int(l) - int(filtered_lbl[-1])) >= min_gap:
            filtered_pos.append(p)
            filtered_lbl.append(l)
    ax.set_xticks(filtered_pos)
    ax.set_xticklabels(filtered_lbl)


def plot_series(ax, cycles, values, title, color, ylabel, mode="step"):
    """Draw a single time series.  mode='step' for per-cycle, 'line' for cumulative."""
    if mode == "step":
        ax.step(cycles, values, where="post", color=color, linewidth=2.0)
        ax.fill_between(cycles, 0, values, step="post", color=color, alpha=0.15)
    else:
        ax.plot(cycles, values, color=color, linewidth=2.0)
        ax.fill_between(cycles, 0, values, color=color, alpha=0.10)
    vmax = float(np.max(values)) if len(values) else 0
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax * 1.08))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_outstanding_loads(ax, cycles, outstanding, title):
    """Draw outstanding loads."""
    line = ax.step(cycles, outstanding, where="post", color="#7F3C8D",
                      linewidth=2.0, label="Outstanding loads")[0]
    from matplotlib.ticker import MultipleLocator
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.fill_between(cycles, 0, outstanding, step="post", color="#7F3C8D", alpha=0.15)
    vmax_out = float(np.max(outstanding)) if len(outstanding) else 0.0
    legend_handles = [
        Line2D([0], [0], color="#7F3C8D", linewidth=4.0,
                   label="Outstanding loads"),
    ]
    ax.legend(handles=legend_handles, loc="upper left",
              frameon=True, fancybox=True, edgecolor="#CCCCCC",
              facecolor="white", framealpha=0.9,
              handleheight=1.1, handlelength=3.0)
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax_out * 1.08))
    ax.set_title(title)
    ax.set_ylabel("Outstanding requests")
    ax.grid(True, axis="y", alpha=0.25)


def plot_issue_stall(ax, cycles, issue, stall, title, ylabel, mode="step"):
    """Draw the issue vs stall dual-series with legend."""
    from matplotlib.ticker import MaxNLocator
    if mode == "step":
        # Stacked area: issuing on bottom, stalled on top
        ax.stackplot(cycles, [issue, stall],
                     labels=["issuing", "stalled"],
                     colors=[STALL_COLORS["issue"], "#BFBFBF"],
                     alpha=0.85, step="post")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    else:
        il = ax.plot(cycles, issue, color=STALL_COLORS["issue"], lw=2.2)[0]
        ax.fill_between(cycles, 0, issue, color=STALL_COLORS["issue"], alpha=0.12)
        sl = ax.plot(cycles, stall, color="#2F2F2F", lw=2.0)[0]
        ax.fill_between(cycles, 0, stall, color="#2F2F2F", alpha=0.08)
    vmax = float(max(np.max(issue), np.max(stall))) if len(issue) else 0
    if mode == "step":
        vmax = float(np.max(issue + stall)) if len(issue) else 0
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax * 1.08))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    if mode == "step":
        ax.legend(loc="upper right", frameon=False)
    else:
        ax.legend([il, sl], ["issuing", "stalled"], loc="upper right", frameon=False)


def plot_rolling_fraction(ax, cycles, numerator, denominator, title, ylabel,
                          window=64, color="#78C679"):
    """Draw a rolling percentage for a ratio-like series over a cycle window."""
    if len(cycles) == 0:
        return

    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    ratio = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)

    window = max(1, min(int(window), len(ratio)))
    valid = np.isfinite(ratio).astype(float)
    filled = np.nan_to_num(ratio, nan=0.0)
    kernel = np.ones(window, dtype=float)
    weighted = np.convolve(filled, kernel, mode="same")
    counts = np.convolve(valid, kernel, mode="same")
    rolling = np.divide(weighted, counts, out=np.full_like(weighted, np.nan), where=counts > 0)
    percent = rolling * 100.0

    ax.plot(cycles, percent, color=color, linewidth=2.2)
    ax.fill_between(cycles, 0, percent, color=color, alpha=0.14)
    ax.axhline(50.0, color="#8C8C8C", linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, 100)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)


def plot_memory_accounting(ax, cycles, load_iss, load_ret, store_iss, title):
    """Draw cumulative memory-request accounting (3 lines)."""
    ax.plot(cycles, load_iss, color="#4C78A8", lw=2.0, label="load-like issues")
    ax.fill_between(cycles, 0, load_iss, color="#4C78A8", alpha=0.10)
    ax.plot(cycles, load_ret, color="#54A24B", lw=2.0, label="load-like returns")
    ax.fill_between(cycles, 0, load_ret, color="#54A24B", alpha=0.10)
    ax.plot(cycles, store_iss, color="#F58518", lw=2.0, label="store issues")
    ax.fill_between(cycles, 0, store_iss, color="#F58518", alpha=0.10)
    ax.set_title(title)
    ax.set_ylabel("Accumulated requests")
    vmax = float(np.max([load_iss.max(), load_ret.max(), store_iss.max()]))
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax * 1.05))
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=False)


def plot_categories_current(ax, cycles, cat_current, present, title, ylabel="Stalled cores"):
    """Stacked area of active stall categories per cycle."""
    from matplotlib.ticker import MaxNLocator
    arrays = [cat_current[c] for c in present]
    colors = [STALL_COLORS[c] for c in present]
    labels = [stall_label(c) for c in present]
    ax.stackplot(cycles, arrays, labels=labels, colors=colors, alpha=0.85, step="post")
    total = np.sum(np.vstack(arrays), axis=0) if arrays else np.zeros_like(cycles)
    vmax = float(np.max(total)) if len(total) else 0
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax * 1.08))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", frameon=True, fancybox=True,
              edgecolor="#CCCCCC", facecolor="white", framealpha=0.68,
              handleheight=1.1, handlelength=2.5)


def plot_categories_cumulative(ax, cycles, cat_cumulative, present, title):
    """Overlay all active stall categories as cumulative lines on one axis.

    Fill is only drawn where the cumulative is rising (= stall active),
    highlighting exactly when each category contributes."""
    legend_handles = []
    for c in present:
        vals = cat_cumulative[c]
        ax.plot(cycles, vals, color=STALL_COLORS[c],
                lw=2.0, label=stall_label(c))
        rising = np.diff(vals, prepend=vals[0]) > 0
        ax.fill_between(cycles, 0, vals, where=rising,
                        color=STALL_COLORS[c], alpha=0.22)
        legend_handles.append(
            Line2D([0], [0], color=STALL_COLORS[c], linewidth=4.0,
                   label=stall_label(c))
        )
    vmax = max((float(np.max(cat_cumulative[c])) for c in present), default=0)
    ax.set_xlim(cycles[0], cycles[-1])
    ax.set_ylim(0, max(1.0, vmax * 1.05))
    ax.set_title(title)
    ax.set_ylabel("Accumulated core-cycles")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(handles=legend_handles, loc="upper left", frameon=True, fancybox=True,
              edgecolor="#CCCCCC", facecolor="white", framealpha=0.9,
              handleheight=1.1, handlelength=3.0)

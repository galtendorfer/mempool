#!/usr/bin/env python3
"""
Plot DAS load-throughput sweep results.

Usage:
    python3 plot_das_sweep.py das_sweep_20260305_123456

Produces:
  - load_throughput.pdf  : throughput & latency vs offered load, one curve per group_tiles
  - port_bottleneck.pdf  : per-port accept counts at selected load points (if port data exists)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob, sys, os

result_dir = sys.argv[1]

# ── Read combined CSV ──
csv_path = os.path.join(result_dir, "results.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    # Fallback: read per-group result files
    rows = []
    for f in sorted(glob.glob(os.path.join(result_dir, "group_tiles_*/results_group*"))):
        gt = int(f.split("group_tiles_")[1].split("/")[0])
        data = np.loadtxt(f)
        if data.ndim == 1:
            data = data.reshape(1, -1)
        for row in data:
            rows.append({"group_tiles": gt, "req_prob": row[0],
                         "avg_latency": row[1], "throughput": row[2]})
    df = pd.DataFrame(rows)

# ── Plot 1: Load-Throughput + Load-Latency ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

colors = {1: "tab:blue", 2: "tab:orange", 4: "tab:green", 8: "tab:red", 16: "tab:purple"}

for gt, grp in df.groupby("group_tiles"):
    grp = grp.sort_values("req_prob")
    c = colors.get(gt, None)
    label = f"group_tiles={gt}" if gt > 1 else "baseline (no grouping)"

    ax1.plot(grp["req_prob"], grp["throughput"], "o-", color=c, label=label, markersize=3)
    ax2.plot(grp["throughput"], grp["avg_latency"], "o-", color=c, label=label, markersize=3)

ax1.set_xlabel("Offered load (req probability)")
ax1.set_ylabel("Throughput (req/core/cycle)")
ax1.set_title("Load vs Throughput")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.set_xlabel("Throughput (req/core/cycle)")
ax2.set_ylabel("Average latency (cycles)")
ax2.set_title("Throughput vs Latency")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_pdf = os.path.join(result_dir, "load_throughput.pdf")
plt.savefig(out_pdf, bbox_inches="tight")
print(f"Saved: {out_pdf}")

# ── Plot 2: Per-port bottleneck (if data exists) ──
port_csv = os.path.join(result_dir, "port_counters.csv")
if os.path.exists(port_csv):
    pdf = pd.read_csv(port_csv)
    if len(pdf) > 0:
        fig2, axes = plt.subplots(1, len(df["group_tiles"].unique()), figsize=(6*len(df["group_tiles"].unique()), 5), squeeze=False)

        for idx, (gt, gt_df) in enumerate(pdf.groupby("group_tiles")):
            ax = axes[0, idx]
            # Pick a representative high-load point (e.g., req_prob~0.3)
            req_probs_avail = sorted(gt_df["req_prob"].unique())
            target_rp = min(req_probs_avail, key=lambda x: abs(x - 0.3))

            sub = gt_df[gt_df["req_prob"] == target_rp]
            # Aggregate across all tiles: sum accepts per port
            port_totals = sub.groupby("port")[["accepts", "backpressure"]].sum()

            port_totals.plot.bar(ax=ax, rot=0)
            ax.set_title(f"group_tiles={gt}, req_prob={target_rp}")
            ax.set_xlabel("Remote port index")
            ax.set_ylabel("Total count (all tiles)")
            ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        port_pdf_path = os.path.join(result_dir, "port_bottleneck.pdf")
        plt.savefig(port_pdf_path, bbox_inches="tight")
        print(f"Saved: {port_pdf_path}")

plt.show()

#!/usr/bin/env python3
"""Compare source-side route-port fan-in across classifier runs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


RUN_COLORS = ("#D55E00", "#0072B2", "#009E73", "#6F63C6", "#9E9E9E")


@dataclass
class BucketStats:
    tile_cycles: int = 0
    requests: int = 0
    fires: int = 0
    stalls: int = 0
    high_fanin_tile_cycles: int = 0
    high_fanin_requests: int = 0


@dataclass
class RunStats:
    label: str
    source: Path
    buckets: dict[int, BucketStats] = field(default_factory=lambda: defaultdict(BucketStats))
    total_active_tile_cycles: int = 0
    total_requests: int = 0
    total_fires: int = 0
    total_stalls: int = 0
    total_high_fanin_tile_cycles: int = 0
    total_high_fanin_requests: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        nargs="+",
        type=Path,
        help="classification directory, result directory, or *_tile_cycles.csv; pass multiple runs to compare",
    )
    parser.add_argument("--label", action="append", help="run label; repeat once per input path")
    parser.add_argument("--max-bucket", type=int, default=4, help="largest explicit fan-in bucket before overflow")
    parser.add_argument("--output-dir", type=Path, help="output directory; defaults to <result_dir>/plots/port_pressure")
    parser.add_argument("--prefix", default="port0_fanin_mechanism", help="output filename prefix")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="figure formats to write",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args()


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value, 0)


def resolve_tile_cycles_csv(input_path: Path) -> Path:
    if input_path.is_file():
        if input_path.name.endswith("_tile_cycles.csv"):
            return input_path
        raise SystemExit(f"Expected *_tile_cycles.csv, got {input_path}")
    candidates = [
        input_path,
        input_path / "port0_source_target_classification",
        input_path / "analysis" / "path_graph" / "port0_source_target_classification",
    ]
    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            matches.extend(sorted(candidate.glob("*_tile_cycles.csv")))
    matches = sorted(set(matches))
    if not matches:
        raise SystemExit(f"No *_tile_cycles.csv found from {input_path}")
    if len(matches) > 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise SystemExit(f"Multiple *_tile_cycles.csv files found; pass one explicitly:\n  {joined}")
    return matches[0]


def default_output_dir(tile_cycles_csv: Path) -> Path:
    for candidate in (tile_cycles_csv.parent, *tile_cycles_csv.parents):
        if candidate.name == "path_graph" and candidate.parent.name == "analysis":
            return candidate.parent.parent / "plots" / "port_pressure"
    return tile_cycles_csv.parent / "plots"


def infer_label(path: Path) -> str:
    text = str(path).lower()
    if "back2local" in text:
        return "Back2Local"
    if "/das_" in text or text.endswith("/das"):
        return "DAS"
    return path.parent.name.replace("_", " ")


def bucket_for_fanin(fanin: int, max_bucket: int) -> int:
    if fanin > max_bucket:
        return max_bucket + 1
    return fanin


def bucket_label(bucket: int, max_bucket: int) -> str:
    if bucket == max_bucket + 1:
        return f">{max_bucket}"
    return str(bucket)


def read_run_stats(tile_cycles_csv: Path, label: str, max_bucket: int) -> RunStats:
    stats = RunStats(label=label, source=tile_cycles_csv)
    with tile_cycles_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"fanin_requests", "fanin_stalls", "fanin_fires"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {tile_cycles_csv}: {', '.join(sorted(missing))}")
        for row in reader:
            fanin = parse_int(row.get("fanin_requests"))
            if fanin <= 0:
                continue
            stalls = parse_int(row.get("fanin_stalls"))
            fires = parse_int(row.get("fanin_fires"))
            high_fanin = parse_int(row.get("high_fanin"))
            bucket = bucket_for_fanin(fanin, max_bucket)
            bucket_stats = stats.buckets[bucket]
            bucket_stats.tile_cycles += 1
            bucket_stats.requests += fanin
            bucket_stats.fires += fires
            bucket_stats.stalls += stalls
            if high_fanin:
                bucket_stats.high_fanin_tile_cycles += 1
                bucket_stats.high_fanin_requests += fanin
            stats.total_active_tile_cycles += 1
            stats.total_requests += fanin
            stats.total_fires += fires
            stats.total_stalls += stalls
            if high_fanin:
                stats.total_high_fanin_tile_cycles += 1
                stats.total_high_fanin_requests += fanin
    if stats.total_active_tile_cycles == 0:
        raise SystemExit(f"No active tile-cycles found in {tile_cycles_csv}")
    return stats


def percent(value: float, _position: int | None = None) -> str:
    return f"{value:.0f}%"


def short_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return str(value)


def ensure_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise SystemExit(f"Output exists: {path} (use --force to overwrite)")


def figure_path(output_base: Path, extension: str) -> Path:
    suffix = extension.lstrip(".")
    if suffix == "pdf":
        return output_base.parent / "pdf" / f"{output_base.name}.pdf"
    return output_base.with_suffix(f".{suffix}")


def data_path(output_dir: Path, filename: str) -> Path:
    return output_dir / "data" / filename


def write_summary_csv(path: Path, runs: list[RunStats], buckets: list[int], max_bucket: int, force: bool) -> None:
    ensure_outputs([path], force)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "source_csv",
        "fanin_bucket",
        "tile_cycles",
        "tile_cycle_share",
        "requests",
        "fires",
        "stalls",
        "stall_rate",
        "high_fanin_tile_cycles",
        "high_fanin_requests",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for bucket in buckets:
                bucket_stats = run.buckets.get(bucket, BucketStats())
                writer.writerow(
                    {
                        "run": run.label,
                        "source_csv": run.source,
                        "fanin_bucket": bucket_label(bucket, max_bucket),
                        "tile_cycles": bucket_stats.tile_cycles,
                        "tile_cycle_share": f"{bucket_stats.tile_cycles / run.total_active_tile_cycles:.6f}",
                        "requests": bucket_stats.requests,
                        "fires": bucket_stats.fires,
                        "stalls": bucket_stats.stalls,
                        "stall_rate": f"{bucket_stats.stalls / bucket_stats.requests:.6f}" if bucket_stats.requests else "0.000000",
                        "high_fanin_tile_cycles": bucket_stats.high_fanin_tile_cycles,
                        "high_fanin_requests": bucket_stats.high_fanin_requests,
                    }
                )


def plot_fanin_comparison(runs: list[RunStats], buckets: list[int], max_bucket: int) -> plt.Figure:
    fig = plt.figure(figsize=(14.5, 9.4))
    grid = fig.add_gridspec(3, 1, height_ratios=[0.95, 2.4, 2.4], hspace=0.34)
    kpi_ax = fig.add_subplot(grid[0])
    share_ax = fig.add_subplot(grid[1])
    stall_ax = fig.add_subplot(grid[2], sharex=share_ax)
    fig.suptitle("Port 0 source-side fan-in mechanism", fontsize=23, y=0.985)
    fig.text(
        0.5,
        0.945,
        "Back2Local removes multi-source admission contention, while later pipeline/latency costs remain separate.",
        ha="center",
        va="center",
        fontsize=12.5,
        color="0.35",
    )

    kpi_ax.axis("off")
    for index, run in enumerate(runs):
        color = RUN_COLORS[index % len(RUN_COLORS)]
        stall_rate = run.total_stalls / run.total_requests if run.total_requests else 0.0
        multi_fanin_share = run.total_high_fanin_tile_cycles / run.total_active_tile_cycles
        x = 0.02 + index * (0.96 / max(len(runs), 1))
        width = 0.9 / max(len(runs), 1)
        text = (
            f"{run.label}\n"
            f"requests {short_count(run.total_requests)}   blocked {short_count(run.total_stalls)} ({stall_rate:.1%})\n"
            f"active tile-cycles {short_count(run.total_active_tile_cycles)}   multi-fan-in cycles {multi_fanin_share:.1%}"
        )
        kpi_ax.text(
            x,
            0.45,
            text,
            transform=kpi_ax.transAxes,
            ha="left",
            va="center",
            fontsize=12,
            color="0.08",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": color, "linewidth": 2.0},
        )
        if index == 0 and len(runs) > 1:
            kpi_ax.plot([x + width, x + width], [0.18, 0.78], transform=kpi_ax.transAxes, color="0.85", lw=1.0)

    x_positions = np.arange(len(buckets), dtype=float)
    bar_width = min(0.34, 0.78 / max(len(runs), 1))
    offsets = (np.arange(len(runs)) - (len(runs) - 1) / 2.0) * bar_width
    for run_index, run in enumerate(runs):
        color = RUN_COLORS[run_index % len(RUN_COLORS)]
        shares = [100.0 * run.buckets.get(bucket, BucketStats()).tile_cycles / run.total_active_tile_cycles for bucket in buckets]
        bars = share_ax.bar(x_positions + offsets[run_index], shares, width=bar_width, label=run.label, color=color, alpha=0.86)
        for bar, share in zip(bars, shares):
            if share >= 4.0:
                share_ax.text(bar.get_x() + bar.get_width() / 2, share + 1.3, f"{share:.0f}%", ha="center", va="bottom", fontsize=10)

    share_ax.set_ylabel("Active tile-cycle share")
    share_ax.yaxis.set_major_formatter(FuncFormatter(percent))
    share_ax.set_ylim(0, 108)
    share_ax.grid(axis="y", alpha=0.25)
    share_ax.legend(loc="upper right", frameon=False)
    share_ax.set_title("Where active source-tile cycles land by fan-in count", loc="left", fontsize=14)

    for run_index, run in enumerate(runs):
        color = RUN_COLORS[run_index % len(RUN_COLORS)]
        stall_rates: list[float] = []
        for bucket in buckets:
            bucket_stats = run.buckets.get(bucket, BucketStats())
            if bucket_stats.requests:
                stall_rates.append(100.0 * bucket_stats.stalls / bucket_stats.requests)
            else:
                stall_rates.append(np.nan)
        stall_ax.plot(x_positions, stall_rates, marker="o", markersize=8, linewidth=2.4, label=run.label, color=color)
        for x_pos, rate in zip(x_positions, stall_rates):
            if not np.isnan(rate) and rate > 0.0:
                stall_ax.text(x_pos, rate + 3.0, f"{rate:.0f}%", ha="center", va="bottom", fontsize=10, color=color)

    stall_ax.set_ylabel("Blocked / requests")
    stall_ax.yaxis.set_major_formatter(FuncFormatter(percent))
    stall_ax.set_xlabel("Source tile fan-in on port 0 in one cycle")
    stall_ax.set_xticks(x_positions)
    stall_ax.set_xticklabels([bucket_label(bucket, max_bucket) for bucket in buckets])
    stall_ax.set_ylim(0, 100)
    stall_ax.grid(axis="y", alpha=0.25)
    stall_ax.set_title("Why fan-in matters: stall probability jumps once multiple sources contend", loc="left", fontsize=14)

    for axis in (share_ax, stall_ax):
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.08, top=0.89)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, prefix: str, formats: list[str], force: bool) -> list[Path]:
    paths = [figure_path(output_dir / f"{prefix}_comparison", fmt) for fmt in formats]
    ensure_outputs(paths, force)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    tile_cycles_csvs = [resolve_tile_cycles_csv(path) for path in args.input_path]
    if args.label and len(args.label) != len(tile_cycles_csvs):
        raise SystemExit("Pass exactly one --label per input path")
    labels = args.label or [infer_label(path) for path in tile_cycles_csvs]
    runs = [read_run_stats(path, label, args.max_bucket) for path, label in zip(tile_cycles_csvs, labels)]
    bucket_set = set(range(1, args.max_bucket + 1))
    for run in runs:
        bucket_set.update(run.buckets.keys())
    buckets = sorted(bucket_set)

    output_dir = args.output_dir or default_output_dir(tile_cycles_csvs[0])
    summary_path = data_path(output_dir, f"{args.prefix}_comparison.csv")
    write_summary_csv(summary_path, runs, buckets, args.max_bucket, args.force)
    written = [summary_path]
    written.extend(save_figure(plot_fanin_comparison(runs, buckets, args.max_bucket), output_dir, args.prefix, args.formats, args.force))
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

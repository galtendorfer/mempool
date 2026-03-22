#!/usr/bin/env python3
import argparse
import glob
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from _plotting_common import comparable_run_dirs, format_sweep_label, get_shared_axis_upper, latest_view_dir


CURVE_EXTREMA_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(description="Plot latency/throughput curves from a sweep result directory.")
    parser.add_argument("result_dir", help="Path to a throughput sweep result directory")
    parser.add_argument("--hide-titles", action="store_true", help="Suppress subplot titles for cleaner comparison sheets.")
    parser.add_argument("--x-min", type=float, default=None, help="Optional shared lower x-axis bound.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional shared upper x-axis bound.")
    parser.add_argument("--latency-y-upper", type=float, default=None, help="Optional shared upper y-axis bound for latency.")
    parser.add_argument("--throughput-y-upper", type=float, default=None, help="Optional shared upper y-axis bound for throughput.")
    return parser.parse_args()


def find_result_files(results_glob_base: str):
    result_files = sorted(glob.glob(f"{results_glob_base}/results_partitionprob*"))
    if not result_files:
        result_files = sorted(glob.glob(f"{results_glob_base}/results_seqprob*"))
    return result_files


def data_dir_for_run(result_dir: Path) -> Path:
    data_dir = result_dir / "data"
    if data_dir.is_dir():
        return data_dir

    summary_dir = result_dir / "summary"
    if summary_dir.is_dir():
        return summary_dir

    return result_dir


def collect_curve_extrema(result_dir: Path):
    cache_key = str(result_dir.resolve())
    if cache_key in CURVE_EXTREMA_CACHE:
        return CURVE_EXTREMA_CACHE[cache_key]

    data_root = data_dir_for_run(result_dir)
    result_files = find_result_files(str(data_root))
    all_latency = []
    all_throughput = []
    for result_file in result_files:
        data = np.loadtxt(result_file)
        data = np.atleast_2d(data)
        all_latency.extend(data[:, 1].tolist())
        all_throughput.extend(data[:, 2].tolist())
    CURVE_EXTREMA_CACHE[cache_key] = (all_latency, all_throughput)
    return CURVE_EXTREMA_CACHE[cache_key]


def shared_curve_y_uppers(result_dir: Path, fallback_latency, fallback_throughput):
    latency_values = list(fallback_latency)
    throughput_values = list(fallback_throughput)

    for comparable_dir in comparable_run_dirs(result_dir):
        try:
            latencies, throughputs = collect_curve_extrema(comparable_dir)
        except Exception:
            continue
        latency_values.extend(latencies)
        throughput_values.extend(throughputs)

    latency_upper = max(latency_values) * 1.05 if latency_values else 1.0
    throughput_upper = max(throughput_values) * 1.05 if throughput_values else 1.0
    return latency_upper, throughput_upper


def sync_latest_plots(result_dir: Path, plots_dir: Path) -> None:
    latest_root = latest_view_dir(result_dir)
    if latest_root is None or latest_root == result_dir:
        return

    tile_range_tag = result_dir.parent.name
    throughput_dir = latest_root / "plots" / "throughput"
    throughput_dir.mkdir(parents=True, exist_ok=True)

    for pdf_path in plots_dir.glob("load_throughput*.pdf"):
        target_path = throughput_dir / f"{tile_range_tag}_{pdf_path.name}"
        shutil.copy2(pdf_path, target_path)


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    results_glob_base = data_dir_for_run(result_dir)
    plots_dir = result_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.4, 6.1), constrained_layout=True)

    result_files = find_result_files(str(results_glob_base))
    if not result_files:
        raise SystemExit(f"No throughput summary files found in {results_glob_base}")

    all_req_prob = []
    all_latency = []
    all_throughput = []

    for result_file in result_files:
        basename = os.path.basename(result_file)
        if "partitionprob" in basename:
            prob = basename.split("partitionprob")[-1]
            label = format_sweep_label("partition_prob", prob)
        else:
            prob = basename.split("seqprob")[-1]
            label = format_sweep_label("seq_prob", prob)

        data = np.loadtxt(result_file)
        data = np.atleast_2d(data)
        req_prob, avg_lat, throughput = data[:, 0], data[:, 1], data[:, 2]

        all_req_prob.extend(req_prob.tolist())
        all_latency.extend(avg_lat.tolist())
        all_throughput.extend(throughput.tolist())

        ax1.plot(req_prob, avg_lat, "o-", label=label)
        ax2.plot(req_prob, throughput, "o-", label=label)

    x_min = args.x_min if args.x_min is not None else min(all_req_prob)
    x_max = args.x_max if args.x_max is not None else max(all_req_prob)
    shared_latency_upper, shared_throughput_upper = shared_curve_y_uppers(result_dir, all_latency, all_throughput)
    manifest_latency_upper = get_shared_axis_upper(result_dir, "throughput", "latency_upper")
    manifest_throughput_upper = get_shared_axis_upper(result_dir, "throughput", "throughput_upper")
    latency_y_upper = args.latency_y_upper if args.latency_y_upper is not None else (manifest_latency_upper or shared_latency_upper)
    throughput_y_upper = args.throughput_y_upper if args.throughput_y_upper is not None else (manifest_throughput_upper or shared_throughput_upper)

    ax1.set_xlabel("Offered load (req probability)")
    ax1.set_ylabel("Average latency (cycles)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0.0, latency_y_upper)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, title="Sweep")

    ax2.set_xlabel("Offered load (req probability)")
    ax2.set_ylabel("Throughput (req/core/cycle)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0.0, throughput_y_upper)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, title="Sweep")

    if not args.hide_titles:
        ax1.set_title("Load-Latency Curve")
        ax2.set_title("Load-Throughput Curve")

    fig.savefig(plots_dir / "load_throughput.pdf")
    plt.close(fig)
    sync_latest_plots(result_dir, plots_dir)


if __name__ == "__main__":
    main()

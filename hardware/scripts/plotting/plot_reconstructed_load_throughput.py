#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt

from _plotting_common import comparable_run_dirs, format_sweep_label, get_shared_axis_upper, latest_view_dir


RECONSTRUCTED_EXTREMA_CACHE = {}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Plot reconstructed latency/throughput curves from audit/latency_wraps outputs "
            "and save them as clearly marked provisional plots without overwriting the original plots."
        )
    )
    parser.add_argument("result_dir", help="Path to a throughput result directory")
    parser.add_argument("--hide-titles", action="store_true", help="Suppress subplot titles.")
    parser.add_argument("--x-min", type=float, default=None, help="Optional shared lower x-axis bound.")
    parser.add_argument("--x-max", type=float, default=None, help="Optional shared upper x-axis bound.")
    parser.add_argument(
        "--latency-y-upper",
        type=float,
        default=None,
        help="Optional shared upper y-axis bound for latency.",
    )
    parser.add_argument(
        "--throughput-y-upper",
        type=float,
        default=None,
        help="Optional shared upper y-axis bound for throughput.",
    )
    parser.add_argument(
        "--output-stem",
        default="load_throughput_provisional",
        help="Output filename stem placed in the run's plots directory.",
    )
    parser.add_argument(
        "--keep-analysis-copy",
        action="store_true",
        help="Also write a compatibility copy into analysis/latency_wraps.",
    )
    return parser.parse_args()


def audit_dir_for_run(result_dir: Path) -> Path:
    audit_dir = result_dir / "audit" / "latency_wraps"
    if audit_dir.is_dir():
        return audit_dir

    legacy_analysis_dir = result_dir / "analysis" / "latency_wraps"
    if legacy_analysis_dir.is_dir():
        return legacy_analysis_dir

    return audit_dir


def collect_reconstructed_extrema(result_dir: Path):
    cache_key = str(result_dir.resolve())
    if cache_key in RECONSTRUCTED_EXTREMA_CACHE:
        return RECONSTRUCTED_EXTREMA_CACHE[cache_key]

    analysis_dir = audit_dir_for_run(result_dir)
    diagnostic_files = sorted(analysis_dir.glob("*_diagnostic.csv"))
    all_reconstructed = []
    all_throughput = []
    for diagnostic_file in diagnostic_files:
        rows = read_diagnostic_csv(diagnostic_file)
        all_reconstructed.extend([row["reconstructed_latency"] for row in rows])
        all_throughput.extend([row["throughput"] for row in rows])
    RECONSTRUCTED_EXTREMA_CACHE[cache_key] = (all_reconstructed, all_throughput)
    return RECONSTRUCTED_EXTREMA_CACHE[cache_key]


def shared_reconstructed_y_uppers(result_dir: Path, fallback_latency, fallback_throughput):
    latency_values = list(fallback_latency)
    throughput_values = list(fallback_throughput)

    for comparable_dir in comparable_run_dirs(result_dir):
        try:
            latencies, throughputs = collect_reconstructed_extrema(comparable_dir)
        except Exception:
            continue
        latency_values.extend(latencies)
        throughput_values.extend(throughputs)

    latency_upper = max(latency_values) * 1.05 if latency_values else 1.0
    throughput_upper = max(throughput_values) * 1.05 if throughput_values else 1.0
    return latency_upper, throughput_upper


def tile_range_tag_for_result(result_dir: Path) -> str:
    if result_dir.parent.name.startswith("tilerange"):
        return result_dir.parent.name
    if result_dir.name.startswith("tilerange"):
        return result_dir.name
    return result_dir.name


def read_reconstructed_csv(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "req_prob": float(row["req_prob"]),
                    "reconstructed_latency": float(row["reconstructed_latency"]),
                    "throughput": float(row["throughput"]),
                    "wraps_added": int(row.get("wraps_added", 0)),
                    "total_wraps": int(row.get("total_wraps", 0)),
                }
            )
    return rows


def write_curated_reconstructed_csv(audit_dir: Path, output_path: Path) -> None:
    reconstructed_files = sorted(audit_dir.glob("*_reconstructed.csv"))
    with output_path.open("w", newline="") as handle:
        fieldnames = [
            "sweep_kind",
            "sweep_value",
            "req_prob",
            "reconstructed_latency",
            "throughput",
            "wraps_added",
            "total_wraps",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for reconstructed_file in reconstructed_files:
            sweep_value = partition_value(reconstructed_file)
            sweep_kind = "partition_prob" if "partitionprob" in reconstructed_file.name else "seq_prob"
            for row in read_reconstructed_csv(reconstructed_file):
                writer.writerow(
                    {
                        "sweep_kind": sweep_kind,
                        "sweep_value": sweep_value,
                        "req_prob": row["req_prob"],
                        "reconstructed_latency": row["reconstructed_latency"],
                        "throughput": row["throughput"],
                        "wraps_added": row["wraps_added"],
                        "total_wraps": row["total_wraps"],
                    }
                )


def sync_latest_outputs(result_dir: Path, audit_dir: Path, plots_dir: Path) -> None:
    latest_root = latest_view_dir(result_dir)
    if latest_root is None or latest_root == result_dir:
        return

    tile_range_tag = tile_range_tag_for_result(result_dir)
    throughput_dir = latest_root / "plots" / "throughput"
    throughput_dir.mkdir(parents=True, exist_ok=True)
    latest_data_dir = latest_root / "data" / "throughput"
    latest_data_dir.mkdir(parents=True, exist_ok=True)

    write_curated_reconstructed_csv(audit_dir, latest_data_dir / f"reconstructed_{tile_range_tag}.csv")

    for pdf_path in plots_dir.glob("load_throughput*.pdf"):
        target_path = throughput_dir / f"{tile_range_tag}_{pdf_path.name}"
        shutil.copy2(pdf_path, target_path)


def partition_value(path: Path) -> str:
    name = path.name
    if "partitionprob" in name:
        return name.split("partitionprob", 1)[1].split("_", 1)[0]
    if "seqprob" in name:
        return name.split("seqprob", 1)[1].split("_", 1)[0]
    return name


def read_diagnostic_csv(path: Path):
    rows = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "req_prob": float(row["req_prob"]),
                    "avg_latency": float(row["avg_latency"]),
                    "throughput": float(row["throughput"]),
                    "reconstructed_latency": float(row["reconstructed_latency"]),
                    "likely_wrap": row["likely_wrap"].strip().lower() == "true",
                    "wraps_added": int(row["wraps_added"]),
                }
            )
    return rows


def main():
    args = parse_args()
    result_dir = Path(args.result_dir)
    analysis_dir = audit_dir_for_run(result_dir)
    plots_dir = result_dir / "plots"
    if not analysis_dir.is_dir():
        raise SystemExit(f"Missing analysis directory: {analysis_dir}")

    diagnostic_files = sorted(analysis_dir.glob("*_diagnostic.csv"))
    if not diagnostic_files:
        raise SystemExit(f"No diagnostic CSV files found in {analysis_dir}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14.8, 6.2), constrained_layout=True)

    all_req = []
    all_reconstructed = []
    all_thr = []

    for diagnostic_file in diagnostic_files:
        prob = partition_value(diagnostic_file)
        rows = read_diagnostic_csv(diagnostic_file)

        req_prob = [row["req_prob"] for row in rows]
        reconstructed_latency = [row["reconstructed_latency"] for row in rows]
        throughput = [row["throughput"] for row in rows]

        label = format_sweep_label("partition_prob", prob)
        ax1.plot(req_prob, reconstructed_latency, "o-", label=label)
        ax2.plot(req_prob, throughput, "o-", label=label)

        all_req.extend(req_prob)
        all_reconstructed.extend(reconstructed_latency)
        all_thr.extend(throughput)

    x_min = args.x_min if args.x_min is not None else min(all_req)
    x_max = args.x_max if args.x_max is not None else max(all_req)
    shared_latency_upper, shared_throughput_upper = shared_reconstructed_y_uppers(result_dir, all_reconstructed, all_thr)
    manifest_latency_upper = get_shared_axis_upper(result_dir, "reconstructed", "latency_upper")
    manifest_throughput_upper = get_shared_axis_upper(result_dir, "reconstructed", "throughput_upper")
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
        ax1.set_title("Provisional Load-Latency Curve")
        ax2.set_title("Provisional Load-Throughput Curve")

    output_path = plots_dir / f"{args.output_stem}.pdf"
    fig.savefig(output_path)
    if args.keep_analysis_copy:
        legacy_analysis_dir = result_dir / "analysis" / "latency_wraps"
        legacy_analysis_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(legacy_analysis_dir / f"{args.output_stem}.pdf")
    plt.close(fig)
    sync_latest_outputs(result_dir, analysis_dir, plots_dir)
    print(output_path)


if __name__ == "__main__":
    main()
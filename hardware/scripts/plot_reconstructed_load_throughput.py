#!/usr/bin/env python3
import argparse
import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt


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


def latest_view_dir(result_dir: Path):
    if result_dir.parent.parent.name == "runs":
        return result_dir.parent.parent.parent / "latest" / result_dir.parent.name
    if result_dir.parent.name == "latest":
        return result_dir
    return None


def sync_latest_outputs(result_dir: Path, audit_dir: Path, plots_dir: Path) -> None:
    latest_dir = latest_view_dir(result_dir)
    if latest_dir is None or latest_dir == result_dir:
        return

    latest_dir.mkdir(parents=True, exist_ok=True)

    latest_plots_dir = latest_dir / "plots"
    if latest_plots_dir.exists():
        shutil.rmtree(latest_plots_dir)
    shutil.copytree(plots_dir, latest_plots_dir)

    latest_audit_dir = latest_dir / "audit"
    if latest_audit_dir.exists():
        shutil.rmtree(latest_audit_dir)
    shutil.copytree(audit_dir.parent, latest_audit_dir)


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
    result_dir = Path(args.result_dir).resolve()
    analysis_dir = audit_dir_for_run(result_dir)
    plots_dir = result_dir / "plots"
    if not analysis_dir.is_dir():
        raise SystemExit(f"Missing analysis directory: {analysis_dir}")

    diagnostic_files = sorted(analysis_dir.glob("*_diagnostic.csv"))
    if not diagnostic_files:
        raise SystemExit(f"No diagnostic CSV files found in {analysis_dir}")

    plots_dir.mkdir(parents=True, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4), constrained_layout=True)

    all_req = []
    all_reconstructed = []
    all_thr = []

    for diagnostic_file in diagnostic_files:
        prob = partition_value(diagnostic_file)
        rows = read_diagnostic_csv(diagnostic_file)

        req_prob = [row["req_prob"] for row in rows]
        reconstructed_latency = [row["reconstructed_latency"] for row in rows]
        throughput = [row["throughput"] for row in rows]

        label = f"partition_prob={prob}"
        ax1.plot(req_prob, reconstructed_latency, "o-", label=label)
        ax2.plot(req_prob, throughput, "o-", label=label)

        all_req.extend(req_prob)
        all_reconstructed.extend(reconstructed_latency)
        all_thr.extend(throughput)

    x_min = args.x_min if args.x_min is not None else min(all_req)
    x_max = args.x_max if args.x_max is not None else max(all_req)
    latency_y_upper = args.latency_y_upper if args.latency_y_upper is not None else max(all_reconstructed) * 1.05
    throughput_y_upper = args.throughput_y_upper if args.throughput_y_upper is not None else max(all_thr) * 1.05

    ax1.set_xlabel("Offered load (req probability)")
    ax1.set_ylabel("Average latency (cycles)")
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(0.0, latency_y_upper)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.set_xlabel("Offered load (req probability)")
    ax2.set_ylabel("Throughput (req/core/cycle)")
    ax2.set_xlim(x_min, x_max)
    ax2.set_ylim(0.0, throughput_y_upper)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

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
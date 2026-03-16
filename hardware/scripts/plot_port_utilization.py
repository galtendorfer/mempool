#!/usr/bin/env python3
import argparse
import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


BOUNDARY_FIELDS = {
    "tile": "tile_util_csv",
    "group": "group_util_csv",
    "subgroup": "subgroup_util_csv",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot per-port utilization heatmaps from a throughput result directory."
    )
    parser.add_argument("result_dir", help="Path to a load_thru_questa_* result directory")
    parser.add_argument(
        "--boundary",
        choices=["all", "tile", "group", "subgroup"],
        default="all",
        help="Which boundary type to plot",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Also show the generated figures interactively",
    )
    parser.add_argument(
        "--plots-per-page",
        type=int,
        default=6,
        help="How many port heatmaps to place on each PDF page",
    )
    return parser.parse_args()


def get_selected_boundaries(boundary_arg):
    if boundary_arg == "all":
        return ["tile", "group", "subgroup"]
    return [boundary_arg]


def detect_subgroup_port_count(result_dir):
    summary_path = os.path.join(result_dir, "run_summary.csv")
    if not os.path.exists(summary_path):
        return 0

    subgroup_port_count = 0
    with open(summary_path, newline="") as summary_file:
        for run in csv.DictReader(summary_file):
            if run.get("status") != "ok":
                continue
            csv_relpath = run.get("subgroup_util_csv", "")
            if not csv_relpath:
                continue
            csv_path = os.path.join(result_dir, os.path.basename(csv_relpath))
            if not os.path.exists(csv_path):
                continue
            with open(csv_path, newline="") as port_file:
                for row in csv.DictReader(port_file):
                    subgroup_port_count = max(subgroup_port_count, int(row["remote_subgroup"]))
            if subgroup_port_count:
                break

    return subgroup_port_count


def make_port_label(boundary, row):
    if boundary == "tile":
        return f"tile{row['tile']}_port{row['port']}"
    if boundary == "group":
        subgroup = row.get("subgroup", "-1")
        if subgroup == "-1":
            return (
                f"group{row['group']}_remote{row['remote_group']}"
                f"_tile{row['tile']}"
            )
        return (
            f"group{row['group']}_remote{row['remote_group']}"
            f"_subgroup{subgroup}_tile{row['tile']}"
        )
    return (
        f"group{row['group']}_subgroup{row['subgroup']}"
        f"_remote{row['remote_subgroup']}_tile{row['tile']}"
    )


def make_type_label(boundary, row, subgroup_port_count):
    if boundary == "tile":
        if subgroup_port_count and int(row["port"]) < subgroup_port_count:
            return "tile_to_subgroup_boundary"
        return "tile_to_group_boundary"
    if boundary == "group":
        return "group_boundary"
    return "subgroup_boundary"


def load_boundary_data(result_dir, boundary, subgroup_port_count):
    summary_path = os.path.join(result_dir, "run_summary.csv")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"Missing run_summary.csv in {result_dir}")

    field_name = BOUNDARY_FIELDS[boundary]
    entries = []

    with open(summary_path, newline="") as summary_file:
        for run in csv.DictReader(summary_file):
            if run["status"] != "ok":
                continue

            csv_relpath = run.get(field_name, "")
            if not csv_relpath:
                continue

            csv_path = os.path.join(result_dir, os.path.basename(csv_relpath))
            if not os.path.exists(csv_path):
                continue

            with open(csv_path, newline="") as port_file:
                rows = list(csv.DictReader(port_file))

            if not rows:
                continue

            partition_prob = float(run["partition_prob"])
            req_prob = float(run["req_prob"])

            for row in rows:
                entries.append(
                    {
                        "partition_prob": partition_prob,
                        "req_prob": req_prob,
                        "util_pct": float(row["util_pct"]),
                        "label": make_port_label(boundary, row),
                        "type_label": make_type_label(boundary, row, subgroup_port_count),
                    }
                )

    return entries


def build_heatmaps(entries):
    partition_probs = sorted({entry["partition_prob"] for entry in entries})
    req_probs = sorted({entry["req_prob"] for entry in entries})
    labels = sorted({entry["label"] for entry in entries})

    partition_index = {value: idx for idx, value in enumerate(partition_probs)}
    req_index = {value: idx for idx, value in enumerate(req_probs)}

    matrices = {
        label: np.full((len(partition_probs), len(req_probs)), np.nan) for label in labels
    }

    for entry in entries:
        matrices[entry["label"]][
            partition_index[entry["partition_prob"]], req_index[entry["req_prob"]]
        ] = entry["util_pct"]

    return partition_probs, req_probs, matrices


def build_aggregate_entries(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["type_label"], entry["partition_prob"], entry["req_prob"])].append(entry["util_pct"])

    aggregate_entries = []
    for (type_label, partition_prob, req_prob), values in grouped.items():
        aggregate_entries.append(
            {
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "util_pct": float(np.mean(values)),
                "label": type_label,
            }
        )

    return aggregate_entries


def render_boundary_pdf(result_dir, boundary, entries, plots_per_page, show):
    partition_probs, req_probs, matrices = build_heatmaps(entries)
    output_path = os.path.join(result_dir, f"{boundary}_port_utilization.pdf")

    labels = sorted(matrices)
    pages = int(math.ceil(len(labels) / plots_per_page))
    cols = 2
    rows = int(math.ceil(plots_per_page / cols))

    with PdfPages(output_path) as pdf:
        figures = []
        for page_idx in range(pages):
            fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
            figures.append(fig)
            axes_flat = axes.flatten()
            page_labels = labels[page_idx * plots_per_page : (page_idx + 1) * plots_per_page]

            image = None
            for ax, label in zip(axes_flat, page_labels):
                image = ax.imshow(
                    matrices[label],
                    origin="lower",
                    aspect="auto",
                    vmin=0.0,
                    vmax=100.0,
                    cmap="viridis",
                )
                ax.set_title(label)
                ax.set_xlabel("req_prob")
                ax.set_ylabel("partition_prob")
                ax.set_xticks(range(len(req_probs)))
                ax.set_xticklabels([f"{value:.2f}" for value in req_probs], rotation=45, ha="right")
                ax.set_yticks(range(len(partition_probs)))
                ax.set_yticklabels([f"{value:.1f}" for value in partition_probs])

            for ax in axes_flat[len(page_labels) :]:
                ax.axis("off")

            if image is not None:
                fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.85, label="util_pct")

            fig.suptitle(f"{boundary.capitalize()} Port Utilization")
            fig.tight_layout()
            pdf.savefig(fig)

        if show:
            for fig in figures:
                fig.show()
        else:
            for fig in figures:
                plt.close(fig)

    return output_path


def render_port_type_pdf(result_dir, entries, plots_per_page, show):
    aggregate_entries = build_aggregate_entries(entries)
    if not aggregate_entries:
        return None

    partition_probs, req_probs, matrices = build_heatmaps(aggregate_entries)
    output_path = os.path.join(result_dir, "port_type_utilization.pdf")

    labels = sorted(matrices)
    cols = 2
    rows = int(math.ceil(min(len(labels), plots_per_page) / cols)) or 1

    fig, axes = plt.subplots(rows, cols, figsize=(12, 4 * rows), squeeze=False)
    axes_flat = axes.flatten()
    image = None

    for ax, label in zip(axes_flat, labels):
        image = ax.imshow(
            matrices[label],
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=100.0,
            cmap="viridis",
        )
        ax.set_title(label)
        ax.set_xlabel("req_prob")
        ax.set_ylabel("partition_prob")
        ax.set_xticks(range(len(req_probs)))
        ax.set_xticklabels([f"{value:.2f}" for value in req_probs], rotation=45, ha="right")
        ax.set_yticks(range(len(partition_probs)))
        ax.set_yticklabels([f"{value:.1f}" for value in partition_probs])

    for ax in axes_flat[len(labels) :]:
        ax.axis("off")

    if image is not None:
        fig.colorbar(image, ax=axes_flat.tolist(), shrink=0.85, label="avg util_pct")

    fig.suptitle("Port Type Utilization")
    fig.tight_layout()
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def main():
    args = parse_args()
    result_dir = os.path.abspath(args.result_dir)
    subgroup_port_count = detect_subgroup_port_count(result_dir)

    generated = []
    all_entries = []
    for boundary in get_selected_boundaries(args.boundary):
        entries = load_boundary_data(result_dir, boundary, subgroup_port_count)
        if not entries:
            continue
        all_entries.extend(entries)
        generated.append(render_boundary_pdf(result_dir, boundary, entries, args.plots_per_page, args.show))

    type_pdf = render_port_type_pdf(result_dir, all_entries, args.plots_per_page, args.show)
    if type_pdf is not None:
        generated.append(type_pdf)

    if not generated:
        raise SystemExit("No utilization CSV data found to plot.")

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
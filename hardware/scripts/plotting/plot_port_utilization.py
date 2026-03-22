#!/usr/bin/env python3
import argparse
import csv
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

from _plotting_common import comparable_run_dirs, format_direction_label, format_partition_label, format_port_semantic_label, get_shared_axis_upper, latest_view_dir

BOUNDARY_FIELDS = {
    "tile": "tile_util_csv",
    "group": "group_util_csv",
    "subgroup": "subgroup_util_csv",
}

SAME_GROUP_COLOR = "#1f77b4"
REMOTE_COLORS = ["#ff7f0e", "#2ca02c", "#d62728", "#8c564b", "#e377c2"]
UNUSED_COLOR = "#4d4d4d"
FORCE_NORMALIZED_Y = False
HIDE_FIGURE_TITLES = False
SHARED_Y_UPPERS = {}
FORCED_TOPOLOGY = "auto"
TOPOLOGY_INFO_CACHE = {}
BOUNDARY_DATA_CACHE = {}
SHARED_METRIC_UPPER_CACHE = {}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Plot average utilization per boundary port index from a throughput result directory."
        )
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
        "--include-secondary",
        action="store_true",
        help="Also generate secondary per-port-by-index diagnostic plots",
    )
    parser.add_argument(
        "--plot-set",
        choices=["merged-report", "directional-utilization", "full"],
        default="merged-report",
        help=(
            "Plot set to generate. 'merged-report' emits merged incoming+outgoing per-port "
            "utilization and backpressure plots averaged across all instances of each boundary "
            "type; 'directional-utilization' keeps the older incoming/outgoing split; 'full' "
            "keeps the wider legacy figure set."
        ),
    )
    parser.add_argument(
        "--topology",
        choices=["auto", "mempool", "terapool"],
        default="auto",
        help="Override topology detection for MemPool or TeraPool result directories",
    )
    parser.add_argument(
        "--force-normalized-y",
        action="store_true",
        help="Force normalized metrics such as utilization and backpressure to use a fixed y-axis range of [0, 1].",
    )
    parser.add_argument(
        "--hide-figure-titles",
        action="store_true",
        help="Suppress figure-level titles so generated PDFs are cleaner for comparison sheets.",
    )
    parser.add_argument(
        "--shared-y-upper",
        action="append",
        default=[],
        metavar="OUTPUT_SUFFIX=VALUE",
        help="Force a shared y-axis ceiling for a specific output suffix, for example ports_utilization=0.62.",
    )
    return parser.parse_args(argv)


def parse_shared_y_uppers(raw_values):
    shared = {}
    for raw_value in raw_values:
        name, separator, value = raw_value.partition("=")
        if not separator:
            raise SystemExit(f"Invalid --shared-y-upper value: {raw_value}")
        name = name.strip()
        if not name:
            raise SystemExit(f"Invalid --shared-y-upper name: {raw_value}")
        try:
            shared[name] = float(value)
        except ValueError as exc:
            raise SystemExit(f"Invalid --shared-y-upper numeric value: {raw_value}") from exc
    return shared


def normalized_y_upper(metric, y_max, output_suffix=None):
    if output_suffix is not None and output_suffix in SHARED_Y_UPPERS:
        return SHARED_Y_UPPERS[output_suffix]
    if metric in {"utilization", "backpressure"}:
        if FORCE_NORMALIZED_Y:
            return 1.0
        return min(1.0, max(0.01, y_max * 1.05))
    return max(1.0, y_max * 1.05)


def subplot_grid(item_count, max_cols, base_width, base_height, wide_width=5.8, wide_height=5.6, collapse_two_rows=True):
    cols = min(max_cols, max(1, item_count))
    rows = int(np.ceil(item_count / cols))
    if collapse_two_rows and rows == 2:
        cols = max(1, item_count)
        rows = 1
        return rows, cols, (wide_width * cols, wide_height)
    return rows, cols, (base_width * cols, base_height * rows)


def partition_subplot_max_cols(topology_info):
    if topology_info["name"] == "mempool":
        return 4
    return 3


def find_summary_path(result_dir, filename):
    data_candidate = os.path.join(result_dir, "data", filename)
    if os.path.exists(data_candidate):
        return data_candidate
    summary_candidate = os.path.join(result_dir, "summary", filename)
    if os.path.exists(summary_candidate):
        return summary_candidate
    return os.path.join(result_dir, filename)


def shared_metric_y_upper(result_dir, boundary, metric, output_suffix, fallback_y_max, transform_key, transform):
    if transform_key == "average_port_entries":
        manifest_axis_key = None
        if metric == "utilization":
            manifest_axis_key = "utilization_upper"
        elif metric == "backpressure":
            manifest_axis_key = "backpressure_upper"
        if manifest_axis_key is not None:
            manifest_upper = get_shared_axis_upper(result_dir, "boundary", manifest_axis_key, boundary=boundary)
            if manifest_upper is not None:
                return manifest_upper

    cache_key = (
        os.path.abspath(result_dir),
        boundary,
        metric,
        output_suffix,
        transform_key,
        FORCED_TOPOLOGY,
        FORCE_NORMALIZED_Y,
        tuple(sorted(SHARED_Y_UPPERS.items())),
        fallback_y_max,
    )
    if cache_key in SHARED_METRIC_UPPER_CACHE:
        return SHARED_METRIC_UPPER_CACHE[cache_key]

    y_values = [fallback_y_max]
    for comparable_dir in comparable_run_dirs(result_dir):
        try:
            topology_info = detect_topology_info(str(comparable_dir), FORCED_TOPOLOGY)
            entries = load_boundary_data(str(comparable_dir), boundary)
            transformed_entries = transform(entries, topology_info)
        except Exception:
            continue

        metric_values = [entry[metric] for entry in transformed_entries if entry.get(metric) is not None]
        if metric_values:
            y_values.append(max(metric_values))

    SHARED_METRIC_UPPER_CACHE[cache_key] = normalized_y_upper(metric, max(y_values) if y_values else fallback_y_max, output_suffix)
    return SHARED_METRIC_UPPER_CACHE[cache_key]


def sync_latest_plots(result_dir):
    latest_root = latest_view_dir(result_dir)
    result_path = Path(result_dir)
    if latest_root is None or latest_root == result_path:
        return

    source_plots_dir = result_path / "plots"
    if not source_plots_dir.is_dir():
        return

    latest_plots_root = latest_root / "plots"
    latest_plots_root.mkdir(parents=True, exist_ok=True)

    tile_range_prefix = tile_range_prefix_for_result(result_dir)

    for boundary in ("tile", "group", "subgroup"):
        source_boundary_dir = source_plots_dir / boundary
        if not source_boundary_dir.is_dir():
            continue
        target_boundary_dir = latest_plots_root / boundary
        target_boundary_dir.mkdir(parents=True, exist_ok=True)
        if tile_range_prefix is not None:
            for existing_pdf in target_boundary_dir.glob(f"{tile_range_prefix}*.pdf"):
                existing_pdf.unlink()
        for pdf_path in source_boundary_dir.glob("*.pdf"):
            shutil.copy2(pdf_path, target_boundary_dir / pdf_path.name)


def resolve_run_artifact(result_dir, csv_relpath):
    if not csv_relpath:
        return None
    if os.path.isabs(csv_relpath):
        return csv_relpath

    direct_path = os.path.join(result_dir, csv_relpath)
    if os.path.exists(direct_path):
        return direct_path

    legacy_path = os.path.join(result_dir, os.path.basename(csv_relpath))
    if os.path.exists(legacy_path):
        return legacy_path

    return direct_path


def get_selected_boundaries(boundary_arg):
    if boundary_arg == "all":
        return ["tile", "group", "subgroup"]
    return [boundary_arg]


def tile_range_prefix_for_result(result_dir):
    match = re.search(r"(tilerange\d+)", os.path.abspath(result_dir))
    if match is None:
        return None
    return f"{match.group(1)}_"


def clear_boundary_plot_outputs(result_dir, boundary):
    tile_range_prefix = tile_range_prefix_for_result(result_dir)
    if tile_range_prefix is None:
        return

    boundary_dir = Path(result_dir) / "plots" / boundary
    if not boundary_dir.is_dir():
        return

    for pdf_path in boundary_dir.glob(f"{tile_range_prefix}*.pdf"):
        pdf_path.unlink()


def make_output_path(result_dir, stem, subdir=None):
    result_path = os.path.abspath(result_dir)
    match = re.search(r"(tilerange\d+)", result_path)
    prefix = f"{match.group(1)}_" if match else ""
    output_dir = os.path.join(result_dir, "plots")
    if subdir:
        output_dir = os.path.join(output_dir, subdir)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, f"{prefix}{stem}.pdf")


def normalize_direction(direction):
    if not direction:
        return "outgoing"
    return direction.strip().lower()


def get_available_directions(entries):
    preferred_order = ["outgoing", "incoming"]
    present = {entry.get("direction", "outgoing") for entry in entries}
    return [direction for direction in preferred_order if direction in present]


def average_utilization_label(entries):
    available_directions = get_available_directions(entries)
    if len(available_directions) > 1:
        return "avg accepted tx/dir/cycle"
    return "accepted tx/cycle"


def total_throughput_label(entries):
    available_directions = get_available_directions(entries)
    if len(available_directions) > 1:
        return "total accepted tx/cycle"
    return "accepted tx/cycle"


def qualify_output_suffix(direction, available_directions, base_suffix):
    if available_directions == ["outgoing"] and direction == "outgoing":
        return base_suffix
    return f"{direction}_{base_suffix}"


def qualify_title(direction, available_directions, base_title):
    if available_directions == ["outgoing"] and direction == "outgoing":
        return base_title
    return f"{format_direction_label(direction)} {base_title}"


def filter_entries_by_direction(entries, direction):
    if direction is None:
        return entries
    return [entry for entry in entries if entry.get("direction", "outgoing") == direction]


def topology_title(topology_info):
    return "TeraPool" if topology_info["name"] == "terapool" else "MemPool"


def detect_topology_info(result_dir, forced_topology):
    cache_key = (os.path.abspath(result_dir), forced_topology)
    if cache_key in TOPOLOGY_INFO_CACHE:
        return TOPOLOGY_INFO_CACHE[cache_key]

    if forced_topology != "auto":
        TOPOLOGY_INFO_CACHE[cache_key] = {
            "name": forced_topology,
            "num_groups": None,
            "num_subgroups_per_group": 4 if forced_topology == "terapool" else 1,
        }
        return TOPOLOGY_INFO_CACHE[cache_key]

    info = {
        "name": "mempool",
        "num_groups": None,
        "num_subgroups_per_group": 1,
    }
    summary_path = find_summary_path(result_dir, "run_summary.csv")
    if not os.path.exists(summary_path):
        TOPOLOGY_INFO_CACHE[cache_key] = info
        return info

    with open(summary_path, newline="") as summary_file:
        for run in csv.DictReader(summary_file):
            if run["status"] != "ok":
                continue

            group_csv_relpath = run.get(BOUNDARY_FIELDS["group"], "")
            if group_csv_relpath:
                group_csv_path = resolve_run_artifact(result_dir, group_csv_relpath)
                if os.path.exists(group_csv_path):
                    with open(group_csv_path, newline="") as group_file:
                        for row in csv.DictReader(group_file):
                            info["num_groups"] = max(
                                info["num_groups"] or 0,
                                int(row["remote_group"]) + 1,
                            )
                            subgroup = int(row["subgroup"])
                            if subgroup >= 0:
                                info["name"] = "terapool"
                                info["num_subgroups_per_group"] = max(
                                    info["num_subgroups_per_group"],
                                    subgroup + 1,
                                )

            subgroup_csv_relpath = run.get(BOUNDARY_FIELDS["subgroup"], "")
            if subgroup_csv_relpath:
                subgroup_csv_path = resolve_run_artifact(result_dir, subgroup_csv_relpath)
                if os.path.exists(subgroup_csv_path):
                    with open(subgroup_csv_path, newline="") as subgroup_file:
                        for row in csv.DictReader(subgroup_file):
                            info["name"] = "terapool"
                            info["num_subgroups_per_group"] = max(
                                info["num_subgroups_per_group"],
                                int(row["subgroup"]) + 1,
                                int(row["remote_subgroup"]) + 1,
                            )
                            break

            if info["name"] == "terapool" and info["num_groups"] is not None:
                break

    TOPOLOGY_INFO_CACHE[cache_key] = info
    return info


def get_port_index(boundary, row):
    if boundary == "tile":
        return int(row["port"])
    if boundary == "group":
        return int(row["remote_group"]) - 1
    return int(row["remote_subgroup"]) - 1


def format_port_label(boundary, topology_info, port_index, num_ports):
    del num_ports
    return format_port_semantic_label(boundary, topology_info, port_index)


def get_port_color(boundary, port_index):
    if boundary == "tile" and port_index == 0:
        return SAME_GROUP_COLOR
    remote_index = port_index - 1 if boundary == "tile" else port_index
    return REMOTE_COLORS[remote_index % len(REMOTE_COLORS)]


def get_class_color(boundary, class_name):
    if class_name == "same-group":
        return SAME_GROUP_COLOR
    return REMOTE_COLORS[0]


def load_boundary_data(result_dir, boundary):
    cache_key = (os.path.abspath(result_dir), boundary)
    if cache_key in BOUNDARY_DATA_CACHE:
        return BOUNDARY_DATA_CACHE[cache_key]

    summary_path = find_summary_path(result_dir, "run_summary.csv")
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

            csv_path = resolve_run_artifact(result_dir, csv_relpath)
            if not os.path.exists(csv_path):
                continue

            with open(csv_path, newline="") as port_file:
                rows = list(csv.DictReader(port_file))

            if not rows:
                continue

            partition_prob = float(run["partition_prob"])
            req_prob = float(run["req_prob"])

            for row in rows:
                port_index = get_port_index(boundary, row)
                if port_index < 0:
                    continue
                entries.append(
                    {
                        "partition_prob": partition_prob,
                        "req_prob": req_prob,
                        "direction": normalize_direction(row.get("direction", "outgoing")),
                        "source_group": int(row["group"]) if "group" in row and row["group"] else -1,
                        "cycles": float(row["cycles"]),
                        "accepts": float(row["accepts"]),
                        "utilization": float(row["accepts"]) / float(row["cycles"]) if float(row["cycles"]) else 0.0,
                        "stalls": float(row["stalls"]),
                        "port_index": port_index,
                        "subgroup": int(row["subgroup"]) if "subgroup" in row and row["subgroup"] else -1,
                    }
                )

    BOUNDARY_DATA_CACHE[cache_key] = entries
    return BOUNDARY_DATA_CACHE[cache_key]


def build_average_port_entries(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["port_index"], entry["partition_prob"], entry["req_prob"])].append(entry)

    averaged_entries = []
    for (port_index, partition_prob, req_prob), values in sorted(grouped.items()):
        total_cycles = sum(value["cycles"] for value in values)
        base_cycles = max(value["cycles"] for value in values)
        total_accepts = sum(value["accepts"] for value in values)
        total_stalls = sum(value["stalls"] for value in values)
        attempts = total_accepts + total_stalls
        direction_count = len({value.get("direction", "outgoing") for value in values}) or 1
        averaged_entries.append(
            {
                "port_index": port_index,
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "direction_count": direction_count,
                "throughput": (total_accepts / base_cycles) if base_cycles else 0.0,
                "utilization": (total_accepts / total_cycles) if total_cycles else 0.0,
                "stalls": float(np.mean([value["stalls"] for value in values])),
                "backpressure": (total_stalls / attempts) if attempts else 0.0,
            }
        )

    return averaged_entries


def build_source_group_port_entries(entries):
    grouped = defaultdict(list)
    for entry in entries:
        source_group = entry.get("source_group", -1)
        if source_group < 0:
            continue
        grouped[(source_group, entry["port_index"], entry["partition_prob"], entry["req_prob"])].append(entry)

    averaged_entries = []
    for (source_group, port_index, partition_prob, req_prob), values in sorted(grouped.items()):
        total_cycles = sum(value["cycles"] for value in values)
        base_cycles = max(value["cycles"] for value in values)
        total_accepts = sum(value["accepts"] for value in values)
        total_stalls = sum(value["stalls"] for value in values)
        attempts = total_accepts + total_stalls
        averaged_entries.append(
            {
                "source_group": source_group,
                "port_index": port_index,
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "throughput": (total_accepts / base_cycles) if base_cycles else 0.0,
                "utilization": (total_accepts / total_cycles) if total_cycles else 0.0,
                "backpressure": (total_stalls / attempts) if attempts else 0.0,
            }
        )

    return averaged_entries


def build_source_group_aggregate_entries(entries):
    grouped = defaultdict(list)
    for entry in entries:
        source_group = entry.get("source_group", -1)
        if source_group < 0:
            continue
        grouped[(source_group, entry["partition_prob"], entry["req_prob"])].append(entry)

    aggregate_entries = []
    for (source_group, partition_prob, req_prob), values in sorted(grouped.items()):
        total_cycles = sum(value["cycles"] for value in values)
        base_cycles = max(value["cycles"] for value in values)
        total_accepts = sum(value["accepts"] for value in values)
        total_stalls = sum(value["stalls"] for value in values)
        attempts = total_accepts + total_stalls
        aggregate_entries.append(
            {
                "source_group": source_group,
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "throughput": (total_accepts / base_cycles) if base_cycles else 0.0,
                "utilization": (total_accepts / total_cycles) if total_cycles else 0.0,
                "backpressure": (total_stalls / attempts) if attempts else 0.0,
            }
        )

    return aggregate_entries


def build_series(entries, metric):
    partition_probs = sorted({entry["partition_prob"] for entry in entries})
    port_indices = sorted({entry["port_index"] for entry in entries})

    series = defaultdict(list)
    for entry in sorted(entries, key=lambda item: (item["port_index"], item["partition_prob"], item["req_prob"])):
        if entry[metric] is None:
            continue
        series[(entry["port_index"], entry["partition_prob"])].append(
            (entry["req_prob"], entry[metric])
        )

    return partition_probs, port_indices, series


def build_summary_entries(entries):
    grouped = defaultdict(list)
    for entry in entries:
        grouped[(entry["partition_prob"], entry["req_prob"])].append(entry)

    summary_entries = []
    for (partition_prob, req_prob), values in sorted(grouped.items()):
        utilizations = [value["utilization"] for value in values]
        backpressures = [value["backpressure"] for value in values]
        max_utilization = max(utilizations) if utilizations else 0.0
        mean_utilization = float(np.mean(utilizations)) if utilizations else 0.0
        summary_entries.append(
            {
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "bottleneck_utilization": max_utilization,
                "imbalance": max_utilization - mean_utilization,
                "worst_backpressure": max(backpressures) if backpressures else 0.0,
            }
        )

    return summary_entries


def build_port_class_entries(boundary, entries, topology_info):
    averaged_entries = build_average_port_entries(entries)
    grouped = defaultdict(list)
    total_ports = max(1, len({entry["port_index"] for entry in averaged_entries}))

    for entry in averaged_entries:
        if boundary == "tile":
            if topology_info["name"] == "terapool":
                if entry["port_index"] < topology_info["num_subgroups_per_group"]:
                    port_class = "same-group total"
                else:
                    port_class = "remote-group total"
            else:
                port_class = "same-group" if entry["port_index"] == 0 else "remote-group total"
        elif boundary == "group":
            port_class = "remote-group total"
        else:
            port_class = "remote-subgroup total"
        grouped[(entry["partition_prob"], entry["req_prob"], port_class)].append(entry)

    class_entries = []
    for (partition_prob, req_prob, port_class), values in sorted(grouped.items()):
        class_entries.append(
            {
                "partition_prob": partition_prob,
                "req_prob": req_prob,
                "port_class": port_class,
                "throughput": float(np.sum([value["throughput"] for value in values])) / total_ports,
                "utilization": float(np.sum([value["utilization"] for value in values])) / total_ports,
                "backpressure": float(np.mean([value["backpressure"] for value in values])),
            }
        )

    return class_entries


def render_port_class_pdf(result_dir, boundary, entries, topology_info, show, metric, output_suffix, ylabel, title_suffix):
    class_entries = build_port_class_entries(boundary, entries, topology_info)
    if not class_entries:
        return None

    output_path = make_output_path(result_dir, f"{boundary}_{output_suffix}", boundary)
    partition_probs = sorted({entry["partition_prob"] for entry in class_entries})
    class_names = sorted(
        {entry["port_class"] for entry in class_entries},
        key=lambda name: 0 if name == "same-group" else 1,
    )
    all_req_prob = sorted({entry["req_prob"] for entry in class_entries})
    all_values = [entry[metric] for entry in class_entries]

    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0
    y_max = max(all_values) if all_values else 0.0
    y_upper = shared_metric_y_upper(
        result_dir,
        boundary,
        metric,
        output_suffix,
        y_max,
        "port_class",
        lambda comparable_entries, comparable_topology: build_port_class_entries(boundary, comparable_entries, comparable_topology),
    )

    rows, cols, figsize = subplot_grid(len(partition_probs), 3, 7.2, 5.2)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, partition_prob in zip(axes_flat, partition_probs):
        for class_name in class_names:
            points = [
                entry
                for entry in class_entries
                if entry["partition_prob"] == partition_prob and entry["port_class"] == class_name
            ]
            if not points:
                continue
            req_prob = [point["req_prob"] for point in points]
            values = [point[metric] for point in points]
            ax.plot(
                req_prob,
                values,
                "o-",
                label=class_name,
                color=get_class_color(boundary, class_name),
            )

        ax.set_title(format_partition_label(partition_prob))
        ax.set_xlabel("req_prob")
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc="upper left", framealpha=0.85, title="Class")

    for ax in axes_flat[len(partition_probs):]:
        ax.axis("off")

    if not HIDE_FIGURE_TITLES:
        fig.suptitle(
            f"{topology_title(topology_info)} {boundary.capitalize()} Boundary Port Split\n"
            f"{title_suffix}\nClass lines show total class share of boundary capacity"
        )
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def render_capacity_breakdown_pdf(result_dir, boundary, entries, topology_info, show, output_suffix, title_suffix):
    averaged_entries = build_average_port_entries(entries)
    if not averaged_entries:
        return None

    output_path = make_output_path(result_dir, f"{boundary}_{output_suffix}", boundary)
    partition_probs = sorted({entry["partition_prob"] for entry in averaged_entries})
    port_indices = sorted({entry["port_index"] for entry in averaged_entries})
    num_ports = len(port_indices)
    req_probs = sorted({entry["req_prob"] for entry in averaged_entries})

    rows, cols, figsize = subplot_grid(
        len(partition_probs),
        partition_subplot_max_cols(topology_info),
        8.2,
        6.0,
        collapse_two_rows=False,
    )
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, partition_prob in zip(axes_flat, partition_probs):
        partition_entries = [
            entry for entry in averaged_entries if entry["partition_prob"] == partition_prob
        ]
        utilization_by_req = {
            (entry["req_prob"], entry["port_index"]): entry["utilization"]
            for entry in partition_entries
        }

        for port_index in port_indices:
            values = [
                utilization_by_req.get((req_prob, port_index), 0.0) / num_ports
                for req_prob in req_probs
            ]
            ax.plot(
                req_probs,
                values,
                "o-",
                color=get_port_color(boundary, port_index),
                label=format_port_label(boundary, topology_info, port_index, num_ports),
            )

        used = np.array([
            sum(utilization_by_req.get((req_prob, port_index), 0.0) for port_index in port_indices) / num_ports
            for req_prob in req_probs
        ])
        unused = np.clip(1.0 - used, 0.0, 1.0)
        ax.plot(
            req_probs,
            unused,
            "o--",
            color=UNUSED_COLOR,
            linewidth=2.0,
            label="unused capacity",
        )

        ax.set_title(format_partition_label(partition_prob))
        ax.set_xlabel("req_prob")
        ax.set_ylabel("share of total port capacity")
        ax.set_ylim(0.0, 1.0)
        ax.set_xlim(min(req_probs), max(req_probs))
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.9, title="Port")

    for ax in axes_flat[len(partition_probs):]:
        ax.axis("off")

    fig.suptitle(
        f"{topology_title(topology_info)} {boundary.capitalize()} Boundary Capacity Breakdown\n"
        f"{title_suffix}\nPer req_prob, normalized port shares plus unused capacity; all shares sum to 100%"
    )
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


# Secondary plots kept for optional deeper analysis.
def render_summary_pdf(result_dir, boundary, entries, show, metric, output_suffix, ylabel, title_suffix):
    summary_entries = build_summary_entries(build_average_port_entries(entries))
    if not summary_entries:
        return None

    output_path = make_output_path(result_dir, f"{boundary}_{output_suffix}", boundary)
    partition_probs = sorted({entry["partition_prob"] for entry in summary_entries})
    all_req_prob = sorted({entry["req_prob"] for entry in summary_entries})
    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0
    y_max = max(entry[metric] for entry in summary_entries)
    y_upper = normalized_y_upper(metric, y_max, output_suffix)

    fig, ax = plt.subplots(figsize=(8.2, 5.8), constrained_layout=True)

    for partition_prob in partition_probs:
        points = [
            entry for entry in summary_entries if entry["partition_prob"] == partition_prob
        ]
        req_prob = [point["req_prob"] for point in points]
        values = [point[metric] for point in points]
        ax.plot(req_prob, values, "o-", label=format_partition_label(partition_prob))

    ax.set_xlabel("req_prob")
    ax.set_ylabel(ylabel)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0.0, y_upper)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10, ncol=2, loc="upper left", framealpha=0.85, title="Sweep")
    if not HIDE_FIGURE_TITLES:
        fig.suptitle(f"{boundary.capitalize()} Boundary Summary\n{title_suffix}")
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def render_partition_comparison_pdf(
    result_dir,
    boundary,
    entries,
    topology_info,
    show,
    metric,
    output_suffix,
    ylabel,
    title_suffix,
):
    averaged_entries = build_average_port_entries(entries)
    plotted_entries = [entry for entry in averaged_entries if entry[metric] is not None]
    if not plotted_entries:
        return None

    output_path = make_output_path(result_dir, f"{boundary}_{output_suffix}", boundary)
    partition_probs = sorted({entry["partition_prob"] for entry in plotted_entries})
    port_indices = sorted({entry["port_index"] for entry in plotted_entries})
    all_req_prob = sorted({entry["req_prob"] for entry in plotted_entries})
    all_values = [entry[metric] for entry in plotted_entries]

    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0
    y_max = max(all_values) if all_values else 0.0
    y_upper = shared_metric_y_upper(
        result_dir,
        boundary,
        metric,
        output_suffix,
        y_max,
        "average_port_entries",
        lambda comparable_entries, comparable_topology: build_average_port_entries(comparable_entries),
    )

    rows, cols, figsize = subplot_grid(len(partition_probs), 3, 7.4, 5.4)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, partition_prob in zip(axes_flat, partition_probs):
        for port_index in port_indices:
            points = [
                entry
                for entry in plotted_entries
                if entry["partition_prob"] == partition_prob and entry["port_index"] == port_index
            ]
            if not points:
                continue

            req_prob = [point["req_prob"] for point in points]
            values = [point[metric] for point in points]
            ax.plot(
                req_prob,
                values,
                "o-",
                color=get_port_color(boundary, port_index),
                label=format_port_label(boundary, topology_info, port_index, len(port_indices)),
            )

        ax.set_title(format_partition_label(partition_prob))
        ax.set_xlabel("req_prob")
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85, title="Port")

    for ax in axes_flat[len(partition_probs):]:
        ax.axis("off")

    if not HIDE_FIGURE_TITLES:
        fig.suptitle(
            f"{topology_title(topology_info)} {boundary.capitalize()} Boundary Port Comparison\n"
            f"{title_suffix}\nEach subplot fixes part_prob and compares port indices directly"
        )
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def render_group_source_pdf(
    result_dir,
    entries,
    topology_info,
    show,
    metric,
    output_suffix,
    ylabel,
    title_suffix,
):
    averaged_entries = build_source_group_port_entries(entries)
    if not averaged_entries:
        return None

    output_path = make_output_path(result_dir, f"group_{output_suffix}", "group")
    partition_probs = sorted({entry["partition_prob"] for entry in averaged_entries})
    source_groups = sorted({entry["source_group"] for entry in averaged_entries})
    port_indices = sorted({entry["port_index"] for entry in averaged_entries})
    all_req_prob = sorted({entry["req_prob"] for entry in averaged_entries})
    all_values = [entry[metric] for entry in averaged_entries]

    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0
    y_max = max(all_values) if all_values else 0.0
    y_upper = shared_metric_y_upper(
        result_dir,
        "group",
        metric,
        output_suffix,
        y_max,
        "source_group_port_entries",
        lambda comparable_entries, comparable_topology: build_source_group_port_entries(comparable_entries),
    )
    rows, cols, figsize = subplot_grid(len(source_groups), 2, 7.4, 5.4)

    with PdfPages(output_path) as pdf:
        figures = []
        for partition_prob in partition_probs:
            fig, axes = plt.subplots(
                rows,
                cols,
                figsize=figsize,
                squeeze=False,
                constrained_layout=True,
            )
            figures.append(fig)
            axes_flat = axes.flatten()

            for ax, source_group in zip(axes_flat, source_groups):
                for port_index in port_indices:
                    points = [
                        entry
                        for entry in averaged_entries
                        if entry["partition_prob"] == partition_prob
                        and entry["source_group"] == source_group
                        and entry["port_index"] == port_index
                    ]
                    if not points:
                        continue

                    req_prob = [point["req_prob"] for point in points]
                    values = [point[metric] for point in points]
                    ax.plot(
                        req_prob,
                        values,
                        "o-",
                        color=get_port_color("group", port_index),
                        label=format_port_label("group", topology_info, port_index, len(port_indices)),
                    )

                ax.set_title(f"group {source_group}")
                ax.set_xlabel("req_prob")
                ax.set_ylabel(ylabel)
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(0.0, y_upper)
                ax.grid(True, alpha=0.3)
                ax.legend(fontsize=9, loc="upper left", framealpha=0.85, title="Port")

            for ax in axes_flat[len(source_groups):]:
                ax.axis("off")

            fig.suptitle(
                f"{topology_title(topology_info)} Group Boundary Traffic per Group\n"
                f"{title_suffix}; part_prob={partition_prob:.1f}\n"
                "Each subplot shows one group; lines average over the group's tiles"
            )
            pdf.savefig(fig)

        if show:
            for fig in figures:
                fig.show()
        else:
            for fig in figures:
                plt.close(fig)

    return output_path


def render_group_aggregate_pdf(
    result_dir,
    entries,
    topology_info,
    show,
    metric,
    output_suffix,
    ylabel,
    title_suffix,
):
    aggregate_entries = build_source_group_aggregate_entries(entries)
    if not aggregate_entries:
        return None

    output_path = make_output_path(result_dir, f"group_{output_suffix}", "group")
    partition_probs = sorted({entry["partition_prob"] for entry in aggregate_entries})
    source_groups = sorted({entry["source_group"] for entry in aggregate_entries})
    all_req_prob = sorted({entry["req_prob"] for entry in aggregate_entries})
    all_values = [entry[metric] for entry in aggregate_entries]

    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0
    y_max = max(all_values) if all_values else 0.0
    y_upper = shared_metric_y_upper(
        result_dir,
        "group",
        metric,
        output_suffix,
        y_max,
        "source_group_aggregate_entries",
        lambda comparable_entries, comparable_topology: build_source_group_aggregate_entries(comparable_entries),
    )
    rows, cols, figsize = subplot_grid(len(source_groups), 2, 7.4, 5.4)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, source_group in zip(axes_flat, source_groups):
        for partition_prob in partition_probs:
            points = [
                entry
                for entry in aggregate_entries
                if entry["source_group"] == source_group
                and entry["partition_prob"] == partition_prob
            ]
            if not points:
                continue

            req_prob = [point["req_prob"] for point in points]
            values = [point[metric] for point in points]
            ax.plot(req_prob, values, "o-", label=format_partition_label(partition_prob))

        ax.set_title(f"group {source_group}")
        ax.set_xlabel("req_prob")
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.85, title="Sweep")

    for ax in axes_flat[len(source_groups):]:
        ax.axis("off")

    if not HIDE_FIGURE_TITLES:
        fig.suptitle(
            f"{topology_title(topology_info)} Group Boundary Traffic per Group\n"
            f"{title_suffix}\n"
            "Each subplot shows one group; lines average over the group's ports and tiles"
        )
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def render_boundary_pdf(result_dir, boundary, entries, topology_info, show, metric, output_suffix, ylabel, title_suffix):
    averaged_entries = build_average_port_entries(entries)
    partition_probs, port_indices, series = build_series(averaged_entries, metric)
    plotted_entries = [entry for entry in averaged_entries if entry[metric] is not None]
    if not plotted_entries:
        return None
    output_path = make_output_path(result_dir, f"{boundary}_{output_suffix}", boundary)
    all_values = [entry[metric] for entry in plotted_entries]
    y_max = max(all_values) if all_values else 0.0
    y_upper = shared_metric_y_upper(
        result_dir,
        boundary,
        metric,
        output_suffix,
        y_max,
        "average_port_entries",
        lambda comparable_entries, comparable_topology: build_average_port_entries(comparable_entries),
    )
    all_req_prob = sorted({entry["req_prob"] for entry in plotted_entries})
    x_min = min(all_req_prob) if all_req_prob else 0.0
    x_max = max(all_req_prob) if all_req_prob else 1.0

    rows, cols, figsize = subplot_grid(len(port_indices), 3, 7.2, 5.2)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        squeeze=False,
        constrained_layout=True,
    )
    axes_flat = axes.flatten()

    for ax, port_index in zip(axes_flat, port_indices):
        for partition_prob in partition_probs:
            points = series.get((port_index, partition_prob), [])
            if not points:
                continue
            req_prob = [point[0] for point in points]
            values = [point[1] for point in points]
            ax.plot(req_prob, values, "o-", label=format_partition_label(partition_prob))

        ax.set_title(format_port_label(boundary, topology_info, port_index, len(port_indices)))
        ax.set_xlabel("req_prob")
        ax.set_ylabel(ylabel)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0.0, y_upper)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, ncol=2, loc="upper left", framealpha=0.85, title="Sweep")

    for ax in axes_flat[len(port_indices) :]:
        ax.axis("off")

    fig.suptitle(
        f"{topology_title(topology_info)} {boundary.capitalize()} Boundary Port Metrics\n"
        f"{title_suffix}\nAverage across all endpoints with the same port index"
    )
    fig.savefig(output_path)

    if show:
        fig.show()
    else:
        plt.close(fig)

    return output_path


def main(argv=None):
    args = parse_args(argv)
    global FORCE_NORMALIZED_Y, HIDE_FIGURE_TITLES, SHARED_Y_UPPERS, FORCED_TOPOLOGY
    FORCE_NORMALIZED_Y = args.force_normalized_y
    HIDE_FIGURE_TITLES = args.hide_figure_titles
    SHARED_Y_UPPERS = parse_shared_y_uppers(args.shared_y_upper)
    FORCED_TOPOLOGY = args.topology
    result_dir = os.path.abspath(args.result_dir)
    topology_info = detect_topology_info(result_dir, args.topology)

    generated = []
    for boundary in get_selected_boundaries(args.boundary):
        clear_boundary_plot_outputs(result_dir, boundary)
        entries = load_boundary_data(result_dir, boundary)
        if not entries:
            continue
        available_directions = get_available_directions(entries)
        average_util_label = average_utilization_label(entries)
        if args.plot_set == "merged-report":
            generated.append(
                render_partition_comparison_pdf(
                    result_dir,
                    boundary,
                    entries,
                    topology_info,
                    args.show,
                    "utilization",
                    "ports_utilization",
                    average_util_label,
                    "Per-port utilization averaged across merged incoming + outgoing traffic",
                )
            )
            generated.append(
                render_partition_comparison_pdf(
                    result_dir,
                    boundary,
                    entries,
                    topology_info,
                    args.show,
                    "backpressure",
                    "ports_backpressure",
                    "stall ratio",
                    "Per-port backpressure across merged incoming + outgoing traffic",
                )
            )
            continue
        if args.plot_set == "directional-utilization":
            for direction in available_directions:
                directional_entries = filter_entries_by_direction(entries, direction)
                generated.append(
                    render_partition_comparison_pdf(
                        result_dir,
                        boundary,
                        directional_entries,
                        topology_info,
                        args.show,
                        "utilization",
                        qualify_output_suffix(direction, available_directions, "ports_utilization"),
                        "accepted tx/cycle",
                        qualify_title(direction, available_directions, "Per-port utilization averaged across all instances"),
                    )
                )
            continue
        # Legacy full-output generation path is intentionally disabled.
        # The reduced merged-report flow above is the supported report mode.
        raise SystemExit("--plot-set full is disabled; use the default merged-report flow.")

    generated = [path for path in generated if path is not None]

    if not generated:
        raise SystemExit("No utilization CSV data found to plot.")

    sync_latest_plots(result_dir)

    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
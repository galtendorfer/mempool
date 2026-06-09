#!/usr/bin/env python3
"""Plot operand source-target offset concentration for one or more runs.

This plotter is a thesis-facing companion to `plot_source_target_classification.py`.
It reads one or more `*_source_target_matrix.csv` files from
`classify_source_targets.py` and builds one combined figure:

* absolute accepted requests and blocked request-cycles, stacked by operand;
* per-operand offset shape, normalized within each operand.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from _plot_output_paths import data_path, figure_path

TRACE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_ANALYSIS_DIR))

from operand_regions import add_classified_operand_provenance_args, validate_classified_operand_provenance


OPERAND_COLORS = {
    "A": "#E68600",
    "B": "#6F63C6",
    "other": "#9E9E9E",
}

OPERAND_TITLES = {
    "A": "Operand A",
    "B": "Operand B",
    "other": "Other",
}

NEIGHBOR_OFFSETS = (-1, 1)

DIRECTION_COLORS = {
    -1: "#3B7EA1",
    1: "#D97904",
}

DIRECTION_TITLES = {
    -1: "target offset -1",
    1: "target offset +1",
}


@dataclass
class OperandStats:
    requests_by_offset: Counter[int] = field(default_factory=Counter)
    stalls_by_offset: Counter[int] = field(default_factory=Counter)
    fires_by_offset: Counter[int] = field(default_factory=Counter)
    high_fanin_by_offset: Counter[int] = field(default_factory=Counter)
    total_requests: int = 0
    total_stalls: int = 0
    total_fires: int = 0
    total_high_fanin: int = 0


@dataclass
class RunStats:
    label: str
    matrix_csv: Path
    operands: dict[str, OperandStats] = field(default_factory=dict)
    source_operands: dict[tuple[int, int], dict[str, OperandStats]] = field(default_factory=dict)
    source_tile_in_group: dict[tuple[int, int], int] = field(default_factory=dict)
    total_requests: int = 0
    total_stalls: int = 0
    total_fires: int = 0
    total_high_fanin: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        nargs="+",
        type=Path,
        help="result dir, path_graph dir, classification dir, or *_source_target_matrix.csv",
    )
    parser.add_argument("--label", action="append", help="run label; repeat once per input path")
    parser.add_argument("--port", type=int, default=0, help="port classifier directory to resolve; default is 0")
    parser.add_argument("--tiles-per-group", type=int, default=16, help="tile slots per group; default is 16")
    parser.add_argument(
        "--offset-mode",
        choices=("raw", "wrapped"),
        default="raw",
        help="plot direct target-source offsets or wrapped shortest offsets",
    )
    parser.add_argument(
        "--operand",
        action="append",
        choices=("A", "B", "other"),
        help="operand column to include; repeat to override the default A/B columns",
    )
    parser.add_argument("--output-dir", type=Path, help="output directory; defaults near the first input")
    parser.add_argument("--prefix", default="port0_operand_offset_sweep", help="output filename prefix")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="figure formats to write",
    )
    add_classified_operand_provenance_args(parser)
    parser.add_argument("--force", action="store_true", help="overwrite existing outputs")
    return parser.parse_args()


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value, 0)


def resolve_matrix_csv(input_path: Path, port: int) -> Path:
    if input_path.is_file():
        if input_path.name.endswith("_source_target_matrix.csv"):
            return input_path
        raise SystemExit(f"Expected *_source_target_matrix.csv, got {input_path}")

    candidates = [
        input_path,
        input_path / f"port{port}_source_target_classification",
        input_path / "analysis" / "path_graph" / f"port{port}_source_target_classification",
    ]
    matches: list[Path] = []
    for candidate in candidates:
        if candidate.is_dir():
            matches.extend(sorted(candidate.glob("*_source_target_matrix.csv")))
    matches = sorted(set(matches))
    if not matches:
        raise SystemExit(f"No *_source_target_matrix.csv found from {input_path}")
    if len(matches) > 1:
        joined = "\n  ".join(str(path) for path in matches)
        raise SystemExit(f"Multiple matrix CSVs found; pass one explicitly:\n  {joined}")
    return matches[0]


def default_output_dir(matrix_csvs: list[Path]) -> Path:
    result_dirs: list[Path] = []
    for matrix_csv in matrix_csvs:
        for candidate in matrix_csv.parents:
            if candidate.name == "path_graph" and candidate.parent.name == "analysis":
                result_dirs.append(candidate.parent.parent)
                break
    if len(result_dirs) > 1:
        try:
            common_parent = Path(os.path.commonpath([str(path) for path in result_dirs]))
        except ValueError:
            common_parent = result_dirs[0].parent
        if common_parent.name != "path_graph":
            return common_parent / "plots" / "port_pressure"
    if result_dirs:
        return result_dirs[0] / "plots" / "port_pressure"
    return matrix_csvs[0].parent / "plots"


def infer_label(path: Path, matrix_csv: Path) -> str:
    for candidate in matrix_csv.parents:
        if candidate.name == "path_graph" and candidate.parent.name == "analysis":
            return candidate.parent.parent.name
    return path.stem.replace("_source_target_matrix", "").replace("_", " ")


def wrapped_signed_offset(source_tile_in_group: int, target_tile_in_group: int, tiles_per_group: int) -> int:
    offset = target_tile_in_group - source_tile_in_group
    while offset <= -(tiles_per_group // 2):
        offset += tiles_per_group
    while offset > tiles_per_group // 2:
        offset -= tiles_per_group
    return offset


def signed_offset(source_tile_in_group: int, target_tile_in_group: int, tiles_per_group: int, offset_mode: str) -> int:
    if offset_mode == "wrapped":
        return wrapped_signed_offset(source_tile_in_group, target_tile_in_group, tiles_per_group)
    return target_tile_in_group - source_tile_in_group


def offset_values(tiles_per_group: int, offset_mode: str) -> list[int]:
    if offset_mode == "wrapped":
        return list(range(-(tiles_per_group // 2) + 1, tiles_per_group // 2 + 1))
    return list(range(-(tiles_per_group - 1), tiles_per_group))


def offset_axis_label(offset_mode: str) -> str:
    if offset_mode == "wrapped":
        return "wrapped signed target offset"
    return "signed target offset (target tile - source tile)"


def add_offset_counts(
    stats: OperandStats,
    offset: int,
    requests: int,
    stalls: int,
    fires: int,
    high_fanin: int,
) -> None:
    stats.requests_by_offset[offset] += requests
    stats.stalls_by_offset[offset] += stalls
    stats.fires_by_offset[offset] += fires
    stats.high_fanin_by_offset[offset] += high_fanin
    stats.total_requests += requests
    stats.total_stalls += stalls
    stats.total_fires += fires
    stats.total_high_fanin += high_fanin


def read_run_stats(
    input_path: Path,
    matrix_csv: Path,
    label: str,
    tiles_per_group: int,
    offset_mode: str,
) -> RunStats:
    run = RunStats(label=label, matrix_csv=matrix_csv)
    required = {
        "source_tile_in_group",
        "target_tile_in_group",
        "operand",
        "requests",
        "stalls",
        "fires",
        "high_fanin_requests",
    }
    with matrix_csv.open(newline="") as file:
        reader = csv.DictReader(file)
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {matrix_csv}: {', '.join(sorted(missing))}")
        for row in reader:
            source_local = parse_int(row.get("source_tile_in_group"), -1)
            target_local = parse_int(row.get("target_tile_in_group"), -1)
            if source_local < 0 or target_local < 0:
                continue
            operand = row.get("operand") or "other"
            requests = parse_int(row.get("requests"))
            stalls = parse_int(row.get("stalls"))
            fires = parse_int(row.get("fires"))
            high_fanin = parse_int(row.get("high_fanin_requests"))
            if requests <= 0:
                continue
            offset = signed_offset(source_local, target_local, tiles_per_group, offset_mode)
            source_group = parse_int(row.get("source_group"), 0)
            source_tile = parse_int(row.get("source_tile"), -1)
            if source_tile < 0:
                source_tile = source_local
            stats = run.operands.setdefault(operand, OperandStats())
            add_offset_counts(stats, offset, requests, stalls, fires, high_fanin)
            source_key = (source_group, source_tile)
            run.source_tile_in_group[source_key] = source_local
            source_stats = run.source_operands.setdefault(source_key, {}).setdefault(operand, OperandStats())
            add_offset_counts(source_stats, offset, requests, stalls, fires, high_fanin)
            run.total_requests += requests
            run.total_stalls += stalls
            run.total_fires += fires
            run.total_high_fanin += high_fanin
    if run.total_requests == 0:
        raise SystemExit(f"No request rows found in {matrix_csv} resolved from {input_path}")
    return run


def percent(value: float, _position: int | None = None) -> str:
    return f"{value:.0f}%"


def short_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if value >= 1000:
        return f"{value / 1000:.0f}k"
    return str(value)


def short_quantity(value: float, _position: int | None = None) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if magnitude >= 10_000:
        return f"{value / 1000:.0f}k"
    if magnitude >= 1000:
        return f"{value / 1000:.1f}k"
    if magnitude >= 100 or abs(value - round(value)) < 0.05:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def nice_percent_limit(value: float) -> float:
    if value <= 10:
        return 10
    if value <= 25:
        return 25
    if value <= 60:
        return 60
    return 100


def metric_counter(stats: OperandStats, metric: str) -> Counter[int]:
    if metric == "accepted":
        return stats.fires_by_offset
    if metric == "blocked":
        return stats.stalls_by_offset
    if metric == "valid":
        return stats.requests_by_offset
    raise ValueError(f"Unknown metric: {metric}")


def operand_metric_total(stats: OperandStats, metric: str) -> int:
    if metric == "accepted":
        return stats.total_fires
    if metric == "blocked":
        return stats.total_stalls
    if metric == "valid":
        return stats.total_requests
    raise ValueError(f"Unknown metric: {metric}")


def run_metric_total(run: RunStats, metric: str) -> int:
    if metric == "accepted":
        return run.total_fires
    if metric == "blocked":
        return run.total_stalls
    if metric == "valid":
        return run.total_requests
    raise ValueError(f"Unknown metric: {metric}")


def operand_offset_share(stats: OperandStats, offset: int, metric: str = "accepted") -> float:
    total = operand_metric_total(stats, metric)
    if total == 0:
        return 0.0
    return 100.0 * metric_counter(stats, metric)[offset] / total


def operand_neighbor_share(stats: OperandStats, metric: str = "accepted") -> float:
    total = operand_metric_total(stats, metric)
    if total == 0:
        return 0.0
    values = metric_counter(stats, metric)
    requests = values[-1] + values[1]
    return 100.0 * requests / total


def source_tile_operand_stats(run: RunStats, operand: str, metric: str) -> list[OperandStats]:
    stats: list[OperandStats] = []
    for source_stats in run.source_operands.values():
        operand_stats = source_stats.get(operand)
        if operand_stats and operand_metric_total(operand_stats, metric) > 0:
            stats.append(operand_stats)
    return stats


def source_tile_count(run: RunStats) -> int:
    return len(run.source_operands)


def source_tile_keys(run: RunStats) -> list[tuple[int, int]]:
    return sorted(
        run.source_operands,
        key=lambda key: (key[0], run.source_tile_in_group.get(key, key[1]), key[1]),
    )


def source_tile_in_group_values(run: RunStats) -> list[int]:
    return sorted(set(run.source_tile_in_group.values()))


def source_groups_for_tile_in_group(run: RunStats, source_local: int) -> list[int]:
    return sorted(
        {
            source_group
            for source_group, source_tile in run.source_operands
            if run.source_tile_in_group.get((source_group, source_tile)) == source_local
        }
    )


def source_tile_axis_label(_run: RunStats) -> str:
    return "source -> target tile in group"


def local_source_direction_value(
    run: RunStats,
    source_local: int,
    operand: str,
    offset: int,
    metric: str,
) -> int:
    return sum(
        source_direction_value(run, key, operand, offset, metric)
        for key in source_tile_keys(run)
        if run.source_tile_in_group.get(key) == source_local
    )


def local_source_direction_rate(run: RunStats, source_local: int, operand: str, offset: int) -> float:
    valid = local_source_direction_value(run, source_local, operand, offset, "valid")
    if valid == 0:
        return 0.0
    blocked = local_source_direction_value(run, source_local, operand, offset, "blocked")
    return 100.0 * blocked / valid


def source_tile_pair_label(run: RunStats, source_local: int, tiles_per_group: int, operand: str = "A") -> str:
    active_offsets = [
        offset
        for offset in NEIGHBOR_OFFSETS
        if local_source_direction_value(run, source_local, operand, offset, "valid") > 0
    ]
    if len(active_offsets) == 1:
        target_local = (source_local + active_offsets[0]) % tiles_per_group
        return f"{source_local}->{target_local}"
    return f"{source_local}"


def source_direction_value(
    run: RunStats,
    key: tuple[int, int],
    operand: str,
    offset: int,
    metric: str,
) -> int:
    stats = run.source_operands.get(key, {}).get(operand, OperandStats())
    return metric_counter(stats, metric)[offset]


def source_tile_mean_offset_count(run: RunStats, operand: str, offset: int, metric: str = "accepted") -> float:
    count = source_tile_count(run)
    if count == 0:
        return 0.0
    stats = run.operands.get(operand)
    if not stats:
        return 0.0
    return metric_counter(stats, metric)[offset] / count


def source_tile_mean_neighbor_count(run: RunStats, operand: str, metric: str = "accepted") -> float:
    count = source_tile_count(run)
    if count == 0:
        return 0.0
    stats = run.operands.get(operand)
    if not stats:
        return 0.0
    values = metric_counter(stats, metric)
    return (values[-1] + values[1]) / count


def source_tile_mean_operand_total(run: RunStats, operand: str, metric: str = "accepted") -> float:
    count = source_tile_count(run)
    if count == 0:
        return 0.0
    stats = run.operands.get(operand)
    if not stats:
        return 0.0
    return operand_metric_total(stats, metric) / count


def source_tile_mean_offset_share(run: RunStats, operand: str, offset: int, metric: str = "accepted") -> float:
    source_stats = source_tile_operand_stats(run, operand, metric)
    if not source_stats:
        return 0.0
    shares = [
        100.0 * metric_counter(stats, metric)[offset] / operand_metric_total(stats, metric)
        for stats in source_stats
    ]
    return sum(shares) / len(shares)


def source_tile_mean_neighbor_share(run: RunStats, operand: str, metric: str = "accepted") -> float:
    source_stats = source_tile_operand_stats(run, operand, metric)
    if not source_stats:
        return 0.0
    shares = [operand_neighbor_share(stats, metric) for stats in source_stats]
    return sum(shares) / len(shares)


def choose_offsets(tiles_per_group: int, offset_mode: str, runs: list[RunStats], operands: list[str]) -> list[int]:
    offsets = set(offset_values(tiles_per_group, offset_mode))
    for run in runs:
        for operand in operands:
            stats = run.operands.get(operand)
            if stats:
                offsets.update(stats.requests_by_offset.keys())
    return sorted(offsets)


def x_ticks_for_offsets(offsets: list[int]) -> list[int]:
    return list(offsets)


def write_summary_csv(path: Path, runs: list[RunStats], operands: list[str], offsets: list[int], force: bool) -> None:
    ensure_outputs([path], force)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "matrix_csv",
        "operand",
        "offset",
        "offset_valid_observations",
        "offset_valid_share",
        "offset_accepted_requests",
        "offset_accepted_share",
        "offset_accepted_per_source_tile",
        "offset_blocked_request_cycles",
        "offset_blocked_share",
        "offset_blocked_per_valid",
        "offset_high_fanin_requests",
        "operand_valid_observations",
        "operand_valid_share_of_port",
        "operand_accepted_requests",
        "operand_accepted_share_of_port",
        "operand_accepted_per_source_tile",
        "operand_blocked_request_cycles",
        "operand_blocked_share_of_port",
        "operand_blocked_per_valid",
        "operand_pm1_accepted_share",
        "run_valid_observations",
        "run_accepted_requests",
        "run_source_tiles",
        "run_accepted_per_source_tile",
        "run_blocked_request_cycles",
        "run_blocked_per_valid",
        "run_high_fanin_requests",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            source_tiles = source_tile_count(run)
            run_blocked_per_valid = run.total_stalls / run.total_requests if run.total_requests else 0.0
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                operand_valid_share = stats.total_requests / run.total_requests if run.total_requests else 0.0
                operand_accepted_share = stats.total_fires / run.total_fires if run.total_fires else 0.0
                operand_blocked_share = stats.total_stalls / run.total_stalls if run.total_stalls else 0.0
                operand_blocked_per_valid = stats.total_stalls / stats.total_requests if stats.total_requests else 0.0
                pm1_accepted_share = operand_neighbor_share(stats, "accepted") / 100.0
                for offset in offsets:
                    offset_valid = stats.requests_by_offset[offset]
                    offset_accepted = stats.fires_by_offset[offset]
                    offset_blocked = stats.stalls_by_offset[offset]
                    writer.writerow(
                        {
                            "run": run.label,
                            "matrix_csv": run.matrix_csv,
                            "operand": operand,
                            "offset": offset,
                            "offset_valid_observations": offset_valid,
                            "offset_valid_share": f"{offset_valid / stats.total_requests:.6f}" if stats.total_requests else "0.000000",
                            "offset_accepted_requests": offset_accepted,
                            "offset_accepted_share": f"{offset_accepted / stats.total_fires:.6f}" if stats.total_fires else "0.000000",
                            "offset_accepted_per_source_tile": f"{offset_accepted / source_tiles:.6f}" if source_tiles else "0.000000",
                            "offset_blocked_request_cycles": offset_blocked,
                            "offset_blocked_share": f"{offset_blocked / stats.total_stalls:.6f}" if stats.total_stalls else "0.000000",
                            "offset_blocked_per_valid": f"{offset_blocked / offset_valid:.6f}" if offset_valid else "0.000000",
                            "offset_high_fanin_requests": stats.high_fanin_by_offset[offset],
                            "operand_valid_observations": stats.total_requests,
                            "operand_valid_share_of_port": f"{operand_valid_share:.6f}",
                            "operand_accepted_requests": stats.total_fires,
                            "operand_accepted_share_of_port": f"{operand_accepted_share:.6f}",
                            "operand_accepted_per_source_tile": f"{stats.total_fires / source_tiles:.6f}" if source_tiles else "0.000000",
                            "operand_blocked_request_cycles": stats.total_stalls,
                            "operand_blocked_share_of_port": f"{operand_blocked_share:.6f}",
                            "operand_blocked_per_valid": f"{operand_blocked_per_valid:.6f}",
                            "operand_pm1_accepted_share": f"{pm1_accepted_share:.6f}",
                            "run_valid_observations": run.total_requests,
                            "run_accepted_requests": run.total_fires,
                            "run_source_tiles": source_tiles,
                            "run_accepted_per_source_tile": f"{run.total_fires / source_tiles:.6f}" if source_tiles else "0.000000",
                            "run_blocked_request_cycles": run.total_stalls,
                            "run_blocked_per_valid": f"{run_blocked_per_valid:.6f}",
                            "run_high_fanin_requests": run.total_high_fanin,
                        }
                    )


def write_a_neighbor_source_csv(path: Path, runs: list[RunStats], tiles_per_group: int, force: bool) -> None:
    ensure_outputs([path], force)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run",
        "matrix_csv",
        "source_groups",
        "source_group_count",
        "source_tile_in_group",
        "source_target_pair_in_group",
        "target_offset",
        "target_tile_in_group",
        "offset_valid_observations",
        "offset_accepted_requests",
        "offset_blocked_request_cycles",
        "offset_blocked_per_valid",
    ]
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            for source_local in source_tile_in_group_values(run):
                source_groups = source_groups_for_tile_in_group(run, source_local)
                for offset in NEIGHBOR_OFFSETS:
                    target_local = (source_local + offset) % tiles_per_group
                    valid = local_source_direction_value(run, source_local, "A", offset, "valid")
                    accepted = local_source_direction_value(run, source_local, "A", offset, "accepted")
                    blocked = local_source_direction_value(run, source_local, "A", offset, "blocked")
                    writer.writerow(
                        {
                            "run": run.label,
                            "matrix_csv": run.matrix_csv,
                            "source_groups": " ".join(str(source_group) for source_group in source_groups),
                            "source_group_count": len(source_groups),
                            "source_tile_in_group": source_local,
                            "source_target_pair_in_group": f"{source_local}->{target_local}",
                            "target_offset": offset,
                            "target_tile_in_group": target_local,
                            "offset_valid_observations": valid,
                            "offset_accepted_requests": accepted,
                            "offset_blocked_request_cycles": blocked,
                            "offset_blocked_per_valid": f"{blocked / valid:.6f}" if valid else "0.000000",
                        }
                    )


def ensure_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise SystemExit(f"Output exists: {path} (use --force to overwrite)")


def plot_joined_operand_offset_sweep(
    runs: list[RunStats],
    operands: list[str],
    offsets: list[int],
    offset_mode: str,
) -> plt.Figure:
    metrics = (
        ("accepted", "Accepted requests", "accepted request count"),
        ("blocked", "Blocked request-cycles", "blocked request-cycle count"),
    )
    figure_width = 13.5
    figure_height = max(7.2, 2.05 * len(runs) + 1.9)
    fig, axes = plt.subplots(len(runs), len(metrics), figsize=(figure_width, figure_height), sharex=True, squeeze=False)

    y_limits: dict[str, float] = {}
    for metric, _title, _ylabel in metrics:
        max_total = 0
        for run in runs:
            for offset in offsets:
                offset_total = 0
                for operand in operands:
                    stats = run.operands.get(operand)
                    if stats:
                        offset_total += metric_counter(stats, metric)[offset]
                max_total = max(max_total, offset_total)
        y_limits[metric] = max_total * 1.18 if max_total else 1.0

    fig.suptitle("Port 0 accepted and blocked traffic by operand offset", fontsize=22, y=0.985)
    fig.text(
        0.5,
        0.955,
        "Accepted requests are fire handshakes; blocked traffic counts stalled request-cycles.",
        ha="center",
        va="center",
        fontsize=12,
        color="0.35",
    )

    x_tick_values = x_ticks_for_offsets(offsets)
    for row_index, run in enumerate(runs):
        for col_index, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row_index][col_index]
            bottom = [0] * len(offsets)
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                values = [metric_counter(stats, metric)[offset] for offset in offsets]
                ax.bar(
                    offsets,
                    values,
                    width=0.82,
                    bottom=bottom,
                    color=OPERAND_COLORS.get(operand, "#9E9E9E"),
                    alpha=0.92,
                    linewidth=0,
                    label=OPERAND_TITLES.get(operand, operand),
                )
                bottom = [base + value for base, value in zip(bottom, values)]

            metric_total = run_metric_total(run, metric)
            valid_total = run.total_requests
            blocked_total = run.total_stalls
            pm1_total = 0
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                counts = metric_counter(stats, metric)
                pm1_total += counts[-1] + counts[1]
            pm1_share = 100.0 * pm1_total / metric_total if metric_total else 0.0
            operand_parts = []
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                operand_total = operand_metric_total(stats, metric)
                operand_share = 100.0 * operand_total / metric_total if metric_total else 0.0
                operand_parts.append(f"{operand} {operand_share:.0f}%")
            if metric == "accepted":
                text = (
                    f"{short_count(metric_total)} accepted ({', '.join(operand_parts)})\n"
                    f"+/-1 {pm1_share:.1f}%   valid obs {short_count(valid_total)}"
                )
            else:
                blocked_per_valid = 100.0 * blocked_total / valid_total if valid_total else 0.0
                text = (
                    f"{short_count(metric_total)} blocked cycles ({', '.join(operand_parts)})\n"
                    f"+/-1 {pm1_share:.1f}%   blocked/valid {blocked_per_valid:.1f}%"
                )

            ax.text(
                0.02,
                0.91,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                color="0.12",
                bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0, "alpha": 0.88},
            )
            if col_index == 0:
                ax.text(
                    -0.16,
                    0.5,
                    run.label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    color="0.08",
                )
            if row_index == 0:
                ax.set_title(title, fontsize=14, pad=10)
            if row_index == len(runs) - 1:
                ax.set_xlabel(offset_axis_label(offset_mode), fontsize=10)
            ax.set_ylabel(ylabel if row_index == len(runs) // 2 else "")
            ax.set_ylim(0, y_limits[metric])
            ax.set_xticks(x_tick_values)
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="y", alpha=0.22)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if metric == "accepted" and row_index == 0 and col_index == 0:
                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.915, 0.955), frameon=False)

    fig.subplots_adjust(left=0.14, right=0.91, bottom=0.09, top=0.91, hspace=0.38, wspace=0.22)
    return fig


def plot_source_tile_mean_operand_sweep(
    runs: list[RunStats],
    operands: list[str],
    offsets: list[int],
    offset_mode: str,
) -> plt.Figure:
    figure_width = 13.5
    figure_height = max(7.0, 1.85 * len(runs) + 1.8)
    fig, axes = plt.subplots(len(runs), 1, figsize=(figure_width, figure_height), sharex=True, squeeze=False)

    max_value = 0.0
    for run in runs:
        for operand in operands:
            for offset in offsets:
                max_value = max(max_value, source_tile_mean_offset_count(run, operand, offset, "accepted"))
    y_limit = max_value * 1.18 if max_value else 1.0

    fig.suptitle("Port 0 accepted requests per source tile by operand offset", fontsize=20, y=0.99)
    fig.text(
        0.5,
        0.945,
        "Bars divide accepted fire handshakes in each offset bin by the number of observed source tiles.",
        ha="center",
        va="center",
        fontsize=12,
        color="0.35",
    )

    x_tick_values = x_ticks_for_offsets(offsets)
    bar_width = min(0.38, 0.78 / max(len(operands), 1))
    bar_offsets = [(index - (len(operands) - 1) / 2.0) * bar_width for index in range(len(operands))]

    for row_index, run in enumerate(runs):
        ax = axes[row_index][0]
        for operand_index, operand in enumerate(operands):
            values = [source_tile_mean_offset_count(run, operand, offset, "accepted") for offset in offsets]
            shifted_offsets = [offset + bar_offsets[operand_index] for offset in offsets]
            ax.bar(
                shifted_offsets,
                values,
                width=bar_width * 0.92,
                color=OPERAND_COLORS.get(operand, "#9E9E9E"),
                alpha=0.9,
                linewidth=0,
                label=OPERAND_TITLES.get(operand, operand),
            )

        ax.text(
            -0.09,
            0.5,
            run.label,
            transform=ax.transAxes,
            ha="right",
            va="center",
            fontsize=12,
            color="0.08",
        )
        source_tiles = source_tile_count(run)
        summary_parts = []
        for operand in operands:
            total_per_tile = source_tile_mean_operand_total(run, operand, "accepted")
            pm1_per_tile = source_tile_mean_neighbor_count(run, operand, "accepted")
            summary_parts.append(
                f"{operand}: {short_quantity(total_per_tile)}/tile accepted, +/-1 {short_quantity(pm1_per_tile)}/tile"
            )
        ax.text(
            0.02,
            0.91,
            f"{source_tiles} source tiles | " + " | ".join(summary_parts),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.2,
            color="0.12",
            bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0, "alpha": 0.88},
        )
        if row_index == len(runs) - 1:
            ax.set_xlabel(offset_axis_label(offset_mode), fontsize=10)
        if row_index == len(runs) // 2:
            ax.set_ylabel("accepted requests per source tile", fontsize=10)
        ax.set_ylim(0, y_limit)
        ax.set_xticks(x_tick_values)
        ax.tick_params(axis="x", labelsize=7, labelrotation=90)
        ax.yaxis.set_major_formatter(FuncFormatter(short_quantity))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row_index == 0:
            ax.legend(loc="upper right", frameon=False)

    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.10, top=0.88, hspace=0.36)
    return fig


def plot_operand_offset_sweep(
    runs: list[RunStats],
    operands: list[str],
    offsets: list[int],
    offset_mode: str,
) -> plt.Figure:
    figure_width = max(12.0, 5.8 * len(operands))
    figure_height = max(7.0, 2.0 * len(runs) + 1.8)
    fig, axes = plt.subplots(
        len(runs),
        len(operands),
        figsize=(figure_width, figure_height),
        sharex=True,
        squeeze=False,
    )

    y_limits: dict[str, float] = {}
    for operand in operands:
        max_share = 0.0
        for run in runs:
            stats = run.operands.get(operand)
            if stats:
                max_share = max(max_share, *(operand_offset_share(stats, offset) for offset in offsets))
        y_limits[operand] = nice_percent_limit(max_share * 1.12)

    fig.suptitle("Port 0 per-operand accepted request offsets", fontsize=22, y=0.985)
    fig.text(
        0.5,
        0.955,
        "Bars are normalized within each operand using accepted fire handshakes.",
        ha="center",
        va="center",
        fontsize=12,
        color="0.35",
    )

    x_tick_values = x_ticks_for_offsets(offsets)
    for row_index, run in enumerate(runs):
        run_stall = 100.0 * run.total_stalls / run.total_requests if run.total_requests else 0.0
        for col_index, operand in enumerate(operands):
            ax = axes[row_index][col_index]
            stats = run.operands.get(operand, OperandStats())
            color = OPERAND_COLORS.get(operand, "#9E9E9E")
            shares = [operand_offset_share(stats, offset) for offset in offsets]
            bars = ax.bar(offsets, shares, width=0.82, color=color, alpha=0.9, linewidth=0)
            ax.axhline(0, color="0.2", linewidth=0.9)
            ax.set_ylim(0, y_limits[operand])
            ax.grid(axis="y", alpha=0.22)
            ax.yaxis.set_major_formatter(FuncFormatter(percent))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            if row_index == 0:
                ax.set_title(OPERAND_TITLES.get(operand, operand), fontsize=14, pad=10)
            if col_index == 0:
                ax.text(
                    -0.16,
                    0.5,
                    run.label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=12,
                    color="0.08",
                )
            if row_index == len(runs) - 1:
                ax.set_xlabel(offset_axis_label(offset_mode), fontsize=10)
            ax.set_xticks(x_tick_values)
            ax.tick_params(axis="x", labelsize=7, labelrotation=90)

            operand_share = 100.0 * stats.total_fires / run.total_fires if run.total_fires else 0.0
            operand_stall = 100.0 * stats.total_stalls / stats.total_requests if stats.total_requests else 0.0
            pm1_share = operand_neighbor_share(stats, "accepted")
            text = (
                f"{short_count(stats.total_fires)} accepted\n"
                f"{operand_share:.1f}% of port\n"
                f"+/-1 {pm1_share:.1f}% | blk/valid {operand_stall:.1f}%"
            )
            ax.text(
                0.02,
                0.91,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.3,
                color="0.12",
                bbox={"facecolor": "white", "edgecolor": "0.82", "pad": 3.0, "alpha": 0.88},
            )
            if col_index == len(operands) - 1:
                ax.text(
                    1.01,
                    0.5,
                    f"port blocked\n{run_stall:.1f}%",
                    transform=ax.transAxes,
                    ha="left",
                    va="center",
                    fontsize=9.2,
                    color="0.35",
                )

    fig.subplots_adjust(left=0.14, right=0.91, bottom=0.09, top=0.91, hspace=0.38, wspace=0.18)
    return fig


def axis_summary(ax: plt.Axes, text: str, fontsize: float = 7.8) -> None:
    ax.text(
        0.02,
        0.94,
        text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=fontsize,
        linespacing=1.12,
        color="0.12",
        bbox={"facecolor": "white", "edgecolor": "0.84", "pad": 2.6, "alpha": 0.88},
    )


def operand_share_summary(run: RunStats, operands: list[str], metric: str) -> str:
    metric_total = run_metric_total(run, metric)
    parts = []
    for operand in operands:
        stats = run.operands.get(operand, OperandStats())
        operand_total = operand_metric_total(stats, metric)
        operand_share = 100.0 * operand_total / metric_total if metric_total else 0.0
        parts.append(f"{operand} {operand_share:.0f}%")
    return ", ".join(parts)


def stacked_metric_offset_max(runs: list[RunStats], operands: list[str], offsets: list[int], metric: str) -> int:
    max_total = 0
    for run in runs:
        for offset in offsets:
            offset_total = 0
            for operand in operands:
                stats = run.operands.get(operand)
                if stats:
                    offset_total += metric_counter(stats, metric)[offset]
            max_total = max(max_total, offset_total)
    return max_total


def source_tile_offset_max(runs: list[RunStats], operands: list[str], offsets: list[int]) -> float:
    max_value = 0.0
    for run in runs:
        for operand in operands:
            for offset in offsets:
                max_value = max(max_value, source_tile_mean_offset_count(run, operand, offset, "accepted"))
    return max_value


def operand_share_offset_max(runs: list[RunStats], operand: str, offsets: list[int]) -> float:
    max_share = 0.0
    for run in runs:
        stats = run.operands.get(operand)
        if stats:
            max_share = max(max_share, *(operand_offset_share(stats, offset) for offset in offsets))
    return max_share


def style_offset_axis(
    ax: plt.Axes,
    offsets: list[int],
    show_x_labels: bool,
    formatter: FuncFormatter | None = None,
) -> None:
    ax.set_xticks(x_ticks_for_offsets(offsets))
    ax.tick_params(axis="x", labelsize=6.2, labelrotation=90, labelbottom=show_x_labels)
    ax.tick_params(axis="y", labelsize=8.5)
    if formatter:
        ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(axis="y", alpha=0.20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined_operand_offsets(
    runs: list[RunStats],
    operands: list[str],
    offsets: list[int],
    offset_mode: str,
) -> plt.Figure:
    section_columns = max(2, len(operands))
    section_rows = len(runs)
    total_rows = section_rows * 2
    figure_width = max(13.5, 5.8 * section_columns)
    figure_height = max(8.6, 2.35 * total_rows + 2.0)
    fig = plt.figure(figsize=(figure_width, figure_height))
    grid = fig.add_gridspec(total_rows, section_columns)

    y_limits: dict[tuple[str, str | None], float] = {}
    accepted_max = stacked_metric_offset_max(runs, operands, offsets, "accepted")
    blocked_max = stacked_metric_offset_max(runs, operands, offsets, "blocked")
    y_limits[("accepted", None)] = accepted_max * 1.18 if accepted_max else 1.0
    y_limits[("blocked", None)] = blocked_max * 1.22 if blocked_max else 1.0
    for operand in operands:
        y_limits[("shape", operand)] = nice_percent_limit(operand_share_offset_max(runs, operand, offsets) * 1.12)

    fig.suptitle("Port 0 operand offset traffic", fontsize=18, y=0.982)
    fig.text(
        0.5,
        0.945,
        "Counts use accepted fire handshakes; blocked counts stalled request-cycles; shape panels normalize within each operand.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="0.35",
    )

    count_formatter = FuncFormatter(short_quantity)
    percent_formatter = FuncFormatter(percent)

    for row_index, run in enumerate(runs):
        pressure_row = row_index
        shape_row = section_rows + row_index

        accepted_span = slice(0, max(1, section_columns // 2))
        blocked_span = slice(max(1, section_columns // 2), section_columns)
        accepted_ax = fig.add_subplot(grid[pressure_row, accepted_span])
        blocked_ax = fig.add_subplot(grid[pressure_row, blocked_span], sharex=accepted_ax)
        pressure_axes = [("accepted", accepted_ax), ("blocked", blocked_ax)]
        for metric, ax in pressure_axes:
            bottom = [0] * len(offsets)
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                values = [metric_counter(stats, metric)[offset] for offset in offsets]
                ax.bar(
                    offsets,
                    values,
                    width=0.82,
                    bottom=bottom,
                    color=OPERAND_COLORS.get(operand, "#9E9E9E"),
                    alpha=0.92,
                    linewidth=0,
                    label=OPERAND_TITLES.get(operand, operand),
                )
                bottom = [base + value for base, value in zip(bottom, values)]

            metric_total = run_metric_total(run, metric)
            counts_pm1 = 0
            for operand in operands:
                stats = run.operands.get(operand, OperandStats())
                counts = metric_counter(stats, metric)
                counts_pm1 += counts[-1] + counts[1]
            pm1_share = 100.0 * counts_pm1 / metric_total if metric_total else 0.0
            if metric == "accepted":
                summary = (
                    f"{short_count(metric_total)} accepted\n"
                    f"{operand_share_summary(run, operands, metric)}\n"
                    f"+/-1 {pm1_share:.1f}%"
                )
            else:
                blocked_per_valid = 100.0 * run.total_stalls / run.total_requests if run.total_requests else 0.0
                summary = (
                    f"{short_count(metric_total)} blocked\n"
                    f"{operand_share_summary(run, operands, metric)}\n"
                    f"blocked/valid {blocked_per_valid:.1f}%"
                )
            axis_summary(ax, summary, fontsize=8.5)
            ax.set_ylim(0, y_limits[(metric, None)])
            style_offset_axis(ax, offsets, True, count_formatter)
            ax.set_xlabel(offset_axis_label(offset_mode), fontsize=8.5)
            if metric == "accepted":
                ax.set_ylabel("accepted requests", fontsize=9)
            else:
                ax.set_ylabel("blocked request-cycles", fontsize=9)
            ax.legend(loc="upper right", frameon=False, fontsize=8.2, ncol=max(1, len(operands)))

        if row_index == 0:
            accepted_ax.set_title("Absolute accepted requests", fontsize=12, pad=8)
            blocked_ax.set_title("Blocked request-cycles", fontsize=12, pad=8)

        shape_axes: list[plt.Axes] = []
        for operand_index, operand in enumerate(operands):
            shape_ax = fig.add_subplot(grid[shape_row, operand_index], sharex=accepted_ax)
            shape_axes.append(shape_ax)
            stats = run.operands.get(operand, OperandStats())
            shares = [operand_offset_share(stats, offset) for offset in offsets]
            shape_ax.bar(
                offsets,
                shares,
                width=0.82,
                color=OPERAND_COLORS.get(operand, "#9E9E9E"),
                alpha=0.9,
                linewidth=0,
                label=OPERAND_TITLES.get(operand, operand),
            )
            operand_share = 100.0 * stats.total_fires / run.total_fires if run.total_fires else 0.0
            operand_stall = 100.0 * stats.total_stalls / stats.total_requests if stats.total_requests else 0.0
            pm1_share = operand_neighbor_share(stats, "accepted")
            summary = (
                f"{short_count(stats.total_fires)} accepted\n"
                f"{operand_share:.1f}% of port\n"
                f"+/-1 {pm1_share:.1f}%, blk/valid {operand_stall:.1f}%"
            )
            axis_summary(shape_ax, summary, fontsize=8.5)
            shape_ax.set_ylim(0, y_limits[("shape", operand)])
            style_offset_axis(shape_ax, offsets, True, percent_formatter)
            shape_ax.set_xlabel(offset_axis_label(offset_mode), fontsize=8.5)
            shape_ax.set_ylabel("share within operand", fontsize=9)
            shape_ax.legend(loc="upper right", frameon=False, fontsize=8.2)
            if row_index == 0:
                shape_ax.set_title(f"{OPERAND_TITLES.get(operand, operand)} normalized offset shape", fontsize=12, pad=8)

        for unused_col in range(len(operands), section_columns):
            empty_ax = fig.add_subplot(grid[shape_row, unused_col])
            empty_ax.axis("off")

        accepted_ax.text(
            -0.17,
            0.5,
            run.label,
            transform=accepted_ax.transAxes,
            ha="right",
            va="center",
            fontsize=10.5,
            color="0.08",
        )
        shape_axes[0].text(
            -0.17,
            0.5,
            run.label,
            transform=shape_axes[0].transAxes,
            ha="right",
            va="center",
            fontsize=10.5,
            color="0.08",
        )

    fig.subplots_adjust(left=0.13, right=0.975, bottom=0.085, top=0.875, hspace=0.98, wspace=0.22)
    return fig


def direction_totals(run: RunStats, operand: str, metric: str) -> dict[int, int]:
    return {
        offset: sum(source_direction_value(run, key, operand, offset, metric) for key in source_tile_keys(run))
        for offset in NEIGHBOR_OFFSETS
    }


def direction_blocked_rates(run: RunStats, operand: str) -> dict[int, float]:
    totals_valid = direction_totals(run, operand, "valid")
    totals_blocked = direction_totals(run, operand, "blocked")
    return {
        offset: 100.0 * totals_blocked[offset] / totals_valid[offset] if totals_valid[offset] else 0.0
        for offset in NEIGHBOR_OFFSETS
    }


def direction_ratio_text(positive: int, negative: int) -> str:
    if positive == 0 and negative == 0:
        return "no blocked A neighbor cycles"
    if negative == 0:
        return "+1/-1 blocked ratio n/a"
    return f"+1/-1 blocked ratio {positive / negative:.1f}x"


def a_neighbor_axis_summary(run: RunStats, metric: str) -> str:
    if metric == "blocked_rate":
        rates = direction_blocked_rates(run, "A")
        delta = rates[1] - rates[-1]
        return f"+1 {rates[1]:.2f}%, -1 {rates[-1]:.2f}%\ndelta {delta:+.2f} pp"
    totals = direction_totals(run, "A", metric)
    if metric == "accepted":
        delta = totals[1] - totals[-1]
        return f"+1 {short_count(totals[1])}, -1 {short_count(totals[-1])}\ndelta {delta:+d} requests"
    return f"+1 {short_count(totals[1])}, -1 {short_count(totals[-1])}\n{direction_ratio_text(totals[1], totals[-1])}"


def plot_a_neighbor_source_tiles(runs: list[RunStats], tiles_per_group: int) -> plt.Figure:
    metrics = (
        ("accepted", "Accepted A neighbor requests", "accepted requests"),
        ("blocked", "Blocked A neighbor request-cycles", "blocked request-cycles"),
        ("blocked_rate", "A neighbor blocked/valid rate", "blocked / valid"),
    )
    figure_width = 15.0
    figure_height = max(6.2, 2.55 * len(runs) + 2.0)
    fig, axes = plt.subplots(len(runs), len(metrics), figsize=(figure_width, figure_height), squeeze=False)

    fig.suptitle("Operand A neighbor traffic by source-target tile pair", fontsize=18, y=0.985)
    fig.text(
        0.5,
        0.948,
        "Offset +1 and -1 are plotted per source tile; accepted requests show demand, blocked cycles show backpressure.",
        ha="center",
        va="center",
        fontsize=10.2,
        color="0.35",
    )

    count_formatter = FuncFormatter(short_quantity)
    percent_formatter = FuncFormatter(percent)
    bar_width = 0.34
    bar_offsets = {
        -1: -bar_width / 2.0,
        1: bar_width / 2.0,
    }

    for row_index, run in enumerate(runs):
        source_locals = source_tile_in_group_values(run)
        x_positions = list(range(len(source_locals)))
        x_labels = [source_tile_pair_label(run, source_local, tiles_per_group) for source_local in source_locals]

        for col_index, (metric, title, ylabel) in enumerate(metrics):
            ax = axes[row_index][col_index]
            max_value = 0.0
            for offset in NEIGHBOR_OFFSETS:
                if metric == "blocked_rate":
                    values = [local_source_direction_rate(run, source_local, "A", offset) for source_local in source_locals]
                else:
                    values = [
                        local_source_direction_value(run, source_local, "A", offset, metric)
                        for source_local in source_locals
                    ]
                max_value = max(max_value, *values, 0.0)
                shifted_x = [position + bar_offsets[offset] for position in x_positions]
                ax.bar(
                    shifted_x,
                    values,
                    width=bar_width,
                    color=DIRECTION_COLORS[offset],
                    alpha=0.9,
                    linewidth=0,
                    label=DIRECTION_TITLES[offset],
                )

            if metric == "blocked_rate":
                ax.set_ylim(0, nice_percent_limit(max_value * 1.15))
                formatter = percent_formatter
            else:
                ax.set_ylim(0, max_value * 1.20 if max_value else 1.0)
                formatter = count_formatter

            axis_summary(ax, a_neighbor_axis_summary(run, metric), fontsize=8.2)
            if row_index == 0:
                ax.set_title(title, fontsize=11.5, pad=8)
            if col_index == 0:
                ax.text(
                    -0.17,
                    0.5,
                    run.label,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=10.5,
                    color="0.08",
                )
            ax.set_xlabel(source_tile_axis_label(run), fontsize=8.5)
            ax.set_ylabel(ylabel, fontsize=9)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(x_labels, rotation=90, fontsize=6.8)
            ax.tick_params(axis="y", labelsize=8.5)
            ax.yaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis="y", alpha=0.20)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper right", frameon=False, fontsize=8.0)

    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.105, top=0.885, hspace=0.68, wspace=0.25)
    return fig


def save_figure(fig: plt.Figure, output_dir: Path, prefix: str, formats: list[str], force: bool) -> list[Path]:
    paths = [figure_path(output_dir / prefix, fmt) for fmt in formats]
    ensure_outputs(paths, force)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    matrix_csvs = [resolve_matrix_csv(path, args.port) for path in args.input_path]
    validate_classified_operand_provenance(matrix_csvs, args.allow_legacy_route_operands)
    if args.label and len(args.label) != len(matrix_csvs):
        raise SystemExit("Pass exactly one --label per input path")
    labels = args.label or [infer_label(path, matrix_csv) for path, matrix_csv in zip(args.input_path, matrix_csvs)]
    operands = args.operand or ["A", "B"]
    runs = [
        read_run_stats(path, matrix_csv, label, args.tiles_per_group, args.offset_mode)
        for path, matrix_csv, label in zip(args.input_path, matrix_csvs, labels)
    ]
    offsets = choose_offsets(args.tiles_per_group, args.offset_mode, runs, operands)
    output_dir = args.output_dir or default_output_dir(matrix_csvs)

    summary_path = data_path(output_dir, f"{args.prefix}.csv")
    a_neighbor_prefix = f"{args.prefix}_a_neighbor_source_tiles"
    a_neighbor_summary_path = data_path(output_dir, f"{a_neighbor_prefix}.csv")
    write_summary_csv(summary_path, runs, operands, offsets, args.force)
    write_a_neighbor_source_csv(a_neighbor_summary_path, runs, args.tiles_per_group, args.force)
    written = [summary_path, a_neighbor_summary_path]
    written.extend(
        save_figure(
            plot_combined_operand_offsets(runs, operands, offsets, args.offset_mode),
            output_dir,
            args.prefix,
            args.formats,
            args.force,
        )
    )
    written.extend(
        save_figure(
            plot_a_neighbor_source_tiles(runs, args.tiles_per_group),
            output_dir,
            a_neighbor_prefix,
            args.formats,
            args.force,
        )
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()

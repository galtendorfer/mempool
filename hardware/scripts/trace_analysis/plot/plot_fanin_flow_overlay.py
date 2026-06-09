#!/usr/bin/env python3
"""Compare source-core fan-in with stalls and fires for route-port traffic."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import FuncFormatter

from _fanin_heatmap_style import add_fanin_colorbar, fanin_cmap_norm
from _plot_output_paths import data_path, figure_path

TRACE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_ANALYSIS_DIR))

from operand_regions import add_operand_region_args, classify_operand, load_operand_regions, operand_address_from_row


OPERANDS = ("A", "B", "C", "other")
COUNT_COLUMNS = ("requests", "stalls", "fires", *OPERANDS)


@dataclass
class TileCycleStats:
    requests: int = 0
    stalls: int = 0
    fires: int = 0
    operands: Counter[str] = field(default_factory=Counter)
    rows: list[dict[str, str]] = field(default_factory=list)


@dataclass(frozen=True)
class FocusTileMetrics:
    tile: int
    port: int
    threshold_cycles: int
    threshold_requests: int
    threshold_stalls: int
    threshold_fires: int
    total_requests: int
    total_stalls: int
    total_fires: int
    peak_requests: int


@dataclass(frozen=True)
class FocusTileChoice:
    requested_focus_tile: str
    tile: int
    port: int
    requested_threshold: int
    effective_threshold: int
    threshold_met: bool
    requested_threshold_cycles: int
    reason: str
    metrics: FocusTileMetrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="path_graph directory, cycle_node_state.csv, or result directory containing analysis/path_graph",
    )
    parser.add_argument("--cycle-start", type=int, required=True, help="first cycle to include")
    parser.add_argument("--cycle-end", type=int, required=True, help="last cycle to include")
    parser.add_argument("--window", type=int, default=1, help="cycle averaging window for heatmaps")
    parser.add_argument(
        "--window-stat",
        choices=("max", "mean"),
        default="mean",
        help="window reduction for heatmap cells when --window > 1; mean rounds average fan-in to 0.25-step colours, max keeps peak 0..4 values",
    )
    parser.add_argument(
        "--port",
        type=int,
        help="route port to analyze; omit for all-port fan-in mechanism output",
    )
    parser.add_argument("--node-point", default="tcdm_remote", help="cycle_node_state point to count")
    parser.add_argument("--tiles-per-group", type=int, default=16, help="source tiles per group")
    parser.add_argument("--group", type=int, help="only include source tiles from this group")
    parser.add_argument("--tile", action="append", help="source tile id(s), comma-separated; may be repeated")
    parser.add_argument(
        "--focus-tile",
        default="auto",
        help="tile for worst-cycle drilldown tables, or 'auto' to choose the tile with strongest fan-in evidence",
    )
    parser.add_argument("--threshold", type=int, default=3, help="preferred minimum focus-tile fan-in for drilldown rows")
    parser.add_argument("--max-cores", type=int, default=4, help="maximum source cores per tile")
    add_operand_region_args(parser)
    parser.add_argument("--output-dir", type=Path, help="output directory; defaults under plots/port_pressure")
    parser.add_argument("--prefix", help="output filename prefix; defaults to all_ports_fanin_flow or port<N>_fanin_flow")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="figure formats to write",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser.parse_args()


def parse_int(value: str | None, default: int | None = None) -> int | None:
    if value in (None, ""):
        return default
    try:
        return int(value, 0)
    except ValueError:
        try:
            return int(value, 16)
        except ValueError:
            return default


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    parsed: list[int] = []
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.append(int(part, 0))
    return sorted(set(parsed))


def parse_focus_tile(value: str) -> int | None:
    if value.lower() == "auto":
        return None
    parsed = parse_int(value)
    if parsed is None:
        raise SystemExit(f"--focus-tile must be an integer tile id or 'auto', got {value!r}")
    return parsed


def resolve_graph_dir(input_path: Path) -> Path:
    if input_path.is_dir() and (input_path / "cycle_node_state.csv").is_file():
        return input_path
    if input_path.is_file() and input_path.name == "cycle_node_state.csv":
        return input_path.parent
    nested = input_path / "analysis" / "path_graph"
    if nested.is_dir() and (nested / "cycle_node_state.csv").is_file():
        return nested
    direct = input_path.parent if input_path.is_file() else input_path
    if (direct / "cycle_node_state.csv").is_file():
        return direct
    raise SystemExit(f"Could not find cycle_node_state.csv from {input_path}")


def default_port_pressure_dir(graph_dir: Path) -> Path:
    if graph_dir.name == "path_graph" and graph_dir.parent.name == "analysis":
        return graph_dir.parent.parent / "plots" / "port_pressure"
    return graph_dir / "plots" / "port_pressure"


def discover_tiles(graph_dir: Path, explicit_tiles: list[int] | None, group: int | None, tiles_per_group: int) -> list[int]:
    if explicit_tiles:
        return explicit_tiles
    if group is not None:
        first_tile = group * tiles_per_group
        return list(range(first_tile, first_tile + tiles_per_group))

    tiles: set[int] = set()
    nodes_path = graph_dir / "nodes.csv"
    if nodes_path.is_file():
        with nodes_path.open(newline="") as file:
            reader = csv.DictReader(file)
            for row in reader:
                if row.get("point") != "tcdm_remote":
                    continue
                tile = parse_int(row.get("tile"))
                if tile is not None and tile >= 0:
                    tiles.add(tile)
    if tiles:
        return sorted(tiles)

    with (graph_dir / "cycle_node_state.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            tile = parse_int(row.get("tile"))
            if tile is not None and tile >= 0:
                tiles.add(tile)
    if not tiles:
        raise SystemExit("Could not infer source tiles")
    return sorted(tiles)


def read_stats(
    graph_dir: Path,
    tiles: list[int],
    cycle_start: int,
    cycle_end: int,
    port: int | None,
    node_point: str,
    operand_regions,
) -> tuple[dict[tuple[int, int, int], TileCycleStats], list[int]]:
    tile_set = set(tiles)
    stats: dict[tuple[int, int, int], TileCycleStats] = defaultdict(TileCycleStats)
    observed_ports: set[int] = set()
    with (graph_dir / "cycle_node_state.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "tile", "point", "core", "port", "valid", "ready", "fire", "stall", "state", "addr"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in cycle_node_state.csv: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("point") != node_point:
                continue
            row_port = parse_int(row.get("port"))
            if row_port is None:
                continue
            if port is not None and row_port != port:
                continue
            cycle = parse_int(row.get("cycle"))
            tile = parse_int(row.get("tile"))
            if cycle is None or tile is None:
                continue
            if cycle < cycle_start or cycle > cycle_end or tile not in tile_set:
                continue

            valid = parse_int(row.get("valid"), 0) or 0
            stall = parse_int(row.get("stall"), 0) or 0
            fire = parse_int(row.get("fire"), 0) or 0
            if valid <= 0 and stall <= 0 and fire <= 0:
                continue

            observed_ports.add(row_port)
            entry = stats[(cycle, tile, row_port)]
            operand_addr = operand_address_from_row(row, operand_regions)
            operand = classify_operand(operand_addr, operand_regions)
            entry.requests += valid
            entry.stalls += stall
            entry.fires += fire
            if valid > 0:
                entry.operands[operand] += valid
                entry.rows.append(dict(row, operand=operand, operand_addr=operand_addr))
    return stats, sorted(observed_ports)


def collapse_port_stats(
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    ports: list[int],
) -> dict[tuple[int, int], TileCycleStats]:
    collapsed: dict[tuple[int, int], TileCycleStats] = defaultdict(TileCycleStats)
    for cycle in cycles:
        for tile in tiles:
            entry = collapsed[(cycle, tile)]
            for port in ports:
                port_entry = stats[(cycle, tile, port)]
                entry.requests += port_entry.requests
                entry.stalls += port_entry.stalls
                entry.fires += port_entry.fires
                entry.operands.update(port_entry.operands)
                entry.rows.extend(port_entry.rows)
    return collapsed


def make_windows(cycles: list[int], window: int) -> list[tuple[int, int, int]]:
    if window < 1:
        raise SystemExit("--window must be >= 1")
    return [
        (chunk[0], chunk[-1], len(chunk))
        for index in range(0, len(cycles), window)
        if (chunk := cycles[index : index + window])
    ]


def count_value(entry: TileCycleStats, column: str) -> int:
    if column == "requests":
        return entry.requests
    if column == "stalls":
        return entry.stalls
    if column == "fires":
        return entry.fires
    return entry.operands[column]


def aggregate_for_plot(
    stats: dict[tuple[int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    window: int,
    window_stat: str,
) -> tuple[dict[tuple[int, int, str], float], list[int], list[tuple[int, int, int]]]:
    windows = make_windows(cycles, window)
    aggregated: dict[tuple[int, int, str], float] = defaultdict(float)
    for window_start, window_end, window_cycles in windows:
        for tile in tiles:
            for column in COUNT_COLUMNS:
                values = [count_value(stats[(cycle, tile)], column) for cycle in range(window_start, window_end + 1)]
                if window_stat == "mean":
                    aggregated[(window_start, tile, column)] = sum(values) / window_cycles
                else:
                    aggregated[(window_start, tile, column)] = float(max(values, default=0))
    return aggregated, [window_start for window_start, _, _ in windows], windows


def ensure_can_write(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {path} (use --force)")


def save_figure(fig, output_base: Path, formats: list[str], force: bool) -> list[Path]:
    written: list[Path] = []
    for extension in formats:
        output_path = figure_path(output_base, extension)
        if output_path.exists() and not force:
            raise SystemExit(f"Refusing to overwrite existing figure: {output_path} (use --force)")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300)
        written.append(output_path)
    plt.close(fig)
    return written


def matrix_for(aggregated: dict[tuple[int, int, str], float], cycles: list[int], tiles: list[int], column: str) -> np.ndarray:
    matrix = np.zeros((len(tiles), len(cycles)), dtype=float)
    for row_index, tile in enumerate(tiles):
        for col_index, cycle in enumerate(cycles):
            matrix[row_index, col_index] = aggregated[(cycle, tile, column)]
    return matrix


def add_group_separators(ax, tiles: list[int], tiles_per_group: int) -> None:
    for index in range(1, len(tiles)):
        if tiles[index] // tiles_per_group != tiles[index - 1] // tiles_per_group:
            ax.axhline(index - 0.5, color="white", linewidth=1.1, alpha=0.9)


def window_axis_label(window: int, window_stat: str) -> str:
    if window == 1:
        return "cycle"
    label = "peak" if window_stat == "max" else "average"
    return f"cycle window start ({window}-cycle {label})"


def configure_axis(ax, cycles: list[int], tiles: list[int], tiles_per_group: int, window: int, window_stat: str) -> None:
    ax.set_ylabel("source tile")
    ax.set_xlabel(window_axis_label(window, window_stat))
    x_ticks = np.linspace(0, len(cycles) - 1, min(10, len(cycles)), dtype=int)
    ax.set_xticks(x_ticks, [str(cycles[position]) for position in x_ticks])
    if len(tiles) <= 20:
        ax.set_yticks(range(len(tiles)), [str(tile) for tile in tiles])
    else:
        y_ticks = np.linspace(0, len(tiles) - 1, min(16, len(tiles)), dtype=int)
        ax.set_yticks(y_ticks, [str(tiles[position]) for position in y_ticks])
    add_group_separators(ax, tiles, tiles_per_group)


def plot_overlay_heatmap(
    aggregated: dict[tuple[int, int, str], float],
    cycles: list[int],
    tiles: list[int],
    output_base: Path,
    formats: list[str],
    force: bool,
    port_label: str,
    max_cores: int,
    tiles_per_group: int,
    window: int,
    window_stat: str,
) -> list[Path]:
    columns = (
        ("requests", "Requests", "Blues"),
        ("stalls", "Stalls", "Reds"),
        ("fires", "Fires", "Greens"),
    )
    height = max(8.0, min(14.0, 4.0 + 0.08 * len(tiles)))
    fig, axes = plt.subplots(3, 1, figsize=(13.0, height), sharex=True, constrained_layout=True)
    images = []
    averaged = window > 1 and window_stat == "mean"
    if window == 1:
        suffix = ""
    elif window_stat == "mean":
        suffix = f" ({window}-cycle average, 0.25-step)"
    else:
        suffix = f" ({window}-cycle peak)"
    for ax, (column, label, cmap) in zip(axes, columns):
        cmap_obj, norm = fanin_cmap_norm(max_cores, cmap, averaged)
        image = ax.imshow(
            matrix_for(aggregated, cycles, tiles, column),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap_obj,
            norm=norm,
        )
        images.append(image)
        ax.set_title(f"{label} on {port_label}{suffix}")
        configure_axis(ax, cycles, tiles, tiles_per_group, window, window_stat)
    add_fanin_colorbar(
        fig,
        images[0],
        list(axes),
        max_cores,
        averaged,
        "cores per tile" if window_stat == "max" else "average cores per tile per cycle (nearest 0.25)",
        pad=0.012,
        shrink=0.86,
    )
    return save_figure(fig, output_base, formats, force)


def write_overlay_csv(
    output_path: Path,
    aggregated: dict[tuple[int, int, str], float],
    cycles: list[int],
    windows: list[tuple[int, int, int]],
    tiles: list[int],
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windowed = any(width > 1 for _, _, width in windows)
    fieldnames = (
        ("window_start", "window_end", "window_cycles", "tile", *COUNT_COLUMNS)
        if windowed
        else ("cycle", "tile", *COUNT_COLUMNS)
    )
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cycle, (window_start, window_end, window_cycles) in zip(cycles, windows):
            for tile in tiles:
                if windowed:
                    row = {"window_start": window_start, "window_end": window_end, "window_cycles": window_cycles, "tile": tile}
                else:
                    row = {"cycle": cycle, "tile": tile}
                for column in COUNT_COLUMNS:
                    value = aggregated[(cycle, tile, column)]
                    row[column] = int(value) if value.is_integer() else f"{value:.6g}"
                writer.writerow(row)


def write_exact_tile_cycle_csv(
    output_path: Path,
    stats: dict[tuple[int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "cycle",
        "tile",
        "requests",
        "stalls",
        "fires",
        "A",
        "B",
        "C",
        "other",
        "stall_per_request",
        "fire_per_request",
    )
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cycle in cycles:
            for tile in tiles:
                entry = stats[(cycle, tile)]
                writer.writerow(
                    {
                        "cycle": cycle,
                        "tile": tile,
                        "requests": entry.requests,
                        "stalls": entry.stalls,
                        "fires": entry.fires,
                        "A": entry.operands["A"],
                        "B": entry.operands["B"],
                        "C": entry.operands["C"],
                        "other": entry.operands["other"],
                        "stall_per_request": f"{entry.stalls / entry.requests:.6f}" if entry.requests else "0",
                        "fire_per_request": f"{entry.fires / entry.requests:.6f}" if entry.requests else "0",
                    }
                )


def write_by_request_summary(
    output_path: Path,
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    ports: list[int],
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    grouped: dict[int, Counter[str]] = defaultdict(Counter)
    for cycle in cycles:
        for tile in tiles:
            for port in ports:
                entry = stats[(cycle, tile, port)]
                requests = entry.requests
                grouped[requests]["tile_port_cycles"] += 1
                grouped[requests]["requests"] += requests
                grouped[requests]["stalls"] += entry.stalls
                grouped[requests]["fires"] += entry.fires
    with output_path.open("w", newline="") as file:
        fieldnames = (
            "request_count",
            "tile_port_cycles",
            "tile_cycles",
            "total_requests",
            "total_stalls",
            "total_fires",
            "stall_per_request",
            "fire_per_request",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for request_count in sorted(grouped):
            row = grouped[request_count]
            requests = row["requests"]
            writer.writerow(
                {
                    "request_count": request_count,
                    "tile_port_cycles": row["tile_port_cycles"],
                    "tile_cycles": row["tile_port_cycles"],
                    "total_requests": requests,
                    "total_stalls": row["stalls"],
                    "total_fires": row["fires"],
                    "stall_per_request": f"{row['stalls'] / requests:.6f}" if requests else "0",
                    "fire_per_request": f"{row['fires'] / requests:.6f}" if requests else "0",
                }
            )


def write_by_tile_demand_balance(
    output_path: Path,
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    ports: list[int],
    max_cores: int,
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bucket_fields = tuple(f"request_count_{count}_tile_port_cycles" for count in range(max_cores + 1))
    fieldnames = (
        "tile",
        "idle_tile_port_cycles",
        "one_core_tile_port_cycles",
        "multi_core_tile_port_cycles",
        "active_tile_port_cycles",
        "multi_core_share",
        "multi_minus_one_tile_port_cycles",
        "one_core_requests",
        "one_core_stalls",
        "one_core_fires",
        "multi_core_requests",
        "multi_core_stalls",
        "multi_core_fires",
        "one_core_stall_rate",
        "multi_core_stall_rate",
        "stall_uplift",
        "multi_core_stall_pressure",
        "same_port_hotspot_score",
        "total_requests",
        "total_stalls",
        "total_fires",
        "stall_per_request",
        "fire_per_request",
        *bucket_fields,
        "request_count_over_max_tile_port_cycles",
    )
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for tile in tiles:
            buckets: Counter[int] = Counter()
            over_max = 0
            total_requests = 0
            total_stalls = 0
            total_fires = 0
            one_core_requests = 0
            one_core_stalls = 0
            one_core_fires = 0
            multi_core_requests = 0
            multi_core_stalls = 0
            multi_core_fires = 0
            for cycle in cycles:
                for port in ports:
                    entry = stats[(cycle, tile, port)]
                    request_count = entry.requests
                    if request_count <= max_cores:
                        buckets[request_count] += 1
                    else:
                        over_max += 1
                    total_requests += request_count
                    total_stalls += entry.stalls
                    total_fires += entry.fires
                    if request_count == 1:
                        one_core_requests += entry.requests
                        one_core_stalls += entry.stalls
                        one_core_fires += entry.fires
                    elif request_count >= 2:
                        multi_core_requests += entry.requests
                        multi_core_stalls += entry.stalls
                        multi_core_fires += entry.fires

            one_core = buckets[1]
            multi_core = sum(buckets[count] for count in range(2, max_cores + 1)) + over_max
            active = one_core + multi_core
            multi_core_share = multi_core / active if active else 0.0
            one_core_stall_rate = one_core_stalls / one_core_requests if one_core_requests else 0.0
            multi_core_stall_rate = multi_core_stalls / multi_core_requests if multi_core_requests else 0.0
            stall_uplift = multi_core_stall_rate - one_core_stall_rate
            multi_core_stall_pressure = multi_core_stalls / active if active else 0.0
            hotspot_score = multi_core_share * max(0.0, stall_uplift)
            row = {
                "tile": tile,
                "idle_tile_port_cycles": buckets[0],
                "one_core_tile_port_cycles": one_core,
                "multi_core_tile_port_cycles": multi_core,
                "active_tile_port_cycles": active,
                "multi_core_share": f"{multi_core_share:.6f}",
                "multi_minus_one_tile_port_cycles": multi_core - one_core,
                "one_core_requests": one_core_requests,
                "one_core_stalls": one_core_stalls,
                "one_core_fires": one_core_fires,
                "multi_core_requests": multi_core_requests,
                "multi_core_stalls": multi_core_stalls,
                "multi_core_fires": multi_core_fires,
                "one_core_stall_rate": f"{one_core_stall_rate:.6f}",
                "multi_core_stall_rate": f"{multi_core_stall_rate:.6f}",
                "stall_uplift": f"{stall_uplift:.6f}",
                "multi_core_stall_pressure": f"{multi_core_stall_pressure:.6f}",
                "same_port_hotspot_score": f"{hotspot_score:.6f}",
                "total_requests": total_requests,
                "total_stalls": total_stalls,
                "total_fires": total_fires,
                "stall_per_request": f"{total_stalls / total_requests:.6f}" if total_requests else "0",
                "fire_per_request": f"{total_fires / total_requests:.6f}" if total_requests else "0",
                "request_count_over_max_tile_port_cycles": over_max,
            }
            row.update({field: buckets[index] for index, field in enumerate(bucket_fields)})
            writer.writerow(row)


def focus_tile_metrics(
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    tile: int,
    port: int,
    threshold: int,
) -> FocusTileMetrics:
    threshold_cycles = 0
    threshold_requests = 0
    threshold_stalls = 0
    threshold_fires = 0
    total_requests = 0
    total_stalls = 0
    total_fires = 0
    peak_requests = 0
    for cycle in cycles:
        entry = stats[(cycle, tile, port)]
        total_requests += entry.requests
        total_stalls += entry.stalls
        total_fires += entry.fires
        peak_requests = max(peak_requests, entry.requests)
        if entry.requests >= threshold:
            threshold_cycles += 1
            threshold_requests += entry.requests
            threshold_stalls += entry.stalls
            threshold_fires += entry.fires
    return FocusTileMetrics(
        tile=tile,
        port=port,
        threshold_cycles=threshold_cycles,
        threshold_requests=threshold_requests,
        threshold_stalls=threshold_stalls,
        threshold_fires=threshold_fires,
        total_requests=total_requests,
        total_stalls=total_stalls,
        total_fires=total_fires,
        peak_requests=peak_requests,
    )


def focus_tile_sort_key(metrics: FocusTileMetrics) -> tuple[int, int, int, int, int, int, int, int]:
    return (
        metrics.threshold_cycles,
        metrics.threshold_stalls,
        metrics.threshold_requests,
        metrics.peak_requests,
        metrics.total_stalls,
        metrics.total_requests,
        -metrics.tile,
        -metrics.port,
    )


def fallback_focus_tile_sort_key(metrics: FocusTileMetrics) -> tuple[int, int, int, int, int]:
    return (
        metrics.peak_requests,
        metrics.total_stalls,
        metrics.total_requests,
        -metrics.tile,
        -metrics.port,
    )


def choose_focus_tile(
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    tiles: list[int],
    ports: list[int],
    requested_focus_tile: str,
    focus_tile: int | None,
    threshold: int,
) -> FocusTileChoice:
    if focus_tile is not None:
        tile_metrics = [focus_tile_metrics(stats, cycles, focus_tile, port, threshold) for port in ports]
        metrics = max(tile_metrics, key=focus_tile_sort_key)
        return FocusTileChoice(
            requested_focus_tile=requested_focus_tile,
            tile=focus_tile,
            port=metrics.port,
            requested_threshold=threshold,
            effective_threshold=threshold,
            threshold_met=metrics.threshold_cycles > 0,
            requested_threshold_cycles=metrics.threshold_cycles,
            reason="explicit focus tile from --focus-tile",
            metrics=metrics,
        )

    threshold_metrics = [
        focus_tile_metrics(stats, cycles, tile, port, threshold)
        for tile in tiles
        for port in ports
    ]
    eligible = [metrics for metrics in threshold_metrics if metrics.threshold_cycles > 0]
    if eligible:
        selected = max(eligible, key=focus_tile_sort_key)
        return FocusTileChoice(
            requested_focus_tile=requested_focus_tile,
            tile=selected.tile,
            port=selected.port,
            requested_threshold=threshold,
            effective_threshold=threshold,
            threshold_met=True,
            requested_threshold_cycles=selected.threshold_cycles,
            reason=f"auto selected tile with most cycles at fan-in >= {threshold}",
            metrics=selected,
        )

    selected_at_requested = max(threshold_metrics, key=fallback_focus_tile_sort_key)
    effective_threshold = selected_at_requested.peak_requests if selected_at_requested.peak_requests > 0 else threshold
    effective_metrics = focus_tile_metrics(stats, cycles, selected_at_requested.tile, selected_at_requested.port, effective_threshold)
    return FocusTileChoice(
        requested_focus_tile=requested_focus_tile,
        tile=selected_at_requested.tile,
        port=selected_at_requested.port,
        requested_threshold=threshold,
        effective_threshold=effective_threshold,
        threshold_met=False,
        requested_threshold_cycles=0,
        reason=(
            f"no tile reached fan-in >= {threshold}; auto selected tile with best available peak fan-in "
            f"({selected_at_requested.peak_requests})"
        ),
        metrics=effective_metrics,
    )


def write_focus_tile_selection(output_path: Path, choice: FocusTileChoice, force: bool) -> None:
    ensure_can_write(output_path, force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "requested_focus_tile",
        "selected_tile",
        "selected_port",
        "requested_threshold",
        "effective_threshold",
        "requested_threshold_met",
        "requested_threshold_cycles",
        "effective_threshold_cycles",
        "effective_threshold_requests",
        "effective_threshold_stalls",
        "effective_threshold_fires",
        "total_requests",
        "total_stalls",
        "total_fires",
        "peak_requests",
        "reason",
    )
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "requested_focus_tile": choice.requested_focus_tile,
                "selected_tile": choice.tile,
                "selected_port": choice.port,
                "requested_threshold": choice.requested_threshold,
                "effective_threshold": choice.effective_threshold,
                "requested_threshold_met": int(choice.threshold_met),
                "requested_threshold_cycles": choice.requested_threshold_cycles,
                "effective_threshold_cycles": choice.metrics.threshold_cycles,
                "effective_threshold_requests": choice.metrics.threshold_requests,
                "effective_threshold_stalls": choice.metrics.threshold_stalls,
                "effective_threshold_fires": choice.metrics.threshold_fires,
                "total_requests": choice.metrics.total_requests,
                "total_stalls": choice.metrics.total_stalls,
                "total_fires": choice.metrics.total_fires,
                "peak_requests": choice.metrics.peak_requests,
                "reason": choice.reason,
            }
        )


def plot_by_request_summary(csv_path: Path, output_base: Path, formats: list[str], force: bool) -> list[Path]:
    request_counts: list[int] = []
    tile_port_cycles: list[int] = []
    stall_per_request: list[float] = []
    fire_per_request: list[float] = []
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            request_count = int(row["request_count"])
            if request_count == 0:
                continue
            request_counts.append(request_count)
            tile_port_cycles.append(int(row.get("tile_port_cycles") or row["tile_cycles"]))
            stall_per_request.append(float(row["stall_per_request"]))
            fire_per_request.append(float(row["fire_per_request"]))
    fig, ax_count = plt.subplots(figsize=(7.8, 4.8))
    ax_rate = ax_count.twinx()
    ax_count.bar(request_counts, tile_port_cycles, color="#9ecae1", label="tile-port-cycles")
    ax_rate.plot(request_counts, stall_per_request, marker="o", color="#cb181d", label="stall/request")
    ax_rate.plot(request_counts, fire_per_request, marker="o", color="#238b45", label="fire/request")
    ax_count.set_xlabel("number of cores requesting the same port per cycle")
    ax_count.set_ylabel("tile-port-cycles")
    ax_rate.set_ylabel("fraction of requests")
    ax_count.set_xticks(request_counts)
    ax_rate.set_ylim(0, 1.05)
    lines_1, labels_1 = ax_count.get_legend_handles_labels()
    lines_2, labels_2 = ax_rate.get_legend_handles_labels()
    ax_count.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")
    ax_count.set_title("Same-Port Demand by Number of Requesting Cores")
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def plot_by_tile_demand_balance(csv_path: Path, output_base: Path, formats: list[str], force: bool) -> list[Path]:
    rows: list[tuple[int, int, int, int, int]] = []
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            active = int(row["active_tile_port_cycles"])
            if active <= 0:
                continue
            one_core = int(row["one_core_tile_port_cycles"])
            multi_core = int(row["multi_core_tile_port_cycles"])
            balance = int(row["multi_minus_one_tile_port_cycles"])
            rows.append((int(row["tile"]), one_core, multi_core, balance, active))
    if not rows:
        return []

    rows.sort(key=lambda item: (item[3], item[2], item[4], -item[0]), reverse=True)
    tiles = [row[0] for row in rows]
    one_core_cycles = [row[1] for row in rows]
    multi_core_cycles = [row[2] for row in rows]
    y_positions = np.arange(len(rows))
    max_cycles = max(max(one_core_cycles, default=0), max(multi_core_cycles, default=0))
    fig_height = max(5.0, min(18.0, 2.0 + 0.22 * len(rows)))
    fig, ax = plt.subplots(figsize=(9.0, fig_height))
    ax.barh(y_positions, [-count for count in one_core_cycles], color="#6baed6", label="1 core")
    ax.barh(y_positions, multi_core_cycles, color="#f16913", label="2+ cores")
    ax.axvline(0, color="#4d4d4d", linewidth=0.9)
    ax.set_yticks(y_positions, [str(tile) for tile in tiles])
    ax.invert_yaxis()
    if max_cycles > 0:
        ax.set_xlim(-max_cycles * 1.08, max_cycles * 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _position: f"{abs(int(value))}"))
    ax.set_xlabel("observed tile-port-cycles")
    ax.set_ylabel("source tile")
    ax.set_title("Same-Port Demand Balance by Tile")
    ax.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def plot_by_tile_hotspots(
    csv_path: Path,
    output_base: Path,
    formats: list[str],
    force: bool,
    tiles_per_group: int,
) -> list[Path]:
    rows: list[dict[str, float | int]] = []
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            active = int(row["active_tile_port_cycles"])
            if active <= 0:
                continue
            rows.append(
                {
                    "tile": int(row["tile"]),
                    "score": float(row["same_port_hotspot_score"]),
                    "multi_core_share": float(row["multi_core_share"]),
                    "stall_uplift": float(row["stall_uplift"]),
                    "multi_core_stall_pressure": float(row["multi_core_stall_pressure"]),
                    "active": active,
                }
            )
    if not rows:
        return []

    max_tile = max(int(row["tile"]) for row in rows)
    group_count = max_tile // tiles_per_group + 1
    score_matrix = np.full((group_count, tiles_per_group), np.nan, dtype=float)
    for row in rows:
        tile = int(row["tile"])
        score_matrix[tile // tiles_per_group, tile % tiles_per_group] = float(row["score"])

    score_max = max(float(row["score"]) for row in rows)
    color_max = score_max if score_max > 0 else 1.0
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#f0f0f0")

    top_rows = sorted(
        rows,
        key=lambda row: (
            float(row["score"]),
            float(row["multi_core_stall_pressure"]),
            int(row["active"]),
            -int(row["tile"]),
        ),
        reverse=True,
    )[: min(16, len(rows))]

    fig_height = max(4.8, min(8.5, 3.6 + 0.18 * len(top_rows)))
    fig, (ax_map, ax_rank) = plt.subplots(
        1,
        2,
        figsize=(12.2, fig_height),
        gridspec_kw={"width_ratios": [1.25, 1.0]},
    )

    image = ax_map.imshow(np.ma.masked_invalid(score_matrix), aspect="auto", cmap=cmap, vmin=0, vmax=color_max)
    ax_map.set_title("Topology view")
    ax_map.set_xlabel("tile in group")
    ax_map.set_ylabel("group")
    ax_map.set_xticks(range(tiles_per_group))
    ax_map.set_yticks(range(group_count))
    ax_map.set_xticks(np.arange(-0.5, tiles_per_group, 1), minor=True)
    ax_map.set_yticks(np.arange(-0.5, group_count, 1), minor=True)
    ax_map.grid(which="minor", color="white", linewidth=0.8)
    ax_map.tick_params(which="minor", bottom=False, left=False)
    colorbar = fig.colorbar(image, ax=ax_map, fraction=0.046, pad=0.04)
    colorbar.set_label("hotspot score")

    y_positions = np.arange(len(top_rows))
    scores = [float(row["score"]) for row in top_rows]
    labels = [str(int(row["tile"])) for row in top_rows]
    colors = [cmap(score / color_max if color_max else 0.0) for score in scores]
    ax_rank.barh(y_positions, scores, color=colors)
    ax_rank.set_yticks(y_positions, labels)
    ax_rank.invert_yaxis()
    ax_rank.set_xlabel("multi-core share x stall-rate uplift")
    ax_rank.set_ylabel("source tile")
    ax_rank.set_title("Highest-score tiles")
    ax_rank.grid(axis="x", color="#d9d9d9", linewidth=0.6, alpha=0.7)
    rank_limit = max(scores, default=0.0) * 1.22
    if rank_limit <= 0:
        rank_limit = 1.0
    ax_rank.set_xlim(0, rank_limit)
    for position, row in enumerate(top_rows):
        score = float(row["score"])
        share = float(row["multi_core_share"])
        uplift = float(row["stall_uplift"])
        label = f"{share:.0%}, +{max(0.0, uplift):.0%}"
        ax_rank.text(score + rank_limit * 0.015, position, label, va="center", fontsize=8)

    fig.suptitle("Same-Port Demand Hotspots by Tile")
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def summarize_focus_cycle(entry: TileCycleStats) -> tuple[str, str, str, str, str]:
    cores = []
    operands = []
    route_addresses = []
    source_addresses = []
    operand_addresses = []
    for row in sorted(entry.rows, key=lambda item: parse_int(item.get("core"), -1) or -1):
        cores.append(row.get("core", ""))
        operands.append(row.get("operand", "other"))
        route_addresses.append(row.get("addr", ""))
        source_addresses.append(row.get("source_addr", ""))
        operand_addresses.append(row.get("operand_addr", ""))
    return ";".join(cores), ";".join(operands), ";".join(route_addresses), ";".join(source_addresses), ";".join(operand_addresses)


def write_focus_tables(
    output_dir: Path,
    prefix: str,
    stats: dict[tuple[int, int, int], TileCycleStats],
    cycles: list[int],
    focus_tile: int,
    focus_port: int,
    threshold: int,
    force: bool,
) -> tuple[Path, Path]:
    focus_prefix = prefix if prefix.startswith(f"port{focus_port}_") else f"{prefix}_port{focus_port}"
    row_path = output_dir / f"{focus_prefix}_tile{focus_tile}_fanin_ge{threshold}_rows.csv"
    summary_path = output_dir / f"{focus_prefix}_tile{focus_tile}_fanin_ge{threshold}_summary.csv"
    ensure_can_write(row_path, force)
    ensure_can_write(summary_path, force)
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_cycles = [cycle for cycle in cycles if stats[(cycle, focus_tile, focus_port)].requests >= threshold]
    focus_cycles.sort(key=lambda cycle: (-stats[(cycle, focus_tile, focus_port)].requests, -stats[(cycle, focus_tile, focus_port)].stalls, cycle))
    with summary_path.open("w", newline="") as file:
        fieldnames = (
            "cycle", "tile", "port", "requests", "stalls", "fires", "stall_per_request", "fire_per_request",
            "cores", "operands", "route_addresses", "source_addresses", "operand_addresses",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cycle in focus_cycles:
            entry = stats[(cycle, focus_tile, focus_port)]
            cores, operands, route_addresses, source_addresses, operand_addresses = summarize_focus_cycle(entry)
            writer.writerow(
                {
                    "cycle": cycle,
                    "tile": focus_tile,
                    "port": focus_port,
                    "requests": entry.requests,
                    "stalls": entry.stalls,
                    "fires": entry.fires,
                    "stall_per_request": f"{entry.stalls / entry.requests:.6f}" if entry.requests else "0",
                    "fire_per_request": f"{entry.fires / entry.requests:.6f}" if entry.requests else "0",
                    "cores": cores,
                    "operands": operands,
                    "route_addresses": route_addresses,
                    "source_addresses": source_addresses,
                    "operand_addresses": operand_addresses,
                }
            )

    with row_path.open("w", newline="") as file:
        fieldnames = (
            "cycle",
            "tile",
            "port",
            "cycle_requests",
            "cycle_stalls",
            "cycle_fires",
            "core",
            "operand",
            "addr",
            "source_addr",
            "operand_addr",
            "valid",
            "ready",
            "fire",
            "stall",
            "state",
            "bank",
            "meta_id",
            "payload_core",
            "write",
            "back2local",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for cycle in focus_cycles:
            entry = stats[(cycle, focus_tile, focus_port)]
            for row in sorted(entry.rows, key=lambda item: parse_int(item.get("core"), -1) or -1):
                writer.writerow(
                    {
                        "cycle": cycle,
                        "tile": focus_tile,
                        "port": focus_port,
                        "cycle_requests": entry.requests,
                        "cycle_stalls": entry.stalls,
                        "cycle_fires": entry.fires,
                        "core": row.get("core", ""),
                        "operand": row.get("operand", "other"),
                        "addr": row.get("addr", ""),
                        "source_addr": row.get("source_addr", ""),
                        "operand_addr": row.get("operand_addr", ""),
                        "valid": row.get("valid", ""),
                        "ready": row.get("ready", ""),
                        "fire": row.get("fire", ""),
                        "stall": row.get("stall", ""),
                        "state": row.get("state", ""),
                        "bank": row.get("bank", ""),
                        "meta_id": row.get("meta_id", ""),
                        "payload_core": row.get("payload_core", ""),
                        "write": row.get("write", ""),
                        "back2local": row.get("back2local", ""),
                    }
                )
    return row_path, summary_path


def main() -> int:
    args = parse_args()
    if args.window < 1:
        raise SystemExit("--window must be >= 1")
    if args.max_cores <= 0:
        raise SystemExit("--max-cores must be > 0")
    if args.threshold <= 0:
        raise SystemExit("--threshold must be > 0")
    graph_dir = resolve_graph_dir(args.input_path)
    operand_regions = load_operand_regions(graph_dir, args)
    explicit_tiles = parse_int_list(args.tile)
    tiles = discover_tiles(graph_dir, explicit_tiles, args.group, args.tiles_per_group)
    requested_focus_tile = args.focus_tile
    focus_tile = parse_focus_tile(requested_focus_tile)
    if focus_tile is not None and focus_tile not in tiles:
        tiles = sorted(set(tiles) | {focus_tile})
    cycles = list(range(args.cycle_start, args.cycle_end + 1))

    stats_by_port, ports = read_stats(graph_dir, tiles, args.cycle_start, args.cycle_end, args.port, args.node_point, operand_regions)
    if not ports:
        port_text = f"port {args.port}" if args.port is not None else "any route port"
        raise SystemExit(f"No rows matched {args.node_point} on {port_text}")
    tile_stats = collapse_port_stats(stats_by_port, cycles, tiles, ports)
    focus_choice = choose_focus_tile(stats_by_port, cycles, tiles, ports, requested_focus_tile, focus_tile, args.threshold)
    aggregated, plot_cycles, windows = aggregate_for_plot(tile_stats, cycles, tiles, args.window, args.window_stat)

    output_dir = args.output_dir or default_port_pressure_dir(graph_dir)
    if args.prefix:
        prefix = args.prefix
        if args.port is None and prefix.startswith("port0"):
            prefix = prefix.replace("port0", "all_ports", 1)
        elif args.port is not None:
            prefix = prefix.replace("port0", f"port{args.port}", 1)
    else:
        prefix = "all_ports_fanin_flow" if args.port is None else f"port{args.port}_fanin_flow"
    port_label = "all route ports" if args.port is None else f"port {args.port}"
    by_request_outputs_enabled = args.port is None
    written: list[Path] = []

    exact_tile_cycle_csv = data_path(output_dir, f"{prefix}_exact_tile_cycle.csv")
    overlay_csv = data_path(output_dir, f"{prefix}_overlay_w{args.window}.csv")
    focus_selection_csv = data_path(output_dir, f"{prefix}_focus_tile_selection.csv")
    write_exact_tile_cycle_csv(exact_tile_cycle_csv, tile_stats, cycles, tiles, args.force)
    write_overlay_csv(overlay_csv, aggregated, plot_cycles, windows, tiles, args.force)
    write_focus_tile_selection(focus_selection_csv, focus_choice, args.force)
    if by_request_outputs_enabled:
        by_request_csv = data_path(output_dir, f"{prefix}_by_request_count.csv")
        by_tile_demand_csv = data_path(output_dir, f"{prefix}_by_tile_demand_balance.csv")
        write_by_request_summary(by_request_csv, stats_by_port, cycles, tiles, ports, args.force)
        write_by_tile_demand_balance(by_tile_demand_csv, stats_by_port, cycles, tiles, ports, args.max_cores, args.force)
    written.extend(plot_overlay_heatmap(
        aggregated,
        plot_cycles,
        tiles,
        output_dir / f"{prefix}_overlay_heatmap",
        args.formats,
        args.force,
        port_label,
        args.max_cores,
        args.tiles_per_group,
        args.window,
        args.window_stat,
    ))
    if by_request_outputs_enabled:
        written.extend(plot_by_request_summary(by_request_csv, output_dir / f"{prefix}_by_request_count", args.formats, args.force))
        written.extend(plot_by_tile_hotspots(
            by_tile_demand_csv,
            output_dir / f"{prefix}_by_tile_hotspots",
            args.formats,
            args.force,
            args.tiles_per_group,
        ))
    focus_row_path, focus_summary_path = write_focus_tables(
        output_dir / "data",
        prefix,
        stats_by_port,
        cycles,
        focus_choice.tile,
        focus_choice.port,
        focus_choice.effective_threshold,
        args.force,
    )

    print("Wrote fan-in flow overlay outputs:")
    for path in written:
        print(f"  {path}")
    data_outputs = [exact_tile_cycle_csv, overlay_csv]
    if by_request_outputs_enabled:
        data_outputs.append(by_request_csv)
        data_outputs.append(by_tile_demand_csv)
    data_outputs.extend([focus_selection_csv, focus_summary_path, focus_row_path])
    for path in data_outputs:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Plot per-tile source-core fan-in for one route port.

For the default MemPool load monitor data, each tile/cycle cell counts how many
cores in that tile have a valid `tcdm_remote` request for the selected port.
For MemPool this value is in [0, 4]. With --window > 1, each cell is the
per-cycle average over that cycle window, still on the same [0, 4] scale.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from _fanin_heatmap_style import add_fanin_colorbar, fanin_cmap_norm
from _plot_output_paths import data_path, figure_path

TRACE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_ANALYSIS_DIR))

from operand_regions import add_operand_region_args, classify_operand, load_operand_regions, operand_address_from_row


OPERANDS = ("A", "B", "C", "other")
COUNT_COLUMNS = ("total", *OPERANDS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="path_graph directory, cycle_node_state.csv, or result directory containing analysis/path_graph",
    )
    parser.add_argument("--cycle-start", type=int, help="first cycle to include")
    parser.add_argument("--cycle-end", type=int, help="last cycle to include")
    parser.add_argument(
        "--window",
        type=int,
        default=1,
        help="cycle aggregation window; values are averaged per cycle when window > 1",
    )
    parser.add_argument(
        "--window-stat",
        choices=("max", "mean"),
        default="mean",
        help="window reduction for heatmap cells when --window > 1; mean rounds average fan-in to 0.25-step colours, max keeps peak 0..4 values",
    )
    parser.add_argument("--port", type=int, default=0, help="route port to plot")
    parser.add_argument("--node-point", default="tcdm_remote", help="cycle_node_state point to count")
    parser.add_argument("--tiles-per-group", type=int, default=16, help="source tiles per group")
    parser.add_argument("--group", type=int, help="only include source tiles from this group")
    parser.add_argument("--tile", action="append", help="source tile id(s), comma-separated; may be repeated")
    parser.add_argument("--max-cores", type=int, default=4, help="maximum source cores per tile")
    add_operand_region_args(parser)
    parser.add_argument("--output-dir", type=Path, help="output directory; defaults under plots/port_pressure")
    parser.add_argument("--prefix", default="port0_tile_fanin", help="output filename prefix")
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


def read_counts(
    graph_dir: Path,
    tiles: list[int],
    cycle_start_filter: int | None,
    cycle_end_filter: int | None,
    port: int,
    node_point: str,
    operand_regions,
) -> tuple[dict[tuple[int, int, str], int], int, int]:
    tile_set = set(tiles)
    counts: dict[tuple[int, int, str], int] = defaultdict(int)
    observed_min: int | None = None
    observed_max: int | None = None
    with (graph_dir / "cycle_node_state.csv").open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "tile", "point", "port", "valid", "addr"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in cycle_node_state.csv: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("point") != node_point:
                continue
            if parse_int(row.get("port"), -1) != port:
                continue
            cycle = parse_int(row.get("cycle"))
            tile = parse_int(row.get("tile"))
            if cycle is None or tile is None or tile not in tile_set:
                continue
            if cycle_start_filter is not None and cycle < cycle_start_filter:
                continue
            if cycle_end_filter is not None and cycle > cycle_end_filter:
                continue
            valid = parse_int(row.get("valid"), 0) or 0
            if valid <= 0:
                continue
            operand = classify_operand(operand_address_from_row(row, operand_regions), operand_regions)
            counts[(cycle, tile, "total")] += valid
            counts[(cycle, tile, operand)] += valid
            observed_min = cycle if observed_min is None else min(observed_min, cycle)
            observed_max = cycle if observed_max is None else max(observed_max, cycle)
    if observed_min is None or observed_max is None:
        raise SystemExit("No matching node-state rows found")
    return counts, observed_min, observed_max


def make_windows(cycles: list[int], window: int) -> list[tuple[int, int, int]]:
    if window < 1:
        raise SystemExit("--window must be >= 1")
    return [
        (chunk[0], chunk[-1], len(chunk))
        for index in range(0, len(cycles), window)
        if (chunk := cycles[index : index + window])
    ]


def aggregate_counts(
    counts: dict[tuple[int, int, str], int],
    cycles: list[int],
    tiles: list[int],
    window: int,
    window_stat: str,
) -> tuple[dict[tuple[int, int, str], float], list[int], list[tuple[int, int, int]]]:
    windows = make_windows(cycles, window)
    if window == 1:
        return defaultdict(float, {key: float(value) for key, value in counts.items()}), cycles, windows

    aggregated: dict[tuple[int, int, str], float] = defaultdict(float)
    for window_start, window_end, window_cycles in windows:
        for tile in tiles:
            for column in COUNT_COLUMNS:
                values = [counts[(cycle, tile, column)] for cycle in range(window_start, window_end + 1)]
                if window_stat == "mean":
                    aggregated[(window_start, tile, column)] = sum(values) / window_cycles
                else:
                    aggregated[(window_start, tile, column)] = float(max(values, default=0))
    return aggregated, [window_start for window_start, _, _ in windows], windows


def build_matrix(counts: dict[tuple[int, int, str], float], cycles: list[int], tiles: list[int], column: str) -> np.ndarray:
    matrix = np.zeros((len(tiles), len(cycles)), dtype=float)
    for row_index, tile in enumerate(tiles):
        for col_index, cycle in enumerate(cycles):
            matrix[row_index, col_index] = counts[(cycle, tile, column)]
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


def configure_heatmap_axis(ax, cycles: list[int], tiles: list[int], tiles_per_group: int, window: int, window_stat: str) -> None:
    ax.set_ylabel("source tile")
    ax.set_xlabel(window_axis_label(window, window_stat))
    tick_count = min(10, len(cycles))
    tick_positions = np.linspace(0, len(cycles) - 1, tick_count, dtype=int)
    ax.set_xticks(tick_positions, [str(cycles[position]) for position in tick_positions])
    if len(tiles) <= 20:
        ax.set_yticks(range(len(tiles)), [str(tile) for tile in tiles])
    else:
        tick_count_y = min(16, len(tiles))
        y_positions = np.linspace(0, len(tiles) - 1, tick_count_y, dtype=int)
        ax.set_yticks(y_positions, [str(tiles[position]) for position in y_positions])
    add_group_separators(ax, tiles, tiles_per_group)


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


def heatmap_style(max_cores: int, colors: list[str], averaged: bool = False) -> tuple[ListedColormap, BoundaryNorm]:
    return fanin_cmap_norm(max_cores, colors, averaged)


def plot_total_heatmap(
    counts: dict[tuple[int, int, str], float],
    cycles: list[int],
    tiles: list[int],
    output_base: Path,
    formats: list[str],
    force: bool,
    port: int,
    max_cores: int,
    tiles_per_group: int,
    window: int,
    window_stat: str,
) -> list[Path]:
    averaged = window > 1 and window_stat == "mean"
    cmap, norm = heatmap_style(max_cores, ["#f7fbff", "#c6dbef", "#6baed6", "#2171b5", "#08306b"], averaged)
    height = max(4.4, min(12.0, 2.2 + 0.12 * len(tiles)))
    fig, ax = plt.subplots(figsize=(13.0, height))
    image = ax.imshow(
        build_matrix(counts, cycles, tiles, "total"),
        aspect="auto",
        interpolation="nearest",
        origin="lower",
        cmap=cmap,
        norm=norm,
    )
    if window == 1:
        suffix = ""
    elif window_stat == "mean":
        suffix = f" ({window}-cycle average, 0.25-step)"
    else:
        suffix = f" ({window}-cycle peak)"
    ax.set_title(f"Per-Tile Source-Core Requests for Port {port}{suffix}")
    configure_heatmap_axis(ax, cycles, tiles, tiles_per_group, window, window_stat)
    add_fanin_colorbar(
        fig,
        image,
        ax,
        max_cores,
        averaged,
        f"cores requesting port {port}" if window_stat == "max" else f"avg cores requesting port {port} per cycle (nearest 0.25)",
        pad=0.012,
    )
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def plot_operand_heatmaps(
    counts: dict[tuple[int, int, str], float],
    cycles: list[int],
    tiles: list[int],
    output_base: Path,
    formats: list[str],
    force: bool,
    port: int,
    max_cores: int,
    tiles_per_group: int,
    window: int,
    window_stat: str,
) -> list[Path]:
    averaged = window > 1 and window_stat == "mean"
    cmap, norm = heatmap_style(max_cores, ["#fff7ec", "#fdd49e", "#fc8d59", "#d7301f", "#7f0000"], averaged)
    fig_height = max(7.5, min(14.0, 4.2 + 0.08 * len(tiles)))
    fig, axes = plt.subplots(
        len(OPERANDS),
        1,
        figsize=(13.0, fig_height),
        sharex=True,
        constrained_layout=True,
    )
    image = None
    for ax, operand in zip(axes, OPERANDS):
        image = ax.imshow(
            build_matrix(counts, cycles, tiles, operand),
            aspect="auto",
            interpolation="nearest",
            origin="lower",
            cmap=cmap,
            norm=norm,
        )
        if window == 1:
            suffix = ""
        elif window_stat == "mean":
            suffix = f" ({window}-cycle average, 0.25-step)"
        else:
            suffix = f" ({window}-cycle peak)"
        ax.set_title(f"{operand} requests on port {port}{suffix}")
        configure_heatmap_axis(ax, cycles, tiles, tiles_per_group, window, window_stat)
    assert image is not None
    add_fanin_colorbar(
        fig,
        image,
        list(axes),
        max_cores,
        averaged,
        f"cores requesting port {port}" if window_stat == "max" else f"avg cores requesting port {port} per cycle (nearest 0.25)",
        pad=0.012,
        shrink=0.86,
    )
    return save_figure(fig, output_base, formats, force)


def ensure_can_write(output_path: Path, force: bool) -> None:
    if output_path.exists() and not force:
        raise SystemExit(f"Refusing to overwrite existing file: {output_path} (use --force)")


def format_count(value: float) -> int | str:
    if value.is_integer():
        return int(value)
    return f"{value:.6g}"


def write_csv(
    output_path: Path,
    counts: dict[tuple[int, int, str], float],
    cycles: list[int],
    windows: list[tuple[int, int, int]],
    tiles: list[int],
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    windowed = any(window_cycles > 1 for _, _, window_cycles in windows)
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
                    row[column] = format_count(counts[(cycle, tile, column)])
                writer.writerow(row)


def write_summary(
    output_path: Path,
    counts: dict[tuple[int, int, str], float],
    cycles: list[int],
    windows: list[tuple[int, int, int]],
    tiles: list[int],
    window_stat: str,
    force: bool,
) -> None:
    ensure_can_write(output_path, force)
    window = max(window_cycles for _, _, window_cycles in windows)
    windowed = any(window_cycles > 1 for _, _, window_cycles in windows)
    window_cycles_by_start = {window_start: window_cycles for window_start, _, window_cycles in windows}
    tile_summary = []
    all_values = []
    for tile in tiles:
        values = [counts[(cycle, tile, "total")] for cycle in cycles]
        all_values.extend(values)
        high_2 = sum(1 for value in values if value >= 2)
        high_3 = sum(1 for value in values if value >= 3)
        high_4 = sum(1 for value in values if value >= 4)
        tile_summary.append((tile, sum(values), high_2, high_3, high_4, max(values) if values else 0))
    tile_sample_count = len(cycles) * len(tiles)
    with output_path.open("w") as file:
        file.write(f"cycles={windows[0][0]}..{windows[-1][1]}\n")
        file.write(f"window={window}\n")
        if not windowed:
            file.write("value=valid_request_count\n")
        elif window_stat == "mean":
            file.write("value=avg_valid_requests_per_cycle\n")
        else:
            file.write("value=peak_valid_requests_per_cycle_window\n")
        file.write(f"tiles={tiles[0]}..{tiles[-1]} count={len(tiles)}\n")
        if not windowed:
            hist = Counter(int(value) for value in all_values)
            file.write(f"tile_cycle_distribution_0_to_4={[hist[index] for index in range(5)]}\n")
        else:
            if window_stat == "mean":
                values_array = np.array(all_values, dtype=float)
                file.write(
                    "tile_window_value_stats="
                    f"min={values_array.min():.3f} mean={values_array.mean():.3f} "
                    f"p95={np.percentile(values_array, 95):.3f} max={values_array.max():.3f}\n"
                )
            else:
                hist = Counter(int(value) for value in all_values)
                file.write(f"tile_window_peak_distribution_0_to_4={[hist[index] for index in range(5)]}\n")
        for threshold in (2, 3, 4):
            high_count = sum(1 for value in all_values if value >= threshold)
            pct = 100.0 * high_count / tile_sample_count if tile_sample_count else 0.0
            sample_name = "tile_windows" if windowed else "tile_cycles"
            file.write(f"{sample_name}_ge_{threshold}={high_count} ({pct:.2f}%)\n")
        if windowed and window_stat == "mean":
            file.write("\ntop_tiles_by_ge2_avg_fanin\n")
        elif windowed:
            file.write("\ntop_tiles_by_ge2_peak_fanin\n")
        else:
            file.write("\ntop_tiles_by_ge2_fanin\n")
        for tile, total, high_2, high_3, high_4, max_value in sorted(tile_summary, key=lambda item: (item[2], item[1]), reverse=True)[:20]:
            if windowed and window_stat == "mean":
                file.write(f"tile={tile} sum_avg={total:.3f} ge2={high_2} ge3={high_3} ge4={high_4} peak_avg={max_value:.3f}\n")
            elif windowed:
                file.write(f"tile={tile} sum_peak={int(total)} ge2={high_2} ge3={high_3} ge4={high_4} peak={int(max_value)}\n")
            else:
                file.write(f"tile={tile} total={int(total)} ge2={high_2} ge3={high_3} ge4={high_4} max={int(max_value)}\n")
        file.write("\noperand_total_valid_requests\n")
        for operand in OPERANDS:
            total = sum(
                counts[(cycle, tile, operand)] * window_cycles_by_start[cycle]
                for cycle in cycles
                for tile in tiles
            )
            file.write(f"{operand}={int(round(total))}\n")


def main() -> int:
    args = parse_args()
    if args.max_cores <= 0:
        raise SystemExit("--max-cores must be > 0")
    if args.window < 1:
        raise SystemExit("--window must be >= 1")
    graph_dir = resolve_graph_dir(args.input_path)
    operand_regions = load_operand_regions(graph_dir, args)
    explicit_tiles = parse_int_list(args.tile)
    tiles = discover_tiles(graph_dir, explicit_tiles, args.group, args.tiles_per_group)
    counts, observed_min, observed_max = read_counts(
        graph_dir,
        tiles,
        args.cycle_start,
        args.cycle_end,
        args.port,
        args.node_point,
        operand_regions,
    )
    display_start = args.cycle_start if args.cycle_start is not None else observed_min
    display_end = args.cycle_end if args.cycle_end is not None else observed_max
    cycles = list(range(display_start, display_end + 1))
    if not cycles:
        raise SystemExit("No cycles to plot")
    plot_counts, plot_cycles, windows = aggregate_counts(counts, cycles, tiles, args.window, args.window_stat)

    output_dir = args.output_dir or default_port_pressure_dir(graph_dir)
    if args.group is not None:
        output_dir = output_dir / f"group{args.group}"
    if explicit_tiles and len(explicit_tiles) == 1:
        output_dir = output_dir / f"tile{explicit_tiles[0]}"

    prefix = args.prefix.replace("port0", f"port{args.port}")
    written = []
    written.extend(plot_total_heatmap(
        plot_counts,
        plot_cycles,
        tiles,
        output_dir / f"{prefix}_total_heatmap",
        args.formats,
        args.force,
        args.port,
        args.max_cores,
        args.tiles_per_group,
        args.window,
        args.window_stat,
    ))
    written.extend(plot_operand_heatmaps(
        plot_counts,
        plot_cycles,
        tiles,
        output_dir / f"{prefix}_operand_heatmap",
        args.formats,
        args.force,
        args.port,
        args.max_cores,
        args.tiles_per_group,
        args.window,
        args.window_stat,
    ))
    csv_path = data_path(output_dir, f"{prefix}_heatmap.csv")
    summary_path = data_path(output_dir, f"{prefix}_summary.txt")
    write_csv(csv_path, plot_counts, plot_cycles, windows, tiles, args.force)
    write_summary(summary_path, plot_counts, plot_cycles, windows, tiles, args.window_stat, args.force)

    print("Wrote port fan-in heatmap outputs:")
    for path in written:
        print(f"  {path}")
    print(f"  {csv_path}")
    print(f"  {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
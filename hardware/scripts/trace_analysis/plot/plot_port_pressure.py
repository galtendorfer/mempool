#!/usr/bin/env python3
"""Plot per-port request and blocked pressure from route checkpoint CSVs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, StrMethodFormatter

from _plot_output_paths import data_path, figure_path


PORT_COLORS = {
    0: "#0072B2",
    1: "#D55E00",
    2: "#009E73",
    3: "#CC79A7",
}


@dataclass(frozen=True)
class PlotScope:
    label: str
    source_tiles: set[int] | None
    output_subdir: str


@dataclass(frozen=True)
class CycleWindow:
    start: int
    end: int
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot request and blocked counts per route port over time and in aggregate.",
    )
    parser.add_argument(
        "input_path",
        type=Path,
        help="route_bottlenecks_all_tiles.csv, a path_graph directory, or a result directory containing analysis/path_graph",
    )
    parser.add_argument(
        "--metric-source",
        choices=("route-summary", "node-state"),
        default="route-summary",
        help="count route-summary rows or valid/stalled/fired observations from cycle_node_state.csv",
    )
    parser.add_argument(
        "--node-point",
        default="tcdm_remote",
        help="cycle_node_state point to count when --metric-source=node-state",
    )
    parser.add_argument(
        "--per-tile-average",
        action="store_true",
        help="plot counts divided by the number of source tiles in the scope",
    )
    parser.add_argument(
        "--slots-per-tile",
        type=int,
        help="maximum request slots per source tile per cycle for utilization axes; defaults to 4 for per-core node points and 1 otherwise",
    )
    parser.add_argument(
        "--port",
        action="append",
        help="only include route port(s), comma-separated; may be repeated",
    )
    parser.add_argument("--cycle-start", type=int, help="first cycle to include")
    parser.add_argument("--cycle-end", type=int, help="last cycle to include")
    parser.add_argument(
        "--average-section",
        type=int,
        help=(
            "use this stall_timeseries_benchmark.csv section as the metric/denominator window; "
            "--cycle-start/--cycle-end remain the visual plot bounds"
        ),
    )
    parser.add_argument(
        "--average-cycle-start",
        type=int,
        help="first cycle for metric/denominator calculations; defaults to --average-section or the visual bounds",
    )
    parser.add_argument(
        "--average-cycle-end",
        type=int,
        help="last cycle for metric/denominator calculations; defaults to --average-section or the visual bounds",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=50,
        help="cycle aggregation window for the time-series plot; use 1 for exact per-cycle counts",
    )
    parser.add_argument(
        "--num-tiles",
        type=int,
        help="number of source tiles that can use each port; defaults to distinct source_tile values in the CSV",
    )
    parser.add_argument("--tiles-per-group", type=int, default=16, help="source tiles per group")
    parser.add_argument("--group", type=int, help="only include source tiles from this group")
    parser.add_argument("--all-groups", action="store_true", help="write one scoped plot set per source group")
    parser.add_argument("--tile", type=int, help="only include this absolute source tile ID")
    parser.add_argument("--tile-local", type=int, help="only include this tile index within --group")
    parser.add_argument("--output-dir", type=Path, help="output directory; defaults beside the input CSV")
    parser.add_argument("--prefix", default="port_requests_blocked", help="output filename prefix")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="figure formats to write",
    )
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser.parse_args()


def parse_int_list(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    parsed: set[int] = set()
    for value in values:
        for part in value.split(","):
            part = part.strip()
            if part:
                parsed.add(int(part, 0))
    return parsed


def resolve_input_csv(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    direct = input_path / "route_bottlenecks_all_tiles.csv"
    if direct.is_file():
        return direct
    nested = input_path / "analysis" / "path_graph" / "route_bottlenecks_all_tiles.csv"
    if nested.is_file():
        return nested
    raise SystemExit(f"Could not find route_bottlenecks_all_tiles.csv from {input_path}")


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


def default_port_pressure_dir(path: Path) -> Path:
    for candidate in (path, *path.parents):
        if candidate.name == "path_graph" and candidate.parent.name == "analysis":
            return candidate.parent.parent / "plots" / "port_pressure"
    return path / "plots" / "port_pressure"


def result_dir_from_path(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if candidate.name == "path_graph" and candidate.parent.name == "analysis":
            return candidate.parent.parent
        if (candidate / "data" / "stall_timeseries_benchmark.csv").is_file():
            return candidate
    return None


def section_cycle_window(result_dir: Path | None, section: int) -> CycleWindow:
    if result_dir is None:
        raise SystemExit("--average-section requires an input under a benchmark result directory")
    csv_path = result_dir / "data" / "stall_timeseries_benchmark.csv"
    if not csv_path.is_file():
        raise SystemExit(f"--average-section requires {csv_path}")

    first_cycle: int | None = None
    last_cycle: int | None = None
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"section", "cycle"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                row_section = int(row["section"])
                cycle = int(row["cycle"])
            except ValueError:
                continue
            if row_section != section:
                continue
            first_cycle = cycle if first_cycle is None else min(first_cycle, cycle)
            last_cycle = cycle if last_cycle is None else max(last_cycle, cycle)

    if first_cycle is None or last_cycle is None:
        raise SystemExit(f"No rows for section {section} in {csv_path}")
    return CycleWindow(first_cycle, last_cycle, f"section {section} from stall_timeseries_benchmark.csv")


def resolve_average_window(
    args: argparse.Namespace,
    section_window: CycleWindow | None,
    display_start: int,
    display_end: int,
) -> CycleWindow:
    source_parts: list[str] = []
    if section_window is not None:
        average_start = section_window.start
        average_end = section_window.end
        source_parts.append(section_window.source)
    else:
        average_start = display_start
        average_end = display_end
        source_parts.append("visual cycle bounds")

    if args.average_cycle_start is not None:
        average_start = args.average_cycle_start
        source_parts.append("--average-cycle-start")
    if args.average_cycle_end is not None:
        average_end = args.average_cycle_end
        source_parts.append("--average-cycle-end")
    if average_end < average_start:
        raise SystemExit(f"Average window end is before start: {average_start}..{average_end}")
    return CycleWindow(average_start, average_end, " + ".join(source_parts))


def window_start(cycle: int, cycle_start: int, window: int) -> int:
    return cycle_start + ((cycle - cycle_start) // window) * window


def read_counts(
    csv_path: Path,
    cycle_start_filter: int | None,
    cycle_end_filter: int | None,
    window: int,
    source_tiles_filter: set[int] | None,
    ports_filter: set[int] | None,
) -> tuple[
    Counter[tuple[int, int]],
    Counter[tuple[int, int]],
    Counter[tuple[int, int]],
    Counter[int],
    Counter[int],
    Counter[int],
    int,
    int,
    set[int],
]:
    if window < 1:
        raise SystemExit("--window must be >= 1")

    raw_rows: list[tuple[int, int, bool, bool]] = []
    observed_min: int | None = None
    observed_max: int | None = None
    observed_source_tiles: set[int] = set()
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "source_tile", "port", "source_master_state", "source_master_fire"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                cycle = int(row["cycle"])
                source_tile = int(row["source_tile"])
                port = int(row["port"])
            except ValueError:
                continue
            if cycle_start_filter is not None and cycle < cycle_start_filter:
                continue
            if cycle_end_filter is not None and cycle > cycle_end_filter:
                continue
            if source_tiles_filter is not None and source_tile not in source_tiles_filter:
                continue
            if ports_filter is not None and port not in ports_filter:
                continue
            if port < 0:
                continue
            is_blocked = row["source_master_state"] == "blocked"
            is_fired = row["source_master_fire"] == "1"
            raw_rows.append((cycle, port, is_blocked, is_fired))
            observed_source_tiles.add(source_tile)
            observed_min = cycle if observed_min is None else min(observed_min, cycle)
            observed_max = cycle if observed_max is None else max(observed_max, cycle)

    if observed_min is None or observed_max is None:
        raise SystemExit("No route rows matched the requested cycle range")

    origin = cycle_start_filter if cycle_start_filter is not None else observed_min
    requests_by_window: Counter[tuple[int, int]] = Counter()
    blocked_by_window: Counter[tuple[int, int]] = Counter()
    fired_by_window: Counter[tuple[int, int]] = Counter()
    requests_by_port: Counter[int] = Counter()
    blocked_by_port: Counter[int] = Counter()
    fired_by_port: Counter[int] = Counter()
    for cycle, port, is_blocked, is_fired in raw_rows:
        start = window_start(cycle, origin, window)
        requests_by_window[(start, port)] += 1
        requests_by_port[port] += 1
        if is_blocked:
            blocked_by_window[(start, port)] += 1
            blocked_by_port[port] += 1
        if is_fired:
            fired_by_window[(start, port)] += 1
            fired_by_port[port] += 1
    return (
        requests_by_window,
        blocked_by_window,
        fired_by_window,
        requests_by_port,
        blocked_by_port,
        fired_by_port,
        observed_min,
        observed_max,
        observed_source_tiles,
    )


def read_node_counts(
    node_csv_path: Path,
    cycle_start_filter: int | None,
    cycle_end_filter: int | None,
    window: int,
    source_tiles_filter: set[int] | None,
    node_point: str,
    ports_filter: set[int] | None,
) -> tuple[
    Counter[tuple[int, int]],
    Counter[tuple[int, int]],
    Counter[tuple[int, int]],
    Counter[int],
    Counter[int],
    Counter[int],
    int,
    int,
    set[int],
]:
    if window < 1:
        raise SystemExit("--window must be >= 1")

    raw_rows: list[tuple[int, int, int, int, int]] = []
    observed_min: int | None = None
    observed_max: int | None = None
    observed_source_tiles: set[int] = set()
    with node_csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "tile", "port", "point", "valid", "stall", "fire"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {node_csv_path}: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("point") != node_point:
                continue
            try:
                cycle = int(row["cycle"])
                source_tile = int(row["tile"])
                port = int(row["port"])
            except ValueError:
                continue
            if cycle_start_filter is not None and cycle < cycle_start_filter:
                continue
            if cycle_end_filter is not None and cycle > cycle_end_filter:
                continue
            if source_tiles_filter is not None and source_tile not in source_tiles_filter:
                continue
            if ports_filter is not None and port not in ports_filter:
                continue
            if port < 0:
                continue
            valid_count = int(row.get("valid") or 0)
            stall_count = int(row.get("stall") or 0)
            fire_count = int(row.get("fire") or 0)
            raw_rows.append((cycle, port, valid_count, stall_count, fire_count))
            observed_source_tiles.add(source_tile)
            observed_min = cycle if observed_min is None else min(observed_min, cycle)
            observed_max = cycle if observed_max is None else max(observed_max, cycle)

    if observed_min is None or observed_max is None:
        raise SystemExit("No node-state rows matched the requested cycle range")

    origin = cycle_start_filter if cycle_start_filter is not None else observed_min
    requests_by_window: Counter[tuple[int, int]] = Counter()
    blocked_by_window: Counter[tuple[int, int]] = Counter()
    fired_by_window: Counter[tuple[int, int]] = Counter()
    requests_by_port: Counter[int] = Counter()
    blocked_by_port: Counter[int] = Counter()
    fired_by_port: Counter[int] = Counter()
    for cycle, port, valid_count, stall_count, fire_count in raw_rows:
        start = window_start(cycle, origin, window)
        requests_by_window[(start, port)] += valid_count
        blocked_by_window[(start, port)] += stall_count
        fired_by_window[(start, port)] += fire_count
        requests_by_port[port] += valid_count
        blocked_by_port[port] += stall_count
        fired_by_port[port] += fire_count
    return (
        requests_by_window,
        blocked_by_window,
        fired_by_window,
        requests_by_port,
        blocked_by_port,
        fired_by_port,
        observed_min,
        observed_max,
        observed_source_tiles,
    )


def build_window_axis(first_cycle: int, last_cycle: int, origin: int, window: int) -> list[int]:
    first_window = window_start(first_cycle, origin, window)
    last_window = window_start(last_cycle, origin, window)
    return list(range(first_window, last_window + 1, window))


def cycles_in_window(window_start_cycle: int, display_start: int, display_end: int, window: int) -> int:
    window_end_cycle = window_start_cycle + window - 1
    overlap_start = max(window_start_cycle, display_start)
    overlap_end = min(window_end_cycle, display_end)
    return max(0, overlap_end - overlap_start + 1)


def discover_source_tiles(csv_path: Path, cycle_start_filter: int | None, cycle_end_filter: int | None) -> set[int]:
    source_tiles: set[int] = set()
    with csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "source_tile"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {csv_path}: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                cycle = int(row["cycle"])
                source_tile = int(row["source_tile"])
            except ValueError:
                continue
            if cycle_start_filter is not None and cycle < cycle_start_filter:
                continue
            if cycle_end_filter is not None and cycle > cycle_end_filter:
                continue
            source_tiles.add(source_tile)
    return source_tiles


def discover_node_source_tiles(
    node_csv_path: Path,
    cycle_start_filter: int | None,
    cycle_end_filter: int | None,
    node_point: str,
) -> set[int]:
    source_tiles: set[int] = set()
    with node_csv_path.open(newline="") as file:
        reader = csv.DictReader(file)
        required = {"cycle", "tile", "point"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Missing required columns in {node_csv_path}: {', '.join(sorted(missing))}")
        for row in reader:
            if row.get("point") != node_point:
                continue
            try:
                cycle = int(row["cycle"])
                source_tile = int(row["tile"])
            except ValueError:
                continue
            if cycle_start_filter is not None and cycle < cycle_start_filter:
                continue
            if cycle_end_filter is not None and cycle > cycle_end_filter:
                continue
            source_tiles.add(source_tile)
    return source_tiles


def group_tiles(group: int, tiles_per_group: int) -> set[int]:
    first_tile = group * tiles_per_group
    return set(range(first_tile, first_tile + tiles_per_group))


def build_scopes(args: argparse.Namespace, observed_source_tiles: set[int]) -> list[PlotScope]:
    if args.tiles_per_group <= 0:
        raise SystemExit("--tiles-per-group must be > 0")
    if args.all_groups and (args.group is not None or args.tile is not None or args.tile_local is not None):
        raise SystemExit("--all-groups cannot be combined with --group, --tile, or --tile-local")
    if args.tile is not None and args.tile_local is not None:
        raise SystemExit("Use either --tile for an absolute tile or --tile-local with --group, not both")
    if args.tile_local is not None and args.group is None:
        raise SystemExit("--tile-local requires --group")
    if args.tile_local is not None and not 0 <= args.tile_local < args.tiles_per_group:
        raise SystemExit(f"--tile-local must be in [0, {args.tiles_per_group - 1}]")

    if args.all_groups:
        groups = sorted({tile // args.tiles_per_group for tile in observed_source_tiles})
        return [PlotScope(f"group {group}", group_tiles(group, args.tiles_per_group), f"group{group}") for group in groups]

    if args.tile is not None:
        return [PlotScope(f"source tile {args.tile}", {args.tile}, f"tile{args.tile}")]

    if args.tile_local is not None:
        source_tile = args.group * args.tiles_per_group + args.tile_local
        return [PlotScope(f"group {args.group}, tile {args.tile_local} (source tile {source_tile})", {source_tile}, f"group{args.group}_tile{args.tile_local}")]

    if args.group is not None:
        return [PlotScope(f"group {args.group}", group_tiles(args.group, args.tiles_per_group), f"group{args.group}")]

    return [PlotScope("all source tiles", None, "")]


def scoped_title(title: str, scope_label: str) -> str:
    if scope_label == "all source tiles":
        return title
    return f"{title} ({scope_label})"


def style_axes(ax, formatter: str = "{x:,.0f}") -> None:
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_major_formatter(StrMethodFormatter(formatter))


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


def ensure_data_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise SystemExit(f"Refusing to overwrite existing output: {path} (use --force)")


def pressure_summary_paths(output_dir: Path, prefix: str) -> list[Path]:
    return [
        data_path(output_dir, f"{prefix}_summary.csv"),
        data_path(output_dir, f"{prefix}_caption.txt"),
    ]


def pressure_caption(
    ports: list[int],
    metric_source: str,
    node_point: str,
    display_start: int,
    display_end: int,
    average_window: CycleWindow,
    all_observed_ports: bool,
) -> str:
    source_text = (
        f"{node_point} monitor valid/stall/fire observations"
        if metric_source == "node-state"
        else "route checkpoint source-master observations"
    )
    if all_observed_ports:
        role = (
            "Route-port pressure guardrail figure. Port-specific operand/source-target plots show "
            "what goes through one selected port; this all-port view shows whether pressure is "
            "removed or shifted to other route ports."
        )
    elif len(ports) > 1:
        port_list = ", ".join(str(port) for port in ports)
        role = (
            f"Route-port pressure drilldown for selected ports {port_list}. Use an unfiltered "
            "all-port guardrail figure before making whole-system pressure claims."
        )
    else:
        role = (
            f"Route-port pressure drilldown for port {ports[0]}. Use together with the all-port "
            "guardrail figure before making whole-system pressure claims."
        )
    return (
        f"{role}\n"
        f"Source: {source_text}.\n"
        f"Visual cycle window: {display_start}..{display_end}.\n"
        f"Metric/denominator window: {average_window.start}..{average_window.end} ({average_window.source}).\n"
        "Requests are valid observations, accepted requests are fire handshakes, "
        "and blocked traffic is counted as stalled request-cycles."
    )


def write_pressure_summary(
    output_dir: Path,
    prefix: str,
    force: bool,
    requests_by_port: Counter[int],
    blocked_by_port: Counter[int],
    fired_by_port: Counter[int],
    ports: list[int],
    scope_label: str,
    metric_source: str,
    node_point: str,
    window: int,
    num_tiles: int,
    slots_per_tile: int,
    display_start: int,
    display_end: int,
    average_window: CycleWindow,
    all_observed_ports: bool,
) -> list[Path]:
    summary_path, caption_path = pressure_summary_paths(output_dir, prefix)
    ensure_data_outputs([summary_path, caption_path], force)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    cycles = max(0, average_window.end - average_window.start + 1)
    capacity = num_tiles * slots_per_tile * cycles
    if all_observed_ports:
        thesis_role = "main_guardrail"
    elif len(ports) > 1:
        thesis_role = "multi_port_drilldown"
    else:
        thesis_role = "port_drilldown"
    fieldnames = [
        "scope",
        "thesis_role",
        "metric_source",
        "node_point",
        "cycle_start",
        "cycle_end",
        "display_cycle_start",
        "display_cycle_end",
        "average_cycle_start",
        "average_cycle_end",
        "average_window_source",
        "window",
        "observed_source_tiles",
        "slots_per_tile",
        "source_slot_capacity",
        "port",
        "valid_observations",
        "blocked_request_cycles",
        "accepted_requests",
        "blocked_per_valid",
        "accepted_per_valid",
        "requested_source_slot_share",
        "accepted_source_slot_share",
        "note",
    ]
    if all_observed_ports:
        note = "all-port guardrail: shows pressure distribution across route ports"
    elif len(ports) > 1:
        note = "selected-port drilldown: pair with all-port guardrail"
    else:
        note = "single-port drilldown: pair with all-port guardrail"
    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        rows: list[tuple[str, int, int, int]] = [
            (
                str(port),
                requests_by_port[port],
                blocked_by_port[port],
                fired_by_port[port],
            )
            for port in ports
        ]
        if len(ports) > 1:
            rows.append((
                "all",
                sum(requests_by_port[port] for port in ports),
                sum(blocked_by_port[port] for port in ports),
                sum(fired_by_port[port] for port in ports),
            ))
        for port_label, valid_count, blocked_count, accepted_count in rows:
            writer.writerow({
                "scope": scope_label,
                "thesis_role": thesis_role,
                "metric_source": metric_source,
                "node_point": node_point if metric_source == "node-state" else "",
                "cycle_start": display_start,
                "cycle_end": display_end,
                "display_cycle_start": display_start,
                "display_cycle_end": display_end,
                "average_cycle_start": average_window.start,
                "average_cycle_end": average_window.end,
                "average_window_source": average_window.source,
                "window": window,
                "observed_source_tiles": num_tiles,
                "slots_per_tile": slots_per_tile,
                "source_slot_capacity": capacity,
                "port": port_label,
                "valid_observations": valid_count,
                "blocked_request_cycles": blocked_count,
                "accepted_requests": accepted_count,
                "blocked_per_valid": f"{blocked_count / valid_count:.6f}" if valid_count else "0.000000",
                "accepted_per_valid": f"{accepted_count / valid_count:.6f}" if valid_count else "0.000000",
                "requested_source_slot_share": f"{valid_count / capacity:.6f}" if capacity else "0.000000",
                "accepted_source_slot_share": f"{accepted_count / capacity:.6f}" if capacity else "0.000000",
                "note": note,
            })
    caption_path.write_text(
        pressure_caption(ports, metric_source, node_point, display_start, display_end, average_window, all_observed_ports) + "\n",
        encoding="utf-8",
    )
    return [summary_path, caption_path]


def plot_timeseries(
    output_base: Path,
    formats: list[str],
    force: bool,
    requests_by_window: Counter[tuple[int, int]],
    blocked_by_window: Counter[tuple[int, int]],
    ports: list[int],
    windows: list[int],
    window: int,
    scope_label: str,
    per_tile_average: bool,
    num_tiles: int,
    slots_per_tile: int,
) -> list[Path]:
    fig, axes = plt.subplots(2, 1, figsize=(11.5, 6.2), sharex=True)
    axis_label = "cycle" if window == 1 else f"{window}-cycle window start"
    if per_tile_average:
        y_label = "avg count / tile / cycle" if window == 1 else f"avg count / tile / {window} cycles"
    else:
        y_label = "count / cycle" if window == 1 else f"count / {window} cycles"
    formatter = "{x:,.2f}" if per_tile_average else "{x:,.0f}"
    expected_max = slots_per_tile * window if per_tile_average else None

    for port in ports:
        color = PORT_COLORS.get(port, "#555555")
        if per_tile_average:
            req_values = [requests_by_window[(cycle, port)] / num_tiles for cycle in windows]
            blocked_values = [blocked_by_window[(cycle, port)] / num_tiles for cycle in windows]
        else:
            req_values = [requests_by_window[(cycle, port)] for cycle in windows]
            blocked_values = [blocked_by_window[(cycle, port)] for cycle in windows]
        axes[0].plot(windows, req_values, color=color, linewidth=1.6, label=f"port {port}")
        axes[1].plot(windows, blocked_values, color=color, linewidth=1.6, label=f"port {port}")

    axes[0].set_title(scoped_title("Route Requests Per Port Over Time", scope_label))
    axes[0].set_ylabel(y_label)
    axes[1].set_title(scoped_title("Blocked Route Requests Per Port Over Time", scope_label))
    axes[1].set_ylabel(y_label)
    axes[1].set_xlabel(axis_label)
    for ax in axes:
        style_axes(ax, formatter)
        if expected_max is not None:
            observed_top = max((line.get_ydata().max() for line in ax.lines), default=0)
            ax.set_ylim(0, max(expected_max, observed_top) * 1.05)
        ax.legend(ncol=min(len(ports), 4), frameon=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def plot_aggregate(
    output_base: Path,
    formats: list[str],
    force: bool,
    requests_by_port: Counter[int],
    blocked_by_port: Counter[int],
    ports: list[int],
    scope_label: str,
    per_tile_average: bool,
    num_tiles: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), gridspec_kw={"width_ratios": [2.2, 1.0]})
    x_positions = list(range(len(ports)))
    width = 0.36
    if per_tile_average:
        request_values = [requests_by_port[port] / num_tiles for port in ports]
        blocked_values = [blocked_by_port[port] / num_tiles for port in ports]
    else:
        request_values = [requests_by_port[port] for port in ports]
        blocked_values = [blocked_by_port[port] for port in ports]
    blocked_pct = [100.0 * blocked_by_port[port] / requests_by_port[port] if requests_by_port[port] else 0.0 for port in ports]

    axes[0].bar([x - width / 2 for x in x_positions], request_values, width, label="requests", color="#9ecae1")
    axes[0].bar([x + width / 2 for x in x_positions], blocked_values, width, label="blocked", color="#de2d26")
    axes[0].set_title(scoped_title("Total Requests and Blocked Requests Per Port", scope_label))
    axes[0].set_ylabel("avg count / tile" if per_tile_average else "count")
    axes[0].set_xticks(x_positions, [f"port {port}" for port in ports])
    axes[0].legend(frameon=False)
    style_axes(axes[0], "{x:,.2f}" if per_tile_average else "{x:,.0f}")
    for x, request_count, blocked_count, pct in zip(x_positions, request_values, blocked_values, blocked_pct):
        axes[0].text(x + width / 2, blocked_count, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5)
        request_label = f"{request_count:.1f}" if per_tile_average else f"{request_count:,.0f}"
        axes[0].text(x - width / 2, request_count, request_label, ha="center", va="bottom", fontsize=8)

    axes[1].bar(x_positions, blocked_pct, color=[PORT_COLORS.get(port, "#555555") for port in ports], width=0.58)
    axes[1].set_title(scoped_title("Blocked Share", scope_label))
    axes[1].set_ylabel("blocked requests [%]")
    axes[1].set_xticks(x_positions, [f"p{port}" for port in ports])
    axes[1].yaxis.set_major_locator(MaxNLocator(nbins=6))
    style_axes(axes[1])
    for x, pct in zip(x_positions, blocked_pct):
        axes[1].text(x, pct, f"{pct:.1f}%", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def plot_utilization(
    output_base: Path,
    formats: list[str],
    force: bool,
    requests_by_window: Counter[tuple[int, int]],
    ports: list[int],
    windows: list[int],
    window: int,
    num_tiles: int,
    display_start: int,
    display_end: int,
    average_start: int,
    average_end: int,
    blocked_by_window: Counter[tuple[int, int]],
    fired_by_window: Counter[tuple[int, int]],
    scope_label: str,
    slots_per_tile: int,
) -> list[Path]:
    fig, axes = plt.subplots(4, 1, figsize=(11.5, 9.2), sharex=True)
    axis_label = "cycle" if window == 1 else f"{window}-cycle window start"

    for port in ports:
        color = PORT_COLORS.get(port, "#555555")
        demand_values: list[float] = []
        fired_values: list[float] = []
        blocked_values: list[float] = []
        no_request_values: list[float] = []
        for cycle in windows:
            capacity = num_tiles * slots_per_tile * cycles_in_window(cycle, average_start, average_end, window)
            request_count = requests_by_window[(cycle, port)]
            fired_count = fired_by_window[(cycle, port)]
            blocked_count = blocked_by_window[(cycle, port)]
            demand = 100.0 * request_count / capacity if capacity else 0.0
            demand_values.append(demand)
            fired_values.append(100.0 * fired_count / capacity if capacity else 0.0)
            blocked_values.append(100.0 * blocked_count / request_count if request_count else 0.0)
            no_request_values.append(max(0.0, 100.0 - demand))
        axes[0].plot(windows, demand_values, color=color, linewidth=1.6, label=f"port {port}")
        axes[1].plot(windows, fired_values, color=color, linewidth=1.6, label=f"port {port}")
        axes[2].plot(windows, blocked_values, color=color, linewidth=1.6, label=f"port {port}")
        axes[3].plot(windows, no_request_values, color=color, linewidth=1.6, label=f"port {port}")

    axes[0].set_title(scoped_title("Port Request Demand Over Time", scope_label))
    axes[0].set_ylabel("requested slots [%]")
    axes[1].set_title(scoped_title("Accepted Port Requests Over Time", scope_label))
    axes[1].set_ylabel("fired slots [%]")
    axes[2].set_title(scoped_title("Blocked Share of Requested Port Traffic Over Time", scope_label))
    axes[2].set_ylabel("blocked / requested [%]")
    axes[3].set_title(scoped_title("No-Request Source-Port Capacity Over Time", scope_label))
    axes[3].set_ylabel("no request slots [%]")
    axes[3].set_xlabel(axis_label)
    for ax in axes:
        style_axes(ax)
        ax.set_ylim(0, 100)
        ax.legend(ncol=min(len(ports), 4), frameon=False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    fig.tight_layout()
    return save_figure(fig, output_base, formats, force)


def default_slots_per_tile(args: argparse.Namespace) -> int:
    if args.slots_per_tile is not None:
        if args.slots_per_tile <= 0:
            raise SystemExit("--slots-per-tile must be > 0")
        return args.slots_per_tile
    if args.metric_source == "node-state" and args.node_point in {"core_q", "tcdm_preroute", "tcdm_remote"}:
        return 4
    return 1


def render_scope(
    args: argparse.Namespace,
    input_path: Path,
    base_output_dir: Path,
    scope: PlotScope,
    section_window: CycleWindow | None,
) -> list[Path]:
    output_dir = base_output_dir / scope.output_subdir if scope.output_subdir else base_output_dir
    ports_filter = parse_int_list(args.port)
    if args.metric_source == "node-state":
        counts = read_node_counts(
            input_path / "cycle_node_state.csv",
            args.cycle_start,
            args.cycle_end,
            args.window,
            scope.source_tiles,
            args.node_point,
            ports_filter,
        )
    else:
        counts = read_counts(input_path, args.cycle_start, args.cycle_end, args.window, scope.source_tiles, ports_filter)
    (
        requests_by_window,
        blocked_by_window,
        fired_by_window,
        requests_by_port,
        blocked_by_port,
        fired_by_port,
        first_cycle,
        last_cycle,
        observed_source_tiles,
    ) = counts
    ports = sorted(requests_by_port)
    if not ports:
        raise SystemExit(f"No route rows matched the requested port filter for {scope.label}")
    origin = args.cycle_start if args.cycle_start is not None else first_cycle
    display_start = args.cycle_start if args.cycle_start is not None else first_cycle
    display_end = args.cycle_end if args.cycle_end is not None else last_cycle
    average_window = resolve_average_window(args, section_window, display_start, display_end)
    windows = build_window_axis(display_start, display_end, origin, args.window)
    if args.num_tiles is not None:
        num_tiles = args.num_tiles
    elif scope.source_tiles is not None:
        num_tiles = len(scope.source_tiles)
    else:
        num_tiles = len(observed_source_tiles)
    if num_tiles <= 0:
        raise SystemExit(f"Could not infer --num-tiles from the CSV for {scope.label}")
    slots_per_tile = default_slots_per_tile(args)
    aggregate_outputs_enabled = ports_filter is None
    if aggregate_outputs_enabled:
        ensure_data_outputs(pressure_summary_paths(output_dir, args.prefix), args.force)

    if average_window.start == display_start and average_window.end == display_end:
        metric_requests_by_port = requests_by_port
        metric_blocked_by_port = blocked_by_port
        metric_fired_by_port = fired_by_port
    elif args.metric_source == "node-state":
        metric_counts = read_node_counts(
            input_path / "cycle_node_state.csv",
            average_window.start,
            average_window.end,
            args.window,
            scope.source_tiles,
            args.node_point,
            ports_filter,
        )
        metric_requests_by_port = metric_counts[3]
        metric_blocked_by_port = metric_counts[4]
        metric_fired_by_port = metric_counts[5]
    else:
        metric_counts = read_counts(
            input_path,
            average_window.start,
            average_window.end,
            args.window,
            scope.source_tiles,
            ports_filter,
        )
        metric_requests_by_port = metric_counts[3]
        metric_blocked_by_port = metric_counts[4]
        metric_fired_by_port = metric_counts[5]

    written = []
    written.extend(plot_timeseries(
        output_dir / f"{args.prefix}_timeseries",
        args.formats,
        args.force,
        requests_by_window,
        blocked_by_window,
        ports,
        windows,
        args.window,
        scope.label,
        args.per_tile_average,
        num_tiles,
        slots_per_tile,
    ))
    if aggregate_outputs_enabled:
        written.extend(plot_aggregate(
            output_dir / f"{args.prefix}_aggregate",
            args.formats,
            args.force,
            metric_requests_by_port,
            metric_blocked_by_port,
            ports,
            scope.label,
            args.per_tile_average,
            num_tiles,
        ))
    written.extend(plot_utilization(
        output_dir / f"{args.prefix}_utilization",
        args.formats,
        args.force,
        requests_by_window,
        ports,
        windows,
        args.window,
        num_tiles,
        display_start,
        display_end,
        average_window.start,
        average_window.end,
        blocked_by_window,
        fired_by_window,
        scope.label,
        slots_per_tile,
    ))
    if aggregate_outputs_enabled:
        written.extend(write_pressure_summary(
            output_dir,
            args.prefix,
            args.force,
            metric_requests_by_port,
            metric_blocked_by_port,
            metric_fired_by_port,
            ports,
            scope.label,
            args.metric_source,
            args.node_point,
            args.window,
            num_tiles,
            slots_per_tile,
            display_start,
            display_end,
            average_window,
            len(ports) > 1,
        ))
    return written


def main() -> None:
    args = parse_args()
    if args.metric_source == "node-state":
        graph_dir = resolve_graph_dir(args.input_path)
        input_path = graph_dir
        output_dir = args.output_dir or default_port_pressure_dir(graph_dir)
        result_dir = result_dir_from_path(graph_dir)
        observed_source_tiles = discover_node_source_tiles(
            graph_dir / "cycle_node_state.csv",
            args.cycle_start,
            args.cycle_end,
            args.node_point,
        )
    else:
        csv_path = resolve_input_csv(args.input_path)
        input_path = csv_path
        output_dir = args.output_dir or default_port_pressure_dir(csv_path.parent)
        result_dir = result_dir_from_path(csv_path)
        observed_source_tiles = discover_source_tiles(csv_path, args.cycle_start, args.cycle_end)
    section_window = section_cycle_window(result_dir, args.average_section) if args.average_section is not None else None
    scopes = build_scopes(args, observed_source_tiles)
    written = []
    for scope in scopes:
        written.extend(render_scope(args, input_path, output_dir, scope, section_window))
    print("Wrote port pressure plots:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()

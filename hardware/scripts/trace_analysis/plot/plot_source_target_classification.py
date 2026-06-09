#!/usr/bin/env python3
"""Plot source-target classification CSVs.

This is the figure companion for `classify_source_targets.py`.  It reads a
`*_source_target_matrix.csv` file and produces compact views of which source
tiles target which local tiles, and which traffic classes dominate by signed
relative destination offset.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

from _plot_output_paths import figure_path

TRACE_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(TRACE_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(TRACE_ANALYSIS_DIR))

from operand_regions import add_classified_operand_provenance_args, validate_classified_operand_provenance


CLASS_ORDER = (
    "A_local_owner",
    "A_neighbor",
    "A_other",
    "B_same_tile",
    "B_same_group",
    "B_remote",
    "other",
)

CLASS_COLORS = {
    "A_local_owner": "#0072B2",
    "A_neighbor": "#E68600",
    "A_other": "#56B4E9",
    "B_same_tile": "#009E73",
    "B_same_group": "#6F63C6",
    "B_remote": "#9B59B6",
    "other": "#9E9E9E",
}

CLASS_LABELS = {
    "A_local_owner": "A: local owner",
    "A_neighbor": "A: neighbor tile",
    "A_other": "A: other",
    "B_same_tile": "B: same tile",
    "B_same_group": "B: same group",
    "B_remote": "B: remote",
    "other": "other",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_path",
        type=Path,
        help="classification directory or *_source_target_matrix.csv from classify_source_targets.py",
    )
    parser.add_argument("--output-dir", type=Path, help="directory for figures; defaults beside the matrix CSV")
    parser.add_argument("--prefix", help="output filename prefix; defaults to the matrix CSV stem without _source_target_matrix")
    parser.add_argument("--tiles-per-group", type=int, default=16, help="tile slots per group for local source/target axes")
    parser.add_argument(
        "--offset-mode",
        choices=("raw", "wrapped"),
        default="raw",
        help="plot raw target-source offsets or wrapped shortest offsets within the group",
    )
    parser.add_argument(
        "--combine",
        action="store_true",
        help="when input_path contains multiple matrix CSVs, combine them into one all-port view",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        choices=("png", "pdf"),
        help="figure formats to write",
    )
    add_classified_operand_provenance_args(parser)
    parser.add_argument("--force", action="store_true", help="overwrite existing output files")
    return parser.parse_args()


def parse_int(value: str | None, default: int = 0) -> int:
    if value in (None, ""):
        return default
    return int(value, 0)


def resolve_matrix_csvs(input_path: Path, combine: bool) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    matches = sorted(input_path.glob("*_source_target_matrix.csv"))
    matches.extend(sorted(input_path.glob("port*_source_target_classification/*_source_target_matrix.csv")))
    nested_graph = input_path / "analysis" / "path_graph"
    if nested_graph.is_dir():
        matches.extend(sorted(nested_graph.glob("port*_source_target_classification/*_source_target_matrix.csv")))
    matches = sorted(set(matches))
    if not matches:
        raise SystemExit(f"No *_source_target_matrix.csv found in {input_path}")
    if len(matches) > 1 and not combine:
        joined = "\n  ".join(str(path) for path in matches)
        raise SystemExit(f"Multiple matrix CSVs found; pass one explicitly or use --combine:\n  {joined}")
    return matches if combine else [matches[0]]


def default_prefix(matrix_csv: Path) -> str:
    stem = matrix_csv.stem
    suffix = "_source_target_matrix"
    if stem.endswith(suffix):
        return stem[: -len(suffix)]
    return stem


def default_output_dir(matrix_csv: Path) -> Path:
    for candidate in (matrix_csv.parent, *matrix_csv.parents):
        if candidate.name == "path_graph" and candidate.parent.name == "analysis":
            return candidate.parent.parent / "plots" / "port_pressure"
    return matrix_csv.parent / "plots"


def read_matrix_rows(matrix_csvs: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    required = {
        "source_tile_in_group",
        "target_tile_in_group",
        "request_class",
        "requests",
        "stalls",
        "fires",
        "high_fanin_requests",
    }
    for matrix_csv in matrix_csvs:
        with matrix_csv.open(newline="") as file:
            reader = csv.DictReader(file)
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise SystemExit(f"Missing required columns in {matrix_csv}: {', '.join(sorted(missing))}")
            rows.extend(reader)
    return rows


def wrapped_signed_offset(source_tile_in_group: int, target_tile_in_group: int, tiles_per_group: int) -> int:
    offset = target_tile_in_group - source_tile_in_group
    while offset <= -(tiles_per_group // 2):
        offset += tiles_per_group
    while offset > tiles_per_group // 2:
        offset -= tiles_per_group
    return offset


def signed_offset(
    source_tile_in_group: int,
    target_tile_in_group: int,
    tiles_per_group: int,
    offset_mode: str,
) -> int:
    if offset_mode == "wrapped":
        return wrapped_signed_offset(source_tile_in_group, target_tile_in_group, tiles_per_group)
    return target_tile_in_group - source_tile_in_group


def offset_values(tiles_per_group: int, offset_mode: str) -> list[int]:
    if offset_mode == "wrapped":
        return list(range(-(tiles_per_group // 2) + 1, tiles_per_group // 2 + 1))
    return list(range(-(tiles_per_group - 1), tiles_per_group))


def offset_axis_label(offset_mode: str) -> str:
    if offset_mode == "wrapped":
        return "wrapped signed destination offset inside group (shortest target - source)"
    return "raw signed destination offset inside group: target tile - source tile"


def sorted_classes(classes: set[str]) -> list[str]:
    ordered = [name for name in CLASS_ORDER if name in classes]
    ordered.extend(sorted(classes.difference(ordered)))
    return ordered


def ensure_outputs(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise SystemExit(f"Output exists: {path} (use --force to overwrite)")


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, formats: list[str], force: bool) -> list[Path]:
    paths = [figure_path(output_dir / stem, fmt) for fmt in formats]
    ensure_outputs(paths, force)
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return paths


def aggregate_rows(rows: list[dict[str, str]], tiles_per_group: int, offset_mode: str) -> dict[str, Counter]:
    matrix_requests: Counter[tuple[int, int]] = Counter()
    matrix_stalls: Counter[tuple[int, int]] = Counter()
    by_offset_class_requests: Counter[tuple[int, str]] = Counter()
    by_offset_class_stalls: Counter[tuple[int, str]] = Counter()
    by_offset_class_fires: Counter[tuple[int, str]] = Counter()
    by_offset_class_high_fanin: Counter[tuple[int, str]] = Counter()
    by_class_requests: Counter[str] = Counter()
    by_class_stalls: Counter[str] = Counter()
    by_class_fires: Counter[str] = Counter()
    by_class_high_fanin: Counter[str] = Counter()

    for row in rows:
        source_local = parse_int(row.get("source_tile_in_group"), -1)
        target_local = parse_int(row.get("target_tile_in_group"), -1)
        if source_local < 0 or target_local < 0:
            continue
        request_class = row.get("request_class") or "other"
        requests = parse_int(row.get("requests"))
        stalls = parse_int(row.get("stalls"))
        fires = parse_int(row.get("fires"))
        high_fanin = parse_int(row.get("high_fanin_requests"))

        matrix_requests[(source_local, target_local)] += requests
        matrix_stalls[(source_local, target_local)] += stalls
        offset = signed_offset(source_local, target_local, tiles_per_group, offset_mode)
        by_offset_class_requests[(offset, request_class)] += requests
        by_offset_class_stalls[(offset, request_class)] += stalls
        by_offset_class_fires[(offset, request_class)] += fires
        by_offset_class_high_fanin[(offset, request_class)] += high_fanin
        by_class_requests[request_class] += requests
        by_class_stalls[request_class] += stalls
        by_class_fires[request_class] += fires
        by_class_high_fanin[request_class] += high_fanin

    return {
        "matrix_requests": matrix_requests,
        "matrix_stalls": matrix_stalls,
        "by_offset_class_requests": by_offset_class_requests,
        "by_offset_class_stalls": by_offset_class_stalls,
        "by_offset_class_fires": by_offset_class_fires,
        "by_offset_class_high_fanin": by_offset_class_high_fanin,
        "by_class_requests": by_class_requests,
        "by_class_stalls": by_class_stalls,
        "by_class_fires": by_class_fires,
        "by_class_high_fanin": by_class_high_fanin,
    }


def infer_port_label(rows: list[dict[str, str]]) -> str:
    ports = sorted({
        parse_int(row.get("port"), -1)
        for row in rows
        if parse_int(row.get("port"), -1) >= 0
    })
    if not ports:
        return "route-port traffic"
    if len(ports) == 1:
        return f"port {ports[0]}"
    if ports == list(range(ports[0], ports[-1] + 1)):
        return f"ports {ports[0]}-{ports[-1]}"
    return "ports " + ", ".join(str(port) for port in ports)


def plot_matrix(counter: Counter[tuple[int, int]], tiles_per_group: int, title: str, colorbar_label: str) -> plt.Figure:
    grid = [
        [counter[(source_local, target_local)] for target_local in range(tiles_per_group)]
        for source_local in range(tiles_per_group)
    ]
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    image = ax.imshow(grid, origin="lower", aspect="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Target tile in source group")
    ax.set_ylabel("Source tile in group")
    ax.set_xticks(range(tiles_per_group))
    ax.set_yticks(range(tiles_per_group))
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(colorbar_label)
    fig.tight_layout()
    return fig


def class_label(request_class: str) -> str:
    return CLASS_LABELS.get(request_class, request_class)


def format_k(value: float, _position: int | None = None) -> str:
    if abs(value) < 0.5:
        return "0"
    if abs(value) >= 1000:
        return f"{value / 1000:.0f}k"
    return f"{value:.0f}"


def format_rate(value: float, _position: int | None = None) -> str:
    return f"{100 * value:.0f}%"


def total_for_offset(counter: Counter[tuple[int, str]], offset: int, classes: list[str]) -> int:
    return sum(counter[(offset, request_class)] for request_class in classes)


def plot_stacked_offset_counts(
    ax: plt.Axes,
    counter: Counter[tuple[int, str]],
    offsets: list[int],
    classes: list[str],
    ylabel: str,
    y_limit: float | None = None,
    annotate_large: bool = True,
) -> None:
    bottoms = [0] * len(offsets)
    for request_class in classes:
        values = [counter[(offset, request_class)] for offset in offsets]
        ax.bar(
            offsets,
            values,
            bottom=bottoms,
            label=class_label(request_class),
            color=CLASS_COLORS.get(request_class, None),
            width=0.72,
        )
        bottoms = [left + right for left, right in zip(bottoms, values)]
    max_total = max(bottoms, default=0)
    if max_total == 0:
        ax.set_ylim(0, 1)
        ax.set_yticks([0])
        ax.text(0, 0.5, "no events observed", ha="center", va="center", color="0.45", fontsize=14)
    elif annotate_large:
        for offset, total in zip(offsets, bottoms):
            if total >= max_total * 0.28:
                ax.text(offset, total * 1.015, format_k(total), ha="center", va="bottom", fontsize=14)
    if y_limit is not None:
        ax.set_ylim(0, y_limit)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(format_k))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=7)


def plot_offset_rate(
    ax: plt.Axes,
    numerator: Counter[tuple[int, str]],
    denominator: Counter[tuple[int, str]],
    offsets: list[int],
    classes: list[str],
    ylabel: str,
    color: str,
) -> None:
    rates: list[float] = []
    for offset in offsets:
        total = total_for_offset(denominator, offset, classes)
        value = total_for_offset(numerator, offset, classes)
        rates.append(value / total if total else 0.0)

    ax.bar(offsets, rates, width=0.72, color=color)
    max_rate = max(rates, default=0.0)
    if max_rate == 0:
        ax.set_ylim(0, 0.05)
        ax.text(0, 0.025, "none observed", ha="center", va="center", color="0.45", fontsize=14)
    else:
        ax.set_ylim(0, min(1.0, max(0.05, max_rate * 1.18)))
        for offset, rate in zip(offsets, rates):
            if rate >= max_rate * 0.65 and rate > 0.01:
                ax.text(offset, rate * 1.025, f"{100 * rate:.1f}%", ha="center", va="bottom", fontsize=12)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(FuncFormatter(format_rate))
    ax.grid(axis="y", alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.4)
    ax.spines["bottom"].set_linewidth(1.4)
    ax.tick_params(axis="both", labelsize=15, width=1.2, length=7)


def plot_offset_classes(
    requests: Counter[tuple[int, str]],
    stalls: Counter[tuple[int, str]],
    fires: Counter[tuple[int, str]],
    high_fanin: Counter[tuple[int, str]],
    classes: list[str],
    tiles_per_group: int,
    port_label: str,
    offset_mode: str,
) -> plt.Figure:
    offsets = offset_values(tiles_per_group, offset_mode)
    count_rows: list[tuple[str, Counter[tuple[int, str]]]] = [
        ("valid requests", requests),
        ("accepted requests", fires),
        ("blocked requests", stalls),
    ]
    if sum(high_fanin.values()) > 0:
        count_rows.append(("requests during contention", high_fanin))

    rate_rows: list[tuple[str, Counter[tuple[int, str]], Counter[tuple[int, str]], str]] = [
        ("blocked share", stalls, requests, "#D55E00"),
    ]
    if sum(high_fanin.values()) > 0:
        rate_rows.append(("contention share", high_fanin, requests, "#CC79A7"))

    all_count_totals = [
        total_for_offset(counter, offset, classes)
        for _, counter in count_rows
        for offset in offsets
    ]
    count_y_limit = max(all_count_totals, default=0) * 1.14
    if count_y_limit == 0:
        count_y_limit = None

    num_rows = len(count_rows) + len(rate_rows)
    fig_height = 2.45 * num_rows + 1.7
    fig, axes = plt.subplots(
        num_rows,
        1,
        figsize=(16.8, fig_height),
        sharex=True,
    )
    if num_rows == 1:
        axes = [axes]
    title = f"{port_label.capitalize()} traffic by {offset_mode} destination offset and class"
    fig.suptitle(title, fontsize=26, y=0.99)
    total_requests = sum(requests.values())
    total_fires = sum(fires.values())
    total_stalls = sum(stalls.values())
    total_high_fanin = sum(high_fanin.values())
    summary = (
        f"valid {total_requests:,} · accepted {total_fires:,} · "
        f"blocked {total_stalls:,} ({total_stalls / total_requests:.1%}) · "
        f"contention {total_high_fanin:,} ({total_high_fanin / total_requests:.1%})"
    ) if total_requests else "no valid requests observed"
    fig.text(0.5, 0.958, summary, ha="center", va="center", fontsize=13, color="0.35")
    for ax in axes:
        for offset in (-1, 1):
            if offset in offsets:
                ax.axvspan(offset - 0.5, offset + 0.5, color="#F3C04A", alpha=0.14, zorder=0)
        ax.set_xlim(offsets[0] - 0.5, offsets[-1] + 0.5)

    for ax, (ylabel, counter) in zip(axes, count_rows):
        plot_stacked_offset_counts(ax, counter, offsets, classes, ylabel, y_limit=count_y_limit)

    for ax, (ylabel, numerator, denominator, color) in zip(axes[len(count_rows):], rate_rows):
        plot_offset_rate(ax, numerator, denominator, offsets, classes, ylabel, color)

    axes[0].legend(loc="upper right", ncol=min(3, len(classes)), frameon=False, fontsize=16)
    axes[-1].set_xlabel(offset_axis_label(offset_mode))
    axes[-1].set_xticks(offsets)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_class_totals(
    requests: Counter[str],
    stalls: Counter[str],
    fires: Counter[str],
    high_fanin: Counter[str],
    classes: list[str],
) -> plt.Figure:
    positions = list(range(len(classes)))
    width = 0.22
    fig, ax = plt.subplots(figsize=(11.0, 5.8))
    ax.bar([pos - 1.5 * width for pos in positions], [requests[name] for name in classes], width, label="requests", color="#0072B2")
    ax.bar([pos - 0.5 * width for pos in positions], [stalls[name] for name in classes], width, label="stalls", color="#D55E00")
    ax.bar([pos + 0.5 * width for pos in positions], [fires[name] for name in classes], width, label="fires", color="#009E73")
    ax.bar([pos + 1.5 * width for pos in positions], [high_fanin[name] for name in classes], width, label="contention requests", color="#CC79A7")
    ax.set_title("Request-class totals")
    ax.set_ylabel("Events")
    ax.set_xticks(positions)
    ax.set_xticklabels([class_label(name) for name in classes], rotation=30, ha="right")
    ax.legend()
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    matrix_csvs = resolve_matrix_csvs(args.input_path, args.combine)
    validate_classified_operand_provenance(matrix_csvs, args.allow_legacy_route_operands)
    output_dir = args.output_dir or default_output_dir(matrix_csvs[0])
    if args.prefix:
        prefix = args.prefix
    elif len(matrix_csvs) > 1:
        prefix = "all_ports_source_target"
    else:
        prefix = default_prefix(matrix_csvs[0])

    rows = read_matrix_rows(matrix_csvs)
    if not rows:
        raise SystemExit(f"No rows found in {', '.join(str(path) for path in matrix_csvs)}")
    aggregates = aggregate_rows(rows, args.tiles_per_group, args.offset_mode)
    classes = sorted_classes(set(aggregates["by_class_requests"].keys()))
    if not classes:
        raise SystemExit(f"No classified request rows found in {', '.join(str(path) for path in matrix_csvs)}")
    port_label = infer_port_label(rows)

    written: list[Path] = []
    written.extend(save_figure(
        plot_matrix(
            aggregates["matrix_requests"],
            args.tiles_per_group,
            "Source-target request heatmap",
            "Requests",
        ),
        output_dir,
        f"{prefix}_source_target_requests",
        args.formats,
        args.force,
    ))
    written.extend(save_figure(
        plot_matrix(
            aggregates["matrix_stalls"],
            args.tiles_per_group,
            "Source-target stall heatmap",
            "Stalls",
        ),
        output_dir,
        f"{prefix}_source_target_stalls",
        args.formats,
        args.force,
    ))
    written.extend(save_figure(
        plot_offset_classes(
            aggregates["by_offset_class_requests"],
            aggregates["by_offset_class_stalls"],
            aggregates["by_offset_class_fires"],
            aggregates["by_offset_class_high_fanin"],
            classes,
            args.tiles_per_group,
            port_label,
            args.offset_mode,
        ),
        output_dir,
        f"{prefix}_traffic_class_by_signed_offset",
        args.formats,
        args.force,
    ))
    written.extend(save_figure(
        plot_class_totals(
            aggregates["by_class_requests"],
            aggregates["by_class_stalls"],
            aggregates["by_class_fires"],
            aggregates["by_class_high_fanin"],
            classes,
        ),
        output_dir,
        f"{prefix}_request_class_totals",
        args.formats,
        args.force,
    ))

    for path in written:
        print(path)


if __name__ == "__main__":
    main()

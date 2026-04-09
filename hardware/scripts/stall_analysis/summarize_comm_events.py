#!/usr/bin/env python3
"""Summarize extracted communication events into a few analysis-ready CSVs.

This is the first lightweight analysis layer on top of
`comm_events_benchmark.csv`. It writes three summary tables:

1. source_dest_counts.csv
   Long-form source tile -> destination tile event counts.
2. source_tile_locality.csv
   Local vs remote event breakdown per source tile.
3. dest_tile_load_latency.csv
   Load-return latency statistics per destination tile.

The script accepts either a benchmark result directory or a direct path to a
communication-event CSV.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path


def _parse_int(value: str | None):
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None):
    if value is None or value == "":
        return None
    return float(value)


def _percent(numerator: int, denominator: int) -> float:
    return (100.0 * numerator / denominator) if denominator else 0.0


def _percentile(sorted_values: list[float], q: float):
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lower = math.floor(pos)
    upper = math.ceil(pos)
    if lower == upper:
        return sorted_values[lower]
    weight = pos - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight


def _sortable(value):
    if value is None:
        return (-1, "")
    if isinstance(value, str):
        return (0, value)
    return (0, value)


def _resolve_input(path_str: str) -> tuple[Path, Path]:
    path = Path(path_str).resolve()
    if path.is_dir():
        csv_path = path / "data" / "comm_events_benchmark.csv"
        output_dir = path / "data" / "comm_summary"
    else:
        csv_path = path
        output_dir = csv_path.parent / "comm_summary"
    return csv_path, output_dir


def _load_rows(csv_path: Path, *, sections: set[int] | None):
    rows = []
    seen_sections = set()
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            section = _parse_int(row.get("section"))
            if section is not None:
                seen_sections.add(section)
            if sections and section not in sections:
                continue
            rows.append({
                "section": section,
                "cycle": _parse_int(row.get("cycle")),
                "core": _parse_int(row.get("core")),
                "group": _parse_int(row.get("group")),
                "subgroup": _parse_int(row.get("subgroup")),
                "tile": _parse_int(row.get("tile")),
                "event_type": (row.get("event_type") or "").strip(),
                "region": (row.get("region") or "").strip(),
                "dest_tile": _parse_int(row.get("dest_tile")),
                "dest_group": _parse_int(row.get("dest_group")),
                "dest_subgroup": _parse_int(row.get("dest_subgroup")),
                "is_local": _parse_int(row.get("is_local")),
                "is_same_group": _parse_int(row.get("is_same_group")),
                "is_same_subgroup": _parse_int(row.get("is_same_subgroup")),
                "latency": _parse_float(row.get("latency")),
            })
    return rows, seen_sections


def _build_source_dest_counts(rows: list[dict]):
    counts = Counter()
    for row in rows:
        key = (
            row["section"],
            row["tile"],
            row["group"],
            row["subgroup"],
            row["dest_tile"],
            row["dest_group"],
            row["dest_subgroup"],
            row["region"],
            row["event_type"],
        )
        counts[key] += 1

    out_rows = []
    for key in sorted(counts, key=lambda item: tuple(_sortable(part) for part in item)):
        section, tile, group, subgroup, dest_tile, dest_group, dest_subgroup, region, event_type = key
        out_rows.append({
            "section": section,
            "source_tile": tile,
            "source_group": group,
            "source_subgroup": "" if subgroup is None else subgroup,
            "dest_tile": "" if dest_tile is None else dest_tile,
            "dest_group": "" if dest_group is None else dest_group,
            "dest_subgroup": "" if dest_subgroup is None else dest_subgroup,
            "region": region,
            "event_type": event_type,
            "count": counts[key],
        })
    return out_rows


def _build_source_tile_locality(rows: list[dict]):
    grouped = defaultdict(lambda: Counter())
    for row in rows:
        key = (row["section"], row["tile"], row["group"], row["subgroup"])
        grouped[key]["total_events"] += 1
        grouped[key][f"event_type::{row['event_type']}"] += 1
        if row["is_local"] == 1:
            grouped[key]["local_events"] += 1
        elif row["is_same_subgroup"] == 1:
            grouped[key]["same_subgroup_events"] += 1
            grouped[key]["remote_events"] += 1
        elif row["is_local"] == 0:
            if row["is_same_group"] == 1:
                grouped[key]["same_group_other_subgroup_events"] += 1
            else:
                grouped[key]["remote_group_events"] += 1
            grouped[key]["remote_events"] += 1
        else:
            grouped[key]["unknown_destination_events"] += 1

    out_rows = []
    for key in sorted(grouped, key=lambda item: tuple(_sortable(part) for part in item)):
        section, tile, group, subgroup = key
        counter = grouped[key]
        total = counter["total_events"]
        local = counter["local_events"]
        same_subgroup = counter["same_subgroup_events"]
        same_group_other_subgroup = counter["same_group_other_subgroup_events"]
        remote_group = counter["remote_group_events"]
        remote = counter["remote_events"]
        unknown = counter["unknown_destination_events"]
        out_rows.append({
            "section": section,
            "source_tile": tile,
            "source_group": group,
            "source_subgroup": "" if subgroup is None else subgroup,
            "total_events": total,
            "local_events": local,
            "same_subgroup_events": same_subgroup,
            "same_group_other_subgroup_events": same_group_other_subgroup,
            "remote_group_events": remote_group,
            "remote_events": remote,
            "unknown_destination_events": unknown,
            "local_pct": f"{_percent(local, total):.2f}",
            "same_subgroup_pct": f"{_percent(same_subgroup, total):.2f}",
            "same_group_other_subgroup_pct": f"{_percent(same_group_other_subgroup, total):.2f}",
            "remote_group_pct": f"{_percent(remote_group, total):.2f}",
            "remote_pct": f"{_percent(remote, total):.2f}",
            "load_issue_count": counter["event_type::load_issue"],
            "store_issue_count": counter["event_type::store_issue"],
            "load_return_count": counter["event_type::load_return"],
        })
    return out_rows


def _build_dest_tile_latency(rows: list[dict]):
    grouped = defaultdict(list)
    for row in rows:
        if row["event_type"] != "load_return":
            continue
        if row["dest_tile"] is None:
            continue
        if row["latency"] is None:
            continue
        key = (row["section"], row["dest_tile"], row["dest_group"], row["dest_subgroup"], row["region"])
        grouped[key].append(float(row["latency"]))

    out_rows = []
    for key in sorted(grouped, key=lambda item: tuple(_sortable(part) for part in item)):
        section, dest_tile, dest_group, dest_subgroup, region = key
        values = sorted(grouped[key])
        count = len(values)
        avg = sum(values) / count if count else None
        out_rows.append({
            "section": section,
            "dest_tile": dest_tile,
            "dest_group": "" if dest_group is None else dest_group,
            "dest_subgroup": "" if dest_subgroup is None else dest_subgroup,
            "region": region,
            "load_return_count": count,
            "avg_latency": f"{avg:.3f}" if avg is not None else "",
            "min_latency": f"{values[0]:.3f}" if values else "",
            "p50_latency": f"{_percentile(values, 0.50):.3f}" if values else "",
            "p95_latency": f"{_percentile(values, 0.95):.3f}" if values else "",
            "max_latency": f"{values[-1]:.3f}" if values else "",
        })
    return out_rows


def _write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        if not fieldnames:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_console_summary(source_dest_rows: list[dict], locality_rows: list[dict], latency_rows: list[dict]):
    print(f"Source-destination pairs: {len(source_dest_rows)}")
    print(f"Per-source locality rows: {len(locality_rows)}")
    print(f"Per-destination latency rows: {len(latency_rows)}")

    if locality_rows:
        hottest_remote = max(locality_rows, key=lambda row: int(row["remote_events"]))
        print(
            "Worst remote source tile: "
            f"tile {hottest_remote['source_tile']} (section {hottest_remote['section']}) -> "
            f"{hottest_remote['remote_events']} remote events / {hottest_remote['remote_pct']}%"
        )

    if latency_rows:
        slowest = max(latency_rows, key=lambda row: float(row["avg_latency"]) if row["avg_latency"] else -1.0)
        print(
            "Slowest destination tile: "
            f"tile {slowest['dest_tile']} (section {slowest['section']}) -> "
            f"avg latency {slowest['avg_latency']} cycles"
        )


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Summarize extracted communication events into a few compact CSV tables."
    )
    parser.add_argument(
        "input_path",
        help="Benchmark result directory or a direct path to comm_events_benchmark.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for the summary CSVs (default: <result_dir>/data/comm_summary or <csv-dir>/comm_summary)",
    )
    parser.add_argument(
        "--section",
        type=int,
        action="append",
        help="Only summarize the specified section(s)",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Shortcut for --section 1",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    csv_path, default_output_dir = _resolve_input(args.input_path)
    output_dir = Path(args.output_dir).resolve() if args.output_dir else default_output_dir

    if not csv_path.is_file():
        raise SystemExit(f"Communication CSV not found: {csv_path}")

    sections = set(args.section or [])
    if args.benchmark_only:
        sections.add(1)

    rows, seen_sections = _load_rows(csv_path, sections=sections or None)
    if not rows:
        if sections:
            available = ", ".join(str(section) for section in sorted(seen_sections)) or "none"
            requested = ", ".join(str(section) for section in sorted(sections))
            raise SystemExit(
                f"No communication rows left after filtering; requested section(s): {requested}; available section(s): {available}"
            )
        raise SystemExit("No communication rows found in the input CSV")

    source_dest_rows = _build_source_dest_counts(rows)
    locality_rows = _build_source_tile_locality(rows)
    latency_rows = _build_dest_tile_latency(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    source_dest_path = output_dir / "source_dest_counts.csv"
    locality_path = output_dir / "source_tile_locality.csv"
    latency_path = output_dir / "dest_tile_load_latency.csv"

    _write_csv(source_dest_path, source_dest_rows)
    _write_csv(locality_path, locality_rows)
    _write_csv(latency_path, latency_rows)

    print(f"Input CSV:  {csv_path}")
    print(f"Output dir: {output_dir}")
    print(f"Wrote {source_dest_path}")
    print(f"Wrote {locality_path}")
    print(f"Wrote {latency_path}")
    _print_console_summary(source_dest_rows, locality_rows, latency_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
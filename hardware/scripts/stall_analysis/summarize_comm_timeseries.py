#!/usr/bin/env python3
"""Generate windowed communication timeseries tables from comm_events CSV.

This script is the temporal counterpart to summarize_comm_events.py.
It keeps the time axis so later tile/core plots can be aligned with
communication behavior.

Outputs:
  1. comm_timeseries_tiles.csv
     Per-window, per-tile aggregate communication metrics.
  2. comm_timeseries_edges.csv
     Per-window, per-source-tile -> dest-tile communication counts.
  3. comm_timeseries_metadata.json
     Windowing and layout metadata for later plot splicing/alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def _parse_int(value: str | None):
    if value is None or value == "":
        return None
    return int(value)


def _parse_float(value: str | None):
    if value is None or value == "":
        return None
    return float(value)


def _resolve_paths(input_path: str):
    path = Path(input_path).resolve()
    if path.is_dir():
        csv_path = path / "data" / "comm_events_benchmark.csv"
        output_dir = path / "data" / "comm_timeseries"
        result_dir = path
    else:
        csv_path = path
        output_dir = csv_path.parent / "comm_timeseries"
        result_dir = csv_path.parent.parent if csv_path.parent.name == "data" else None
    return csv_path, output_dir, result_dir


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


def _window_bounds(cycle: int, min_cycle: int, window: int):
    window_index = (cycle - min_cycle) // window
    start_cycle = min_cycle + window_index * window
    end_cycle = start_cycle + window - 1
    center_cycle = start_cycle + window / 2.0
    return window_index, start_cycle, end_cycle, center_cycle


def _latency_fields(total_latency: float, samples: int):
    avg_latency = total_latency / samples if samples else None
    return {
        "avg_latency": f"{avg_latency:.6f}" if avg_latency is not None else "",
        "latency_samples": samples,
    }


def _build_tile_rows(rows: list[dict], *, window: int):
    min_cycle = min(row["cycle"] for row in rows if row["cycle"] is not None)
    max_cycle = max(row["cycle"] for row in rows if row["cycle"] is not None)

    grouped = defaultdict(lambda: {
        "source_group": None,
        "source_subgroup": None,
        "outgoing_events": 0,
        "outgoing_load_issue": 0,
        "outgoing_store_issue": 0,
        "load_returns_seen": 0,
        "incoming_events": 0,
        "incoming_load_issue": 0,
        "incoming_store_issue": 0,
        "incoming_load_returns": 0,
        "local_events": 0,
        "same_subgroup_events": 0,
        "same_group_events": 0,
        "remote_group_events": 0,
        "unknown_dest_events": 0,
        "remote_outgoing_events": 0,
        "local_outgoing_events": 0,
        "outgoing_latency_total": 0.0,
        "outgoing_latency_samples": 0,
        "incoming_latency_total": 0.0,
        "incoming_latency_samples": 0,
    })

    for row in rows:
        cycle = row["cycle"]
        source_tile = row["tile"]
        if cycle is None or source_tile is None:
            continue
        window_index, start_cycle, end_cycle, center_cycle = _window_bounds(cycle, min_cycle, window)

        source_key = (row["section"], window_index, source_tile)
        source_bucket = grouped[source_key]
        source_bucket["source_group"] = row["group"]
        source_bucket["source_subgroup"] = row["subgroup"]
        source_bucket["outgoing_events"] += 1
        event_type = row["event_type"]
        if event_type == "load_issue":
            source_bucket["outgoing_load_issue"] += 1
        elif event_type == "store_issue":
            source_bucket["outgoing_store_issue"] += 1
        elif event_type == "load_return":
            source_bucket["load_returns_seen"] += 1

        if row["is_local"] == 1:
            source_bucket["local_events"] += 1
            source_bucket["local_outgoing_events"] += 1
        elif row["is_same_subgroup"] == 1:
            source_bucket["same_subgroup_events"] += 1
            source_bucket["same_group_events"] += 1
            source_bucket["remote_outgoing_events"] += 1
        elif row["is_same_group"] == 1:
            source_bucket["same_group_events"] += 1
            source_bucket["remote_outgoing_events"] += 1
        elif row["dest_tile"] is None:
            source_bucket["unknown_dest_events"] += 1
        else:
            source_bucket["remote_group_events"] += 1
            source_bucket["remote_outgoing_events"] += 1

        if row["latency"] is not None and event_type == "load_return":
            source_bucket["outgoing_latency_total"] += float(row["latency"])
            source_bucket["outgoing_latency_samples"] += 1

        dest_tile = row["dest_tile"]
        if dest_tile is None:
            continue
        dest_key = (row["section"], window_index, dest_tile)
        dest_bucket = grouped[dest_key]
        dest_bucket["source_group"] = row["dest_group"]
        dest_bucket["incoming_events"] += 1
        if event_type == "load_issue":
            dest_bucket["incoming_load_issue"] += 1
        elif event_type == "store_issue":
            dest_bucket["incoming_store_issue"] += 1
        elif event_type == "load_return":
            dest_bucket["incoming_load_returns"] += 1
        if row["latency"] is not None and event_type == "load_return":
            dest_bucket["incoming_latency_total"] += float(row["latency"])
            dest_bucket["incoming_latency_samples"] += 1

    out_rows = []
    for key in sorted(grouped):
        section, window_index, tile = key
        bucket = grouped[key]
        start_cycle = min_cycle + window_index * window
        end_cycle = start_cycle + window - 1
        center_cycle = start_cycle + window / 2.0
        base = {
            "section": section,
            "window_index": window_index,
            "window_start_cycle": start_cycle,
            "window_end_cycle": end_cycle,
            "window_center_cycle": f"{center_cycle:.1f}",
            "window_size": window,
            "tile": tile,
            "group": "" if bucket["source_group"] is None else bucket["source_group"],
            "subgroup": "" if bucket["source_subgroup"] is None else bucket["source_subgroup"],
            "outgoing_events": bucket["outgoing_events"],
            "outgoing_load_issue": bucket["outgoing_load_issue"],
            "outgoing_store_issue": bucket["outgoing_store_issue"],
            "load_returns_seen": bucket["load_returns_seen"],
            "incoming_events": bucket["incoming_events"],
            "incoming_load_issue": bucket["incoming_load_issue"],
            "incoming_store_issue": bucket["incoming_store_issue"],
            "incoming_load_returns": bucket["incoming_load_returns"],
            "local_events": bucket["local_events"],
            "same_subgroup_events": bucket["same_subgroup_events"],
            "same_group_events": bucket["same_group_events"],
            "remote_group_events": bucket["remote_group_events"],
            "unknown_dest_events": bucket["unknown_dest_events"],
            "local_outgoing_events": bucket["local_outgoing_events"],
            "remote_outgoing_events": bucket["remote_outgoing_events"],
        }
        base.update({
            "outgoing_" + key_name: value
            for key_name, value in _latency_fields(bucket["outgoing_latency_total"], bucket["outgoing_latency_samples"]).items()
        })
        base.update({
            "incoming_" + key_name: value
            for key_name, value in _latency_fields(bucket["incoming_latency_total"], bucket["incoming_latency_samples"]).items()
        })
        out_rows.append(base)

    metadata = {
        "min_cycle": min_cycle,
        "max_cycle": max_cycle,
        "window": window,
        "num_windows": ((max_cycle - min_cycle) // window) + 1,
        "tiles": sorted({row["tile"] for row in rows if row["tile"] is not None}),
        "sections": sorted({row["section"] for row in rows if row["section"] is not None}),
    }
    return out_rows, metadata


def _build_edge_rows(rows: list[dict], *, window: int, min_cycle: int):
    grouped = defaultdict(lambda: {
        "source_group": None,
        "source_subgroup": None,
        "dest_group": None,
        "dest_subgroup": None,
        "event_count": 0,
        "load_issue_count": 0,
        "store_issue_count": 0,
        "load_return_count": 0,
        "latency_total": 0.0,
        "latency_samples": 0,
        "is_local_events": 0,
        "same_subgroup_events": 0,
        "same_group_events": 0,
        "remote_group_events": 0,
    })

    for row in rows:
        cycle = row["cycle"]
        source_tile = row["tile"]
        dest_tile = row["dest_tile"]
        if cycle is None or source_tile is None or dest_tile is None:
            continue
        window_index, _, _, center_cycle = _window_bounds(cycle, min_cycle, window)
        key = (row["section"], window_index, source_tile, dest_tile)
        bucket = grouped[key]
        bucket["source_group"] = row["group"]
        bucket["source_subgroup"] = row["subgroup"]
        bucket["dest_group"] = row["dest_group"]
        bucket["dest_subgroup"] = row["dest_subgroup"]
        bucket["event_count"] += 1
        if row["event_type"] == "load_issue":
            bucket["load_issue_count"] += 1
        elif row["event_type"] == "store_issue":
            bucket["store_issue_count"] += 1
        elif row["event_type"] == "load_return":
            bucket["load_return_count"] += 1
        if row["is_local"] == 1:
            bucket["is_local_events"] += 1
        elif row["is_same_subgroup"] == 1:
            bucket["same_subgroup_events"] += 1
            bucket["same_group_events"] += 1
        elif row["is_same_group"] == 1:
            bucket["same_group_events"] += 1
        else:
            bucket["remote_group_events"] += 1
        if row["latency"] is not None and row["event_type"] == "load_return":
            bucket["latency_total"] += float(row["latency"])
            bucket["latency_samples"] += 1

    out_rows = []
    for key in sorted(grouped):
        section, window_index, source_tile, dest_tile = key
        bucket = grouped[key]
        start_cycle = min_cycle + window_index * window
        end_cycle = start_cycle + window - 1
        center_cycle = start_cycle + window / 2.0
        row = {
            "section": section,
            "window_index": window_index,
            "window_start_cycle": start_cycle,
            "window_end_cycle": end_cycle,
            "window_center_cycle": f"{center_cycle:.1f}",
            "window_size": window,
            "source_tile": source_tile,
            "source_group": "" if bucket["source_group"] is None else bucket["source_group"],
            "source_subgroup": "" if bucket["source_subgroup"] is None else bucket["source_subgroup"],
            "dest_tile": dest_tile,
            "dest_group": "" if bucket["dest_group"] is None else bucket["dest_group"],
            "dest_subgroup": "" if bucket["dest_subgroup"] is None else bucket["dest_subgroup"],
            "event_count": bucket["event_count"],
            "load_issue_count": bucket["load_issue_count"],
            "store_issue_count": bucket["store_issue_count"],
            "load_return_count": bucket["load_return_count"],
            "is_local_events": bucket["is_local_events"],
            "same_subgroup_events": bucket["same_subgroup_events"],
            "same_group_events": bucket["same_group_events"],
            "remote_group_events": bucket["remote_group_events"],
        }
        row.update(_latency_fields(bucket["latency_total"], bucket["latency_samples"]))
        out_rows.append(row)
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


def _write_metadata(path: Path, *, source_csv: Path, result_dir: Path | None, tile_rows: list[dict], edge_rows: list[dict], base_metadata: dict, sections: list[int]):
    payload = {
        "kind": "comm_timeseries",
        "source_csv": str(source_csv),
        "result_dir": str(result_dir) if result_dir is not None else None,
        "window": base_metadata["window"],
        "min_cycle": base_metadata["min_cycle"],
        "max_cycle": base_metadata["max_cycle"],
        "num_windows": base_metadata["num_windows"],
        "sections": sections,
        "tiles": base_metadata["tiles"],
        "outputs": {
            "tile_csv": "comm_timeseries_tiles.csv",
            "edge_csv": "comm_timeseries_edges.csv",
        },
        "schemas": {
            "tile_csv": {
                "row_granularity": "one row per section/window/tile",
                "time_fields": ["window_index", "window_start_cycle", "window_end_cycle", "window_center_cycle", "window_size"],
                "identity_fields": ["section", "tile", "group", "subgroup"],
                "outgoing_fields": [
                    "outgoing_events", "outgoing_load_issue", "outgoing_store_issue", "load_returns_seen",
                    "local_events", "same_subgroup_events", "same_group_events", "remote_group_events", "unknown_dest_events",
                    "local_outgoing_events", "remote_outgoing_events",
                    "outgoing_avg_latency", "outgoing_latency_samples"
                ],
                "incoming_fields": [
                    "incoming_events", "incoming_load_issue", "incoming_store_issue", "incoming_load_returns",
                    "incoming_avg_latency", "incoming_latency_samples"
                ],
            },
            "edge_csv": {
                "row_granularity": "one row per section/window/source_tile/dest_tile",
                "time_fields": ["window_index", "window_start_cycle", "window_end_cycle", "window_center_cycle", "window_size"],
                "identity_fields": ["section", "source_tile", "source_group", "source_subgroup", "dest_tile", "dest_group", "dest_subgroup"],
                "metric_fields": [
                    "event_count", "load_issue_count", "store_issue_count", "load_return_count",
                    "is_local_events", "same_subgroup_events", "same_group_events", "remote_group_events",
                    "avg_latency", "latency_samples"
                ],
            },
        },
        "alignment": {
            "purpose": "Use the same section filter and cycle window when combining with stall plots.",
            "x_axis_field": "window_center_cycle",
            "cycle_range": [base_metadata["min_cycle"], base_metadata["max_cycle"]],
        },
        "row_counts": {
            "tile_csv": len(tile_rows),
            "edge_csv": len(edge_rows),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate windowed communication timeseries tables from comm_events_benchmark.csv."
    )
    parser.add_argument(
        "input_path",
        help="Benchmark result directory or direct path to comm_events_benchmark.csv",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for timeseries CSVs (default: <result_dir>/data/comm_timeseries or <csv-dir>/comm_timeseries)",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=64,
        help="Cycle window size for aggregation (default: 64)",
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
    if args.window <= 0:
        raise SystemExit("--window must be positive")

    csv_path, default_output_dir, result_dir = _resolve_paths(args.input_path)
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

    tile_rows, base_metadata = _build_tile_rows(rows, window=args.window)
    edge_rows = _build_edge_rows(rows, window=args.window, min_cycle=base_metadata["min_cycle"])

    output_dir.mkdir(parents=True, exist_ok=True)
    tile_path = output_dir / "comm_timeseries_tiles.csv"
    edge_path = output_dir / "comm_timeseries_edges.csv"
    metadata_path = output_dir / "comm_timeseries_metadata.json"

    _write_csv(tile_path, tile_rows)
    _write_csv(edge_path, edge_rows)
    _write_metadata(
        metadata_path,
        source_csv=csv_path,
        result_dir=result_dir,
        tile_rows=tile_rows,
        edge_rows=edge_rows,
        base_metadata=base_metadata,
        sections=base_metadata["sections"],
    )

    print(f"Input CSV:     {csv_path}")
    print(f"Output dir:    {output_dir}")
    print(f"Window:        {args.window} cycles")
    print(f"Wrote {tile_path}")
    print(f"Wrote {edge_path}")
    print(f"Wrote {metadata_path}")
    print(f"Tile rows:     {len(tile_rows)}")
    print(f"Edge rows:     {len(edge_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
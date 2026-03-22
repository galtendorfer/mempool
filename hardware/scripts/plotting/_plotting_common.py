#!/usr/bin/env python3

import csv
import json
from collections import defaultdict
from pathlib import Path


COMPARABLE_RUN_DIRS_CACHE = {}
SHARED_AXES_MANIFEST_CACHE = {}


def latest_view_dir(result_dir: Path):
    result_dir = Path(result_dir)
    if result_dir.parent.parent.name == "runs":
        return result_dir.parent.parent.parent / "latest"
    if result_dir.name == "latest":
        return result_dir
    return None


def comparable_run_dirs(result_dir: Path):
    result_dir = Path(result_dir)
    cache_key = str(result_dir.resolve())
    if cache_key in COMPARABLE_RUN_DIRS_CACHE:
        return COMPARABLE_RUN_DIRS_CACHE[cache_key]

    if result_dir.parent.parent.name == "runs":
        runs_root = result_dir.parent.parent.parent / "runs"
        comparable_dirs = []
        for tilerange_dir in sorted(runs_root.glob("tilerange*")):
            if not tilerange_dir.is_dir():
                continue
            run_dirs = sorted(path for path in tilerange_dir.iterdir() if path.is_dir())
            if run_dirs:
                comparable_dirs.append(run_dirs[-1])
        COMPARABLE_RUN_DIRS_CACHE[cache_key] = comparable_dirs
        return comparable_dirs

    if result_dir.parent.name == "latest":
        latest_root = result_dir.parent
        comparable_dirs = sorted(path for path in latest_root.glob("tilerange*") if path.is_dir())
        COMPARABLE_RUN_DIRS_CACHE[cache_key] = comparable_dirs
        return comparable_dirs

    if result_dir.name == "latest":
        runs_root = result_dir.parent / "runs"
    else:
        COMPARABLE_RUN_DIRS_CACHE[cache_key] = []
        return []

    comparable_dirs = []
    for tilerange_dir in sorted(runs_root.glob("tilerange*")):
        if not tilerange_dir.is_dir():
            continue
        run_dirs = sorted(path for path in tilerange_dir.iterdir() if path.is_dir())
        if run_dirs:
            comparable_dirs.append(run_dirs[-1])
    COMPARABLE_RUN_DIRS_CACHE[cache_key] = comparable_dirs
    return comparable_dirs


def format_compact_value(value) -> str:
    numeric_value = float(value)
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return f"{numeric_value:.3f}".rstrip("0").rstrip(".")


def format_sweep_label(sweep_kind: str, sweep_value) -> str:
    compact_value = format_compact_value(sweep_value)
    if sweep_kind in {"partition_prob", "part_prob"}:
        return f"p={compact_value}"
    if sweep_kind == "seq_prob":
        return f"seq={compact_value}"
    return f"{sweep_kind}={compact_value}"


def format_partition_label(partition_prob) -> str:
    return format_sweep_label("partition_prob", partition_prob)


def format_direction_label(direction: str) -> str:
    normalized = direction.strip().lower()
    if normalized == "incoming":
        return "in"
    if normalized == "outgoing":
        return "out"
    return normalized


def format_port_semantic_label(boundary, topology_info, port_index):
    if boundary == "tile":
        if topology_info["name"] == "terapool":
            subgroup_count = max(1, topology_info["num_subgroups_per_group"])
            if port_index == 0:
                return "unused"
            if port_index < subgroup_count:
                return f"local SG{port_index}"
            remote_group_port = port_index - (subgroup_count - 1)
            return f"remote G{remote_group_port}"

        if port_index == 0:
            return "local"
        return f"remote G{port_index}"

    if boundary == "group":
        return f"remote G{port_index + 1}"

    return f"local SG{port_index + 1}"


def _shared_axes_manifest_path(latest_root: Path) -> Path:
    return latest_root / "data" / "meta" / "shared_axes.json"


def _shared_axes_input_files(latest_root: Path):
    patterns = [
        latest_root / "data" / "throughput" / "tilerange*.csv",
        latest_root / "data" / "throughput" / "reconstructed_tilerange*.csv",
        latest_root / "data" / "tile" / "tilerange*.csv",
        latest_root / "data" / "group" / "tilerange*.csv",
        latest_root / "data" / "subgroup" / "tilerange*.csv",
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(pattern.parent.glob(pattern.name)))
    return files


def _normalized_metric_upper(metric_name: str, max_value: float) -> float:
    if metric_name in {"utilization", "backpressure"}:
        return min(1.0, max(0.01, max_value * 1.05))
    return max(1.0, max_value * 1.05)


def _boundary_port_index(boundary: str, row) -> int:
    if boundary == "tile":
        return int(row["port"])
    if boundary == "group":
        return int(row["remote_group"]) - 1
    return int(row["remote_subgroup"]) - 1


def _compute_boundary_metric_upper(boundary_csv: Path, boundary: str, metric_name: str):
    grouped = defaultdict(lambda: {"cycles": 0.0, "accepts": 0.0, "stalls": 0.0})
    with boundary_csv.open(newline="") as handle:
        for row in csv.DictReader(handle):
            port_index = _boundary_port_index(boundary, row)
            if port_index < 0:
                continue
            partition_prob = float(row["partition_prob"])
            req_prob = float(row["req_prob"])
            aggregate = grouped[(port_index, partition_prob, req_prob)]
            aggregate["cycles"] += float(row["cycles"])
            aggregate["accepts"] += float(row["accepts"])
            aggregate["stalls"] += float(row["stalls"])

    max_metric = 0.0
    for aggregate in grouped.values():
        total_cycles = aggregate["cycles"]
        total_accepts = aggregate["accepts"]
        total_stalls = aggregate["stalls"]
        attempts = total_accepts + total_stalls
        if metric_name == "utilization":
            metric_value = (total_accepts / total_cycles) if total_cycles else 0.0
        elif metric_name == "backpressure":
            metric_value = (total_stalls / attempts) if attempts else 0.0
        else:
            metric_value = 0.0
        max_metric = max(max_metric, metric_value)

    return _normalized_metric_upper(metric_name, max_metric)


def _compute_shared_axes_manifest(latest_root: Path):
    manifest = {
        "throughput": {
            "latency_upper": 1.0,
            "throughput_upper": 1.0,
        },
        "reconstructed": {
            "latency_upper": 1.0,
            "throughput_upper": 1.0,
        },
        "boundary": {},
    }

    throughput_latency_max = 0.0
    throughput_values_max = 0.0
    for csv_path in sorted((latest_root / "data" / "throughput").glob("tilerange*.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                throughput_latency_max = max(throughput_latency_max, float(row["avg_latency"]))
                throughput_values_max = max(throughput_values_max, float(row["throughput"]))
    if throughput_latency_max:
        manifest["throughput"]["latency_upper"] = _normalized_metric_upper("latency", throughput_latency_max)
    if throughput_values_max:
        manifest["throughput"]["throughput_upper"] = _normalized_metric_upper("throughput", throughput_values_max)

    reconstructed_latency_max = 0.0
    reconstructed_throughput_max = 0.0
    for csv_path in sorted((latest_root / "data" / "throughput").glob("reconstructed_tilerange*.csv")):
        with csv_path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                reconstructed_latency_max = max(reconstructed_latency_max, float(row["reconstructed_latency"]))
                reconstructed_throughput_max = max(reconstructed_throughput_max, float(row["throughput"]))
    if reconstructed_latency_max:
        manifest["reconstructed"]["latency_upper"] = _normalized_metric_upper("latency", reconstructed_latency_max)
    if reconstructed_throughput_max:
        manifest["reconstructed"]["throughput_upper"] = _normalized_metric_upper("throughput", reconstructed_throughput_max)

    for boundary in ("tile", "group", "subgroup"):
        manifest["boundary"][boundary] = {
            "utilization_upper": 1.0,
            "backpressure_upper": 1.0,
        }
        boundary_files = sorted((latest_root / "data" / boundary).glob("tilerange*.csv"))
        if not boundary_files:
            continue
        utilization_uppers = []
        backpressure_uppers = []
        for csv_path in boundary_files:
            with csv_path.open(newline="") as handle:
                has_rows = next(csv.DictReader(handle), None) is not None
            if not has_rows:
                continue
            utilization_uppers.append(_compute_boundary_metric_upper(csv_path, boundary, "utilization"))
            backpressure_uppers.append(_compute_boundary_metric_upper(csv_path, boundary, "backpressure"))
        if utilization_uppers:
            manifest["boundary"][boundary]["utilization_upper"] = max(utilization_uppers)
        if backpressure_uppers:
            manifest["boundary"][boundary]["backpressure_upper"] = max(backpressure_uppers)

    return manifest


def load_shared_axes_manifest(result_dir):
    latest_root = latest_view_dir(result_dir)
    if latest_root is None:
        return None

    latest_root = Path(latest_root)
    input_files = _shared_axes_input_files(latest_root)
    if not input_files:
        return None

    manifest_path = _shared_axes_manifest_path(latest_root)
    newest_input_mtime = max(path.stat().st_mtime for path in input_files)
    cache_key = str(manifest_path)
    cached = SHARED_AXES_MANIFEST_CACHE.get(cache_key)
    if cached is not None and cached[0] >= newest_input_mtime:
        return cached[1]

    manifest = None
    if manifest_path.is_file() and manifest_path.stat().st_mtime >= newest_input_mtime:
        with manifest_path.open() as handle:
            manifest = json.load(handle)
    else:
        manifest = _compute_shared_axes_manifest(latest_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w") as handle:
            json.dump(manifest, handle, indent=2, sort_keys=True)

    SHARED_AXES_MANIFEST_CACHE[cache_key] = (newest_input_mtime, manifest)
    return manifest


def get_shared_axis_upper(result_dir, axis_group: str, axis_key: str, boundary: str = None):
    manifest = load_shared_axes_manifest(result_dir)
    if manifest is None:
        return None

    if axis_group == "boundary" and boundary is not None:
        return manifest.get("boundary", {}).get(boundary, {}).get(axis_key)
    return manifest.get(axis_group, {}).get(axis_key)
#!/usr/bin/env python3
"""Generate a combined communication-event CSV from all trace files in a folder.

Reads every trace_hart_*.trace in --folder, calls _extract_comm_events.py on
each one, and appends the rows into a single --csv output file.

Normal users can also use `extract_comm_events.py` when they already have a
benchmark result directory and want the safest interface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from _workflow_metadata import (
    TOPOLOGY_KEYS,
    find_result_dir,
    format_topology,
    load_config_topology,
    load_env_topology,
    load_result_dir_topology,
    validate_topology_consistency,
)


def _resolve_topology(args, parser, folder, output_path):
    result_dir = find_result_dir(folder, output_path)

    try:
        env_topology = load_env_topology(os.environ)
    except ValueError as exc:
        parser.error(str(exc))

    requested_topology = None
    if args.topology:
        requested_topology = load_config_topology(args.topology)
        if requested_topology is None:
            parser.error(f"Unknown topology/config: {args.topology}")

    detected_topology = load_result_dir_topology(result_dir)
    topology = requested_topology or detected_topology or env_topology
    if topology is None:
        parser.error(
            "Cannot determine topology. Run inside a benchmark result directory, "
            "pass --topology <config>, or set all of: " + ", ".join(TOPOLOGY_KEYS)
        )

    if detected_topology and requested_topology:
        try:
            validate_topology_consistency(detected_topology, requested_topology)
        except ValueError as exc:
            parser.error(f"--topology disagrees with result metadata: {exc}")

    if detected_topology and env_topology:
        try:
            validate_topology_consistency(detected_topology, env_topology)
        except ValueError as exc:
            parser.error(f"Environment disagrees with result metadata: {exc}")

    if requested_topology and env_topology:
        try:
            validate_topology_consistency(requested_topology, env_topology)
        except ValueError as exc:
            parser.error(f"Environment disagrees with --topology: {exc}")

    trace_files = sorted(folder.glob("trace_hart_*.trace"))
    if trace_files and len(trace_files) != int(topology["NUM_CORES"]):
        print(
            f"Warning: found {len(trace_files)} traces, but topology expects {topology['NUM_CORES']} cores.",
            file=sys.stderr,
        )

    return topology, trace_files


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract combined communication events from all traces in a folder.")
    parser.add_argument("--folder", required=True, help="Folder containing trace_hart_*.trace files")
    parser.add_argument("--csv", required=True, help="Combined CSV output path")
    parser.add_argument("--section", type=int, action="append", help="Emit rows only for the specified section; may be repeated")
    parser.add_argument("--benchmark-only", action="store_true", help="Shortcut for --section 1 (the benchmark bracket)")
    parser.add_argument("-p", "--permissive", action="store_true", help="Ignore malformed non-trace lines when possible")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV output (default: refuse)")
    parser.add_argument("--topology", help="Configuration/topology name (for example: mempool or terapool) when metadata is unavailable")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"--folder is not a directory: {folder}")

    output_path = Path(args.csv)
    topology, trace_files = _resolve_topology(args, argparse.ArgumentParser(prog="extract_comm_events_batch.py"), folder, output_path)
    if not trace_files:
        raise SystemExit(f"No trace_hart_*.trace files found in {folder}")

    if output_path.exists():
        if not args.force:
            raise SystemExit(
                f"Output CSV already exists: {output_path}\n"
                "       Use --force to overwrite."
            )
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    worker = Path(__file__).with_name("_extract_comm_events.py")
    cmd = [sys.executable, str(worker), "--csv", str(output_path)]
    child_env = os.environ.copy()
    for key in TOPOLOGY_KEYS:
        child_env[key] = str(topology[key])

    print(f"Topology: {format_topology(topology)}", file=sys.stderr)
    print(f"Source:   {topology.get('source', 'unknown')}", file=sys.stderr)
    if args.permissive:
        cmd.append("--permissive")
    if args.benchmark_only:
        cmd.append("--benchmark-only")
    for section in args.section or []:
        cmd.extend(["--section", str(section)])

    for index, trace_file in enumerate(trace_files, start=1):
        subprocess.run(cmd + [str(trace_file)], check=True, env=child_env)
        if index % 32 == 0 or index == len(trace_files):
            print(f"Processed {index}/{len(trace_files)} traces", file=sys.stderr)

    print(f"Wrote combined communication events to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
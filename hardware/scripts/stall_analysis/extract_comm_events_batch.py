#!/usr/bin/env python3
"""Generate a combined communication-event CSV from all trace files in a folder.

Reads every supported trace_hart_* file in --folder, calls
_extract_comm_events.py on each one, and appends the rows into a single
--csv output file.

Normal users can also use `extract_comm_events.py` when they already have a
benchmark result directory and want the safest interface.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def _find_trace_files(folder: Path) -> list[Path]:
    trace_files = []
    for path in sorted(folder.glob("trace_hart_*")):
        if not path.is_file():
            continue
        if path.suffix == ".dasm":
            continue
        trace_files.append(path)
    return trace_files


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

    trace_files = _find_trace_files(folder)
    if trace_files and len(trace_files) != int(topology["NUM_CORES"]):
        print(
            f"Warning: found {len(trace_files)} traces, but topology expects {topology['NUM_CORES']} cores.",
            file=sys.stderr,
        )

    return topology, trace_files


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract combined communication events from all traces in a folder.")
    parser.add_argument(
        "--folder",
        required=True,
        help="Folder containing real trace_hart_* files (legacy reconstructed traces may still parse for one-off recovery)",
    )
    parser.add_argument("--csv", required=True, help="Combined CSV output path")
    parser.add_argument("--section", type=int, action="append", help="Emit rows only for the specified section; may be repeated")
    parser.add_argument("--benchmark-only", action="store_true", help="Shortcut for --section 1 (the benchmark bracket)")
    parser.add_argument("-p", "--permissive", action="store_true", help="Ignore malformed non-trace lines when possible")
    parser.add_argument("--force", action="store_true", help="Overwrite existing CSV output (default: refuse)")
    parser.add_argument("--topology", help="Configuration/topology name (for example: mempool or terapool) when metadata is unavailable")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel extraction workers (default: 1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    folder = Path(args.folder)
    if not folder.is_dir():
        raise SystemExit(f"--folder is not a directory: {folder}")

    output_path = Path(args.csv)
    topology, trace_files = _resolve_topology(args, argparse.ArgumentParser(prog="extract_comm_events_batch.py"), folder, output_path)
    if not trace_files:
        raise SystemExit(f"No supported trace_hart_* files found in {folder}")

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

    n_jobs = min(args.jobs, len(trace_files))

    if n_jobs <= 1:
        # Sequential (original behaviour)
        for index, trace_file in enumerate(trace_files, start=1):
            subprocess.run(
                cmd + [str(trace_file)],
                check=True,
                env=child_env,
                stdout=subprocess.DEVNULL,
            )
            if index % 32 == 0 or index == len(trace_files):
                print(f"Processed {index}/{len(trace_files)} traces", file=sys.stderr)
    else:
        # Parallel: each worker writes to a temp CSV, then merge.
        print(f"Extracting {len(trace_files)} traces with {n_jobs} workers ...", file=sys.stderr)
        tmp_dir = tempfile.mkdtemp(prefix="comm_par_")

        def _run_one(item):
            idx, trace_file = item
            tmp_csv = os.path.join(tmp_dir, f"part_{idx:04d}.csv")
            subprocess.run(
                [sys.executable, str(worker), "--csv", tmp_csv, str(trace_file)]
                + (["--permissive"] if args.permissive else [])
                + (["--benchmark-only"] if args.benchmark_only else [])
                + [a for s in (args.section or []) for a in ("--section", str(s))],
                check=True,
                env=child_env,
                stdout=subprocess.DEVNULL,
            )
            return tmp_csv

        done = 0
        tmp_csvs = []
        with ThreadPoolExecutor(max_workers=n_jobs) as pool:
            futures = {pool.submit(_run_one, (i, tf)): i for i, tf in enumerate(trace_files)}
            for future in as_completed(futures):
                future.result()  # raises on error
                done += 1
                if done % 64 == 0 or done == len(trace_files):
                    print(f"Processed {done}/{len(trace_files)} traces", file=sys.stderr)

        # Merge temp CSVs into final output (preserve header from first file only)
        tmp_csvs = sorted(Path(tmp_dir).glob("part_*.csv"))
        header_written = False
        with open(output_path, "w", newline="") as out:
            for tmp_csv in tmp_csvs:
                with open(tmp_csv, "r") as inp:
                    for line_no, line in enumerate(inp):
                        if line_no == 0:
                            if not header_written:
                                out.write(line)
                                header_written = True
                        else:
                            out.write(line)
                tmp_csv.unlink()
        Path(tmp_dir).rmdir()

    print(f"Wrote combined communication events to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
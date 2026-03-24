#!/usr/bin/env python3
"""Generate a combined cycle-by-cycle stall CSV from all trace files in a folder.

Reads every trace_hart_*.trace in --folder, calls _gen_stall_timeseries.py on
each one, and appends the rows into a single --csv output file.

Normal users should prefer one of these public entry points instead:
    - `make benchmark`                  for the full simulation-to-CSV pipeline
    - `make rerun_stall_timeseries`     from hardware/ for CSV-only rebuilds
    - `rerun_stall_timeseries.py`       for a direct result_dir-based wrapper

This underscore-prefixed script is kept as an internal building block.

Required flags:
  --folder <dir>    Directory containing trace_hart_*.trace files.
                    Typically: result_dir/traces/
  --csv <path>      Output CSV path.
                    Typically: result_dir/data/stall_timeseries_benchmark.csv

Optional flags:
  --benchmark-only  Only emit rows for section 1 (the benchmark bracket).
                    Shortcut for --section 1.
  --section N       Emit rows for a specific section (repeatable).
  -p, --permissive  Ignore malformed lines in trace files when possible.
    --topology NAME   Explicit configuration/topology name when the input paths
                                        are outside a benchmark result directory.
  --force           Allow overwriting an existing output CSV.
                    Without this, the script refuses to run if the output
                    file already exists, to prevent accidental data loss.

Topology handling:
    If --folder/--csv point into a benchmark result directory, the script loads
    topology from result_dir/topology.env when available, otherwise from the
    saved result_dir/env + config/<name>.mk. Standalone runs must pass
    --topology or provide the topology env vars explicitly.

Examples:
  # First run
  python _gen_stall_timeseries_batch.py \
      --folder results/matmul_i32_mempool/2x2_xpulpv2/baseline/traces \
      --csv results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/stall_timeseries_benchmark.csv \
      --benchmark-only -p

  # Re-generate (must pass --force)
  python _gen_stall_timeseries_batch.py \
      --folder results/matmul_i32_mempool/2x2_xpulpv2/baseline/traces \\
      --csv results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/stall_timeseries_benchmark.csv \\
      --benchmark-only -p --force

  # Standalone run outside a saved result directory
  python _gen_stall_timeseries_batch.py \
      --folder scratch/traces \
      --csv scratch/data/stall_timeseries_benchmark.csv \
      --benchmark-only -p --topology mempool
"""

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


def resolve_topology(args, parser, folder, output_path):
    result_dir = find_result_dir(folder, output_path)

    try:
        env_topology = load_env_topology(os.environ)
    except ValueError as exc:
        parser.error(str(exc))

    requested_topology = None
    if args.topology:
        requested_topology = load_config_topology(args.topology)
        if requested_topology is None:
            parser.error(f'Unknown topology/config: {args.topology}')

    detected_topology = load_result_dir_topology(result_dir)
    topology = requested_topology or detected_topology or env_topology
    if topology is None:
        parser.error(
            'Cannot determine topology. Run inside a benchmark result directory, '
            'pass --topology <config>, or set all of: ' + ', '.join(TOPOLOGY_KEYS))

    if detected_topology and requested_topology:
        try:
            validate_topology_consistency(detected_topology, requested_topology)
        except ValueError as exc:
            parser.error(f'--topology disagrees with result metadata: {exc}')

    if detected_topology and env_topology:
        try:
            validate_topology_consistency(detected_topology, env_topology)
        except ValueError as exc:
            parser.error(f'Environment disagrees with result metadata: {exc}')

    if requested_topology and env_topology:
        try:
            validate_topology_consistency(requested_topology, env_topology)
        except ValueError as exc:
            parser.error(f'Environment disagrees with --topology: {exc}')

    trace_files = sorted(folder.glob('trace_hart_*.trace'))
    if trace_files and len(trace_files) != int(topology['NUM_CORES']):
        print(
            f'Warning: found {len(trace_files)} traces, but topology expects {topology["NUM_CORES"]} cores.',
            file=sys.stderr,
        )

    return topology, trace_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--folder',
        required=True,
        help='Folder containing trace_hart_*.trace files')
    parser.add_argument(
        '--csv',
        required=True,
        help='Combined CSV output path')
    parser.add_argument(
        '--section',
        type=int,
        action='append',
        help='Emit rows only for the specified section; may be repeated')
    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Shortcut for --section 1 for apps with a single benchmark bracket')
    parser.add_argument(
        '-p',
        '--permissive',
        action='store_true',
        help='Ignore malformed non-trace lines when possible')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite existing CSV output (default: refuse)')
    parser.add_argument(
        '--topology',
        help='Configuration/topology name (for example: mempool or terapool) when metadata is unavailable')
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        parser.error(f'--folder is not a directory: {folder}')

    output_path = Path(args.csv)
    topology, trace_files = resolve_topology(args, parser, folder, output_path)
    if not trace_files:
        parser.error(f'No trace_hart_*.trace files found in {folder}')

    if output_path.exists():
        if not args.force:
            parser.error(
                f'Output CSV already exists: {output_path}\n'
                '       Use --force to overwrite.')
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script_path = Path(__file__).with_name(
        Path(__file__).name.replace('_batch', '')
    )
    common_args = [
        sys.executable,
        str(script_path),
        '--csv',
        str(output_path),
    ]
    child_env = os.environ.copy()
    for key in TOPOLOGY_KEYS:
        child_env[key] = str(topology[key])

    print(f'Topology: {format_topology(topology)}', file=sys.stderr)
    print(f'Source:   {topology.get("source", "unknown")}', file=sys.stderr)
    if args.permissive:
        common_args.append('--permissive')
    if args.benchmark_only:
        common_args.append('--benchmark-only')
    for section in args.section or []:
        common_args.extend(['--section', str(section)])

    for index, trace_file in enumerate(trace_files, start=1):
        subprocess.run(common_args + [str(trace_file)], check=True, env=child_env)
        if index % 32 == 0 or index == len(trace_files):
            print(f'Processed {index}/{len(trace_files)} traces', file=sys.stderr)

    print(f'Wrote combined stall time-series to {output_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
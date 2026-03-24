#!/usr/bin/env python3
"""Public wrapper for re-generating stall_timeseries_benchmark.csv.

Use this script when you already have a benchmark result directory with
`traces/` and want to rebuild only the combined stall CSV safely.

This is the stable user-facing entry point for the reprocessing step.
The underscore-prefixed `_gen_stall_timeseries_batch.py` remains internal.

Typical usage:
    python rerun_stall_timeseries.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline

Examples:
    # Rebuild the default benchmark CSV in-place
    python rerun_stall_timeseries.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --force

    # Write to a separate CSV for comparison
    python rerun_stall_timeseries.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline \
        --csv ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline/data/recheck.csv

The wrapper derives:
  - traces folder: <result_dir>/traces
  - default output: <result_dir>/data/stall_timeseries_benchmark.csv

Topology is loaded automatically from saved result metadata when possible.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description='Re-generate stall_timeseries_benchmark.csv for an existing benchmark result.')
    parser.add_argument(
        'result_dir',
        help='Benchmark result directory containing traces/ and data/')
    parser.add_argument(
        '--csv',
        help='Optional output CSV path (default: <result_dir>/data/stall_timeseries_benchmark.csv)')
    parser.add_argument(
        '--section',
        type=int,
        action='append',
        help='Emit rows only for the specified section; may be repeated')
    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Shortcut for --section 1 (used by the Makefile target)')
    parser.add_argument(
        '-p',
        '--permissive',
        action='store_true',
        help='Ignore malformed non-trace lines when possible')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite the destination CSV if it already exists')
    parser.add_argument(
        '--topology',
        help='Optional fallback topology/config name for standalone or incomplete result directories')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    result_dir = Path(args.result_dir).resolve()
    traces_dir = result_dir / 'traces'
    data_dir = result_dir / 'data'
    output_csv = Path(args.csv).resolve() if args.csv else data_dir / 'stall_timeseries_benchmark.csv'

    if not result_dir.is_dir():
        raise SystemExit(f'Result directory not found: {result_dir}')
    if not traces_dir.is_dir():
        raise SystemExit(f'Traces directory not found: {traces_dir}')

    data_dir.mkdir(parents=True, exist_ok=True)

    batch_script = Path(__file__).with_name('_gen_stall_timeseries_batch.py')
    cmd = [
        sys.executable,
        str(batch_script),
        '--folder', str(traces_dir),
        '--csv', str(output_csv),
    ]
    if args.benchmark_only:
        cmd.append('--benchmark-only')
    for section in args.section or []:
        cmd.extend(['--section', str(section)])
    if args.permissive:
        cmd.append('--permissive')
    if args.force:
        cmd.append('--force')
    if args.topology:
        cmd.extend(['--topology', args.topology])

    print(f'Result dir: {result_dir}')
    print(f'Traces:     {traces_dir}')
    print(f'Output CSV: {output_csv}')
    subprocess.run(cmd, check=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
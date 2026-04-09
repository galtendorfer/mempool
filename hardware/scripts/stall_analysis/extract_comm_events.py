#!/usr/bin/env python3
"""Public wrapper for extracting communication events from an existing result_dir.

Use this script when you already have a benchmark result directory and want to
build a combined source/destination communication CSV from archived real traces.

Typical usage:
    python extract_comm_events.py ../../../results/matmul_i32_mempool/2x2_xpulpv2/baseline
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _resolve_trace_dir(result_dir: Path) -> Path:
    trace_dir = result_dir / "real_traces"
    reconstructed_dir = result_dir / "traces"

    if trace_dir.is_dir():
        return trace_dir

    message = [f"Canonical analysis trace directory not found: {trace_dir}"]
    if reconstructed_dir.is_dir():
        message.append(
            "Reconstructed traces are still present, but they are decommissioned for the standard communication-analysis pipeline."
        )
    raise SystemExit("\n".join(message))


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Extract communication events for an existing benchmark result directory."
    )
    parser.add_argument("result_dir", help="Benchmark result directory containing real_traces/ and data/")
    parser.add_argument("--csv", help="Optional output CSV path (default: <result_dir>/data/comm_events_benchmark.csv)")
    parser.add_argument("--section", type=int, action="append", help="Emit rows only for the specified section; may be repeated")
    parser.add_argument("--benchmark-only", action="store_true", help="Shortcut for --section 1")
    parser.add_argument("-p", "--permissive", action="store_true", help="Ignore malformed non-trace lines when possible")
    parser.add_argument("--force", action="store_true", help="Overwrite the destination CSV if it already exists")
    parser.add_argument("--topology", help="Optional fallback topology/config name for standalone or incomplete result directories")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="Number of parallel extraction workers (default: 1)")
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    result_dir = Path(args.result_dir).resolve()
    data_dir = result_dir / "data"
    output_csv = Path(args.csv).resolve() if args.csv else data_dir / "comm_events_benchmark.csv"

    if not result_dir.is_dir():
        raise SystemExit(f"Result directory not found: {result_dir}")
    traces_dir = _resolve_trace_dir(result_dir)

    data_dir.mkdir(parents=True, exist_ok=True)

    batch_script = Path(__file__).with_name("extract_comm_events_batch.py")
    cmd = [
        sys.executable,
        str(batch_script),
        "--folder",
        str(traces_dir),
        "--csv",
        str(output_csv),
    ]
    if args.benchmark_only:
        cmd.append("--benchmark-only")
    for section in args.section or []:
        cmd.extend(["--section", str(section)])
    if args.permissive:
        cmd.append("--permissive")
    if args.force:
        cmd.append("--force")
    if args.topology:
        cmd.extend(["--topology", args.topology])
    if args.jobs > 1:
        cmd.extend(["-j", str(args.jobs)])

    print(f"Result dir: {result_dir}")
    print("Source:     canonical-analysis-trace archive")
    print(f"Traces:     {traces_dir}")
    print(f"Output CSV: {output_csv}")
    subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Batch plotter: generate tile-detail plots for all tiles in a benchmark result.

Auto-discovers CSV, traces, tile IDs, and topology from the result directory.
Routes each tile's output to the correct group/subgroup folder.

Normal users should prefer `make plots` from hardware/.
This script remains the direct implementation behind that public target.

Required positional argument:
    result_dir          Path to a variant directory, e.g.:
                          results/matmul_i32_mempool/2x2_xpulpv2/baseline
                          results/matmul_i32_terapool/2x2_xpulpv2/das

Optional flags:
    --section N         Filter by section number (repeatable).
                        Typically --section 1 for the benchmark bracket.
    --topology NAME     Force topology: mempool or terapool.
                        Auto-detected from result metadata if omitted,
                        then from the directory name as a fallback.
    --tiles T [T ...]   Only generate for these specific tile IDs.
                        Default: all tiles discovered from the CSV.
    --overview          Also generate the cluster overview page
                        into plots/overview/.
    --window N          Sliding-window width (in cycles) for the timeseries
                        aggregation.  Forwarded to _plot_specific_tile.
                        Default: 64.
    --force             Overwrite existing plot PNGs.
                        Without this, tiles that already have a PNG are
                        silently skipped to prevent accidental overwrites.
    --dry-run           Print what would be done without executing.
                        Useful to verify topology routing before committing.

Examples:
    # All tiles, topology from result metadata
    python plot_all_tiles.py ../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1

    # Only tiles 0 and 35
    python plot_all_tiles.py ../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1 --tiles 0 35

    # Dry run to see routing
    python plot_all_tiles.py ../../results/matmul_i32_terapool/2x2_xpulpv2/baseline --section 1 --dry-run

    # Re-generate all plots (overwrite existing)
    python plot_all_tiles.py ../../results/matmul_i32_mempool/2x2_xpulpv2/baseline --section 1 --force
"""

import argparse
import csv
import multiprocessing
import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
from _workflow_metadata import (
    format_topology,
    infer_named_topology,
    load_result_dir_topology,
)

# ── Topology definitions ──────────────────────────────────────────────────────

TOPOLOGIES = {
    "mempool": {
        "num_groups": 4,
        "tiles_per_group": 16,
        "subgroups_per_group": 0,  # 0 = no subgroups
        "tile_digits": 2,          # tile00..tile63
    },
    "terapool": {
        "num_groups": 4,
        "tiles_per_group": 32,
        "subgroups_per_group": 4,
        "tiles_per_subgroup": 8,
        "tile_digits": 3,          # tile000..tile127
    },
}


def detect_topology(result_dir):
    """Guess topology from directory path."""
    metadata = load_result_dir_topology(result_dir)
    if metadata is not None:
        named = infer_named_topology(metadata)
        if named is not None:
            return named, metadata.get('source', 'result metadata')

    name = str(result_dir).lower()
    if "terapool" in name:
        return "terapool", 'result_dir path'
    if "mempool" in name:
        return "mempool", 'result_dir path'
    return None, None


def tile_output_dir(plots_dir, topology_name, tile_id):
    """Compute the output directory for a given tile ID."""
    topo = TOPOLOGIES[topology_name]
    group = tile_id // topo["tiles_per_group"]

    if topo["subgroups_per_group"] > 0:
        tiles_per_sg = topo["tiles_per_subgroup"]
        subgroup = (tile_id % topo["tiles_per_group"]) // tiles_per_sg
        return plots_dir / f"group{group}" / f"subgroup{subgroup}"
    else:
        return plots_dir / f"group{group}"


def discover_tile_ids(csv_path):
    """Read unique tile IDs from the CSV."""
    tiles = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            val = row.get("tile", "").strip()
            if val:
                tiles.add(int(val))
    return sorted(tiles)


def validate_tiles(topology_name, tile_ids):
    if not tile_ids:
        raise ValueError('No tile IDs found in CSV')
    topo = TOPOLOGIES[topology_name]
    max_tiles = topo['num_groups'] * topo['tiles_per_group']
    invalid = [tile_id for tile_id in tile_ids if tile_id < 0 or tile_id >= max_tiles]
    if invalid:
        raise ValueError(
            f'Tiles {invalid[:5]} do not fit topology {topology_name} '
            f'(expected tile IDs in [0, {max_tiles - 1}])')


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Batch-generate tile detail plots for all tiles.")
    p.add_argument("result_dir",
                   help="Variant directory (e.g. results/matmul_i32_mempool/2x2_xpulpv2/baseline)")
    p.add_argument("--section", type=int, action="append",
                   help="Filter by section (repeatable)")
    p.add_argument("--topology", choices=list(TOPOLOGIES),
                   help="Force topology (auto-detected if omitted)")
    p.add_argument("--tiles", type=int, nargs="+",
                   help="Only these tile IDs (default: all from CSV)")
    p.add_argument("--overview", action="store_true",
                   help="Also generate the cluster overview page")
    p.add_argument("--window", type=int, default=64,
                   help="Sliding-window width for timeseries (default: 64)")
    p.add_argument("--force", action="store_true",
                   help="Overwrite existing plot files (default: skip)")
    p.add_argument("--jobs", "-j", type=int, default=1,
                   help="Number of parallel tile-plot workers (default: 1)")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would be done without executing")
    return p.parse_args(argv)


def _tile_worker(item):
    """Worker for parallel tile plotting (must be at module level for pickling)."""
    import sys
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))
    import _plot_specific_tile

    tid, out_dir, tile_argv, dg = item
    try:
        _plot_specific_tile.main(tile_argv)
        return (tid, None)
    except Exception as e:
        return (tid, str(e))


def main(argv=None):
    args = parse_args(argv)
    result_dir = Path(args.result_dir).resolve()

    # ── Discover paths ────────────────────────────────────────────────
    csv_path = result_dir / "data" / "stall_timeseries_benchmark.csv"
    traces_dir = result_dir / "traces_dasm"
    plots_dir = result_dir / "plots"

    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")
    if not traces_dir.is_dir():
        print(f"Warning: traces dir not found: {traces_dir}", file=sys.stderr)
        traces_dir = None

    # ── Detect topology ───────────────────────────────────────────────
    auto_topology, topology_source = detect_topology(result_dir)
    topology = args.topology or auto_topology
    if topology is None:
        sys.exit("Cannot detect topology from path. Use --topology mempool|terapool")

    result_metadata = load_result_dir_topology(result_dir)
    if result_metadata is not None and args.topology:
        inferred = infer_named_topology(result_metadata)
        if inferred is not None and inferred != args.topology:
            sys.exit(
                f'Forced topology {args.topology} disagrees with result metadata '
                f'({format_topology(result_metadata)} from {result_metadata.get("source")})')

    print(f"Topology: {topology}")
    if topology_source:
        print(f"Source:   {topology_source}")
    if result_metadata is not None:
        print(f"Layout:   {format_topology(result_metadata)}")
    print(f"CSV:      {csv_path}")
    print(f"Traces:   {traces_dir or '(none)'}")
    print(f"Plots:    {plots_dir}")

    # ── Discover tiles ────────────────────────────────────────────────
    if args.tiles:
        tile_ids = sorted(args.tiles)
    else:
        tile_ids = discover_tile_ids(csv_path)

    try:
        validate_tiles(topology, tile_ids)
    except ValueError as exc:
        sys.exit(str(exc))

    print(f"Tiles:    {len(tile_ids)} ({tile_ids[0]}–{tile_ids[-1]})")
    print()

    # ── Import plotting scripts (they're in the same directory) ───────
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import _plot_specific_tile

    # ── Section filter args ───────────────────────────────────────────
    section_args = []
    if args.section:
        for s in args.section:
            section_args += ["--section", str(s)]

    traces_args = ["--traces-dir", str(traces_dir)] if traces_dir else []
    window_args = ["--window", str(args.window)]

    # ── Overview ──────────────────────────────────────────────────────
    if args.overview:
        ov_dir = plots_dir / "overview"
        ov_existing = list(ov_dir.glob("*.png")) if ov_dir.exists() else []
        if ov_existing and not args.force:
            print(f"Overview: skipped ({len(ov_existing)} PNGs already exist, use --force to overwrite)")
        else:
            ov_argv = [str(csv_path)] + section_args + traces_args + window_args + [
                "--overview",
                "--output-dir", str(ov_dir),
            ]
            if args.dry_run:
                print(f"[dry-run] overview → {ov_dir}")
            else:
                print(f"Overview → {ov_dir}")
                ov_dir.mkdir(parents=True, exist_ok=True)
                _plot_specific_tile.main(ov_argv)
                print()

    # ── Per-tile detail ───────────────────────────────────────────────
    digits = TOPOLOGIES[topology]["tile_digits"]
    failed = []
    skipped = 0

    # Build work items
    work_items = []
    for tid in tile_ids:
        out_dir = tile_output_dir(plots_dir, topology, tid)
        tile_png = out_dir / f"tile_detail_tile{tid}.png"
        if tile_png.exists() and not args.force:
            skipped += 1
            continue
        tile_argv = [str(csv_path), str(tid)] + section_args + traces_args + window_args + [
            "--output-dir", str(out_dir),
            "--prefix", "tile_detail",
        ]
        work_items.append((tid, out_dir, tile_argv, digits))

    if args.dry_run:
        for tid, out_dir, _, dg in work_items:
            label = f"tile {tid:0{dg}d} → {out_dir.relative_to(plots_dir)}"
            print(f"[dry-run] {label}")
    elif work_items:
        n_jobs = min(args.jobs, len(work_items))
        # Ensure output dirs exist
        for _, out_dir, _, _ in work_items:
            out_dir.mkdir(parents=True, exist_ok=True)

        if n_jobs <= 1:
            # Sequential (original behaviour)
            for i, (tid, out_dir, tile_argv, dg) in enumerate(work_items, 1):
                label = f"[{i}/{len(work_items)}] tile {tid:0{dg}d} → {out_dir.relative_to(plots_dir)}"
                print(label, end=" ... ", flush=True)
                try:
                    _plot_specific_tile.main(tile_argv)
                    print("ok")
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed.append((tid, str(e)))
        else:
            print(f"Generating {len(work_items)} tile plots with {n_jobs} parallel workers ...")
            # Use spawn context to avoid fork-safety issues with matplotlib
            ctx = multiprocessing.get_context("spawn")

            with ctx.Pool(n_jobs) as pool:
                results = pool.map(_tile_worker, work_items)
            for tid, err in results:
                if err is not None:
                    failed.append((tid, err))
            print(f"  {len(work_items) - len(failed)} ok, {len(failed)} failed")

    # ── Summary ───────────────────────────────────────────────────────
    if not args.dry_run:
        generated = len(tile_ids) - len(failed) - skipped
        parts = [f"{generated} generated"]
        if skipped:
            parts.append(f"{skipped} skipped (exist, use --force)")
        if failed:
            parts.append(f"{len(failed)} failed")
        print(f"\nDone: {' / '.join(parts)}  ({len(tile_ids)} total)")
        if failed:
            print("Failed tiles:")
            for tid, err in failed:
                print(f"  tile {tid}: {err}")


if __name__ == "__main__":
    main()

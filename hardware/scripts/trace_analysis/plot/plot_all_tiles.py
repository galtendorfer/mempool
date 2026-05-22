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
    --overview          Also generate the cluster overview page and per-group
                        IPC/cycle breakdown into plots/overview/.
                        into plots/overview/.
    --group-details     Generate group/subgroup detail pages. These mirror the
                        tile-detail style, but each heatmap row is a tile.
    --skip-tile-details Do not generate per-tile detail pages.
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

import matplotlib.pyplot as plt

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)
from _workflow_metadata import (
    format_topology,
    infer_named_topology,
    load_result_dir_topology,
)
from _stall_plot_common import filter_rows, load_rows

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


def group_detail_specs(plots_dir, topology_name, tile_ids):
    """Return group/subgroup detail output specs for the selected tiles."""
    topo = TOPOLOGIES[topology_name]
    tiles_per_group = topo["tiles_per_group"]
    specs = {}

    for tile_id in tile_ids:
        group = tile_id // tiles_per_group
        if topo["subgroups_per_group"] > 0:
            tiles_per_subgroup = topo["tiles_per_subgroup"]
            subgroup = (tile_id % tiles_per_group) // tiles_per_subgroup
            key = (group, subgroup)
            out_dir = plots_dir / f"group{group}" / f"subgroup{subgroup}"
            png_path = out_dir / f"subgroup_detail_group{group}_subgroup{subgroup}.png"
            title = f"Group {group} Subgroup {subgroup} Detail Report"
        else:
            key = (group, None)
            out_dir = plots_dir / f"group{group}"
            png_path = out_dir / f"group_detail_group{group}.png"
            title = f"Group {group} Detail Report"

        if key not in specs:
            specs[key] = {
                "key": key,
                "out_dir": out_dir,
                "png_path": png_path,
                "title": title,
                "tile_ids": [],
            }
        specs[key]["tile_ids"].append(tile_id)

    for spec in specs.values():
        spec["tile_ids"] = sorted(spec["tile_ids"])
    return [specs[key] for key in sorted(specs)]


def expected_tile_count(topology_name):
    topo = TOPOLOGIES[topology_name]
    return topo["num_groups"] * topo["tiles_per_group"]


def discover_tile_ids(csv_path):
    """Read unique tile IDs from the CSV."""
    tiles = set()
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            val = row.get("tile", "").strip()
            if val:
                tiles.add(int(val))
    return sorted(tiles)


def group_rows_by_tile(rows, selected_tiles=None):
    grouped = {}
    selected = None if selected_tiles is None else set(selected_tiles)
    for row in rows:
        tile_id = row.get("tile")
        if tile_id is None:
            continue
        if selected is not None and tile_id not in selected:
            continue
        grouped.setdefault(tile_id, []).append(row)
    return grouped


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
                   help="Also generate cluster overview and per-group breakdown pages")
    p.add_argument("--group-details", action="store_true",
                   help="Generate group/subgroup detail pages")
    p.add_argument("--skip-tile-details", action="store_true",
                   help="Skip per-tile detail pages")
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
    plots_dir = result_dir / "plots"

    if not csv_path.is_file():
        sys.exit(f"CSV not found: {csv_path}")

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
    print(f"Plots:    {plots_dir}")

    if (not args.tiles and not args.force and not args.dry_run
            and not args.overview and not args.group_details and not args.skip_tile_details):
        existing_tile_pngs = list(plots_dir.rglob("tile_detail_tile*.png")) if plots_dir.exists() else []
        if len(existing_tile_pngs) >= expected_tile_count(topology):
            print(f"Tiles:    {expected_tile_count(topology)} (0–{expected_tile_count(topology) - 1})")
            print()
            print("Nothing to do: all expected tile plots already exist.")
            print(
                f"\nDone: 0 generated / {expected_tile_count(topology)} skipped "
                f"(exist, use --force)  ({expected_tile_count(topology)} total)")
            return

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

    window_args = ["--window", str(args.window)]

    # ── Overview ──────────────────────────────────────────────────────
    overview_needs_work = False
    overview_paths = {}
    if args.overview:
        ov_dir = plots_dir / "overview"
        overview_paths = {
            "workload": ov_dir / "overview_workload.png",
            "group_breakdown": ov_dir / "group_ipc_breakdown.png",
        }
        missing = [path.name for path in overview_paths.values() if not path.exists()]
        if not missing and not args.force:
            print("Overview: skipped (overview outputs already exist, use --force to overwrite)")
        else:
            overview_needs_work = True
            if args.dry_run:
                print(f"[dry-run] overview → {ov_dir}")

    # ── Per-tile detail ───────────────────────────────────────────────
    digits = TOPOLOGIES[topology]["tile_digits"]
    failed = []
    group_failed = []
    group_skipped = 0
    skipped = 0

    # ── Group/subgroup detail ────────────────────────────────────────
    group_work_items = []
    if args.group_details:
        for spec in group_detail_specs(plots_dir, topology, tile_ids):
            if spec["png_path"].exists() and not args.force:
                group_skipped += 1
                continue
            group_work_items.append(spec)
            if args.dry_run:
                rel = spec["png_path"].relative_to(plots_dir)
                print(f"[dry-run] {spec['title']} → {rel}")

    # Build work items
    work_items = []
    if args.skip_tile_details:
        print("Tile details: skipped (--skip-tile-details)")
    else:
        for tid in tile_ids:
            out_dir = tile_output_dir(plots_dir, topology, tid)
            tile_png = out_dir / f"tile_detail_tile{tid}.png"
            if tile_png.exists() and not args.force:
                skipped += 1
                continue
            tile_argv = [str(csv_path), str(tid)] + section_args + window_args + [
                "--output-dir", str(out_dir),
                "--prefix", "tile_detail",
            ]
            work_items.append((tid, out_dir, tile_argv, digits))

    if not args.dry_run and not overview_needs_work and not group_work_items and not work_items:
        print("Nothing to do: all requested plots already exist.")
        print(f"\nDone: 0 generated / {skipped} skipped (exist, use --force)  ({len(tile_ids)} total)")
        return

    rows = None
    rows_by_tile = None
    if not args.dry_run and (overview_needs_work or group_work_items or (work_items and args.jobs <= 1)):
        print("Loading stall CSV once for all tile plots ...", flush=True)
        rows = filter_rows(load_rows(csv_path), section=args.section)
        if not rows:
            sys.exit("No rows after filtering")
        if group_work_items or (work_items and args.jobs <= 1):
            rows_by_tile = group_rows_by_tile(rows, tile_ids)

    if args.dry_run:
        for tid, out_dir, _, dg in work_items:
            label = f"tile {tid:0{dg}d} → {out_dir.relative_to(plots_dir)}"
            print(f"[dry-run] {label}")
    else:
        if overview_needs_work:
            ov_dir.mkdir(parents=True, exist_ok=True)
            filter_desc = _plot_specific_tile._filter_desc(args)

            print(f"Overview → {ov_dir}")
            if args.force or not overview_paths["workload"].exists():
                agg = _plot_specific_tile.aggregate_rows(rows, args.window, context_field="tile")
                fig, _ = _plot_specific_tile.write_overview_page(
                    overview_paths["workload"], agg, filter_desc, args.window)
                plt.close(fig)
            else:
                print("  skipped overview_workload.png (exists)")

            if args.force or not overview_paths["group_breakdown"].exists():
                group_stats = _plot_specific_tile.build_group_overview_stats(rows)
                fig, _ = _plot_specific_tile.write_group_overview_page(
                    overview_paths["group_breakdown"], group_stats, filter_desc)
                plt.close(fig)
            else:
                print("  skipped group_ipc_breakdown.png (exists)")
            print()

        if group_work_items:
            for index, spec in enumerate(group_work_items, 1):
                rel = spec["png_path"].relative_to(plots_dir)
                print(f"[{index}/{len(group_work_items)}] {spec['title']} → {rel}", end=" ... ", flush=True)
                try:
                    spec["out_dir"].mkdir(parents=True, exist_ok=True)
                    group_rows = []
                    for tile_id in spec["tile_ids"]:
                        group_rows.extend(rows_by_tile.get(tile_id, []))
                    if not group_rows:
                        raise ValueError(f"No rows for {spec['title']}")
                    series = _plot_specific_tile.build_group_series(group_rows, spec["title"])
                    fig, _ = _plot_specific_tile.write_group_detail(spec["png_path"], series)
                    plt.close(fig)
                    print("ok")
                except Exception as e:
                    print(f"FAILED: {e}")
                    group_failed.append((spec["title"], str(e)))

    if not args.dry_run and work_items:
        n_jobs = min(args.jobs, len(work_items))
        # Ensure output dirs exist
        for _, out_dir, _, _ in work_items:
            out_dir.mkdir(parents=True, exist_ok=True)

        if n_jobs <= 1:
            # Fast sequential path: reuse the already loaded CSV rows.
            for i, (tid, out_dir, _, dg) in enumerate(work_items, 1):
                label = f"[{i}/{len(work_items)}] tile {tid:0{dg}d} → {out_dir.relative_to(plots_dir)}"
                print(label, end=" ... ", flush=True)
                try:
                    tile_rows = rows_by_tile.get(tid)
                    if not tile_rows:
                        raise ValueError(f"No rows for tile {tid}")
                    ts = _plot_specific_tile.build_tile_series(tile_rows, csv_path, tid)
                    tile_png = out_dir / f"tile_detail_tile{tid}.png"
                    fig, _ = _plot_specific_tile.write_tile_detail(tile_png, ts)
                    plt.close(fig)
                    print("ok")
                except Exception as e:
                    print(f"FAILED: {e}")
                    failed.append((tid, str(e)))
        else:
            print(f"Generating {len(work_items)} tile plots with {n_jobs} parallel workers ...")
            print("Note: parallel workers currently reload the CSV independently; for very large CSVs, jobs=1 may still be faster.")
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
        parts = []
        if args.skip_tile_details:
            parts.append("tile details skipped")
        else:
            generated = len(work_items) - len(failed)
            parts.append(f"{generated} tile details generated")
            if skipped:
                parts.append(f"{skipped} skipped (exist, use --force)")
        if overview_needs_work:
            parts.append("overview generated")
        if args.group_details:
            generated = len(group_work_items) - len(group_failed)
            parts.append(f"{generated} group details generated")
            if group_skipped:
                parts.append(f"{group_skipped} group details skipped (exist, use --force)")
        if failed:
            parts.append(f"{len(failed)} failed")
        if group_failed:
            parts.append(f"{len(group_failed)} group details failed")
        print(f"\nDone: {' / '.join(parts)}  ({len(tile_ids)} total)")
        if failed:
            print("Failed tiles:")
            for tid, err in failed:
                print(f"  tile {tid}: {err}")
        if group_failed:
            print("Failed group details:")
            for title, err in group_failed:
                print(f"  {title}: {err}")


if __name__ == "__main__":
    main()

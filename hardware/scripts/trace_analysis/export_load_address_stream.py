#!/usr/bin/env python3
"""Export source-core load address streams from tile path monitor CSVs.

This is intentionally a plain CSV extractor.  It preserves cycle-level
valid/ready/fire information so later prefetch models can choose whether they
want accepted loads only or every stalled demand attempt.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from _workflow_metadata import format_topology, load_result_dir_topology
from operand_regions import (
    add_operand_region_args,
    classify_operand,
    format_operand_regions,
    load_operand_regions,
    normalized_hex_int,
    operand_address_from_row,
    operand_classification_address_field,
    operand_regions_are_exact,
)


DEFAULT_POINT = 'tcdm_remote'
STREAM_FIELDS = (
    'cycle',
    'time',
    'source_group',
    'source_tile',
    'source_tile_in_group',
    'source_tile_core',
    'source_global_core',
    'point',
    'port',
    'prefetch_domain',
    'remote_lane',
    'bank',
    'valid',
    'ready',
    'fire',
    'blocked',
    'write',
    'back2local',
    'route_addr',
    'source_addr',
    'operand_addr',
    'operand_addr_int',
    'operand',
    'meta_id',
    'payload_core',
    'core_valid_index',
    'core_fire_index',
)

SORT_KEYS = {
    'input': (),
    'cycle': ('cycle', 'source_tile', 'port', 'source_tile_core', 'core_valid_index'),
    'tile-port-cycle-core': ('source_tile', 'port', 'cycle', 'source_tile_core', 'core_valid_index'),
    'tile-port-core-cycle': ('source_tile', 'port', 'source_tile_core', 'cycle', 'core_valid_index'),
    'tile-core-cycle': ('source_tile', 'source_tile_core', 'cycle', 'port', 'core_valid_index'),
    'domain-tile-port-core-cycle': (
        'prefetch_domain',
        'source_tile',
        'port',
        'source_tile_core',
        'cycle',
        'core_valid_index',
    ),
}


def parse_bit(value: str | None) -> int:
    return 1 if value == '1' else 0


def parse_int(value: str | None, default: int = -1) -> int:
    if value in (None, ''):
        return default
    return int(value, 0)


def result_dir_arg(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f'result directory not found: {value}')
    return path


def default_output_path(result_dir: Path) -> Path:
    return result_dir / 'analysis' / 'load_streams' / 'load_address_stream.csv'


def classify_prefetch_domain(port: int) -> tuple[str, str]:
    if port < 0:
        return 'unknown', ''
    if port == 0:
        return 'local_port', ''
    return 'remote_port', str(port - 1)


def sort_value(row: dict[str, object], key: str) -> object:
    value = row.get(key, '')
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value == '':
            return -1
        try:
            return int(value, 0)
        except ValueError:
            return value
    return value


def sort_rows(rows: list[dict[str, object]], sort_mode: str) -> None:
    keys = SORT_KEYS[sort_mode]
    if not keys:
        return
    rows.sort(key=lambda row: tuple(sort_value(row, key) for key in keys))


def monitor_paths(result_dir: Path, monitor_dir: Path | None) -> list[Path]:
    directory = monitor_dir or result_dir / 'monitor'
    if not directory.is_dir():
        raise SystemExit(f'Monitor directory not found: {directory}')
    paths = sorted(directory.glob('tile_path_tile*.csv'))
    if not paths:
        raise SystemExit(f'No tile path monitor CSVs found in {directory}')
    return paths


def graph_dir_for_regions(result_dir: Path) -> Path:
    graph_dir = result_dir / 'analysis' / 'path_graph'
    return graph_dir if graph_dir.is_dir() else result_dir


def write_summary(path: Path, rows: list[tuple[str, object]]) -> None:
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('metric', 'value'))
        for key, value in rows:
            writer.writerow((key, value))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Export cycle-ordered load addresses by source tile/core from tile path monitor CSVs.'
    )
    parser.add_argument('result_dir', type=result_dir_arg, help='benchmark result directory')
    parser.add_argument(
        '--monitor-dir',
        type=Path,
        help='monitor directory; defaults to <result_dir>/monitor',
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='output CSV path; defaults to <result_dir>/analysis/load_streams/load_address_stream.csv',
    )
    parser.add_argument(
        '--point',
        action='append',
        help=f'RTL monitor point to export; may be repeated; default: {DEFAULT_POINT}',
    )
    parser.add_argument(
        '--all-points',
        action='store_true',
        help='export all monitor points instead of the default source remote-load point',
    )
    parser.add_argument(
        '--accepted-only',
        action='store_true',
        help='only export rows where fire=1',
    )
    parser.add_argument(
        '--operand',
        action='append',
        help='only export this operand label; may be repeated, for example --operand B',
    )
    parser.add_argument(
        '--sort',
        choices=sorted(SORT_KEYS),
        default='input',
        help='output ordering; default preserves monitor file order',
    )
    parser.add_argument(
        '--allow-inexact-operands',
        action='store_true',
        help='allow legacy/missing operand metadata instead of requiring exact source-address regions',
    )
    add_operand_region_args(parser)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    output = (args.output or default_output_path(result_dir)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    sidecar = result_dir / 'analysis' / 'operand_regions.json'
    if args.operand_regions_json is None and sidecar.is_file():
        args.operand_regions_json = sidecar
    args.require_exact_operands = not args.allow_inexact_operands
    operand_regions = load_operand_regions(graph_dir_for_regions(result_dir), args)

    topology = load_result_dir_topology(result_dir)
    if topology is None:
        raise SystemExit(f'Could not load topology metadata from {result_dir}')
    cores_per_tile = int(topology['NUM_CORES_PER_TILE'])
    num_cores = int(topology['NUM_CORES'])
    num_groups = int(topology['NUM_GROUPS'])
    num_tiles = num_cores // cores_per_tile
    tiles_per_group = num_tiles // num_groups

    selected_points = None if args.all_points else set(args.point or [DEFAULT_POINT])
    selected_operands = set(args.operand or [])

    valid_index: Counter[int] = Counter()
    fire_index: Counter[int] = Counter()
    counts: Counter[str] = Counter()
    operand_counts: Counter[str] = Counter()
    state_counts: Counter[str] = Counter()
    output_rows: list[dict[str, object]] = []

    for path in monitor_paths(result_dir, args.monitor_dir):
        with path.open(newline='') as csv_file:
            for row in csv.DictReader(csv_file):
                counts['input_rows'] += 1
                point = row.get('point', '')
                if selected_points is not None and point not in selected_points:
                    counts['skipped_point'] += 1
                    continue
                valid = parse_bit(row.get('valid'))
                ready = parse_bit(row.get('ready'))
                fire = parse_bit(row.get('fire'))
                if not valid:
                    counts['skipped_invalid'] += 1
                    continue
                if args.accepted_only and not fire:
                    counts['skipped_not_fired'] += 1
                    continue

                operand_addr = operand_address_from_row(row, operand_regions)
                operand = classify_operand(operand_addr, operand_regions)
                if selected_operands and operand not in selected_operands:
                    counts['skipped_operand'] += 1
                    continue

                tile = parse_int(row.get('tile'))
                tile_core = parse_int(row.get('core'))
                port = parse_int(row.get('port'))
                prefetch_domain, remote_lane = classify_prefetch_domain(port)
                source_global_core = tile * cores_per_tile + tile_core
                source_group = tile // tiles_per_group
                source_tile_in_group = tile % tiles_per_group
                blocked = int(bool(valid and not ready))

                valid_index[source_global_core] += 1
                core_fire_index = ''
                if fire:
                    fire_index[source_global_core] += 1
                    core_fire_index = fire_index[source_global_core]

                operand_int = normalized_hex_int(operand_addr)
                out_row = {
                    'cycle': row.get('cycle', ''),
                    'time': row.get('time', ''),
                    'source_group': source_group,
                    'source_tile': tile,
                    'source_tile_in_group': source_tile_in_group,
                    'source_tile_core': tile_core,
                    'source_global_core': source_global_core,
                    'point': point,
                    'port': port,
                    'prefetch_domain': prefetch_domain,
                    'remote_lane': remote_lane,
                    'bank': row.get('bank', ''),
                    'valid': valid,
                    'ready': ready,
                    'fire': fire,
                    'blocked': blocked,
                    'write': parse_bit(row.get('write')),
                    'back2local': parse_bit(row.get('back2local')),
                    'route_addr': row.get('addr', ''),
                    'source_addr': row.get('source_addr', ''),
                    'operand_addr': operand_addr,
                    'operand_addr_int': '' if operand_int is None else operand_int,
                    'operand': operand,
                    'meta_id': row.get('meta_id', ''),
                    'payload_core': row.get('payload_core', ''),
                    'core_valid_index': valid_index[source_global_core],
                    'core_fire_index': core_fire_index,
                }
                output_rows.append(out_row)
                counts['written_rows'] += 1
                operand_counts[operand] += 1
                counts[f'prefetch_domain_{prefetch_domain}'] += 1
                if blocked:
                    state_counts['blocked'] += 1
                elif fire:
                    state_counts['fire'] += 1
                else:
                    state_counts['valid_no_fire'] += 1

    sort_rows(output_rows, args.sort)

    with output.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=STREAM_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)

    summary_path = output.with_name(output.stem + '_summary.csv')
    write_summary(
        summary_path,
        [
            ('result_dir', result_dir),
            ('output', output),
            ('topology', format_topology(topology)),
            ('monitor_points', 'all' if selected_points is None else ','.join(sorted(selected_points))),
            ('accepted_only', int(args.accepted_only)),
            ('sort', args.sort),
            ('operand_filter', ','.join(sorted(selected_operands)) if selected_operands else 'all'),
            ('operand_region_source', operand_regions.source),
            ('operand_address_field', operand_classification_address_field(operand_regions)),
            ('operand_regions_exact', int(operand_regions_are_exact(operand_regions))),
            ('operand_regions', format_operand_regions(operand_regions)),
            *sorted(counts.items()),
            *[(f'operand_{key}_rows', value) for key, value in sorted(operand_counts.items())],
            *[(f'state_{key}_rows', value) for key, value in sorted(state_counts.items())],
        ],
    )

    print(f'Wrote {counts["written_rows"]} rows to {output}')
    print(f'Wrote summary to {summary_path}')


if __name__ == '__main__':
    main()

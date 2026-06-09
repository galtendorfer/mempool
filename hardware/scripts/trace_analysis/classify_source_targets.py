#!/usr/bin/env python3
"""Classify source-tile requests by operand and decoded target tile.

This script is a CSV-only drilldown for path monitor datasets.  It focuses on
the source-side route selection point, usually `tcdm_remote`, and answers:
which source tile/core requested which route port, which physical target tile
inside the destination group that request decodes to, and whether the request is
part of a multi-core same-port burst.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from path_route_checkpoints import (
    decode_target_from_source,
    infer_bank_count,
    infer_power_of_two_domain,
    infer_tiles_per_group,
)
from path_graph_common import infer_group_and_tile, parse_int, read_csv_rows, require_graph_dir
from operand_regions import (
    add_operand_region_args,
    classify_operand,
    format_operand_regions,
    load_operand_regions,
    normalized_hex_int,
    operand_classification_address_field,
    operand_address_from_row,
    operand_regions_are_exact,
)


DETAIL_FIELDS = (
    'cycle',
    'source_group',
    'source_tile',
    'source_tile_in_group',
    'source_core',
    'source_global_core',
    'rowblock',
    'rowblock_first_row',
    'rowblock_last_row',
    'rowblock_worker_tile_a',
    'rowblock_worker_tile_b',
    'port',
    'fanin_requests',
    'fanin_stalls',
    'fanin_fires',
    'high_fanin',
    'state',
    'valid',
    'stall',
    'fire',
    'write',
    'addr',
    'source_addr',
    'operand_addr',
    'operand',
    'request_class',
    'decoded_target_group',
    'decoded_target_tile',
    'decoded_target_tile_in_group',
    'decoded_target_bank',
    'decoded_target_addr',
    'source_to_target_group_relation',
    'source_to_target_tile_relation',
    'meta_id',
    'payload_core',
    'subject_id',
)

TILE_CYCLE_FIELDS = (
    'cycle',
    'source_group',
    'source_tile',
    'source_tile_in_group',
    'port',
    'fanin_requests',
    'fanin_stalls',
    'fanin_fires',
    'source_arbitration_floor',
    'excess_stalls',
    'high_fanin',
    'operand_mix',
    'request_class_mix',
    'target_tiles',
    'target_tile_in_groups',
    'A_requests',
    'B_requests',
    'C_requests',
    'other_requests',
    'A_local_owner_requests',
    'A_neighbor_requests',
    'A_other_requests',
    'B_same_tile_requests',
    'B_same_group_requests',
    'B_remote_requests',
    'other_requests_classified',
)

MATRIX_FIELDS = (
    'source_group',
    'source_tile',
    'source_tile_in_group',
    'target_group',
    'target_tile',
    'target_tile_in_group',
    'port',
    'operand',
    'request_class',
    'requests',
    'stalls',
    'fires',
    'high_fanin_requests',
    'tile_cycles',
    'high_fanin_tile_cycles',
)

TARGET_IN_GROUP_FIELDS = (
    'source_group',
    'source_tile',
    'source_tile_in_group',
    'target_tile_in_group',
    'port',
    'operand',
    'request_class',
    'requests',
    'stalls',
    'fires',
    'high_fanin_requests',
    'tile_cycles',
    'high_fanin_tile_cycles',
)

SUMMARY_FIELDS = ('metric', 'value')

OPERAND_REGION_AUDIT_FIELDS = (
    'operand',
    'region_source',
    'classification_address_field',
    'configured_start',
    'configured_end',
    'observed_request_rows',
    'observed_valid_observations',
    'observed_addr_min',
    'observed_addr_max',
    'observed_out_of_region_valid_observations',
    'note',
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'input_path',
        type=Path,
        help='path_graph directory, cycle_node_state.csv, or result directory containing analysis/path_graph',
    )
    parser.add_argument('--cycle-start', type=int, help='first cycle to include')
    parser.add_argument('--cycle-end', type=int, help='last cycle to include')
    parser.add_argument('--port', type=int, default=0, help='route port to analyze')
    parser.add_argument('--node-point', default='tcdm_remote', help='cycle_node_state point to classify')
    parser.add_argument(
        '--high-fanin-threshold',
        type=int,
        default=2,
        help='tile-cycle same-port request count treated as contention',
    )
    parser.add_argument('--cores-per-tile', type=int, default=4, help='source cores per tile')
    parser.add_argument('--rowblocks-per-tile-pair', type=int, default=1, help=argparse.SUPPRESS)
    add_operand_region_args(parser)
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='directory for generated CSVs; defaults under the graph directory',
    )
    parser.add_argument('--prefix', default='port0_source_target', help='output filename prefix')
    parser.add_argument('--force', action='store_true', help='overwrite existing output files')
    return parser.parse_args()


def resolve_graph_dir(input_path: Path) -> Path:
    if input_path.is_dir() and (input_path / 'cycle_node_state.csv').is_file():
        return input_path
    if input_path.is_file() and input_path.name == 'cycle_node_state.csv':
        return input_path.parent
    nested = input_path / 'analysis' / 'path_graph'
    if nested.is_dir() and (nested / 'cycle_node_state.csv').is_file():
        return nested
    return require_graph_dir(input_path)


def ensure_can_write(paths: list[Path], force: bool) -> None:
    for path in paths:
        if path.exists() and not force:
            raise SystemExit(f'Output exists: {path} (use --force to overwrite)')


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def format_hex(value: int | None) -> str:
    return '' if value is None else f'0x{value:x}'


def operand_region_note(source: str, has_region: bool) -> str:
    if not has_region:
        return 'observed operand has no configured region'
    if 'legacy-route' in source:
        return 'legacy route-address fallback; not transcript-emitted operand metadata'
    if 'sidecar' in source:
        return 'sidecar operand metadata'
    if 'transcript' in source:
        return 'transcript operand metadata'
    if 'cli' in source:
        return 'command-line operand range override'
    if source == 'none':
        return 'no operand regions configured'
    return 'operand region metadata'


def build_operand_region_audit(detail_rows: list[dict[str, object]], operand_regions) -> list[dict[str, object]]:
    region_by_name = {region.name: region for region in operand_regions.regions}
    stats: dict[str, dict[str, object]] = {}

    for name in region_by_name:
        stats[name] = {
            'rows': 0,
            'valid': 0,
            'min_addr': None,
            'max_addr': None,
            'out_of_region_valid': 0,
        }

    for row in detail_rows:
        operand = str(row.get('operand') or 'other')
        entry = stats.setdefault(operand, {
            'rows': 0,
            'valid': 0,
            'min_addr': None,
            'max_addr': None,
            'out_of_region_valid': 0,
        })
        valid = int(row.get('valid') or 0)
        entry['rows'] = int(entry['rows']) + 1
        entry['valid'] = int(entry['valid']) + valid
        addr = normalized_hex_int(str(row.get('operand_addr') or ''))
        if addr is not None:
            min_addr = entry['min_addr']
            max_addr = entry['max_addr']
            entry['min_addr'] = addr if min_addr is None else min(int(min_addr), addr)
            entry['max_addr'] = addr if max_addr is None else max(int(max_addr), addr)
            region = region_by_name.get(operand)
            if region is not None and not (region.start <= addr <= region.end):
                entry['out_of_region_valid'] = int(entry['out_of_region_valid']) + valid

    rows = []
    for operand in sorted(stats):
        region = region_by_name.get(operand)
        entry = stats[operand]
        rows.append({
            'operand': operand,
            'region_source': operand_regions.source,
            'classification_address_field': operand_classification_address_field(operand_regions),
            'configured_start': format_hex(region.start if region else None),
            'configured_end': format_hex(region.end if region else None),
            'observed_request_rows': entry['rows'],
            'observed_valid_observations': entry['valid'],
            'observed_addr_min': format_hex(entry['min_addr']),
            'observed_addr_max': format_hex(entry['max_addr']),
            'observed_out_of_region_valid_observations': entry['out_of_region_valid'],
            'note': operand_region_note(operand_regions.source, region is not None),
        })
    return rows


def row_count(row: dict[str, str], field: str) -> int:
    value = parse_int(row.get(field), 0)
    return 0 if value is None else value


def row_active(row: dict[str, str]) -> bool:
    return any(row_count(row, field) > 0 for field in ('valid', 'stall', 'fire'))


def format_counter(counter: Counter[str]) -> str:
    return ';'.join(f'{key}:{counter[key]}' for key in sorted(counter) if counter[key])


def format_values(values: set[int | str]) -> str:
    sortable = [value for value in values if value != '']
    return ';'.join(str(value) for value in sorted(sortable, key=lambda item: int(item)))


def new_aggregate() -> dict[str, object]:
    return {
        'requests': 0,
        'stalls': 0,
        'fires': 0,
        'high_fanin_requests': 0,
        'tile_cycles_set': set(),
        'high_fanin_tile_cycles_set': set(),
    }


def infer_tile_configs(graph_dir: Path) -> tuple[list[dict[str, int]], int, int, int]:
    nodes = read_csv_rows(graph_dir / 'nodes.csv')
    tile_ids = sorted({
        tile for row in nodes
        if row.get('point') == 'tcdm_remote'
        for tile in [parse_int(row.get('tile'))]
        if tile is not None and tile >= 0
    })
    if not tile_ids:
        raise SystemExit('Could not infer source tiles from nodes.csv')

    tile_configs = []
    for tile in tile_ids:
        group, tile_in_group = infer_group_and_tile(nodes, tile)
        tile_configs.append({'tile': tile, 'group': group, 'tile_in_group': tile_in_group})
    tiles_per_group = infer_tiles_per_group(tile_configs)
    num_groups = infer_power_of_two_domain(max(config['group'] for config in tile_configs), 1)
    banks_per_tile = infer_bank_count(nodes)
    return tile_configs, num_groups, tiles_per_group, banks_per_tile


def relation_to_target(source_group: int, source_tile: int, decoded: dict[str, int | str] | None) -> tuple[str, str]:
    if decoded is None:
        return '', ''
    target_group = int(decoded['target_group'])
    target_tile = int(decoded['target_tile'])
    if target_group == source_group:
        group_relation = 'same_group'
    else:
        group_relation = 'remote_group'
    if target_tile == source_tile:
        tile_relation = 'same_tile'
    elif target_group == source_group:
        tile_relation = 'same_group_other_tile'
    else:
        tile_relation = 'remote_tile'
    return group_relation, tile_relation


def request_class(
    operand: str,
    source_group: int,
    source_tile: int,
    target_group: int | str,
    target_tile: int | str,
    rowblock_worker_tiles: tuple[int, int],
) -> str:
    if target_group == '' or target_tile == '':
        return f'{operand}_unknown' if operand in {'A', 'B'} else 'other'
    target_group_int = int(target_group)
    target_tile_int = int(target_tile)
    if operand == 'A':
        if target_tile_int == source_tile:
            return 'A_local_owner'
        if target_tile_int in rowblock_worker_tiles:
            return 'A_neighbor'
        return 'A_other'
    if operand == 'B':
        if target_tile_int == source_tile:
            return 'B_same_tile'
        if target_group_int == source_group:
            return 'B_same_group'
        return 'B_remote'
    return 'other'


def add_matrix_count(
    matrix: dict[tuple[int, int, int, int, int, int, int, str, str], dict[str, object]],
    detail: dict[str, object],
    high_fanin: bool,
) -> None:
    target_group = detail.get('decoded_target_group', '')
    target_tile = detail.get('decoded_target_tile', '')
    target_tile_in_group = detail.get('decoded_target_tile_in_group', '')
    if target_group == '' or target_tile == '' or target_tile_in_group == '':
        return
    key = (
        int(detail['source_group']),
        int(detail['source_tile']),
        int(detail['source_tile_in_group']),
        int(target_group),
        int(target_tile),
        int(target_tile_in_group),
        int(detail['port']),
        str(detail['operand']),
        str(detail['request_class']),
    )
    counts = matrix[key]
    counts['requests'] += int(detail['valid'])
    counts['stalls'] += int(detail['stall'])
    counts['fires'] += int(detail['fire'])
    if high_fanin:
        counts['high_fanin_requests'] += int(detail['valid'])
        counts['high_fanin_tile_cycles_set'].add((int(detail['cycle']), int(detail['source_tile'])))
    counts['tile_cycles_set'].add((int(detail['cycle']), int(detail['source_tile'])))


def add_target_in_group_count(
    summary: dict[tuple[int, int, int, int, int, str, str], dict[str, object]],
    detail: dict[str, object],
    high_fanin: bool,
) -> None:
    target_tile_in_group = detail.get('decoded_target_tile_in_group', '')
    if target_tile_in_group == '':
        return
    key = (
        int(detail['source_group']),
        int(detail['source_tile']),
        int(detail['source_tile_in_group']),
        int(target_tile_in_group),
        int(detail['port']),
        str(detail['operand']),
        str(detail['request_class']),
    )
    counts = summary[key]
    counts['requests'] += int(detail['valid'])
    counts['stalls'] += int(detail['stall'])
    counts['fires'] += int(detail['fire'])
    if high_fanin:
        counts['high_fanin_requests'] += int(detail['valid'])
        counts['high_fanin_tile_cycles_set'].add((int(detail['cycle']), int(detail['source_tile'])))
    counts['tile_cycles_set'].add((int(detail['cycle']), int(detail['source_tile'])))


def classify_requests(
    graph_dir: Path,
    cycle_start: int | None,
    cycle_end: int | None,
    port: int,
    node_point: str,
    high_fanin_threshold: int,
    cores_per_tile: int,
    operand_regions,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
]:
    tile_configs, num_groups, tiles_per_group, banks_per_tile = infer_tile_configs(graph_dir)
    tile_config_by_id = {config['tile']: config for config in tile_configs}
    tile_cycle_rows: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    tile_cycle_counts: dict[tuple[int, int], Counter[str]] = defaultdict(Counter)

    with (graph_dir / 'cycle_node_state.csv').open(newline='') as file:
        reader = csv.DictReader(file)
        required = {'cycle', 'tile', 'point', 'core', 'port', 'valid', 'stall', 'fire', 'state', 'addr'}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise SystemExit(f'Missing required columns in cycle_node_state.csv: {", ".join(sorted(missing))}')

        for row_index, row in enumerate(reader, start=1):
            if row.get('point') != node_point:
                continue
            row_port = parse_int(row.get('port'))
            if row_port is None or row_port != port:
                continue
            cycle = parse_int(row.get('cycle'))
            source_tile = parse_int(row.get('tile'))
            source_core = parse_int(row.get('core'))
            if cycle is None or source_tile is None or source_core is None:
                continue
            if cycle_start is not None and cycle < cycle_start:
                continue
            if cycle_end is not None and cycle > cycle_end:
                continue
            if not row_active(row):
                continue
            valid = row_count(row, 'valid')
            if valid <= 0:
                continue

            tile_config = tile_config_by_id.get(source_tile)
            if tile_config is None:
                continue
            source_group = int(tile_config['group'])
            source_tile_in_group = int(tile_config['tile_in_group'])
            source_global_core = source_tile * cores_per_tile + source_core
            rowblock = source_global_core // (2 * cores_per_tile)
            rowblock_worker_tiles = (2 * rowblock, 2 * rowblock + 1)
            operand_addr = operand_address_from_row(row, operand_regions)
            operand = classify_operand(operand_addr, operand_regions)
            decoded = decode_target_from_source(
                source_tile,
                source_group,
                port,
                row.get('addr', ''),
                num_groups=num_groups,
                tiles_per_group=tiles_per_group,
                banks_per_tile=banks_per_tile,
                back2local=row_count(row, 'back2local') > 0,
            )
            group_relation, tile_relation = relation_to_target(source_group, source_tile, decoded)
            target_group = '' if decoded is None else decoded['target_group']
            target_tile = '' if decoded is None else decoded['target_tile']
            target_tile_in_group = '' if decoded is None else decoded['target_tile_in_group']
            target_bank = '' if decoded is None else decoded['target_bank']
            target_addr = '' if decoded is None else decoded['target_addr']
            classification = request_class(
                operand,
                source_group,
                source_tile,
                target_group,
                target_tile,
                rowblock_worker_tiles,
            )

            detail = {
                'cycle': cycle,
                'source_group': source_group,
                'source_tile': source_tile,
                'source_tile_in_group': source_tile_in_group,
                'source_core': source_core,
                'source_global_core': source_global_core,
                'rowblock': rowblock,
                'rowblock_first_row': 4 * rowblock,
                'rowblock_last_row': 4 * rowblock + 3,
                'rowblock_worker_tile_a': rowblock_worker_tiles[0],
                'rowblock_worker_tile_b': rowblock_worker_tiles[1],
                'port': port,
                'fanin_requests': 0,
                'fanin_stalls': 0,
                'fanin_fires': 0,
                'high_fanin': 0,
                'state': row.get('state', ''),
                'valid': valid,
                'stall': row_count(row, 'stall'),
                'fire': row_count(row, 'fire'),
                'write': row.get('write', ''),
                'addr': row.get('addr', ''),
                'source_addr': row.get('source_addr', ''),
                'operand_addr': operand_addr,
                'operand': operand,
                'request_class': classification,
                'decoded_target_group': target_group,
                'decoded_target_tile': target_tile,
                'decoded_target_tile_in_group': target_tile_in_group,
                'decoded_target_bank': target_bank,
                'decoded_target_addr': target_addr,
                'source_to_target_group_relation': group_relation,
                'source_to_target_tile_relation': tile_relation,
                'meta_id': row.get('meta_id', ''),
                'payload_core': row.get('payload_core', ''),
                'subject_id': row.get('subject_id', ''),
            }
            key = (cycle, source_tile)
            tile_cycle_rows[key].append(detail)
            tile_cycle_counts[key]['requests'] += valid
            tile_cycle_counts[key]['stalls'] += int(detail['stall'])
            tile_cycle_counts[key]['fires'] += int(detail['fire'])

            if row_index % 500000 == 0:
                print(f'Scanned {row_index} cycle_node_state rows', flush=True)

    detail_rows: list[dict[str, object]] = []
    tile_cycle_summary_rows: list[dict[str, object]] = []
    source_target_matrix: dict[tuple[int, int, int, int, int, int, int, str, str], dict[str, object]] = defaultdict(new_aggregate)
    target_in_group_summary: dict[tuple[int, int, int, int, int, str, str], dict[str, object]] = defaultdict(new_aggregate)
    overall = Counter()
    overall['num_groups'] = num_groups
    overall['tiles_per_group'] = tiles_per_group
    overall['banks_per_tile'] = banks_per_tile
    overall['operand_region_source'] = operand_regions.source
    overall['operand_address_field'] = operand_classification_address_field(operand_regions)
    overall['operand_regions_exact'] = int(operand_regions_are_exact(operand_regions))
    overall['operand_regions'] = format_operand_regions(operand_regions)
    if operand_regions.sidecar_path is not None:
        overall['operand_regions_json'] = str(operand_regions.sidecar_path)

    for key in sorted(tile_cycle_rows):
        cycle, source_tile = key
        rows = tile_cycle_rows[key]
        counts = tile_cycle_counts[key]
        fanin_requests = counts['requests']
        fanin_stalls = counts['stalls']
        fanin_fires = counts['fires']
        high_fanin = fanin_requests >= high_fanin_threshold
        source_arbitration_floor = max(0, fanin_requests - 1)
        excess_stalls = max(0, fanin_stalls - source_arbitration_floor)
        operand_counts = Counter(str(row['operand']) for row in rows for _ in range(int(row['valid'])))
        class_counts = Counter(str(row['request_class']) for row in rows for _ in range(int(row['valid'])))
        target_tiles = {row['decoded_target_tile'] for row in rows if row['decoded_target_tile'] != ''}
        target_tile_in_groups = {row['decoded_target_tile_in_group'] for row in rows if row['decoded_target_tile_in_group'] != ''}
        source_group = int(rows[0]['source_group'])
        source_tile_in_group = int(rows[0]['source_tile_in_group'])

        for row in rows:
            row['fanin_requests'] = fanin_requests
            row['fanin_stalls'] = fanin_stalls
            row['fanin_fires'] = fanin_fires
            row['high_fanin'] = int(high_fanin)
            detail_rows.append(row)
            add_matrix_count(source_target_matrix, row, high_fanin)
            add_target_in_group_count(target_in_group_summary, row, high_fanin)

        summary_row = {
            'cycle': cycle,
            'source_group': source_group,
            'source_tile': source_tile,
            'source_tile_in_group': source_tile_in_group,
            'port': port,
            'fanin_requests': fanin_requests,
            'fanin_stalls': fanin_stalls,
            'fanin_fires': fanin_fires,
            'source_arbitration_floor': source_arbitration_floor,
            'excess_stalls': excess_stalls,
            'high_fanin': int(high_fanin),
            'operand_mix': format_counter(operand_counts),
            'request_class_mix': format_counter(class_counts),
            'target_tiles': format_values(target_tiles),
            'target_tile_in_groups': format_values(target_tile_in_groups),
            'A_requests': operand_counts['A'],
            'B_requests': operand_counts['B'],
            'C_requests': operand_counts['C'],
            'other_requests': operand_counts['other'],
            'A_local_owner_requests': class_counts['A_local_owner'],
            'A_neighbor_requests': class_counts['A_neighbor'],
            'A_other_requests': class_counts['A_other'],
            'B_same_tile_requests': class_counts['B_same_tile'],
            'B_same_group_requests': class_counts['B_same_group'],
            'B_remote_requests': class_counts['B_remote'],
            'other_requests_classified': class_counts['other'],
        }
        tile_cycle_summary_rows.append(summary_row)
        overall['tile_cycles_with_requests'] += 1
        overall['total_requests'] += fanin_requests
        overall['total_stalls'] += fanin_stalls
        overall['total_fires'] += fanin_fires
        overall['source_arbitration_floor'] += source_arbitration_floor
        overall['excess_stalls'] += excess_stalls
        if high_fanin:
            overall['high_fanin_tile_cycles'] += 1
            overall['high_fanin_requests'] += fanin_requests
            overall['high_fanin_stalls'] += fanin_stalls
        for operand, count in operand_counts.items():
            overall[f'operand_{operand}_requests'] += count
        for class_name, count in class_counts.items():
            overall[f'class_{class_name}_requests'] += count

    matrix_rows = []
    for key, counts in sorted(source_target_matrix.items()):
        (
            source_group,
            source_tile,
            source_tile_in_group,
            target_group,
            target_tile,
            target_tile_in_group,
            port_value,
            operand,
            class_name,
        ) = key
        matrix_rows.append({
            'source_group': source_group,
            'source_tile': source_tile,
            'source_tile_in_group': source_tile_in_group,
            'target_group': target_group,
            'target_tile': target_tile,
            'target_tile_in_group': target_tile_in_group,
            'port': port_value,
            'operand': operand,
            'request_class': class_name,
            'requests': counts['requests'],
            'stalls': counts['stalls'],
            'fires': counts['fires'],
            'high_fanin_requests': counts['high_fanin_requests'],
            'tile_cycles': len(counts['tile_cycles_set']),
            'high_fanin_tile_cycles': len(counts['high_fanin_tile_cycles_set']),
        })

    target_in_group_rows = []
    for key, counts in sorted(target_in_group_summary.items()):
        (
            source_group,
            source_tile,
            source_tile_in_group,
            target_tile_in_group,
            port_value,
            operand,
            class_name,
        ) = key
        target_in_group_rows.append({
            'source_group': source_group,
            'source_tile': source_tile,
            'source_tile_in_group': source_tile_in_group,
            'target_tile_in_group': target_tile_in_group,
            'port': port_value,
            'operand': operand,
            'request_class': class_name,
            'requests': counts['requests'],
            'stalls': counts['stalls'],
            'fires': counts['fires'],
            'high_fanin_requests': counts['high_fanin_requests'],
            'tile_cycles': len(counts['tile_cycles_set']),
            'high_fanin_tile_cycles': len(counts['high_fanin_tile_cycles_set']),
        })

    summary_rows = [{'metric': metric, 'value': value} for metric, value in sorted(overall.items())]
    return detail_rows, tile_cycle_summary_rows, matrix_rows, target_in_group_rows, summary_rows


def main() -> int:
    args = parse_args()
    graph_dir = resolve_graph_dir(args.input_path)
    operand_regions = load_operand_regions(graph_dir, args)
    prefix = args.prefix
    if prefix == 'port0_source_target' and args.port != 0:
        prefix = f'port{args.port}_source_target'
    output_dir = args.output_dir or (graph_dir / f'{prefix}_classification')
    paths = {
        'details': output_dir / f'{prefix}_details.csv',
        'tile_cycles': output_dir / f'{prefix}_tile_cycles.csv',
        'matrix': output_dir / f'{prefix}_source_target_matrix.csv',
        'target_in_group': output_dir / f'{prefix}_target_tile_in_group.csv',
        'summary': output_dir / f'{prefix}_summary.csv',
        'operand_region_audit': output_dir / f'{prefix}_operand_region_audit.csv',
    }
    ensure_can_write(list(paths.values()), args.force)
    (
        detail_rows,
        tile_cycle_rows,
        matrix_rows,
        target_in_group_rows,
        summary_rows,
    ) = classify_requests(
        graph_dir,
        args.cycle_start,
        args.cycle_end,
        args.port,
        args.node_point,
        args.high_fanin_threshold,
        args.cores_per_tile,
        operand_regions,
    )
    write_csv(paths['details'], detail_rows, DETAIL_FIELDS)
    write_csv(paths['tile_cycles'], tile_cycle_rows, TILE_CYCLE_FIELDS)
    write_csv(paths['matrix'], matrix_rows, MATRIX_FIELDS)
    write_csv(paths['target_in_group'], target_in_group_rows, TARGET_IN_GROUP_FIELDS)
    write_csv(paths['summary'], summary_rows, SUMMARY_FIELDS)
    operand_region_audit_rows = build_operand_region_audit(detail_rows, operand_regions)
    write_csv(paths['operand_region_audit'], operand_region_audit_rows, OPERAND_REGION_AUDIT_FIELDS)

    summary = {str(row['metric']): row['value'] for row in summary_rows}
    print(f'Wrote detail rows: {len(detail_rows)} -> {paths["details"]}')
    print(f'Wrote tile-cycle rows: {len(tile_cycle_rows)} -> {paths["tile_cycles"]}')
    print(f'Wrote source-target matrix rows: {len(matrix_rows)} -> {paths["matrix"]}')
    print(f'Wrote target-in-group rows: {len(target_in_group_rows)} -> {paths["target_in_group"]}')
    print(f'Wrote summary -> {paths["summary"]}')
    print(f'Wrote operand-region audit -> {paths["operand_region_audit"]}')
    print(
        'Totals: '
        f'requests={summary.get("total_requests", 0)}, '
        f'stalls={summary.get("total_stalls", 0)}, '
        f'contention_requests={summary.get("high_fanin_requests", 0)}',
    )
    print(f'Operand regions ({operand_regions.source}): {format_operand_regions(operand_regions)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

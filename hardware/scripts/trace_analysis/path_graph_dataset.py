#!/usr/bin/env python3
"""Normalize monitor CSVs into a graph-oriented path dataset.

The raw tile/path monitor files are cycle accurate, but they are organized by
RTL probe rather than by the questions we ask while debugging congestion:

* which observed nodes and lanes exist?
* which of them fired, idled, or blocked in each cycle?
* where does backpressure persist across a chosen window?

This script keeps the raw valid/ready/fire semantics intact and writes a small
schema around them so plotting tools can work from stable graph tables instead
of re-learning every monitor CSV shape.
"""

from __future__ import annotations

import argparse
import csv
import html
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from _workflow_metadata import format_topology, load_result_dir_topology


TILE_POINT_ORDER = {
    'core_q': 10,
    'core_p': 20,
    'tcdm_preroute': 30,
    'tcdm_local': 40,
    'local_xbar_out': 50,
    'bank_req': 60,
    'bank_resp': 70,
    'superbank_resp': 80,
    'local_resp_core': 90,
    'tcdm_remote': 100,
    'remote_xbar_out': 110,
    'tile_master_req_out': 120,
    'tile_slave_req_in': 130,
    'tile_slave_req_postreg': 140,
    'tile_slave_resp_prereg': 150,
    'tile_slave_resp_out': 160,
    'tile_master_resp_in': 170,
    'tile_master_resp_postreg': 180,
    'remote_resp_core': 190,
}

LANE_STAGE_ORDER = {
    'in0': 10,
    'post0': 20,
    'in1': 30,
    'post1': 40,
    'out': 50,
}

STATE_PRIORITY = {
    'inactive': 0,
    'idle_ready': 1,
    'flow': 2,
    'valid_no_fire': 3,
    'mixed_blocked_flow': 4,
    'blocked': 5,
}

COUNT_FIELDS = (
    'observations',
    'valid',
    'ready',
    'fire',
    'stall',
    'idle_ready',
    'inactive',
    'valid_no_fire',
    'write',
    'back2local',
)


class CountBucket(Counter):
    def add_counts(self, counts: dict[str, int]) -> None:
        for field in COUNT_FIELDS:
            self[field] += int(counts.get(field, 0))


def parse_int(value: str | None, default: int | None = 0) -> int | None:
    if value is None or value == '':
        return default
    return int(value, 0)


def parse_bit(value: str | None) -> int:
    return 1 if value == '1' else 0


def in_cycle_range(cycle: int, start: int | None, end: int | None) -> bool:
    if start is not None and cycle < start:
        return False
    if end is not None and cycle > end:
        return False
    return True


def csv_int(value: int | None) -> str:
    if value is None or value < 0:
        return ''
    return str(value)


def monitor_int(row: dict[str, str], key: str) -> int:
    value = parse_int(row.get(key), -1)
    return -1 if value is None else value


def identity_suffix(core: int, port: int, bank: int, index: int) -> str:
    parts = []
    if core >= 0:
        parts.append(f'c{core}')
    if port >= 0:
        parts.append(f'p{port}')
    if bank >= 0:
        parts.append(f'b{bank}')
    if index >= 0 and index not in {core, port, bank}:
        parts.append(f'i{index}')
    return ':' + ':'.join(parts) if parts else ''


def identity_label(core: int, port: int, bank: int, index: int) -> str:
    parts = []
    if core >= 0:
        parts.append(f'C{core}')
    if port >= 0:
        parts.append(f'P{port}')
    if bank >= 0:
        parts.append(f'B{bank}')
    if index >= 0 and index not in {core, port, bank}:
        parts.append(f'I{index}')
    return '/'.join(parts)


def location_label(group: int, tile: int, tile_in_group: int) -> str:
    parts = []
    if group >= 0 and tile_in_group >= 0:
        parts.append(f'G{group}/T{tile_in_group:02d}')
    elif group >= 0:
        parts.append(f'G{group}')
    if tile >= 0:
        parts.append(f'abs{tile:03d}')
    return ' '.join(parts) if parts else 'unknown-location'


def tile_subject_label(group: int, tile: int, tile_in_group: int, point: str,
                       core: int, port: int, bank: int, index: int) -> str:
    label = f'{location_label(group, tile, tile_in_group)} | node {point}'
    suffix = identity_label(core, port, bank, index)
    if suffix:
        label += f' | {suffix}'
    return label


def lane_subject_label(group: int, tile: int, tile_in_group: int, port: int,
                       channel: str, stage: str) -> str:
    label = f'{location_label(group, tile, tile_in_group)} | lane {channel}/{stage}'
    if port >= 0:
        label += f' | P{port}'
    return label


def state_from_bits(valid: int, ready: int, fire: int) -> str:
    if valid and not ready:
        return 'blocked'
    if fire:
        return 'flow'
    if valid:
        return 'valid_no_fire'
    if ready:
        return 'idle_ready'
    return 'inactive'


def state_from_counts(counts: dict[str, int] | Counter) -> str:
    if counts.get('stall', 0) and counts.get('fire', 0):
        return 'mixed_blocked_flow'
    if counts.get('stall', 0):
        return 'blocked'
    if counts.get('fire', 0):
        return 'flow'
    if counts.get('valid_no_fire', 0):
        return 'valid_no_fire'
    if counts.get('idle_ready', 0):
        return 'idle_ready'
    return 'inactive'


def state_counts(valid: int, ready: int, fire: int, write: int = 0, back2local: int = 0) -> dict[str, int]:
    stall = int(bool(valid and not ready))
    idle_ready = int(bool(not valid and ready))
    inactive = int(bool(not valid and not ready))
    valid_no_fire = int(bool(valid and ready and not fire))
    return {
        'observations': 1,
        'valid': valid,
        'ready': ready,
        'fire': fire,
        'stall': stall,
        'idle_ready': idle_ready,
        'inactive': inactive,
        'valid_no_fire': valid_no_fire,
        'write': write,
        'back2local': back2local,
    }


def rate(part: int, total: int) -> str:
    if total == 0:
        return ''
    return f'{part / total:.6f}'


def resolve_monitor_dir(input_path: Path) -> tuple[Path, Path | None]:
    path = input_path.resolve()
    if (path / 'monitor').is_dir():
        return path / 'monitor', path
    if path.is_dir() and any(path.glob('tile_path_tile*.csv')):
        result_dir = path.parent if (path.parent / 'topology.env').is_file() else None
        return path, result_dir
    raise SystemExit(f'Could not find monitor CSVs in {input_path}')


def topology_dims(topology: dict | None) -> tuple[int | None, int | None]:
    if topology is None:
        return None, None
    tiles = int(topology['NUM_CORES']) // int(topology['NUM_CORES_PER_TILE'])
    groups = int(topology['NUM_GROUPS'])
    return tiles, tiles // groups


def tile_group(tile: int, tiles_per_group: int | None) -> tuple[int, int]:
    if tiles_per_group is None or tiles_per_group <= 0 or tile < 0:
        return -1, -1
    return tile // tiles_per_group, tile % tiles_per_group


def absolute_tile(group: int, tile_in_group: int, tiles_per_group: int | None) -> int:
    if tiles_per_group is None or group < 0 or tile_in_group < 0:
        return -1
    return group * tiles_per_group + tile_in_group


def make_tile_node_id(tile: int, point: str, core: int, port: int, bank: int, index: int) -> str:
    return f'tile:t{tile:03d}:{point}{identity_suffix(core, port, bank, index)}'


def make_lane_id(group: int, tile_in_group: int, port: int, channel: str, stage: str) -> str:
    return f'lane:g{group}:t{tile_in_group:02d}:p{port}:{channel}:{stage}'


def update_summary(
    summary: dict[tuple, CountBucket],
    key: tuple,
    counts: dict[str, int],
) -> None:
    summary[key].add_counts(counts)


def update_cycle_summary(
    summary: dict[tuple, CountBucket],
    cycle: int,
    subject_type: str,
    counts: dict[str, int],
) -> None:
    update_summary(summary, (cycle, 'all'), counts)
    update_summary(summary, (cycle, subject_type), counts)


def update_window_summary(
    summary: dict[tuple, CountBucket],
    base_cycle: int,
    window_size: int,
    cycle: int,
    subject_type: str,
    subject_id: str,
    counts: dict[str, int],
) -> None:
    if window_size <= 0:
        return
    window_start = base_cycle + ((cycle - base_cycle) // window_size) * window_size
    window_end = window_start + window_size - 1
    update_summary(summary, (window_start, window_end, subject_type, subject_id), counts)


def write_subject_registry(path: Path, rows: dict[str, dict[str, str]]) -> None:
    fields = (
        'subject_id', 'subject_type', 'domain', 'group', 'tile', 'tile_in_group',
        'point', 'stage', 'channel', 'index', 'core', 'port', 'bank',
        'order', 'label', 'description',
    )
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows.values(), key=lambda item: (int(item['order']), item['subject_id'])):
            writer.writerow(row)


def write_cycle_summary(path: Path, summary: dict[tuple, CountBucket]) -> None:
    fields = ('cycle', 'subject_type', *COUNT_FIELDS, 'state', 'stall_rate', 'fire_rate')
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (cycle, subject_type), counts in sorted(summary.items()):
            row = {'cycle': cycle, 'subject_type': subject_type}
            row.update({field: counts[field] for field in COUNT_FIELDS})
            row['state'] = state_from_counts(counts)
            row['stall_rate'] = rate(counts['stall'], counts['observations'])
            row['fire_rate'] = rate(counts['fire'], counts['observations'])
            writer.writerow(row)


def write_subject_summary(
    path: Path,
    summary: dict[tuple, CountBucket],
    registry: dict[str, dict[str, str]],
) -> None:
    fields = (
        'subject_type', 'subject_id', 'label', 'domain', 'group', 'tile',
        'tile_in_group', 'point', 'stage', 'channel', 'core', 'port', 'bank',
        *COUNT_FIELDS, 'state', 'stall_rate', 'fire_rate',
    )
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (subject_type, subject_id), counts in sorted(
            summary.items(),
            key=lambda item: (-item[1]['stall'], -item[1]['fire'], item[0][0], item[0][1]),
        ):
            meta = registry.get(subject_id, {})
            row = {
                'subject_type': subject_type,
                'subject_id': subject_id,
                'label': meta.get('label', subject_id),
                'domain': meta.get('domain', ''),
                'group': meta.get('group', ''),
                'tile': meta.get('tile', ''),
                'tile_in_group': meta.get('tile_in_group', ''),
                'point': meta.get('point', ''),
                'stage': meta.get('stage', ''),
                'channel': meta.get('channel', ''),
                'core': meta.get('core', ''),
                'port': meta.get('port', ''),
                'bank': meta.get('bank', ''),
            }
            row.update({field: counts[field] for field in COUNT_FIELDS})
            row['state'] = state_from_counts(counts)
            row['stall_rate'] = rate(counts['stall'], counts['observations'])
            row['fire_rate'] = rate(counts['fire'], counts['observations'])
            writer.writerow(row)


def write_window_summary(
    path: Path,
    summary: dict[tuple, CountBucket],
    registry: dict[str, dict[str, str]],
) -> None:
    fields = (
        'window_start', 'window_end', 'subject_type', 'subject_id', 'label',
        'domain', 'point', 'stage', 'channel', *COUNT_FIELDS,
        'state', 'stall_rate', 'fire_rate',
    )
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (window_start, window_end, subject_type, subject_id), counts in sorted(
            summary.items(),
            key=lambda item: (item[0][0], -item[1]['stall'], -item[1]['fire'], item[0][2], item[0][3]),
        ):
            meta = registry.get(subject_id, {})
            row = {
                'window_start': window_start,
                'window_end': window_end,
                'subject_type': subject_type,
                'subject_id': subject_id,
                'label': meta.get('label', subject_id),
                'domain': meta.get('domain', ''),
                'point': meta.get('point', ''),
                'stage': meta.get('stage', ''),
                'channel': meta.get('channel', ''),
            }
            row.update({field: counts[field] for field in COUNT_FIELDS})
            row['state'] = state_from_counts(counts)
            row['stall_rate'] = rate(counts['stall'], counts['observations'])
            row['fire_rate'] = rate(counts['fire'], counts['observations'])
            writer.writerow(row)


def write_edges(path: Path, registry: dict[str, dict[str, str]]) -> int:
    fields = ('edge_id', 'source_id', 'target_id', 'domain', 'kind', 'confidence', 'description')
    edges: dict[str, dict[str, str]] = {}

    def add(source: str, target: str, kind: str, confidence: str, description: str) -> None:
        if source not in registry or target not in registry:
            return
        edge_id = f'{source}->{target}'
        edges[edge_id] = {
            'edge_id': edge_id,
            'source_id': source,
            'target_id': target,
            'domain': registry[source].get('domain', ''),
            'kind': kind,
            'confidence': confidence,
            'description': description,
        }

    tile_nodes = [row for row in registry.values() if row['subject_type'] == 'node']
    by_tile_point: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in tile_nodes:
        by_tile_point[(row['tile'], row['point'])].append(row)

    for tile in sorted({row['tile'] for row in tile_nodes}):
        core_q = {row['core']: row['subject_id'] for row in by_tile_point[(tile, 'core_q')]}
        core_p = {row['core']: row['subject_id'] for row in by_tile_point[(tile, 'core_p')]}
        preroute = {row['core']: row['subject_id'] for row in by_tile_point[(tile, 'tcdm_preroute')]}
        for core, source in core_q.items():
            add(source, core_p.get(core, ''), 'tile_request', 'direct', 'Core request queue to pipeline stage')
            add(core_p.get(core, ''), preroute.get(core, ''), 'tile_request', 'direct', 'Core pipeline stage to route decision')

        for source in by_tile_point[(tile, 'tcdm_local')]:
            for target in by_tile_point[(tile, 'local_xbar_out')]:
                if source['bank'] == target['bank']:
                    add(source['subject_id'], target['subject_id'], 'local_request', 'bank-matched', 'Local request to local crossbar bank lane')
        for source in by_tile_point[(tile, 'local_xbar_out')]:
            for target in by_tile_point[(tile, 'bank_req')]:
                if source['bank'] == target['bank']:
                    add(source['subject_id'], target['subject_id'], 'local_request', 'bank-matched', 'Local crossbar output to bank request')
        for source in by_tile_point[(tile, 'bank_req')]:
            for target in by_tile_point[(tile, 'bank_resp')]:
                if source['bank'] == target['bank']:
                    add(source['subject_id'], target['subject_id'], 'bank', 'bank-matched', 'Bank request to bank response observation')
        for source in by_tile_point[(tile, 'bank_resp')]:
            for target in by_tile_point[(tile, 'superbank_resp')]:
                if source['bank'] == target['bank']:
                    add(source['subject_id'], target['subject_id'], 'bank_response', 'bank-matched', 'Bank response to superbank response')
        for source in by_tile_point[(tile, 'tcdm_remote')]:
            for target in by_tile_point[(tile, 'remote_xbar_out')]:
                if source['port'] == target['port']:
                    add(source['subject_id'], target['subject_id'], 'remote_request', 'port-matched', 'Remote request to remote crossbar output')
        for source in by_tile_point[(tile, 'remote_xbar_out')]:
            for target in by_tile_point[(tile, 'tile_master_req_out')]:
                if source['port'] == target['port']:
                    add(source['subject_id'], target['subject_id'], 'remote_request', 'port-matched', 'Remote crossbar output to tile master request')
        for source in by_tile_point[(tile, 'tile_master_resp_in')]:
            for target in by_tile_point[(tile, 'tile_master_resp_postreg')]:
                if source['port'] == target['port']:
                    add(source['subject_id'], target['subject_id'], 'remote_response', 'port-matched', 'Tile master response input to post-register stage')

    lane_rows = [row for row in registry.values() if row['subject_type'] == 'lane']
    lane_by_base: dict[tuple[str, str, str, str], dict[str, str]] = defaultdict(dict)
    for row in lane_rows:
        lane_by_base[(row['group'], row['tile_in_group'], row['port'], row['channel'])][row['stage']] = row['subject_id']
    for stages in lane_by_base.values():
        add(stages.get('in0', ''), stages.get('post0', ''), 'group_lane', 'direct', 'Path-util input 0 to post stage 0')
        add(stages.get('post0', ''), stages.get('out', ''), 'group_lane', 'direct', 'Path-util post stage 0 to output')
        add(stages.get('in1', ''), stages.get('post1', ''), 'group_lane', 'direct', 'Path-util input 1 to post stage 1')
        add(stages.get('post1', ''), stages.get('out', ''), 'group_lane', 'direct', 'Path-util post stage 1 to output')

    lanes_by_group_tile_port: dict[tuple[str, str, str], str] = {}
    for row in lane_rows:
        if row['channel'] == 'req' and row['stage'] == 'out':
            lanes_by_group_tile_port[(row['group'], row['tile_in_group'], row['port'])] = row['subject_id']
    for row in tile_nodes:
        if row['point'] != 'tile_master_req_out':
            continue
        target = lanes_by_group_tile_port.get((row['group'], row['tile_in_group'], row['port']))
        add(row['subject_id'], target or '', 'tile_to_group_lane', 'metadata-matched', 'Tile master outgoing request to group request output lane')

    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in sorted(edges.values(), key=lambda item: item['edge_id']):
            writer.writerow(row)
    return len(edges)


def write_schema(path: Path) -> None:
    path.write_text(
        '# Path Graph Dataset\n\n'
        'This directory normalizes tile/path monitor CSVs for graph-time analysis.\n\n'
        '## Core Tables\n\n'
        '- `nodes.csv`: observed tile-level handshake nodes from `tile_path_tile*.csv`.\n'
        '- `lanes.csv`: observed group path-util lane stages from `path_util_group*.csv`.\n'
        '- `edges.csv`: inferred static adjacency for route sketches; `confidence` marks whether it is direct or lane-matched.\n'
        '- `cycle_node_state.csv`: per-cycle state for each tile node.\n'
        '- `cycle_lane_state.csv`: per-cycle state for each group lane stage.\n'
        '- `cycle_summary.csv`: per-cycle aggregate pressure for all subjects, tile nodes, and group lanes.\n'
        '- `subject_summary.csv`: whole-window aggregate per subject, sorted by stall pressure.\n'
        '- `window_summary.csv`: per-window aggregate per subject.\n'
        '- `path_timeline.html`: compact heatmap for the highest-stall subjects.\n\n'
        '`subject_id` is a stable machine key for joins. `label` uses the human-readable '\
        '`G#/T## abs### | node/lane ... | qualifiers` form shown in the HTML timeline.\n\n'
        '## State Semantics\n\n'
        '- `flow`: `valid=1`, `ready=1`, `fire=1`.\n'
        '- `blocked`: `valid=1`, `ready=0`, `fire=0`; this is direct backpressure.\n'
        '- `idle_ready`: `valid=0`, `ready=1`, `fire=0`; the path was available but unused.\n'
        '- `inactive`: `valid=0`, `ready=0`, `fire=0`.\n'
        '- `valid_no_fire`: `valid=1`, `ready=1`, `fire=0`; uncommon, kept explicit instead of folded away.\n'
        '- `mixed_blocked_flow`: aggregate row containing both blocked and firing observations.\n',
        encoding='utf-8',
    )


def top_subjects(subject_summary_path: Path, limit: int) -> list[str]:
    subjects = []
    with subject_summary_path.open(newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            subjects.append(row['subject_id'])
            if len(subjects) >= limit:
                break
    return subjects


def read_cycles(paths: Iterable[Path]) -> list[int]:
    cycles = set()
    for path in paths:
        if not path.is_file():
            continue
        with path.open(newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                cycles.add(int(row['cycle']))
    return sorted(cycles)


def write_html_timeline(
    path: Path,
    node_state_path: Path,
    lane_state_path: Path,
    subject_summary_path: Path,
    registry: dict[str, dict[str, str]],
    subject_limit: int,
    cycle_limit: int,
) -> None:
    subjects = top_subjects(subject_summary_path, subject_limit)
    if not subjects:
        return
    cycles = read_cycles((node_state_path, lane_state_path))
    if len(cycles) > cycle_limit > 0:
        stride = max(1, (len(cycles) + cycle_limit - 1) // cycle_limit)
        cycles = cycles[::stride]
    cycle_set = set(cycles)
    subject_set = set(subjects)
    matrix: dict[tuple[str, int], str] = {}

    for state_path in (node_state_path, lane_state_path):
        if not state_path.is_file():
            continue
        with state_path.open(newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                subject_id = row['subject_id']
                cycle = int(row['cycle'])
                if subject_id not in subject_set or cycle not in cycle_set:
                    continue
                current = matrix.get((subject_id, cycle), 'inactive')
                candidate = row['state']
                if STATE_PRIORITY[candidate] > STATE_PRIORITY[current]:
                    matrix[(subject_id, cycle)] = candidate

    legend = ''.join(
        f'<span class="legend-item"><span class="cell {state}"></span>{html.escape(state)}</span>'
        for state in ('blocked', 'mixed_blocked_flow', 'valid_no_fire', 'flow', 'idle_ready', 'inactive')
    )
    header_cells = ''.join(f'<th>{cycle}</th>' for cycle in cycles)
    rows = []
    for subject_id in subjects:
        label = registry.get(subject_id, {}).get('label', subject_id)
        cells = []
        for cycle in cycles:
            state = matrix.get((subject_id, cycle), 'inactive')
            cells.append(f'<td class="cell {state}" title="{html.escape(subject_id)} cycle {cycle}: {state}"></td>')
        rows.append(f'<tr><th title="{html.escape(subject_id)}">{html.escape(label)}</th>{"".join(cells)}</tr>')

    path.write_text(
        '<!doctype html>\n'
        '<meta charset="utf-8">\n'
        '<title>Path Timeline</title>\n'
        '<style>\n'
        'body{font-family:system-ui,sans-serif;margin:24px;color:#18212f;background:#f8fafc}\n'
        'h1{font-size:22px;margin:0 0 8px}\n'
        'p{max-width:900px;line-height:1.45}\n'
        '.legend{display:flex;gap:14px;flex-wrap:wrap;margin:16px 0}\n'
        '.legend-item{display:inline-flex;gap:6px;align-items:center;font-size:13px}\n'
        '.wrap{overflow:auto;border:1px solid #d6dee8;background:white}\n'
        'table{border-collapse:collapse;font-size:11px}\n'
        'th{position:sticky;left:0;background:#eef3f8;text-align:left;white-space:nowrap;padding:3px 6px;border:1px solid #d6dee8;z-index:1}\n'
        'thead th{position:sticky;top:0;left:auto;writing-mode:vertical-rl;text-align:right;z-index:2}\n'
        'thead th:first-child{left:0;z-index:3;writing-mode:horizontal-tb}\n'
        'td.cell,.legend .cell{width:11px;height:11px;min-width:11px;padding:0;border:1px solid #edf1f5}\n'
        '.blocked{background:#d73027}.mixed_blocked_flow{background:#fc8d59}.valid_no_fire{background:#fee08b}\n'
        '.flow{background:#1a9850}.idle_ready{background:#d9ef8b}.inactive{background:#2d3748}\n'
        '</style>\n'
        '<h1>Path Timeline</h1>\n'
        '<p>Top monitored subjects by stall pressure. Red cells are direct backpressure '
        '(<code>valid=1, ready=0</code>); green cells fired; light green cells were idle but ready.</p>\n'
        f'<div class="legend">{legend}</div>\n'
        '<div class="wrap"><table><thead><tr><th>subject</th>'
        f'{header_cells}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>\n',
        encoding='utf-8',
    )


def parse_id_list(values: list[str] | None) -> set[int] | None:
    if not values:
        return None
    ids = set()
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                ids.add(int(part, 0))
    return ids


def parse_str_list(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    strings = set()
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                strings.add(part)
    return strings


def process_tile_paths(
    monitor_dir: Path,
    output_path: Path,
    registry: dict[str, dict[str, str]],
    subject_summary: dict[tuple, CountBucket],
    cycle_summary: dict[tuple, CountBucket],
    window_summary: dict[tuple, CountBucket],
    tiles_per_group: int | None,
    cycle_start: int | None,
    cycle_end: int | None,
    window_size: int,
    tile_filter: set[int] | None,
    tile_group_filter: set[int] | None,
    point_filter: set[str] | None,
    base_cycle: int,
) -> int:
    fields = (
        'cycle', 'time', 'subject_id', 'subject_type', 'domain', 'group', 'tile', 'tile_in_group',
        'point', 'index', 'core', 'port', 'bank', *COUNT_FIELDS, 'state',
        'addr', 'source_addr', 'meta_id', 'payload_core',
    )
    rows = 0
    with output_path.open('w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for path in sorted(monitor_dir.glob('tile_path_tile*.csv')):
            with path.open(newline='') as in_file:
                reader = csv.DictReader(in_file)
                for row in reader:
                    cycle = parse_int(row.get('cycle'))
                    if cycle is None or not in_cycle_range(cycle, cycle_start, cycle_end):
                        continue
                    tile = monitor_int(row, 'tile')
                    if tile_filter is not None and tile not in tile_filter:
                        continue
                    group, tile_in_group = tile_group(tile, tiles_per_group)
                    if tile_group_filter is not None and group not in tile_group_filter:
                        continue
                    point = row.get('point', '')
                    if point_filter is not None and point not in point_filter:
                        continue
                    index = monitor_int(row, 'index')
                    core = monitor_int(row, 'core')
                    port = monitor_int(row, 'port')
                    bank = monitor_int(row, 'bank')
                    subject_id = make_tile_node_id(tile, point, core, port, bank, index)
                    label = tile_subject_label(group, tile, tile_in_group, point, core, port, bank, index)
                    registry.setdefault(subject_id, {
                        'subject_id': subject_id,
                        'subject_type': 'node',
                        'domain': 'tile',
                        'group': csv_int(group),
                        'tile': csv_int(tile),
                        'tile_in_group': csv_int(tile_in_group),
                        'point': point,
                        'stage': '',
                        'channel': '',
                        'index': csv_int(index),
                        'core': csv_int(core),
                        'port': csv_int(port),
                        'bank': csv_int(bank),
                        'order': str(tile * 100000 + TILE_POINT_ORDER.get(point, 900) * 100 + max(index, 0)),
                        'label': label,
                        'description': 'Tile-level valid/ready/fire monitor point',
                    })

                    valid = parse_bit(row.get('valid'))
                    ready = parse_bit(row.get('ready'))
                    fire = parse_bit(row.get('fire'))
                    counts = state_counts(valid, ready, fire, parse_bit(row.get('write')), parse_bit(row.get('back2local')))
                    state = state_from_bits(valid, ready, fire)
                    output = {
                        'cycle': cycle,
                        'time': row.get('time', ''),
                        'subject_id': subject_id,
                        'subject_type': 'node',
                        'domain': 'tile',
                        'group': csv_int(group),
                        'tile': csv_int(tile),
                        'tile_in_group': csv_int(tile_in_group),
                        'point': point,
                        'index': csv_int(index),
                        'core': csv_int(core),
                        'port': csv_int(port),
                        'bank': csv_int(bank),
                        **counts,
                        'state': state,
                        'addr': row.get('addr', ''),
                        'source_addr': row.get('source_addr', ''),
                        'meta_id': row.get('meta_id', ''),
                        'payload_core': row.get('payload_core', ''),
                    }
                    writer.writerow(output)
                    update_summary(subject_summary, ('node', subject_id), counts)
                    update_cycle_summary(cycle_summary, cycle, 'node', counts)
                    update_window_summary(window_summary, base_cycle, window_size, cycle, 'node', subject_id, counts)
                    rows += 1
    return rows


def process_group_lanes(
    monitor_dir: Path,
    output_path: Path,
    registry: dict[str, dict[str, str]],
    subject_summary: dict[tuple, CountBucket],
    cycle_summary: dict[tuple, CountBucket],
    window_summary: dict[tuple, CountBucket],
    tiles_per_group: int | None,
    cycle_start: int | None,
    cycle_end: int | None,
    window_size: int,
    group_filter: set[int] | None,
    lane_channel_filter: set[str] | None,
    lane_stage_filter: set[str] | None,
    base_cycle: int,
) -> int:
    fields = (
        'cycle', 'time', 'subject_id', 'subject_type', 'domain', 'group', 'tile', 'tile_in_group',
        'port', 'channel', 'stage', *COUNT_FIELDS, 'state',
    )
    rows = 0
    with output_path.open('w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fields)
        writer.writeheader()
        for path in sorted(monitor_dir.glob('path_util_group*.csv')):
            with path.open(newline='') as in_file:
                reader = csv.DictReader(in_file)
                for row in reader:
                    cycle = parse_int(row.get('cycle'))
                    if cycle is None or not in_cycle_range(cycle, cycle_start, cycle_end):
                        continue
                    group = monitor_int(row, 'group')
                    if group_filter is not None and group not in group_filter:
                        continue
                    tile_in_group = monitor_int(row, 'tile')
                    tile = absolute_tile(group, tile_in_group, tiles_per_group)
                    port = monitor_int(row, 'port')
                    channel = row.get('channel', '')
                    if lane_channel_filter is not None and channel not in lane_channel_filter:
                        continue
                    for stage in ('in0', 'post0', 'in1', 'post1', 'out'):
                        if lane_stage_filter is not None and stage not in lane_stage_filter:
                            continue
                        valid = parse_bit(row.get(f'{stage}_valid'))
                        ready = parse_bit(row.get(f'{stage}_ready'))
                        fire = parse_bit(row.get(f'{stage}_fire'))
                        back2local = parse_bit(row.get('out_back2local')) if stage == 'out' else 0
                        counts = state_counts(valid, ready, fire, back2local=back2local)
                        subject_id = make_lane_id(group, tile_in_group, port, channel, stage)
                        label = lane_subject_label(group, tile, tile_in_group, port, channel, stage)
                        registry.setdefault(subject_id, {
                            'subject_id': subject_id,
                            'subject_type': 'lane',
                            'domain': 'group_lane',
                            'group': csv_int(group),
                            'tile': csv_int(tile),
                            'tile_in_group': csv_int(tile_in_group),
                            'point': '',
                            'stage': stage,
                            'channel': channel,
                            'index': '',
                            'core': '',
                            'port': csv_int(port),
                            'bank': '',
                            'order': str(10_000_000 + group * 100000 + tile_in_group * 1000 + port * 100 + LANE_STAGE_ORDER[stage]),
                            'label': label,
                            'description': 'Group path-util valid/ready/fire lane stage',
                        })
                        writer.writerow({
                            'cycle': cycle,
                            'time': row.get('time', ''),
                            'subject_id': subject_id,
                            'subject_type': 'lane',
                            'domain': 'group_lane',
                            'group': csv_int(group),
                            'tile': csv_int(tile),
                            'tile_in_group': csv_int(tile_in_group),
                            'port': csv_int(port),
                            'channel': channel,
                            'stage': stage,
                            **counts,
                            'state': state_from_bits(valid, ready, fire),
                        })
                        update_summary(subject_summary, ('lane', subject_id), counts)
                        update_cycle_summary(cycle_summary, cycle, 'lane', counts)
                        update_window_summary(window_summary, base_cycle, window_size, cycle, 'lane', subject_id, counts)
                        rows += 1
    return rows


def write_manifest(
    output_dir: Path,
    input_path: Path,
    monitor_dir: Path,
    result_dir: Path | None,
    topology: dict | None,
    cycle_start: int | None,
    cycle_end: int | None,
    window_size: int,
    node_rows: int,
    lane_rows: int,
    edge_rows: int,
    filters: dict[str, str],
) -> None:
    lines = [
        f'input_path={input_path.resolve()}',
        f'monitor_dir={monitor_dir}',
        f'result_dir={result_dir if result_dir is not None else "n/a"}',
        f'topology={format_topology(topology) if topology is not None else "n/a"}',
        f'cycle_start={cycle_start if cycle_start is not None else "first"}',
        f'cycle_end={cycle_end if cycle_end is not None else "last"}',
        f'window_size={window_size}',
        f'cycle_node_state_rows={node_rows}',
        f'cycle_lane_state_rows={lane_rows}',
        f'edges={edge_rows}',
    ]
    lines.extend(f'{key}={value}' for key, value in sorted(filters.items()))
    (output_dir / 'manifest.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_path', type=Path, help='Monitor directory, or result directory containing monitor/')
    parser.add_argument('--output-dir', type=Path, help='Output directory [default: <result_dir>/analysis/path_graph]')
    parser.add_argument('--cycle-start', type=int, help='First cycle to include')
    parser.add_argument('--cycle-end', type=int, help='Last cycle to include')
    parser.add_argument('--window-size', type=int, default=64, help='Window size for window_summary.csv [default: 64]')
    parser.add_argument('--tile', action='append', help='Tile id(s) to include, comma-separated; may be repeated')
    parser.add_argument('--tile-group', action='append', help='Group id(s) to include for tile monitor nodes, comma-separated; may be repeated')
    parser.add_argument('--point', action='append', help='Tile monitor point(s) to include, comma-separated; may be repeated')
    parser.add_argument('--group', action='append', help='Group id(s) to include for path-util lanes, comma-separated; may be repeated')
    parser.add_argument('--lane-channel', action='append', help='Path-util channel(s) to include, e.g. req or resp; may be repeated')
    parser.add_argument('--lane-stage', action='append', help='Path-util stage(s) to include: in0, post0, in1, post1, out; may be repeated')
    parser.add_argument('--outgoing-only', action='store_true', help='Shortcut for outgoing request view: tile_master_req_out plus req/out group lanes')
    parser.add_argument('--html-subjects', type=int, default=60, help='Top stalled subjects to include in HTML [default: 60]')
    parser.add_argument('--html-cycle-limit', type=int, default=400, help='Max cycle columns in HTML before striding [default: 400]')
    parser.add_argument('--no-html', action='store_true', help='Do not generate path_timeline.html')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    monitor_dir, result_dir = resolve_monitor_dir(args.input_path)
    topology = load_result_dir_topology(result_dir) if result_dir is not None else None
    _, tiles_per_group = topology_dims(topology)
    output_dir = args.output_dir or ((result_dir / 'analysis' / 'path_graph') if result_dir else (monitor_dir / 'path_graph'))
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_filter = parse_id_list(args.tile)
    tile_group_filter = parse_id_list(args.tile_group)
    point_filter = parse_str_list(args.point)
    group_filter = parse_id_list(args.group)
    lane_channel_filter = parse_str_list(args.lane_channel)
    lane_stage_filter = parse_str_list(args.lane_stage)
    if args.outgoing_only:
        if point_filter is None:
            point_filter = {'tile_master_req_out'}
        if lane_channel_filter is None:
            lane_channel_filter = {'req'}
        if lane_stage_filter is None:
            lane_stage_filter = {'out'}
    base_cycle = args.cycle_start or 0

    registry: dict[str, dict[str, str]] = {}
    subject_summary: dict[tuple, CountBucket] = defaultdict(CountBucket)
    cycle_summary: dict[tuple, CountBucket] = defaultdict(CountBucket)
    window_summary: dict[tuple, CountBucket] = defaultdict(CountBucket)

    node_rows = process_tile_paths(
        monitor_dir,
        output_dir / 'cycle_node_state.csv',
        registry,
        subject_summary,
        cycle_summary,
        window_summary,
        tiles_per_group,
        args.cycle_start,
        args.cycle_end,
        args.window_size,
        tile_filter,
        tile_group_filter,
        point_filter,
        base_cycle,
    )
    lane_rows = process_group_lanes(
        monitor_dir,
        output_dir / 'cycle_lane_state.csv',
        registry,
        subject_summary,
        cycle_summary,
        window_summary,
        tiles_per_group,
        args.cycle_start,
        args.cycle_end,
        args.window_size,
        group_filter,
        lane_channel_filter,
        lane_stage_filter,
        base_cycle,
    )

    node_registry = {key: row for key, row in registry.items() if row['subject_type'] == 'node'}
    lane_registry = {key: row for key, row in registry.items() if row['subject_type'] == 'lane'}
    write_subject_registry(output_dir / 'nodes.csv', node_registry)
    write_subject_registry(output_dir / 'lanes.csv', lane_registry)
    edge_rows = write_edges(output_dir / 'edges.csv', registry)
    write_cycle_summary(output_dir / 'cycle_summary.csv', cycle_summary)
    write_subject_summary(output_dir / 'subject_summary.csv', subject_summary, registry)
    write_window_summary(output_dir / 'window_summary.csv', window_summary, registry)
    write_schema(output_dir / 'schema.md')
    write_manifest(
        output_dir,
        args.input_path,
        monitor_dir,
        result_dir,
        topology,
        args.cycle_start,
        args.cycle_end,
        args.window_size,
        node_rows,
        lane_rows,
        edge_rows,
        {
            'tile_filter': ','.join(str(item) for item in sorted(tile_filter)) if tile_filter else 'all',
            'tile_group_filter': ','.join(str(item) for item in sorted(tile_group_filter)) if tile_group_filter else 'all',
            'point_filter': ','.join(sorted(point_filter)) if point_filter else 'all',
            'group_filter': ','.join(str(item) for item in sorted(group_filter)) if group_filter else 'all',
            'lane_channel_filter': ','.join(sorted(lane_channel_filter)) if lane_channel_filter else 'all',
            'lane_stage_filter': ','.join(sorted(lane_stage_filter)) if lane_stage_filter else 'all',
            'outgoing_only': str(bool(args.outgoing_only)).lower(),
        },
    )

    if not args.no_html:
        write_html_timeline(
            output_dir / 'path_timeline.html',
            output_dir / 'cycle_node_state.csv',
            output_dir / 'cycle_lane_state.csv',
            output_dir / 'subject_summary.csv',
            registry,
            args.html_subjects,
            args.html_cycle_limit,
        )

    print(f'Wrote {node_rows} cycle node rows to {output_dir / "cycle_node_state.csv"}')
    print(f'Wrote {lane_rows} cycle lane rows to {output_dir / "cycle_lane_state.csv"}')
    print(f'Wrote {len(node_registry)} nodes, {len(lane_registry)} lanes, and {edge_rows} inferred edges to {output_dir}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
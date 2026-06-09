#!/usr/bin/env python3
"""Shared helpers for path-graph monitor analysis scripts."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path


STATE_PRIORITY = {
    'unobserved': -1,
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


def parse_int(value: str | None, default: int | None = None) -> int | None:
    if value is None or value == '':
        return default
    return int(value, 0)


def in_cycle_range(cycle: int, start: int | None, end: int | None) -> bool:
    if start is not None and cycle < start:
        return False
    if end is not None and cycle > end:
        return False
    return True


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as file:
        return list(csv.DictReader(file))


def require_graph_dir(path: Path) -> Path:
    graph_dir = path.resolve()
    required = ('nodes.csv', 'lanes.csv', 'cycle_node_state.csv', 'cycle_lane_state.csv')
    missing = [name for name in required if not (graph_dir / name).is_file()]
    if missing:
        raise SystemExit(f'Missing graph dataset files in {graph_dir}: {", ".join(missing)}')
    return graph_dir


def state_from_counts(counts: Counter) -> str:
    if not counts or counts['observations'] == 0:
        return 'unobserved'
    if counts['stall'] and counts['fire']:
        return 'mixed_blocked_flow'
    if counts['stall']:
        return 'blocked'
    if counts['fire']:
        return 'flow'
    if counts['valid_no_fire']:
        return 'valid_no_fire'
    if counts['idle_ready']:
        return 'idle_ready'
    return 'inactive'


def packet_status(state: str, counts: Counter) -> str:
    if not counts or counts['observations'] == 0:
        return 'not selected / no row'
    if counts['valid'] == 0:
        return 'no packet'
    if state == 'blocked':
        return 'packet waiting'
    if state == 'mixed_blocked_flow':
        return 'packets mixed'
    if state == 'flow':
        return 'packet moved right'
    return 'packet present'


def infer_group_and_tile(nodes: list[dict[str, str]], tile: int) -> tuple[int, int]:
    for node in nodes:
        if parse_int(node.get('tile')) == tile:
            group = parse_int(node.get('group'), -1)
            tile_in_group = parse_int(node.get('tile_in_group'), -1)
            return group if group is not None else -1, tile_in_group if tile_in_group is not None else -1
    raise SystemExit(f'Tile {tile} is not present in nodes.csv')


def infer_ports(
    nodes: list[dict[str, str]],
    lanes: list[dict[str, str]],
    tile: int,
    group: int,
    tile_in_group: int,
    explicit_ports: list[int] | None,
) -> list[int]:
    if explicit_ports:
        return sorted(explicit_ports)
    ports = set()
    for node in nodes:
        if parse_int(node.get('tile')) == tile and node.get('point') in {'tcdm_remote', 'remote_xbar_out', 'tile_master_req_out'}:
            port = parse_int(node.get('port'))
            if port is not None and port >= 0:
                ports.add(port)
    for lane in lanes:
        if parse_int(lane.get('group')) == group and parse_int(lane.get('tile_in_group')) == tile_in_group:
            if lane.get('channel') == 'req' and lane.get('stage') == 'out':
                port = parse_int(lane.get('port'))
                if port is not None and port >= 0:
                    ports.add(port)
    if not ports:
        raise SystemExit('Could not infer route ports; pass --port explicitly')
    return sorted(ports)


def infer_sources(nodes: list[dict[str, str]], tile: int, explicit_source: int | None) -> list[int]:
    if explicit_source is not None:
        return [explicit_source]
    sources = set()
    for node in nodes:
        if parse_int(node.get('tile')) == tile and node.get('point') == 'core_q':
            source = parse_int(node.get('core'))
            if source is not None and source >= 0:
                sources.add(source)
    if not sources:
        raise SystemExit('Could not infer source lanes; pass --source explicitly')
    return sorted(sources)


def parse_ports(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    ports = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                ports.append(int(part, 0))
    return ports
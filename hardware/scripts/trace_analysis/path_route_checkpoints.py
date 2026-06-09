#!/usr/bin/env python3
"""Build route-checkpoint CSVs from a normalized path graph dataset.

This is the CSV-first companion to the packet-flow HTML view. It emits a detailed
checkpoint table and a compact per-request summary so bottlenecks can be sorted
and inspected without drawing the graph.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

from path_graph_common import (
    COUNT_FIELDS,
    infer_group_and_tile,
    infer_ports,
    infer_sources,
    in_cycle_range,
    packet_status,
    parse_int,
    parse_ports,
    read_csv_rows,
    require_graph_dir,
    state_from_counts,
)


SOURCE_CORE_CHECKPOINTS = (
    ('source_core_q', 'core_q', 'source tile core request q'),
    ('source_preroute', 'tcdm_preroute', 'source tile preroute request'),
)

SOURCE_ROUTE_CHECKPOINT = ('source_remote_select', 'tcdm_remote', 'source tile remote select into tile xbar')

SOURCE_SHARED_CHECKPOINTS = (
    ('source_tile_xbar_out', 'remote_xbar_out', 'source tile xbar output / prereg'),
    ('source_tile_master_out', 'tile_master_req_out', 'source tile master request output'),
)

GROUP_LANE_STAGES = ('in0', 'post0', 'in1', 'post1', 'out')

TARGET_CHECKPOINTS = (
    ('target_slave_req_in', 'tile_slave_req_in', 'target tile slave request input'),
    ('target_slave_req_postreg', 'tile_slave_req_postreg', 'target tile slave request post register'),
    ('target_local_xbar_out', 'local_xbar_out', 'target tile local xbar output'),
    ('target_bank_req', 'bank_req', 'target tile bank request'),
)

DETAIL_FIELDS = (
    'cycle',
    'checkpoint_index',
    'checkpoint',
    'scope',
    'evidence',
    'group',
    'tile',
    'tile_in_group',
    'source_core',
    'port',
    'bank',
    'lane_stage',
    'state',
    'packet',
    *COUNT_FIELDS,
    'addr',
    'decoded_target_group',
    'decoded_target_tile',
    'decoded_target_tile_in_group',
    'decoded_target_bank',
    'decoded_target_addr',
    'meta_id',
    'payload_core',
    'identity_key',
    'is_main_path',
    'is_contention',
    'subject_id',
    'description',
    'note',
)

SUMMARY_FIELDS = (
    'cycle',
    'source_tile',
    'source_core',
    'port',
    'addr',
    'decoded_target_group',
    'decoded_target_tile',
    'decoded_target_tile_in_group',
    'decoded_target_bank',
    'decoded_target_addr',
    'meta_id',
    'payload_core',
    'identity_key',
    'source_master_state',
    'source_master_fire',
    'first_blocked_checkpoint',
    'blocked_checkpoints',
    'active_group_lane_tiles',
    'blocked_group_lane_tiles',
    'active_group_lanes',
    'blocked_group_lanes',
    'same_port_active_lane_count',
    'same_port_blocked_lane_count',
    'target_match_tiles',
    'target_postreg_tiles',
    'target_first_cycle',
    'target_first_delta',
    'target_first_checkpoint',
    'target_first_state',
    'target_match_kind',
    'notes',
)


def counts_from_row(row: dict[str, str]) -> Counter:
    counts: Counter = Counter()
    for field in COUNT_FIELDS:
        counts[field] = parse_int(row.get(field), 0) or 0
    return counts


def row_state(row: dict[str, str]) -> str:
    return row.get('state') or state_from_counts(counts_from_row(row))


def row_packet(row: dict[str, str]) -> str:
    return packet_status(row_state(row), counts_from_row(row))


def is_active(row: dict[str, str]) -> bool:
    return any((parse_int(row.get(field), 0) or 0) > 0 for field in ('valid', 'fire', 'stall'))


def is_blocked(row: dict[str, str]) -> bool:
    return (parse_int(row.get('stall'), 0) or 0) > 0 or row_state(row) == 'blocked'


def identity_key(row: dict[str, str]) -> str:
    addr = row.get('addr', '')
    meta_id = row.get('meta_id', '')
    payload_core = row.get('payload_core', '')
    if not addr and not meta_id and not payload_core:
        return ''
    return f'addr={addr}|meta={meta_id}|payload_core={payload_core}'


def normalized_hex(value: str) -> str:
    value = value.strip().lower()
    if value.startswith('0x'):
        value = value[2:]
    value = value.lstrip('0')
    return value or '0'


def node_matches(row: dict[str, str], *, tile: int | None = None, point: str | None = None,
                 core: int | None = None, port: int | None = None) -> bool:
    if tile is not None and parse_int(row.get('tile')) != tile:
        return False
    if point is not None and row.get('point') != point:
        return False
    if core is not None and parse_int(row.get('core')) != core:
        return False
    if port is not None and parse_int(row.get('port')) != port:
        return False
    return True


def lane_matches(row: dict[str, str], *, group: int | None = None, port: int | None = None,
                 channel: str | None = None, stage: str | None = None) -> bool:
    if group is not None and parse_int(row.get('group')) != group:
        return False
    if port is not None and parse_int(row.get('port')) != port:
        return False
    if channel is not None and row.get('channel') != channel:
        return False
    if stage is not None and row.get('stage') != stage:
        return False
    return True


def build_cycle_indexes(
    node_rows: list[dict[str, str]],
    lane_rows: list[dict[str, str]],
) -> tuple[
    dict[tuple[int, str, int], list[dict[str, str]]],
    dict[tuple[int, str, int, int], list[dict[str, str]]],
    dict[tuple[int, str, int], list[dict[str, str]]],
    dict[tuple[int, int, str], list[dict[str, str]]],
]:
    nodes_by_tile_point_core: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    nodes_by_tile_point_core_port: dict[tuple[int, str, int, int], list[dict[str, str]]] = defaultdict(list)
    nodes_by_tile_point_port: dict[tuple[int, str, int], list[dict[str, str]]] = defaultdict(list)
    lanes_by_group_port_channel: dict[tuple[int, int, str], list[dict[str, str]]] = defaultdict(list)

    for row in node_rows:
        tile = parse_int(row.get('tile'))
        point = row.get('point', '')
        core = parse_int(row.get('core'))
        port = parse_int(row.get('port'))
        if tile is None or not point:
            continue
        if core is not None:
            nodes_by_tile_point_core[(tile, point, core)].append(row)
        if core is not None and port is not None:
            nodes_by_tile_point_core_port[(tile, point, core, port)].append(row)
        if port is not None:
            nodes_by_tile_point_port[(tile, point, port)].append(row)

    for row in lane_rows:
        group = parse_int(row.get('group'))
        port = parse_int(row.get('port'))
        channel = row.get('channel', '')
        if group is None or port is None or not channel:
            continue
        lanes_by_group_port_channel[(group, port, channel)].append(row)

    return (
        nodes_by_tile_point_core,
        nodes_by_tile_point_core_port,
        nodes_by_tile_point_port,
        lanes_by_group_port_channel,
    )


def index_csv_rows_by_cycle(csv_path: Path, cycle_start: int | None,
                            cycle_end: int | None) -> dict[int, list[dict[str, str]]]:
    by_cycle: dict[int, list[dict[str, str]]] = defaultdict(list)
    with csv_path.open(newline='') as file:
        reader = csv.DictReader(file)
        for row_index, row in enumerate(reader, start=1):
            cycle = parse_int(row.get('cycle'))
            if cycle is None or not in_cycle_range(cycle, cycle_start, cycle_end):
                continue
            by_cycle[cycle].append(row)
            if row_index % 500000 == 0:
                print(f'Indexed {row_index} rows from {csv_path.name}', flush=True)
    return by_cycle


def infer_power_of_two_domain(max_value: int, minimum: int = 1) -> int:
    if max_value < minimum:
        return minimum
    return 1 << max(0, math.ceil(math.log2(max_value + 1)))


def infer_tiles_per_group(tile_configs: list[dict[str, object]]) -> int:
    max_tile_in_group = 0
    for tile_config in tile_configs:
        tile_in_group = int(tile_config.get('tile_in_group', 0))
        max_tile_in_group = max(max_tile_in_group, tile_in_group)
    return infer_power_of_two_domain(max_tile_in_group, 1)


def infer_bank_count(nodes: list[dict[str, str]]) -> int:
    max_bank = 0
    for row in nodes:
        bank = parse_int(row.get('bank'))
        if bank is not None and bank > max_bank:
            max_bank = bank
    return infer_power_of_two_domain(max_bank, 1)


def decode_target_from_source(
    source_tile: int,
    source_group: int,
    port: int,
    addr: str,
    *,
    num_groups: int,
    tiles_per_group: int,
    banks_per_tile: int,
    back2local: bool | int = False,
) -> dict[str, int | str] | None:
    addr = normalized_hex(addr)
    if not addr:
        return None
    try:
        raw_addr = int(addr, 16)
    except ValueError:
        return None
    if port < 0 or port >= num_groups:
        return None

    tile_bits = max(0, (tiles_per_group - 1).bit_length())
    tile_mask = (1 << tile_bits) - 1
    target_tile_in_group = raw_addr & tile_mask
    if target_tile_in_group >= tiles_per_group:
        return None

    target_group = source_group if back2local else source_group ^ port
    if target_group >= num_groups:
        return None

    target_addr_int = raw_addr >> tile_bits
    target_tile = target_group * tiles_per_group + target_tile_in_group
    return {
        'target_group': target_group,
        'target_tile': target_tile,
        'target_tile_in_group': target_tile_in_group,
        'target_bank': target_addr_int & (banks_per_tile - 1) if banks_per_tile > 0 else '',
        'target_addr': format(target_addr_int, 'x'),
        'source_tile': source_tile,
    }


def build_target_index(
    nodes_by_cycle: dict[int, list[dict[str, str]]]
) -> dict[tuple[int, int, str, str, str], list[dict[str, str]]]:
    target_points = {point for _, point, _ in TARGET_CHECKPOINTS}
    target_index: dict[tuple[int, int, str, str, str], list[dict[str, str]]] = defaultdict(list)
    for cycle, rows in nodes_by_cycle.items():
        for row in rows:
            if row.get('point') not in target_points:
                continue
            tile = parse_int(row.get('tile'))
            if tile is None:
                continue
            key = (
                cycle,
                tile,
                normalized_hex(row.get('addr', '')),
                row.get('meta_id', ''),
                row.get('payload_core', ''),
            )
            target_index[key].append(row)
    return target_index


def pick_one(rows: list[dict[str, str]]) -> dict[str, str] | None:
    active = [row for row in rows if is_active(row)]
    if active:
        return sorted(active, key=lambda row: (parse_int(row.get('fire'), 0) or 0, row.get('subject_id', '')), reverse=True)[0]
    return rows[0] if rows else None


def detail_row(
    row: dict[str, str],
    *,
    checkpoint_index: int,
    checkpoint: str,
    scope: str,
    evidence: str,
    source_core: int | None,
    lane_stage: str = '',
    main_path: bool,
    contention: bool,
    description: str,
    note: str,
) -> dict[str, str | int]:
    output: dict[str, str | int] = {
        'cycle': row.get('cycle', ''),
        'checkpoint_index': checkpoint_index,
        'checkpoint': checkpoint,
        'scope': scope,
        'evidence': evidence,
        'group': row.get('group', ''),
        'tile': row.get('tile', ''),
        'tile_in_group': row.get('tile_in_group', ''),
        'source_core': '' if source_core is None else source_core,
        'port': row.get('port', ''),
        'bank': row.get('bank', ''),
        'lane_stage': lane_stage,
        'state': row_state(row),
        'packet': row_packet(row),
        'addr': row.get('addr', ''),
        'meta_id': row.get('meta_id', ''),
        'payload_core': row.get('payload_core', ''),
        'identity_key': identity_key(row),
        'is_main_path': int(main_path),
        'is_contention': int(contention),
        'subject_id': row.get('subject_id', ''),
        'description': description,
        'note': note,
    }
    for field in COUNT_FIELDS:
        output[field] = parse_int(row.get(field), 0) or 0
    return output


def empty_summary(cycle: int, source_tile: int, source_core: int | None, port: int,
                  source_row: dict[str, str]) -> dict[str, str | int]:
    return {
        'cycle': cycle,
        'source_tile': source_tile,
        'source_core': '' if source_core is None else source_core,
        'port': port,
        'addr': source_row.get('addr', ''),
        'decoded_target_group': '',
        'decoded_target_tile': '',
        'decoded_target_tile_in_group': '',
        'decoded_target_bank': '',
        'decoded_target_addr': '',
        'meta_id': source_row.get('meta_id', ''),
        'payload_core': source_row.get('payload_core', ''),
        'identity_key': identity_key(source_row),
        'source_master_state': row_state(source_row),
        'source_master_fire': parse_int(source_row.get('fire'), 0) or 0,
        'first_blocked_checkpoint': '',
        'blocked_checkpoints': '',
        'active_group_lane_tiles': '',
        'blocked_group_lane_tiles': '',
        'active_group_lanes': '',
        'blocked_group_lanes': '',
        'same_port_active_lane_count': 0,
        'same_port_blocked_lane_count': 0,
        'target_match_tiles': '',
        'target_postreg_tiles': '',
        'target_first_cycle': '',
        'target_first_delta': '',
        'target_first_checkpoint': '',
        'target_first_state': '',
        'target_match_kind': '',
        'notes': '',
    }


def parse_int_list(values: list[str] | None) -> list[int] | None:
    if not values:
        return None
    items = []
    for value in values:
        for part in value.split(','):
            part = part.strip()
            if part:
                items.append(int(part, 0))
    return items


def infer_source_tiles(nodes: list[dict[str, str]], explicit_tiles: list[int] | None) -> list[int]:
    if explicit_tiles:
        return sorted(set(explicit_tiles))
    tiles = {
        tile for row in nodes
        if row.get('point') == 'tile_master_req_out'
        for tile in [parse_int(row.get('tile'))]
        if tile is not None and tile >= 0
    }
    if not tiles:
        tiles = {
            tile for row in nodes
            for tile in [parse_int(row.get('tile'))]
            if tile is not None and tile >= 0
        }
    if not tiles:
        raise SystemExit('Could not infer source tiles; pass --tile explicitly')
    return sorted(tiles)


def append_source_core_rows(
    detail_rows: list[dict[str, str | int]],
    nodes_by_tile_point_core: dict[tuple[int, str, int], list[dict[str, str]]],
    source_tile: int,
    sources: list[int],
) -> None:
    for source_core in sources:
        for checkpoint_index, (checkpoint, point, description) in enumerate(SOURCE_CORE_CHECKPOINTS):
            selected = pick_one(nodes_by_tile_point_core.get((source_tile, point, source_core), []))
            if selected is None:
                continue
            detail_rows.append(detail_row(
                selected,
                checkpoint_index=checkpoint_index,
                checkpoint=checkpoint,
                scope='source_tile',
                evidence='identity_node',
                source_core=source_core,
                main_path=True,
                contention=False,
                description=description,
                note='per-core source tile checkpoint',
            ))


def append_source_route_rows(
    detail_rows: list[dict[str, str | int]],
    nodes_by_tile_point_core_port: dict[tuple[int, str, int, int], list[dict[str, str]]],
    source_tile: int,
    sources: list[int],
    ports: list[int],
) -> None:
    checkpoint, point, description = SOURCE_ROUTE_CHECKPOINT
    for source_core in sources:
        for port in ports:
            selected = pick_one(nodes_by_tile_point_core_port.get((source_tile, point, source_core, port), []))
            if selected is None:
                continue
            detail_rows.append(detail_row(
                selected,
                checkpoint_index=2,
                checkpoint=checkpoint,
                scope='source_tile',
                evidence='identity_node',
                source_core=source_core,
                main_path=True,
                contention=False,
                description=description,
                note='per-core source tile remote-select checkpoint',
            ))


def append_source_shared_rows(
    detail_rows: list[dict[str, str | int]],
    nodes_by_tile_point_port: dict[tuple[int, str, int], list[dict[str, str]]],
    source_tile: int,
    ports: list[int],
) -> list[tuple[str, dict[str, str]]]:
    shared_rows: list[tuple[str, dict[str, str]]] = []
    for port in ports:
        for checkpoint_index, (checkpoint, point, description) in enumerate(SOURCE_SHARED_CHECKPOINTS, start=3):
            selected = pick_one(nodes_by_tile_point_port.get((source_tile, point, port), []))
            if selected is None:
                continue
            selected_source = parse_int(selected.get('payload_core'))
            shared_rows.append((checkpoint, selected))
            detail_rows.append(detail_row(
                selected,
                checkpoint_index=checkpoint_index,
                checkpoint=checkpoint,
                scope='source_tile',
                evidence='identity_node',
                source_core=selected_source,
                main_path=True,
                contention=False,
                description=description,
                note='shared source tile output checkpoint; source_core comes from payload_core',
            ))
    return shared_rows


def append_group_lane_rows(
    detail_rows: list[dict[str, str | int]],
    lanes_by_group_port_channel: dict[tuple[int, int, str], list[dict[str, str]]],
    group: int,
    source_tile: int,
    source_core: int | None,
    port: int,
    include_idle_lanes: bool,
) -> tuple[list[str], list[str], list[str], list[str], int, int]:
    same_port_lanes = [
        row for row in lanes_by_group_port_channel.get((group, port, 'req'), [])
        if include_idle_lanes or is_active(row) or row.get('stage') == 'out'
    ]
    for lane_row in same_port_lanes:
        lane_stage = lane_row.get('stage', '')
        lane_index = 10 + GROUP_LANE_STAGES.index(lane_stage) if lane_stage in GROUP_LANE_STAGES else 10
        detail_rows.append(detail_row(
            lane_row,
            checkpoint_index=lane_index,
            checkpoint=f'group_req_{lane_stage}',
            scope='group_lane',
            evidence='aggregate_lane',
            source_core=source_core,
            lane_stage=lane_stage,
            main_path=False,
            contention=parse_int(lane_row.get('tile')) != source_tile,
            description='group request lane activity for same output port',
            note='lane aggregate: valid/ready/fire only; packet identity not proven',
        ))
    active_tiles = sorted({row.get('tile', '') for row in same_port_lanes if is_active(row)})
    blocked_tiles = sorted({row.get('tile', '') for row in same_port_lanes if is_blocked(row)})
    active_lanes = sorted({f"{row.get('tile', '')}:{row.get('stage', '')}" for row in same_port_lanes if is_active(row)})
    blocked_lanes = sorted({f"{row.get('tile', '')}:{row.get('stage', '')}" for row in same_port_lanes if is_blocked(row)})
    active_count = sum(1 for row in same_port_lanes if is_active(row))
    blocked_count = sum(1 for row in same_port_lanes if is_blocked(row))
    return active_tiles, blocked_tiles, active_lanes, blocked_lanes, active_count, blocked_count


def append_target_matches(
    detail_rows: list[dict[str, str | int]],
    node_rows: list[dict[str, str]],
    target_index: dict[tuple[int, int, str, str, str], list[dict[str, str]]],
    cycle: int,
    source_core: int | None,
    source_identity: str,
    decoded_target: dict[str, int | str] | None,
    source_row: dict[str, str],
    target_window: int,
) -> tuple[set[str], set[str], list[str], dict[str, str | int]]:
    target_tiles: set[str] = set()
    postreg_tiles: set[str] = set()
    blocked_checkpoints: list[str] = []
    first_match: dict[str, str | int] = {}

    matches_by_point: dict[str, list[dict[str, str]]] = defaultdict(list)
    match_kind = ''
    if decoded_target is not None:
        target_tile = int(decoded_target['target_tile'])
        target_addr = str(decoded_target['target_addr'])
        meta_id = source_row.get('meta_id', '')
        payload_core = source_row.get('payload_core', '')
        for target_cycle in range(cycle, cycle + max(0, target_window) + 1):
            key = (target_cycle, target_tile, target_addr, meta_id, payload_core)
            for row in target_index.get(key, []):
                matches_by_point[row.get('point', '')].append(row)
        if matches_by_point:
            match_kind = 'decoded_target_window'

    if not source_identity:
        return target_tiles, postreg_tiles, blocked_checkpoints, first_match

    if not matches_by_point:
        for _, point, _ in TARGET_CHECKPOINTS:
            matches_by_point[point].extend(
                row for row in node_rows
                if row.get('point') == point and identity_key(row) == source_identity
            )
        if any(matches_by_point.values()):
            match_kind = 'same_cycle_identity'

    for target_checkpoint_index, (checkpoint, point, description) in enumerate(TARGET_CHECKPOINTS, start=20):
        matches = sorted(
            matches_by_point.get(point, []),
            key=lambda row: (parse_int(row.get('cycle'), cycle) or cycle, row.get('subject_id', '')),
        )[:1]
        for match in matches:
            target_tiles.add(match.get('tile', ''))
            if point == 'tile_slave_req_postreg':
                postreg_tiles.add(match.get('tile', ''))
            if is_blocked(match):
                blocked_checkpoints.append(checkpoint)
            if not first_match:
                match_cycle = parse_int(match.get('cycle'), cycle) or cycle
                first_match = {
                    'target_first_cycle': match_cycle,
                    'target_first_delta': match_cycle - cycle,
                    'target_first_checkpoint': checkpoint,
                    'target_first_state': row_state(match),
                    'target_match_kind': match_kind,
                }
            detail_rows.append(detail_row(
                match,
                checkpoint_index=target_checkpoint_index,
                checkpoint=checkpoint,
                scope='target_tile',
                evidence=match_kind or 'identity_node_match',
                source_core=source_core,
                main_path=True,
                contention=False,
                description=description,
                note='matched by decoded target tile/address within target window'
                if match_kind == 'decoded_target_window'
                else 'matched by addr/meta_id/payload_core in same cycle',
            ))
    return target_tiles, postreg_tiles, blocked_checkpoints, first_match


def write_checkpoint_csvs(
    graph_dir: Path,
    output_path: Path,
    summary_path: Path,
    tile_configs: list[dict[str, object]],
    cycle_start: int | None,
    cycle_end: int | None,
    include_idle_lanes: bool,
    target_window: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', newline='') as detail_file, summary_path.open('w', newline='') as summary_file:
        detail_writer = csv.DictWriter(detail_file, fieldnames=DETAIL_FIELDS)
        summary_writer = csv.DictWriter(summary_file, fieldnames=SUMMARY_FIELDS)
        detail_writer.writeheader()
        summary_writer.writeheader()
        detail_file.flush()
        summary_file.flush()

        nodes_by_cycle = index_csv_rows_by_cycle(graph_dir / 'cycle_node_state.csv', cycle_start, cycle_end)
        lanes_by_cycle = index_csv_rows_by_cycle(graph_dir / 'cycle_lane_state.csv', cycle_start, cycle_end)
        target_index = build_target_index(nodes_by_cycle)
        cycles = sorted(set(nodes_by_cycle) | set(lanes_by_cycle))
        total_cycles = len(cycles)
        tiles_per_group = infer_tiles_per_group(tile_configs)
        num_groups = infer_power_of_two_domain(max(int(config['group']) for config in tile_configs), 1)
        banks_per_tile = infer_bank_count(read_csv_rows(graph_dir / 'nodes.csv'))
        print(f'Prepared {total_cycles} cycles for checkpoint extraction', flush=True)

        for cycle_index, cycle in enumerate(cycles, start=1):
            node_rows = nodes_by_cycle.get(cycle, [])
            lane_rows = lanes_by_cycle.get(cycle, [])
            (
                nodes_by_tile_point_core,
                nodes_by_tile_point_core_port,
                nodes_by_tile_point_port,
                lanes_by_group_port_channel,
            ) = build_cycle_indexes(node_rows, lane_rows)
            detail_rows: list[dict[str, str | int]] = []
            summary_rows: list[dict[str, str | int]] = []

            for tile_config in tile_configs:
                source_tile = int(tile_config['tile'])
                group = int(tile_config['group'])
                sources = list(tile_config['sources'])
                ports = list(tile_config['ports'])
                append_source_core_rows(detail_rows, nodes_by_tile_point_core, source_tile, sources)
                append_source_route_rows(detail_rows, nodes_by_tile_point_core_port, source_tile, sources, ports)
                shared_rows = append_source_shared_rows(detail_rows, nodes_by_tile_point_port, source_tile, ports)
                active_master_rows = [
                    row for checkpoint, row in shared_rows
                    if checkpoint == 'source_tile_master_out' and is_active(row)
                ]

                for master_out in active_master_rows:
                    port = parse_int(master_out.get('port'), -1)
                    if port is None or port < 0:
                        continue
                    source_core = parse_int(master_out.get('payload_core'))
                    summary = empty_summary(cycle, source_tile, source_core, port, master_out)
                    decoded_target = decode_target_from_source(
                        source_tile,
                        group,
                        port,
                        master_out.get('addr', ''),
                        num_groups=num_groups,
                        tiles_per_group=tiles_per_group,
                        banks_per_tile=banks_per_tile,
                        back2local=(parse_int(master_out.get('back2local'), 0) or 0) > 0,
                    )
                    if decoded_target is not None:
                        summary['decoded_target_group'] = decoded_target['target_group']
                        summary['decoded_target_tile'] = decoded_target['target_tile']
                        summary['decoded_target_tile_in_group'] = decoded_target['target_tile_in_group']
                        summary['decoded_target_bank'] = decoded_target['target_bank']
                        summary['decoded_target_addr'] = decoded_target['target_addr']
                    blocked_checkpoints = [
                        checkpoint for checkpoint, row in shared_rows
                        if row.get('port') == master_out.get('port') and is_blocked(row)
                    ]
                    active_tiles, blocked_tiles, active_lanes, blocked_lanes, active_count, blocked_count = append_group_lane_rows(
                        detail_rows,
                        lanes_by_group_port_channel,
                        group,
                        source_tile,
                        source_core,
                        port,
                        include_idle_lanes,
                    )
                    target_tiles, postreg_tiles, target_blocked, first_target = append_target_matches(
                        detail_rows,
                        node_rows,
                        target_index,
                        cycle,
                        source_core,
                        identity_key(master_out),
                        decoded_target,
                        master_out,
                        target_window,
                    )
                    blocked_checkpoints.extend(target_blocked)
                    summary['active_group_lane_tiles'] = ';'.join(active_tiles)
                    summary['blocked_group_lane_tiles'] = ';'.join(blocked_tiles)
                    summary['active_group_lanes'] = ';'.join(active_lanes)
                    summary['blocked_group_lanes'] = ';'.join(blocked_lanes)
                    summary['same_port_active_lane_count'] = active_count
                    summary['same_port_blocked_lane_count'] = blocked_count
                    summary['target_match_tiles'] = ';'.join(sorted(tile for tile in target_tiles if tile))
                    summary['target_postreg_tiles'] = ';'.join(sorted(tile for tile in postreg_tiles if tile))
                    summary.update(first_target)
                    summary['blocked_checkpoints'] = ';'.join(dict.fromkeys(blocked_checkpoints))
                    summary['first_blocked_checkpoint'] = blocked_checkpoints[0] if blocked_checkpoints else ''
                    notes = []
                    if active_count or blocked_count:
                        notes.append('group lanes are aggregate same-port evidence')
                    if decoded_target is not None:
                        notes.append('decoded target assumes MemPool xor port routing')
                    if not target_tiles:
                        notes.append('no decoded/identity target tile nodes matched in target window')
                    summary['notes'] = '; '.join(notes)
                    summary_rows.append(summary)

            if detail_rows:
                detail_writer.writerows(detail_rows)
            if summary_rows:
                summary_writer.writerows(summary_rows)
            detail_file.flush()
            summary_file.flush()
            if cycle_index == 1 or cycle_index == total_cycles or cycle_index % 250 == 0:
                print(
                    f'Processed {cycle_index}/{total_cycles} cycles; '
                    f'wrote {len(detail_rows)} detail rows and {len(summary_rows)} summaries for cycle {cycle}',
                    flush=True,
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph_dir', type=Path, help='Path graph dataset directory from path_graph_dataset.py')
    parser.add_argument('--tile', action='append', help='Source tile id(s) to include, comma-separated; default: all observed tiles')
    parser.add_argument('--core', '--source', dest='source', type=int, help='Limit to one local source core')
    parser.add_argument('--port', action='append', help='Output port(s), comma-separated; may be repeated')
    parser.add_argument('--cycle-start', type=int, help='First cycle to include')
    parser.add_argument('--cycle-end', type=int, help='Last cycle to include')
    parser.add_argument('--output', type=Path, help='Detailed checkpoint CSV path')
    parser.add_argument('--summary-output', type=Path, help='Compact summary CSV path')
    parser.add_argument('--include-idle-lanes', action='store_true', help='Include idle same-port group lane rows too')
    parser.add_argument('--target-window', type=int, default=16, help='Cycles to look ahead for decoded target-side evidence')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    graph_dir = require_graph_dir(args.graph_dir)
    nodes = read_csv_rows(graph_dir / 'nodes.csv')
    lanes = read_csv_rows(graph_dir / 'lanes.csv')
    source_tiles = infer_source_tiles(nodes, parse_int_list(args.tile))
    explicit_ports = parse_ports(args.port)
    tile_configs = []
    for source_tile in source_tiles:
        group, tile_in_group = infer_group_and_tile(nodes, source_tile)
        tile_configs.append({
            'tile': source_tile,
            'group': group,
            'tile_in_group': tile_in_group,
            'sources': infer_sources(nodes, source_tile, args.source),
            'ports': infer_ports(nodes, lanes, source_tile, group, tile_in_group, explicit_ports),
        })
    default_stem = f'tile{source_tiles[0]}' if len(source_tiles) == 1 else 'all_tiles'
    output = args.output or (graph_dir / f'route_checkpoints_{default_stem}.csv')
    summary_output = args.summary_output or (graph_dir / f'route_bottlenecks_{default_stem}.csv')
    write_checkpoint_csvs(
        graph_dir,
        output,
        summary_output,
        tile_configs,
        args.cycle_start,
        args.cycle_end,
        args.include_idle_lanes,
        args.target_window,
    )
    print(f'Wrote route checkpoint CSV to {output}')
    print(f'Wrote bottleneck summary CSV to {summary_output}')
    print(f'Source tiles: {", ".join(str(tile) for tile in source_tiles)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
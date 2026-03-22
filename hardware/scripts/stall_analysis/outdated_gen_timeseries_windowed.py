#!/usr/bin/env python3
"""Legacy windowed time-series metric generator.

This script is retained only for the optional `timeline_window` Makefile path.
It produces window-aggregated performance metrics in `timeline.csv` and is
separate from the cycle-accurate stall-analysis pipeline.
"""

import argparse
import csv
import os
import re
import sys
import warnings
from collections import defaultdict, deque

import numpy as np


TRACE_IN_REGEX = r'(\d+)\s+(\d+)\s+(0x[0-9A-Fa-fz]+)\s+([^#;]*)(\s*#;\s*(.*))?'
ANNOTATED_TRACE_REGEX = r'^\s*(\d+)\s+(\d+)(?:\s+(0x[0-9A-Fa-f]+)\s+(.*?))?\s*$'

REG_ABI_NAMES_I = (
    'zero', 'ra', 'sp', 'gp', 'tp',
    't0', 't1', 't2',
    's0', 's1',
    *('a{}'.format(i) for i in range(8)),
    *('s{}'.format(i) for i in range(2, 12)),
    *('t{}'.format(i) for i in range(3, 7))
)

OPER_TYPES = {'gpr': 1, 'csr': 8}
MEM_REGIONS = {'Other': 0, 'Sequential': 1, 'Interleaved': 2}
RAW_TYPES = ['lsu', 'acc']
RAW_PERF_LIST_KEYS = (
    'snitch_load_latency',
    'snitch_load_region',
    'snitch_load_tile',
    'snitch_store_region',
    'snitch_store_tile',
)


def getenv_int(*names, default):
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


NUM_CORES = getenv_int('NUM_CORES', 'num_cores', default=256)
NUM_GROUPS = getenv_int('NUM_GROUPS', 'num_groups', default=1)
NUM_CORES_PER_TILE = getenv_int(
    'NUM_CORES_PER_TILE', 'num_cores_per_tile', default=4)
NUM_TILES = NUM_CORES // NUM_CORES_PER_TILE
SEQ_MEM_SIZE = 4 * getenv_int('SEQ_MEM_SIZE', 'seq_mem_size', default=1024)
TCDM_SIZE = 16 * 1024 * NUM_TILES


def read_annotations(dict_str: str) -> dict:
    annot = {key: int(val, 16) for key, val in
             re.findall(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)", dict_str)}
    annot.update({key: val for key, val in
                  re.findall(r"'([^']+)'\s*:\s*(0x[0-9a-fA-FxX]+)",
                             re.sub(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)",
                                    '', dict_str))})
    return annot


def add_perf_metric(metric: dict, key: str, value: int = 1):
    metric[key] += value


def append_perf_metric(metric: dict, key: str, value):
    metric.setdefault(key, []).append(value)


def clone_perf_metric(metric: dict) -> dict:
    cloned = defaultdict(int)
    for key, value in metric.items():
        cloned[key] = list(value) if isinstance(value, list) else value
    return cloned


def diff_perf_metric(curr_metric: dict, prev_metric: dict | None) -> dict:
    if prev_metric is None:
        delta_metric = clone_perf_metric(curr_metric)
        delta_metric['start'] = curr_metric['start']
        return delta_metric

    delta_metric = defaultdict(int)
    for key in set(curr_metric.keys()) | set(prev_metric.keys()):
        if key in ('core', 'group', 'tile', 'tile_in_group', 'core_in_tile',
                   'core_in_group', 'section', 'sample_index', 'sample_cycle',
                   'sample_type', 'sample_mode', 'start', 'end'):
            continue
        curr_val = curr_metric.get(key, [] if key in RAW_PERF_LIST_KEYS else 0)
        prev_val = prev_metric.get(key, [] if key in RAW_PERF_LIST_KEYS else 0)
        if isinstance(curr_val, list):
            delta_metric[key] = list(curr_val[len(prev_val):])
        else:
            delta_metric[key] = curr_val - prev_val
    delta_metric['start'] = prev_metric['end'] + 1
    return delta_metric


def strip_raw_perf_lists(metric: dict):
    for key in RAW_PERF_LIST_KEYS:
        metric.pop(key, None)


def get_core_hierarchy(core_id: int) -> dict:
    if core_id < 0:
        return {
            'group': -1,
            'tile': -1,
            'tile_in_group': -1,
            'core_in_tile': -1,
            'core_in_group': -1,
        }
    tiles_per_group = NUM_TILES // NUM_GROUPS if NUM_GROUPS else NUM_TILES
    cores_per_group = NUM_CORES // NUM_GROUPS if NUM_GROUPS else NUM_CORES
    tile_id = core_id // NUM_CORES_PER_TILE
    return {
        'group': tile_id // tiles_per_group if tiles_per_group else 0,
        'tile': tile_id,
        'tile_in_group': tile_id % tiles_per_group if tiles_per_group else 0,
        'core_in_tile': core_id % NUM_CORES_PER_TILE,
        'core_in_group': core_id % cores_per_group if cores_per_group else 0,
    }


def add_core_metadata(rows: list, core_id: int, hierarchy: dict):
    for row in rows:
        row['core'] = core_id
        row['group'] = hierarchy['group']
        row['tile'] = hierarchy['tile']
        row['tile_in_group'] = hierarchy['tile_in_group']
        row['core_in_tile'] = hierarchy['core_in_tile']
        row['core_in_group'] = hierarchy['core_in_group']


def addr_to_meta(address):
    region = MEM_REGIONS['Other']
    tile = -1
    if address < SEQ_MEM_SIZE * NUM_TILES:
        region = MEM_REGIONS['Sequential']
        tile = address // SEQ_MEM_SIZE
    elif address < TCDM_SIZE:
        region = MEM_REGIONS['Interleaved']
        tile = (address // 64) % NUM_TILES
    return region, tile


def safe_div(dividend, divisor):
    return dividend / divisor if divisor else None


def eval_perf_metrics(perf_metrics: list, hierarchy: dict):
    tile_id = hierarchy['tile']
    for seg in perf_metrics:
        end = seg['end']
        cycles = end - seg['start'] + 1
        seg.update({
            'snitch_avg_load_latency': np.mean(seg['snitch_load_latency']),
            'snitch_occupancy': safe_div(seg['snitch_issues'], cycles),
        })
        seg['cycles'] = cycles
        seg['total_ipc'] = seg['snitch_occupancy']
        if seg['snitch_loads'] > 0:
            seq_region = [x == MEM_REGIONS['Sequential']
                          for x in seg['snitch_load_region']]
            itl_region = [x == MEM_REGIONS['Interleaved']
                          for x in seg['snitch_load_region']]
            loc_loads = [x == tile_id for x in seg['snitch_load_tile']]
            seq_loads_local = np.logical_and(np.array(seq_region), np.array(loc_loads))
            seq_loads_global = np.logical_and(
                np.array(seq_region), np.invert(np.array(loc_loads)))
            itl_loads_local = np.logical_and(np.array(itl_region), np.array(loc_loads))
            itl_loads_global = np.logical_and(
                np.array(itl_region), np.invert(np.array(loc_loads)))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                seg.update({
                    'seq_loads_local': np.count_nonzero(seq_loads_local),
                    'seq_loads_global': np.count_nonzero(seq_loads_global),
                    'itl_loads_local': np.count_nonzero(itl_loads_local),
                    'itl_loads_global': np.count_nonzero(itl_loads_global),
                    'seq_latency_local': np.mean(
                        np.array(seg['snitch_load_latency'])[seq_loads_local]),
                    'seq_latency_global': np.mean(
                        np.array(seg['snitch_load_latency'])[seq_loads_global]),
                    'itl_latency_local': np.mean(
                        np.array(seg['snitch_load_latency'])[itl_loads_local]),
                    'itl_latency_global': np.mean(
                        np.array(seg['snitch_load_latency'])[itl_loads_global]),
                })
        if seg['snitch_stores'] > 0:
            seq_region = [x == MEM_REGIONS['Sequential']
                          for x in seg['snitch_store_region']]
            itl_region = [x == MEM_REGIONS['Interleaved']
                          for x in seg['snitch_store_region']]
            loc_stores = [x == tile_id for x in seg['snitch_store_tile']]
            seq_stores_local = np.logical_and(np.array(seq_region), np.array(loc_stores))
            seq_stores_global = np.logical_and(
                np.array(seq_region), np.invert(np.array(loc_stores)))
            itl_stores_local = np.logical_and(np.array(itl_region), np.array(loc_stores))
            itl_stores_global = np.logical_and(
                np.array(itl_region), np.invert(np.array(loc_stores)))
            seg.update({
                'seq_stores_local': np.count_nonzero(seq_stores_local),
                'seq_stores_global': np.count_nonzero(seq_stores_global),
                'itl_stores_local': np.count_nonzero(itl_stores_local),
                'itl_stores_global': np.count_nonzero(itl_stores_global),
            })


def build_raw_snapshot(
    perf_metric: dict,
    sample_cycle: int,
    section: int,
    sample_index: int,
    sample_type: str,
):
    snapshot = clone_perf_metric(perf_metric)
    snapshot['end'] = sample_cycle
    snapshot['section'] = section
    snapshot['sample_cycle'] = sample_cycle
    snapshot['sample_index'] = sample_index
    snapshot['sample_type'] = sample_type
    return snapshot


def finalize_snapshot(
    snapshot: dict,
    core_id: int,
    hierarchy: dict,
    sample_mode: str,
):
    snapshot['sample_mode'] = sample_mode
    eval_perf_metrics([snapshot], hierarchy)
    add_core_metadata([snapshot], core_id, hierarchy)
    strip_raw_perf_lists(snapshot)
    return snapshot


def perf_metrics_to_csv(perf_metrics: list, filename: str):
    if not perf_metrics:
        return
    keys = perf_metrics[0].keys()
    known_keys = [
        'core',
        'group',
        'tile',
        'tile_in_group',
        'core_in_tile',
        'core_in_group',
        'section',
        'sample_index',
        'sample_cycle',
        'sample_type',
        'sample_mode',
        'start',
        'end',
        'cycles',
        'snitch_loads',
        'snitch_stores',
        'snitch_avg_load_latency',
        'snitch_occupancy',
        'total_ipc',
        'snitch_issues',
        'stall_tot',
        'stall_ins',
        'stall_raw',
        'stall_raw_lsu',
        'stall_raw_acc',
        'stall_lsu',
        'stall_acc',
        'stall_wfi',
        'seq_loads_local',
        'seq_loads_global',
        'itl_loads_local',
        'itl_loads_global',
        'seq_latency_local',
        'seq_latency_global',
        'itl_latency_local',
        'itl_latency_global',
        'seq_stores_local',
        'seq_stores_global',
        'itl_stores_local',
        'itl_stores_global',
    ]
    for key in keys:
        if key not in known_keys:
            known_keys.append(key)
    write_header = not os.path.exists(filename)
    with open(filename, 'a+') as out:
        dict_writer = csv.DictWriter(out, known_keys)
        if write_header:
            dict_writer.writeheader()
        dict_writer.writerows(perf_metrics)
    print('\nWrote time-series metrics to %s\n' % filename)


def process_trace_line(
    line: str,
    gpr_wb_info: dict,
    perf_metric: dict,
    last_time_info: tuple,
    prev_wfi_time: int,
    retired_reg: dict,
    permissive: bool,
):
    line_stripped = line.strip('\n')
    match = re.search(TRACE_IN_REGEX, line_stripped)
    if match is not None:
        _, cycle_str, _, insn, _, extras_str = match.groups()
        time_info = (int(match.group(1)), int(cycle_str))
        if not extras_str:
            return time_info, 0, retired_reg
        extras = read_annotations(extras_str)
        for key in (
            'stall', 'stall_tot', 'stall_ins', 'stall_raw', 'stall_lsu',
            'stall_acc', 'retire_load', 'retire_acc', 'is_load', 'is_store',
            'is_branch', 'write_rd', 'rd', 'rs1', 'rs2', 'lsu_rd', 'acc_pid',
            'alu_result', 'pc_d', 'ls_size', 'opc_select', 'opa_select',
            'opb_select',
        ):
            extras.setdefault(key, 0)

        raw_stall = {key: 0 for key in RAW_TYPES}
        if perf_metric.get('start') is None:
            perf_metric['start'] = time_info[1] - extras['stall_tot']

        if not extras['stall']:
            for raw_type in RAW_TYPES:
                for reg_name in ('rs1', 'rs2', 'rd'):
                    if extras[reg_name] == retired_reg.get(raw_type, -1):
                        raw_stall[raw_type] = retired_reg[raw_type]
            if extras['is_load']:
                add_perf_metric(perf_metric, 'snitch_loads')
                gpr_wb_info[extras['rd']].appendleft((time_info[1], extras['alu_result']))
            elif extras['is_store']:
                add_perf_metric(perf_metric, 'snitch_stores')
                region, tile = addr_to_meta(extras['alu_result'])
                append_perf_metric(perf_metric, 'snitch_store_region', region)
                append_perf_metric(perf_metric, 'snitch_store_tile', int(tile))

        if extras['retire_load']:
            try:
                start_time, address = gpr_wb_info[extras['lsu_rd']].pop()
                region, tile = addr_to_meta(address)
                append_perf_metric(perf_metric, 'snitch_load_latency', time_info[1] - start_time)
                append_perf_metric(perf_metric, 'snitch_load_region', region)
                append_perf_metric(perf_metric, 'snitch_load_tile', int(tile))
            except IndexError:
                if not permissive:
                    sys.stderr.write(
                        'FATAL: In cycle {}, LSU attempts writeback to {}, but none in flight.\n'.format(
                            time_info[1], REG_ABI_NAMES_I[extras['lsu_rd']]))
                    sys.exit(1)
            retired_reg['lsu'] = extras['lsu_rd']

        if extras['retire_acc'] and extras['acc_pid'] != 0:
            retired_reg['acc'] = extras['acc_pid']

        if not extras['stall']:
            add_perf_metric(perf_metric, 'snitch_issues')
            if extras['stall_tot']:
                add_perf_metric(perf_metric, 'stall_tot', extras['stall_tot'])
                if extras['stall_ins']:
                    add_perf_metric(perf_metric, 'stall_ins', extras['stall_ins'])
                if extras['stall_raw']:
                    add_perf_metric(perf_metric, 'stall_raw', extras['stall_raw'])
                    for raw_type in RAW_TYPES:
                        if raw_stall[raw_type] > 0:
                            add_perf_metric(
                                perf_metric,
                                'stall_raw_{}'.format(raw_type),
                                extras['stall_raw'])
                if extras['stall_lsu']:
                    add_perf_metric(perf_metric, 'stall_lsu', extras['stall_lsu'])
                if extras['stall_acc']:
                    add_perf_metric(perf_metric, 'stall_acc', extras['stall_acc'])
                if prev_wfi_time != 0:
                    add_perf_metric(perf_metric, 'stall_wfi', time_info[1] - prev_wfi_time - 1)
            retired_reg = {key: -1 for key in RAW_TYPES}

        prev_wfi_time = time_info[1] if insn.strip() == 'wfi' and not extras['stall'] else 0
        return time_info, prev_wfi_time, retired_reg

    before, annotation = (line_stripped.split('#;', 1) + [''])[:2]
    annotated_match = re.match(ANNOTATED_TRACE_REGEX, before)
    if annotated_match is None:
        raise ValueError('Not a valid trace line:\n{}'.format(line))

    time_info = (int(annotated_match.group(1)), int(annotated_match.group(2)))
    insn = (annotated_match.group(4) or '').strip()
    annotation = annotation.strip()
    if perf_metric.get('start') is None:
        stall_match = re.search(r'// stall\s+(\d+)\s+cycles', annotation)
        perf_metric['start'] = time_info[1] - int(stall_match.group(1)) if stall_match else time_info[1]

    if annotated_match.group(3) is not None and insn:
        add_perf_metric(perf_metric, 'snitch_issues')

    load_issue = re.search(r'\b([a-z][a-z0-9]*)\s+<~~\s+\w+\[(0x[0-9a-fA-F]+)\]', annotation)
    if load_issue:
        reg_name = load_issue.group(1)
        address = int(load_issue.group(2), 16)
        rd = REG_ABI_NAMES_I.index(reg_name) if reg_name in REG_ABI_NAMES_I else None
        if rd is not None:
            add_perf_metric(perf_metric, 'snitch_loads')
            gpr_wb_info[rd].appendleft((time_info[1], address))

    store_issue = re.search(r'~~>\s+\w+\[(0x[0-9a-fA-F]+)\]', annotation)
    if store_issue:
        address = int(store_issue.group(1), 16)
        region, tile = addr_to_meta(address)
        add_perf_metric(perf_metric, 'snitch_stores')
        append_perf_metric(perf_metric, 'snitch_store_region', region)
        append_perf_metric(perf_metric, 'snitch_store_tile', int(tile))

    retired_load = re.search(r'\(lsu\)\s+([a-z][a-z0-9]*)\s+<--', annotation)
    if retired_load:
        reg_name = retired_load.group(1)
        rd = REG_ABI_NAMES_I.index(reg_name) if reg_name in REG_ABI_NAMES_I else None
        if rd is not None:
            try:
                start_time, address = gpr_wb_info[rd].pop()
                region, tile = addr_to_meta(address)
                append_perf_metric(perf_metric, 'snitch_load_latency', time_info[1] - start_time)
                append_perf_metric(perf_metric, 'snitch_load_region', region)
                append_perf_metric(perf_metric, 'snitch_load_tile', int(tile))
            except IndexError:
                if not permissive:
                    sys.stderr.write(
                        'FATAL: In cycle {}, LSU attempts writeback to {}, but none in flight.\n'.format(
                            time_info[1], reg_name))
                    sys.exit(1)
            retired_reg['lsu'] = rd

    retired_acc = re.search(r'\(acc\)\s+([a-z][a-z0-9]*)\s+<--', annotation)
    if retired_acc:
        reg_name = retired_acc.group(1)
        if reg_name in REG_ABI_NAMES_I:
            retired_reg['acc'] = REG_ABI_NAMES_I.index(reg_name)

    stall_match = re.search(r'// stall\s+(\d+)\s+cycles', annotation)
    if stall_match:
        add_perf_metric(perf_metric, 'stall_tot', int(stall_match.group(1)))
        specific_ins = re.search(r'\((\d+)\s+ins\)', annotation)
        specific_raw = re.search(r'\((\d+)\s+raw', annotation)
        specific_lsu = re.search(r'\((\d+)\s+lsu\)', annotation)
        specific_acc = re.search(r'\((\d+)\s+acc\)', annotation)
        if specific_ins:
            add_perf_metric(perf_metric, 'stall_ins', int(specific_ins.group(1)))
        if specific_raw:
            raw_count = int(specific_raw.group(1))
            add_perf_metric(perf_metric, 'stall_raw', raw_count)
            if 'lsu:' in annotation:
                add_perf_metric(perf_metric, 'stall_raw_lsu', raw_count)
            if 'acc:' in annotation:
                add_perf_metric(perf_metric, 'stall_raw_acc', raw_count)
        if specific_lsu:
            add_perf_metric(perf_metric, 'stall_lsu', int(specific_lsu.group(1)))
        if specific_acc:
            add_perf_metric(perf_metric, 'stall_acc', int(specific_acc.group(1)))
        if prev_wfi_time != 0:
            add_perf_metric(perf_metric, 'stall_wfi', time_info[1] - prev_wfi_time - 1)

    prev_wfi_time = time_info[1] if insn == 'wfi' else 0
    if annotated_match.group(3) is not None and insn:
        retired_reg = {key: -1 for key in RAW_TYPES}
    return time_info, prev_wfi_time, retired_reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infile',
        metavar='infile.dasm',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='A matching ASCII signal dump',
    )
    parser.add_argument(
        '--csv',
        required=True,
        help='CSV file that will receive sampled cumulative metrics')
    parser.add_argument(
        '--window',
        type=int,
        required=True,
        help='Sampling period in cycles')
    parser.add_argument(
        '--section',
        type=int,
        action='append',
        help='Emit samples only for the specified section; may be repeated')
    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Shortcut for --section 1 for apps with a single benchmark bracket')
    parser.add_argument(
        '--delta',
        action='store_true',
        help='Emit per-window deltas instead of cumulative metrics')
    parser.add_argument(
        '-p',
        '--permissive',
        action='store_true',
        help='Ignore some state-related issues when they occur')
    args = parser.parse_args()

    if args.window <= 0:
        parser.error('--window must be > 0')

    selected_sections = set(args.section or [])
    if args.benchmark_only:
        selected_sections.add(1)

    path, filename = os.path.split(args.infile.name)
    core_id_hex = re.search(r'(0x[0-9a-fA-F]+)', filename)
    core_id_dec = re.search(r'([\d]+)', filename)
    if core_id_hex:
        core_id = int(core_id_hex.group(1), 16)
    elif core_id_dec:
        core_id = int(core_id_dec.group(1))
    else:
        core_id = -1

    core_hierarchy = get_core_hierarchy(core_id)
    time_info = (0, 0)
    prev_wfi_time = 0
    retired_reg = {key: -1 for key in RAW_TYPES}
    gpr_wb_info = defaultdict(deque)
    perf_metrics = [defaultdict(int)]
    perf_metrics[0]['start'] = None
    timeline_metrics = []
    next_timeline_cycle = None
    timeline_sample_index = 0
    previous_section_snapshot = {}

    def emit_snapshot(sample_cycle: int, current_section: int, sample_type: str):
        nonlocal timeline_sample_index
        raw_snapshot = build_raw_snapshot(
            perf_metrics[-1],
            sample_cycle,
            current_section,
            timeline_sample_index,
            sample_type,
        )
        if args.delta:
            snapshot = diff_perf_metric(
                raw_snapshot,
                previous_section_snapshot.get(current_section),
            )
            snapshot['end'] = raw_snapshot['end']
            snapshot['section'] = raw_snapshot['section']
            snapshot['sample_cycle'] = raw_snapshot['sample_cycle']
            snapshot['sample_index'] = raw_snapshot['sample_index']
            snapshot['sample_type'] = raw_snapshot['sample_type']
            previous_section_snapshot[current_section] = raw_snapshot
            sample_mode = 'delta'
        else:
            snapshot = raw_snapshot
            sample_mode = 'cumulative'
        timeline_metrics.append(finalize_snapshot(
            snapshot,
            core_id,
            core_hierarchy,
            sample_mode,
        ))
        timeline_sample_index += 1

    def record_timeline_samples(sample_cycle: int, sample_type: str = 'periodic'):
        nonlocal next_timeline_cycle
        current_perf_metric = perf_metrics[-1]
        current_section = len(perf_metrics) - 1
        if current_perf_metric.get('start') is None:
            return
        if next_timeline_cycle is None:
            next_timeline_cycle = current_perf_metric['start']
        while sample_cycle >= next_timeline_cycle:
            if not selected_sections or current_section in selected_sections:
                emit_snapshot(next_timeline_cycle, current_section, sample_type)
            next_timeline_cycle += args.window

    def record_section_end_sample(sample_cycle: int):
        current_section = len(perf_metrics) - 1
        if perf_metrics[-1].get('start') is None:
            return
        record_timeline_samples(sample_cycle)
        if selected_sections and current_section not in selected_sections:
            return
        if (timeline_metrics and
                timeline_metrics[-1]['section'] == current_section and
                timeline_metrics[-1]['sample_cycle'] == sample_cycle):
            return
        emit_snapshot(sample_cycle, current_section, 'section_end')

    for line in iter(args.infile.readline, b''):
        if not line:
            break
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if not re.match(r'^\d', stripped_line):
            continue
        time_info, prev_wfi_time, retired_reg = process_trace_line(
            line,
            gpr_wb_info,
            perf_metrics[-1],
            time_info,
            prev_wfi_time,
            retired_reg,
            args.permissive,
        )
        if perf_metrics[0]['start'] is None:
            perf_metrics[0]['start'] = time_info[1]
        record_timeline_samples(time_info[1])
        if 'trace' in line or 'mcycle' in line:
            record_section_end_sample(time_info[1])
            perf_metrics[-1]['end'] = time_info[1]
            perf_metrics.append(defaultdict(int))
            perf_metrics[-1]['start'] = None
            next_timeline_cycle = None
            timeline_sample_index = 0

    args.infile.close()
    perf_metrics[-1]['end'] = time_info[1]
    if perf_metrics[-1]['start'] is None:
        perf_metrics = perf_metrics[:-1]
    if not perf_metrics or perf_metrics[0]['start'] is None:
        sys.stderr.write('WARNING: Empty trace file ({}).\n'.format(args.infile.name))
        return 0

    record_section_end_sample(perf_metrics[-1]['end'])

    csv_file = args.csv
    if os.path.split(csv_file)[0] == '':
        csv_file = os.path.join(path, csv_file)
    perf_metrics_to_csv(timeline_metrics, csv_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
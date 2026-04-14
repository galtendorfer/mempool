#!/usr/bin/env python3
"""Parse a single MemPool .trace file and emit cycle-by-cycle stall rows to CSV.

Internal script — normally called by _gen_stall_timeseries_batch.py, not
invoked directly.  Each row records the stall state of one core at one cycle,
together with the core's group, tile, and in-tile index.

Positional argument:
    infile              Path to a .trace file (raw or annotated).
                        Reads from stdin if omitted.

Required flag:
    --csv PATH          Output CSV file.  Rows are appended if the file
                        already exists (allows the folder wrapper to
                        concatenate multiple traces into one CSV).

Optional flags:
    --section N         Emit only rows belonging to section N (repeatable).
    --benchmark-only    Shortcut for --section 1 (the benchmark bracket).
    -p, --permissive    Ignore malformed non-trace lines instead of erroring.

Environment variables (set by Makefile via trace_env):
    NUM_CORES           Total core count          [default: 256]
    NUM_GROUPS          Number of groups           [default: 1]
    NUM_CORES_PER_TILE  Cores per tile             [default: 4]
  These determine how each core ID is mapped to group / tile / in-tile index.
  The defaults are safe only for single-group testing; real runs must set them.
"""

import argparse
import csv
import os
import re
import sys


TRACE_IN_REGEX = r'(\d+)\s+(\d+)\s+(0x[0-9A-Fa-fz]+)\s+([^#;]*)(\s*#;\s*(.*))?'
ANNOTATED_TRACE_REGEX = r'^\s*(\d+)\s+(\d+)(?:\s+(0x[0-9A-Fa-f]+)\s+(.*?))?\s*$'


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


def read_annotations(dict_str: str) -> dict:
    annot = {key: int(val, 16) for key, val in
             re.findall(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)", dict_str)}
    annot.update({key: val for key, val in
                  re.findall(r"'([^']+)'\s*:\s*(0x[0-9a-fA-FxX]+)",
                             re.sub(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)",
                                    '', dict_str))})
    return annot


def detect_marker(line: str) -> bool:
    return 'trace' in line or 'mcycle' in line


def parse_stall_annotation(annotation: str, prev_wfi_time: int, cycle: int) -> dict:
    stall_info = {
        'stall_tot': 0,
        'stall_ins': 0,
        'stall_raw': 0,
        'stall_raw_lsu': 0,
        'stall_raw_acc': 0,
        'stall_lsu': 0,
        'stall_acc': 0,
        'stall_wfi': 0,
    }
    stall_match = re.search(r'// stall\s+(\d+)\s+cycles', annotation)
    if not stall_match:
        return stall_info
    stall_info['stall_tot'] = int(stall_match.group(1))
    specific_ins = re.search(r'\((\d+)\s+ins\)', annotation)
    specific_raw = re.search(r'\((\d+)\s+raw', annotation)
    specific_lsu = re.search(r'\((\d+)\s+lsu\)', annotation)
    specific_acc = re.search(r'\((\d+)\s+acc\)', annotation)
    if specific_ins:
        stall_info['stall_ins'] = int(specific_ins.group(1))
    if specific_raw:
        raw_count = int(specific_raw.group(1))
        stall_info['stall_raw'] = raw_count
        if 'lsu:' in annotation:
            stall_info['stall_raw_lsu'] = raw_count
        if 'acc:' in annotation:
            stall_info['stall_raw_acc'] = raw_count
    if specific_lsu:
        stall_info['stall_lsu'] = int(specific_lsu.group(1))
    if specific_acc:
        stall_info['stall_acc'] = int(specific_acc.group(1))
    if prev_wfi_time != 0:
        stall_info['stall_wfi'] = cycle - prev_wfi_time - 1
    return stall_info


def classify_stall(stall_info: dict) -> tuple[str, int]:
    categories = []
    if stall_info['stall_ins'] > 0:
        categories.append('ins')
    if stall_info['stall_raw'] > 0:
        categories.append('raw')
    if stall_info['stall_lsu'] > 0:
        categories.append('lsu')
    if stall_info['stall_acc'] > 0:
        categories.append('acc')
    if stall_info['stall_wfi'] > 0:
        categories.append('wfi')
    if not categories:
        return 'none', 1
    return '+'.join(categories), int(len(categories) <= 1)


def add_common_metadata(row: dict, core_id: int, hierarchy: dict, section: int):
    row['core'] = core_id
    row['group'] = hierarchy['group']
    row['tile'] = hierarchy['tile']
    row['tile_in_group'] = hierarchy['tile_in_group']
    row['core_in_tile'] = hierarchy['core_in_tile']
    row['core_in_group'] = hierarchy['core_in_group']
    row['section'] = section


def parse_trace_event(line: str, prev_wfi_time: int, permissive: bool) -> tuple[dict | None, int]:
    line_stripped = line.strip('\n')
    match = re.search(TRACE_IN_REGEX, line_stripped)
    if match is not None and match.group(6) and re.search(r"'[^']+'\s*:", match.group(6)):
        _, cycle_str, pc_str, insn, _, extras_str = match.groups()
        cycle = int(cycle_str)
        if not extras_str:
            return {
                'cycle': cycle,
                'pc': pc_str,
                'insn': insn.strip(),
                'has_issue': bool(insn.strip()),
                'stall_info': {
                    'stall_tot': 0,
                    'stall_ins': 0,
                    'stall_raw': 0,
                    'stall_raw_lsu': 0,
                    'stall_raw_acc': 0,
                    'stall_lsu': 0,
                    'stall_acc': 0,
                    'stall_wfi': 0,
                },
            }, 0

        extras = read_annotations(extras_str)
        for key in ('stall', 'stall_tot', 'stall_ins', 'stall_raw', 'stall_lsu',
                    'stall_acc'):
            extras.setdefault(key, 0)
        stall_info = {
            'stall_tot': int(extras['stall_tot']),
            'stall_ins': int(extras['stall_ins']),
            'stall_raw': int(extras['stall_raw']),
            'stall_raw_lsu': int(extras['stall_raw']) if int(extras['stall_raw']) > 0 and 'lsu_rd' in extras else 0,
            'stall_raw_acc': 0,
            'stall_lsu': int(extras['stall_lsu']),
            'stall_acc': int(extras['stall_acc']),
            'stall_wfi': cycle - prev_wfi_time - 1 if prev_wfi_time != 0 and int(extras['stall_tot']) > 0 else 0,
        }
        next_prev_wfi_time = cycle if insn.strip() == 'wfi' and not extras['stall'] else 0
        return {
            'cycle': cycle,
            'pc': pc_str,
            'insn': insn.strip(),
            'has_issue': not extras['stall'] and bool(insn.strip()),
            'stall_info': stall_info,
        }, next_prev_wfi_time

    before, annotation = (line_stripped.split('#;', 1) + [''])[:2]
    annotated_match = re.match(ANNOTATED_TRACE_REGEX, before)
    if annotated_match is None:
        if permissive and not line_stripped:
            return None, prev_wfi_time
        raise ValueError('Not a valid trace line:\n{}'.format(line))
    cycle = int(annotated_match.group(2))
    pc = annotated_match.group(3) or ''
    insn = (annotated_match.group(4) or '').strip()
    annotation = annotation.strip()
    stall_info = parse_stall_annotation(annotation, prev_wfi_time, cycle)
    next_prev_wfi_time = cycle if insn == 'wfi' else 0
    return {
        'cycle': cycle,
        'pc': pc,
        'insn': insn,
        'has_issue': bool(pc and insn),
        'stall_info': stall_info,
    }, next_prev_wfi_time


def write_rows(rows: list[dict], filename: str):
    if not rows:
        return
    known_keys = [
        'core', 'group', 'tile', 'tile_in_group', 'core_in_tile',
        'core_in_group', 'section', 'cycle', 'state', 'pc', 'insn',
        'stall_interval_id', 'stall_interval_start', 'stall_interval_end',
        'stall_interval_cycles', 'stall_interval_offset', 'stall_kind',
        'stall_kind_exact', 'stall_tot', 'stall_ins', 'stall_raw',
        'stall_raw_lsu', 'stall_raw_acc', 'stall_lsu', 'stall_acc',
        'stall_wfi',
    ]
    write_header = not os.path.exists(filename)
    with open(filename, 'a+') as out:
        dict_writer = csv.DictWriter(out, known_keys)
        if write_header:
            dict_writer.writeheader()
        dict_writer.writerows(rows)
    print('\nWrote stall time-series to %s\n' % filename)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'infile',
        metavar='infile.trace',
        nargs='?',
        type=argparse.FileType('r'),
        default=sys.stdin,
        help='A raw or annotated MemPool trace')
    parser.add_argument(
        '--csv',
        required=True,
        help='CSV file that will receive cycle-granular stall state rows')
    parser.add_argument(
        '--section',
        type=int,
        action='append',
        help='Emit rows only for the specified section; may be repeated')
    parser.add_argument(
        '--benchmark-only',
        action='store_true',
        help='Shortcut for --section 1 for apps with a single benchmark bracket')
    parser.add_argument(
        '-p',
        '--permissive',
        action='store_true',
        help='Ignore malformed non-trace lines when possible')
    args = parser.parse_args()

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
    hierarchy = get_core_hierarchy(core_id)

    section = 0
    prev_wfi_time = 0
    rows = []
    stall_interval_id = 0

    for line in iter(args.infile.readline, b''):
        if not line:
            break
        stripped_line = line.strip()
        if not stripped_line:
            continue
        if not re.match(r'^\d', stripped_line):
            continue

        event, prev_wfi_time = parse_trace_event(line, prev_wfi_time, args.permissive)
        if event is None:
            continue

        if not selected_sections or section in selected_sections:
            stall_info = event['stall_info']
            if stall_info['stall_tot'] > 0:
                interval_start = event['cycle'] - stall_info['stall_tot']
                interval_end = event['cycle'] - 1
                stall_kind, stall_kind_exact = classify_stall(stall_info)
                for offset, cycle in enumerate(range(interval_start, event['cycle'])):
                    row = {
                        'cycle': cycle,
                        'state': 'stall',
                        'pc': '',
                        'insn': '',
                        'stall_interval_id': stall_interval_id,
                        'stall_interval_start': interval_start,
                        'stall_interval_end': interval_end,
                        'stall_interval_cycles': stall_info['stall_tot'],
                        'stall_interval_offset': offset,
                        'stall_kind': stall_kind,
                        'stall_kind_exact': stall_kind_exact,
                    }
                    row.update(stall_info)
                    add_common_metadata(row, core_id, hierarchy, section)
                    rows.append(row)
                stall_interval_id += 1

            if event['has_issue']:
                row = {
                    'cycle': event['cycle'],
                    'state': 'issue',
                    'pc': event['pc'],
                    'insn': event['insn'],
                    'stall_interval_id': '',
                    'stall_interval_start': '',
                    'stall_interval_end': '',
                    'stall_interval_cycles': 0,
                    'stall_interval_offset': '',
                    'stall_kind': 'none',
                    'stall_kind_exact': 1,
                    'stall_tot': 0,
                    'stall_ins': 0,
                    'stall_raw': 0,
                    'stall_raw_lsu': 0,
                    'stall_raw_acc': 0,
                    'stall_lsu': 0,
                    'stall_acc': 0,
                    'stall_wfi': 0,
                }
                add_common_metadata(row, core_id, hierarchy, section)
                rows.append(row)

        if detect_marker(line):
            section += 1

    args.infile.close()
    csv_file = args.csv
    if os.path.split(csv_file)[0] == '':
        csv_file = os.path.join(path, csv_file)
    write_rows(rows, csv_file)
    return 0


if __name__ == '__main__':
    sys.exit(main())
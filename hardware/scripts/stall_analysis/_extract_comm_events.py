#!/usr/bin/env python3
"""Extract per-event communication rows from a single MemPool trace.

This internal worker parses one trace file and emits a CSV containing
load/store issue events plus load return events with inferred destination
metadata derived from the MemPool address map.

Normal users should prefer one of these public entry points instead:
    - `extract_comm_events.py`         result_dir-based wrapper
    - `extract_comm_events_batch.py`   direct folder-based batch mode

Supported input formats:
    - raw decode-style trace lines with `#; { ... }` annotation dicts
    - annotated `.trace` lines produced by `gen_trace.py`
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict, deque


_TRACE_IN_REGEX = re.compile(
    r"(\d+)\s+(\d+)\s+(0x[0-9A-Fa-fz]+)\s+([^#;]*)(\s*#;\s*(.*))?"
)
_ANNOTATED_TRACE_REGEX = re.compile(
    r"^\s*(\d+)\s+(\d+)(?:\s+(0x[0-9A-Fa-f]+)\s+(.*?))?\s*$"
)
_RAW_ANNOTATION_REGEX = re.compile(r"'[^']+'\s*:")
_LOAD_ISSUE_REGEX = re.compile(
    r"(?P<rd>[A-Za-z0-9]+)\s+<~~\s+"
    r"(?P<size>Byte|Half|Word|Doub)\[(?P<address>[^\]]+)\]"
)
_STORE_ISSUE_REGEX = re.compile(
    r"~~>\s+(?P<size>Byte|Half|Word|Doub)\[(?P<address>[^\]]+)\]"
)
_LOAD_RETURN_REGEX = re.compile(r"\(lsu\)\s+(?P<rd>[A-Za-z0-9]+)\s+<--")

_REG_ABI_NAMES_I = (
    "zero", "ra", "sp", "gp", "tp",
    "t0", "t1", "t2",
    "s0", "s1",
    *(f"a{i}" for i in range(8)),
    *(f"s{i}" for i in range(2, 12)),
    *(f"t{i}" for i in range(3, 7)),
)
_REG_NAME_TO_INDEX = {name: idx for idx, name in enumerate(_REG_ABI_NAMES_I)}
_LS_SIZE_LABELS = {0: "Byte", 1: "Half", 2: "Word", 3: "Doub"}
_LS_SIZE_BYTES = {0: 1, 1: 2, 2: 4, 3: 8}
_SIZE_NAME_TO_BYTES = {name: _LS_SIZE_BYTES[idx] for idx, name in _LS_SIZE_LABELS.items()}
_MEM_REGION_LABELS = {0: "other", 1: "sequential", 2: "interleaved"}

_KNOWN_KEYS = [
    "section",
    "cycle",
    "core",
    "group",
    "tile",
    "tile_in_group",
    "core_in_tile",
    "core_in_group",
    "event_type",
    "request_id",
    "pc",
    "insn",
    "origin_pc",
    "origin_insn",
    "rd",
    "size_bytes",
    "address",
    "region",
    "dest_tile",
    "dest_group",
    "is_local",
    "is_same_group",
    "issue_cycle",
    "return_cycle",
    "latency",
]


def _getenv_int(*names: str, default: int) -> int:
    for name in names:
        value = os.environ.get(name)
        if value is not None:
            return int(value)
    return default


_NUM_CORES = _getenv_int("NUM_CORES", "num_cores", default=256)
_NUM_GROUPS = _getenv_int("NUM_GROUPS", "num_groups", default=1)
_NUM_CORES_PER_TILE = _getenv_int("NUM_CORES_PER_TILE", "num_cores_per_tile", default=4)
_BANKING_FACTOR = _getenv_int("BANKING_FACTOR", "banking_factor", default=4)
_L1_BANK_SIZE = _getenv_int("L1_BANK_SIZE", "l1_bank_size", default=1024)
_NUM_BANKS_PER_TILE = _NUM_CORES_PER_TILE * _BANKING_FACTOR
_SEQ_MEM_SIZE = _NUM_CORES_PER_TILE * _getenv_int("SEQ_MEM_SIZE", "seq_mem_size", default=512)
_NUM_TILES = _NUM_CORES // _NUM_CORES_PER_TILE if _NUM_CORES_PER_TILE else 0
_INTERLEAVE_STRIDE = 4 * _NUM_BANKS_PER_TILE
_TCDM_SIZE = _NUM_BANKS_PER_TILE * _L1_BANK_SIZE * _NUM_TILES


def _read_annotations(dict_str: str) -> dict:
    annot = {key: int(val, 16) for key, val in re.findall(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)", dict_str)}
    annot.update({
        key: val for key, val in re.findall(
            r"'([^']+)'\s*:\s*(0x[0-9a-fA-FxX]+)",
            re.sub(r"'([^']+)'\s*:\s*(0x[0-9a-fA-F]+)", "", dict_str),
        )
    })
    return annot


def _detect_marker(line: str) -> bool:
    return "trace" in line or "mcycle" in line


def _parse_int_literal(raw: str) -> int:
    return int(raw.strip().rstrip(","), 0)


def _format_address(address: int | None) -> str:
    if address is None:
        return ""
    return hex(int(address))


def _reg_name(index: int | None) -> str:
    if index is None or index < 0 or index >= len(_REG_ABI_NAMES_I):
        return ""
    return _REG_ABI_NAMES_I[index]


def _reg_index(name: str | None) -> int | None:
    if not name:
        return None
    return _REG_NAME_TO_INDEX.get(name)


def _get_core_hierarchy(core_id: int) -> dict:
    if core_id < 0:
        return {
            "group": -1,
            "tile": -1,
            "tile_in_group": -1,
            "core_in_tile": -1,
            "core_in_group": -1,
        }
    tiles_per_group = _NUM_TILES // _NUM_GROUPS if _NUM_GROUPS else _NUM_TILES
    cores_per_group = _NUM_CORES // _NUM_GROUPS if _NUM_GROUPS else _NUM_CORES
    tile_id = core_id // _NUM_CORES_PER_TILE if _NUM_CORES_PER_TILE else -1
    return {
        "group": tile_id // tiles_per_group if tiles_per_group else 0,
        "tile": tile_id,
        "tile_in_group": tile_id % tiles_per_group if tiles_per_group else 0,
        "core_in_tile": core_id % _NUM_CORES_PER_TILE if _NUM_CORES_PER_TILE else 0,
        "core_in_group": core_id % cores_per_group if cores_per_group else 0,
    }


def _addr_to_meta(address: int) -> tuple[str, int, int]:
    region_code = 0
    dest_tile = -1
    if 0 <= address < _SEQ_MEM_SIZE * _NUM_TILES:
        region_code = 1
        dest_tile = address // _SEQ_MEM_SIZE if _SEQ_MEM_SIZE else -1
    elif 0 <= address < _TCDM_SIZE:
        region_code = 2
        dest_tile = (address // _INTERLEAVE_STRIDE) % _NUM_TILES if _NUM_TILES else -1
    if dest_tile < 0:
        return _MEM_REGION_LABELS[region_code], -1, -1
    tiles_per_group = _NUM_TILES // _NUM_GROUPS if _NUM_GROUPS else _NUM_TILES
    dest_group = dest_tile // tiles_per_group if tiles_per_group else 0
    return _MEM_REGION_LABELS[region_code], int(dest_tile), int(dest_group)


def _issue_event_type_dict(extras: dict) -> list[dict]:
    events = []
    if not extras.get("stall") and int(extras.get("is_load", 0)):
        ls_size = int(extras.get("ls_size", 0))
        events.append({
            "event_type": "load_issue",
            "rd_index": int(extras.get("rd", 0)),
            "size_bytes": _LS_SIZE_BYTES.get(ls_size, ""),
            "address": int(extras.get("alu_result", 0)),
        })
    elif not extras.get("stall") and int(extras.get("is_store", 0)):
        ls_size = int(extras.get("ls_size", 0))
        events.append({
            "event_type": "store_issue",
            "rd_index": None,
            "size_bytes": _LS_SIZE_BYTES.get(ls_size, ""),
            "address": int(extras.get("alu_result", 0)),
        })
    if int(extras.get("retire_load", 0)):
        events.append({
            "event_type": "load_return",
            "rd_index": int(extras.get("lsu_rd", 0)),
            "size_bytes": "",
            "address": None,
        })
    return events


def _issue_event_type_annotated(annotation: str) -> list[dict]:
    events = []
    for match in _LOAD_ISSUE_REGEX.finditer(annotation):
        events.append({
            "event_type": "load_issue",
            "rd_index": _reg_index(match.group("rd")),
            "size_bytes": _SIZE_NAME_TO_BYTES.get(match.group("size"), ""),
            "address": _parse_int_literal(match.group("address")),
        })
    for match in _STORE_ISSUE_REGEX.finditer(annotation):
        events.append({
            "event_type": "store_issue",
            "rd_index": None,
            "size_bytes": _SIZE_NAME_TO_BYTES.get(match.group("size"), ""),
            "address": _parse_int_literal(match.group("address")),
        })
    for match in _LOAD_RETURN_REGEX.finditer(annotation):
        events.append({
            "event_type": "load_return",
            "rd_index": _reg_index(match.group("rd")),
            "size_bytes": "",
            "address": None,
        })
    return events


def _parse_line_events(line: str, permissive: bool) -> dict | None:
    line = line.rstrip("\n")
    if not line.strip():
        return None
    match = _TRACE_IN_REGEX.search(line)
    if match is not None:
        _, cycle_str, pc_str, insn, _, extras_str = match.groups()
        cycle = int(cycle_str)
        insn = insn.strip()
        extras_str = extras_str or ""
        if extras_str and _RAW_ANNOTATION_REGEX.search(extras_str):
            extras = _read_annotations(extras_str)
            for key in ("stall", "is_load", "is_store", "retire_load", "rd", "lsu_rd", "alu_result", "ls_size"):
                extras.setdefault(key, 0)
            return {
                "cycle": cycle,
                "pc": pc_str,
                "insn": insn,
                "events": _issue_event_type_dict(extras),
                "marker": _detect_marker(line),
            }

    before, annotation = (line.split("#;", 1) + [""])[:2]
    annotated_match = _ANNOTATED_TRACE_REGEX.match(before)
    if annotated_match is None:
        if permissive:
            return None
        raise ValueError(f"Not a valid trace line:\n{line}")
    cycle = int(annotated_match.group(2))
    pc = annotated_match.group(3) or ""
    insn = (annotated_match.group(4) or "").strip()
    return {
        "cycle": cycle,
        "pc": pc,
        "insn": insn,
        "events": _issue_event_type_annotated(annotation.strip()),
        "marker": _detect_marker(line),
    }


def _make_row(section: int, cycle: int, core_id: int, hierarchy: dict, *, event_type: str,
              request_id: int | str, pc: str, insn: str, origin_pc: str, origin_insn: str,
              rd_index: int | None, size_bytes, address: int | None, issue_cycle, return_cycle, latency):
    region, dest_tile, dest_group = _addr_to_meta(address) if address is not None else ("", -1, -1)
    is_local = ""
    is_same_group = ""
    if dest_tile >= 0:
        is_local = int(dest_tile == hierarchy["tile"])
        is_same_group = int(dest_group == hierarchy["group"])
    return {
        "section": section,
        "cycle": cycle,
        "core": core_id,
        "group": hierarchy["group"],
        "tile": hierarchy["tile"],
        "tile_in_group": hierarchy["tile_in_group"],
        "core_in_tile": hierarchy["core_in_tile"],
        "core_in_group": hierarchy["core_in_group"],
        "event_type": event_type,
        "request_id": request_id,
        "pc": pc,
        "insn": insn,
        "origin_pc": origin_pc,
        "origin_insn": origin_insn,
        "rd": _reg_name(rd_index),
        "size_bytes": size_bytes,
        "address": _format_address(address),
        "region": region,
        "dest_tile": dest_tile if dest_tile >= 0 else "",
        "dest_group": dest_group if dest_group >= 0 else "",
        "is_local": is_local,
        "is_same_group": is_same_group,
        "issue_cycle": issue_cycle,
        "return_cycle": return_cycle,
        "latency": latency,
    }


def _iter_comm_rows(infile, core_id: int, permissive: bool):
    hierarchy = _get_core_hierarchy(core_id)
    pending_loads = defaultdict(deque)
    next_request_id = 0
    section = 0

    for line in iter(infile.readline, b""):
        if not line:
            break
        parsed = _parse_line_events(line, permissive)
        if parsed is None:
            continue

        for event in parsed["events"]:
            if event["event_type"] == "load_issue":
                request_id = next_request_id
                next_request_id += 1
                pending = {
                    "request_id": request_id,
                    "issue_cycle": parsed["cycle"],
                    "pc": parsed["pc"],
                    "insn": parsed["insn"],
                    "rd_index": event["rd_index"],
                    "size_bytes": event["size_bytes"],
                    "address": event["address"],
                }
                if event["rd_index"] is not None:
                    pending_loads[event["rd_index"]].appendleft(pending)
                yield _make_row(
                    section,
                    parsed["cycle"],
                    core_id,
                    hierarchy,
                    event_type="load_issue",
                    request_id=request_id,
                    pc=parsed["pc"],
                    insn=parsed["insn"],
                    origin_pc=parsed["pc"],
                    origin_insn=parsed["insn"],
                    rd_index=event["rd_index"],
                    size_bytes=event["size_bytes"],
                    address=event["address"],
                    issue_cycle=parsed["cycle"],
                    return_cycle="",
                    latency="",
                )

            elif event["event_type"] == "store_issue":
                yield _make_row(
                    section,
                    parsed["cycle"],
                    core_id,
                    hierarchy,
                    event_type="store_issue",
                    request_id="",
                    pc=parsed["pc"],
                    insn=parsed["insn"],
                    origin_pc=parsed["pc"],
                    origin_insn=parsed["insn"],
                    rd_index=None,
                    size_bytes=event["size_bytes"],
                    address=event["address"],
                    issue_cycle=parsed["cycle"],
                    return_cycle="",
                    latency="",
                )

            elif event["event_type"] == "load_return":
                try:
                    pending = pending_loads[event["rd_index"]].pop()
                except IndexError:
                    msg = (
                        f"WARNING: In cycle {parsed['cycle']}, LSU return to {_reg_name(event['rd_index'])} "
                        f"had no matching in-flight load in {infile.name}."
                    )
                    if permissive:
                        print(msg, file=sys.stderr)
                        continue
                    raise SystemExit(msg)
                yield _make_row(
                    section,
                    parsed["cycle"],
                    core_id,
                    hierarchy,
                    event_type="load_return",
                    request_id=pending["request_id"],
                    pc=parsed["pc"],
                    insn=parsed["insn"],
                    origin_pc=pending["pc"],
                    origin_insn=pending["insn"],
                    rd_index=event["rd_index"],
                    size_bytes=pending["size_bytes"],
                    address=pending["address"],
                    issue_cycle=pending["issue_cycle"],
                    return_cycle=parsed["cycle"],
                    latency=parsed["cycle"] - pending["issue_cycle"],
                )

        if parsed["marker"]:
            section += 1


def _write_rows(rows: list[dict], filename: str):
    write_header = not os.path.exists(filename)
    with open(filename, "a+", newline="") as out:
        writer = csv.DictWriter(out, _KNOWN_KEYS)
        if write_header:
            writer.writeheader()
        if rows:
            writer.writerows(rows)


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Extract communication events from a single MemPool trace.")
    parser.add_argument(
        "infile",
        metavar="infile.trace",
        nargs="?",
        type=argparse.FileType("r"),
        default=sys.stdin,
        help="A raw or annotated MemPool trace",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="CSV file that will receive communication event rows",
    )
    parser.add_argument(
        "--section",
        type=int,
        action="append",
        help="Emit rows only for the specified section; may be repeated",
    )
    parser.add_argument(
        "--benchmark-only",
        action="store_true",
        help="Shortcut for --section 1 (the benchmark bracket)",
    )
    parser.add_argument(
        "-p",
        "--permissive",
        action="store_true",
        help="Ignore malformed trace lines and unmatched returns when possible",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    selected_sections = set(args.section or [])
    if args.benchmark_only:
        selected_sections.add(1)

    path, filename = os.path.split(args.infile.name)
    core_id_hex = re.search(r"(0x[0-9a-fA-F]+)", filename)
    core_id_dec = re.search(r"([\d]+)", filename)
    if core_id_hex:
        core_id = int(core_id_hex.group(1), 16)
    elif core_id_dec:
        core_id = int(core_id_dec.group(1))
    else:
        core_id = -1

    rows = []
    for row in _iter_comm_rows(args.infile, core_id, args.permissive):
        if selected_sections and row["section"] not in selected_sections:
            continue
        rows.append(row)

    args.infile.close()
    _write_rows(rows, args.csv)
    if rows:
        print(f"Wrote {len(rows)} communication events to {args.csv}")
    else:
        print(f"No communication events found; wrote header-only CSV to {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
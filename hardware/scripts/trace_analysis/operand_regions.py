#!/usr/bin/env python3
"""Operand-region helpers for monitor-derived matmul traffic classification."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperandRegion:
    name: str
    start: int
    end: int
    source: str


@dataclass(frozen=True)
class OperandRegionMap:
    regions: tuple[OperandRegion, ...]
    source: str
    address_field: str
    sidecar_path: Path | None = None

    def classify(self, addr: str | None) -> str:
        raw = normalized_hex_int(addr)
        if raw is None:
            return 'other'
        for region in self.regions:
            if region.start <= raw <= region.end:
                return region.name
        return 'other'


LEGACY_ROUTE_REGIONS = (
    OperandRegion('A', 0x8000, 0x87ff, 'legacy route-address fallback'),
    OperandRegion('B', 0x8800, 0x8fff, 'legacy route-address fallback'),
)

ADDRESS_FIELDS = ('source_addr', 'source_addr_or_addr', 'addr')
SIDECAR_FILENAME = 'operand_regions.json'


REGION_LINE_RE = re.compile(
    r'MATMUL_I32_REGION\s+'
    r'name=(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+'
    r'base=(?P<base>0x[0-9A-Fa-f]+|\d+)\s+'
    r'size=(?P<size>0x[0-9A-Fa-f]+|\d+)\s+'
    r'end=(?P<end>0x[0-9A-Fa-f]+|\d+)'
)


def normalized_hex_int(value: str | None) -> int | None:
    if value in (None, ''):
        return None
    text = value.strip().lower()
    if text.startswith('0x'):
        text = text[2:]
    if not text:
        return None
    try:
        return int(text, 16)
    except ValueError:
        return None


def parse_number(value: str) -> int:
    text = value.strip().lower()
    if text.startswith('0x'):
        return int(text, 16)
    return int(text, 10)


def parse_json_number(value: object, field: str, path: Path) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return parse_number(value)
        except ValueError as error:
            raise SystemExit(f'{path}: {field} is not a valid number: {value!r}') from error
    raise SystemExit(f'{path}: {field} must be an integer or string number')


def parse_range(value: str, name: str, source: str) -> OperandRegion:
    parts = [part.strip() for part in value.replace('..', ':').split(':') if part.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f'{name} range must be START:END, got {value!r}')
    start = parse_number(parts[0])
    end = parse_number(parts[1])
    if end < start:
        raise argparse.ArgumentTypeError(f'{name} range end is before start: {value!r}')
    return OperandRegion(name, start, end, source)


def parse_operand_region(value: str) -> OperandRegion:
    parts = [part.strip() for part in value.split(':') if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError('operand region must be NAME:START:END')
    return parse_range(f'{parts[1]}:{parts[2]}', parts[0], 'cli')


def add_operand_region_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument('--a-range', help='operand A source-address range START:END; overrides transcript metadata')
    parser.add_argument('--b-range', help='operand B source-address range START:END; overrides transcript metadata')
    parser.add_argument('--c-range', help='operand C source-address range START:END; overrides transcript metadata')
    parser.add_argument(
        '--operand-regions-json',
        type=Path,
        help='explicit operand region sidecar; defaults to <result_dir>/analysis/operand_regions.json',
    )
    parser.add_argument(
        '--operand-address-field',
        choices=ADDRESS_FIELDS,
        help='address field used for CLI/transcript operand ranges; defaults to source_addr',
    )
    parser.add_argument(
        '--operand-region',
        action='append',
        type=parse_operand_region,
        default=[],
        help='additional/override operand source-address region NAME:START:END; may be repeated',
    )
    parser.add_argument(
        '--no-legacy-operand-fallback',
        action='store_true',
        help='deprecated compatibility flag; legacy route fallback is disabled by default',
    )
    parser.add_argument(
        '--allow-legacy-route-operands',
        action='store_true',
        help='allow old route-address A/B fallback when no exact source-address metadata is available',
    )
    parser.add_argument(
        '--require-exact-operands',
        action='store_true',
        help='fail unless operand labels come from source_addr plus sidecar/transcript/CLI regions',
    )


def add_classified_operand_provenance_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--allow-legacy-route-operands',
        action='store_true',
        help='allow plotting classifier CSVs produced with legacy route-address operand labels',
    )


def result_dir_from_graph_dir(graph_dir: Path) -> Path | None:
    if graph_dir.name == 'path_graph' and graph_dir.parent.name == 'analysis':
        return graph_dir.parent.parent
    nested = graph_dir / 'analysis' / 'path_graph'
    if nested.is_dir():
        return graph_dir
    return None


def sidecar_candidates(graph_dir: Path, args: argparse.Namespace | None = None) -> list[Path]:
    explicit = getattr(args, 'operand_regions_json', None) if args is not None else None
    if explicit:
        return [Path(explicit)]

    candidates: list[Path] = []
    result_dir = result_dir_from_graph_dir(graph_dir)
    if result_dir is not None:
        candidates.append(result_dir / 'analysis' / SIDECAR_FILENAME)
        candidates.append(result_dir / SIDECAR_FILENAME)
    candidates.append(graph_dir / SIDECAR_FILENAME)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        normalized = path.resolve() if path.exists() else path
        if normalized not in seen:
            seen.add(normalized)
            unique.append(path)
    return unique


def _sidecar_region(name: str, raw: object, path: Path) -> OperandRegion:
    if not isinstance(raw, dict):
        raise SystemExit(f'{path}: region {name!r} must be an object')
    start_value = raw.get('start', raw.get('base'))
    if start_value is None:
        raise SystemExit(f'{path}: region {name!r} needs start or base')
    start = parse_json_number(start_value, f'{name}.start', path)
    end_value = raw.get('end')
    if end_value is None:
        size_value = raw.get('size')
        if size_value is None:
            raise SystemExit(f'{path}: region {name!r} needs end or size')
        size = parse_json_number(size_value, f'{name}.size', path)
        if size <= 0:
            raise SystemExit(f'{path}: region {name!r} size must be positive')
        end = start + size - 1
    else:
        end = parse_json_number(end_value, f'{name}.end', path)
    if end < start:
        raise SystemExit(f'{path}: region {name!r} end is before start')
    return OperandRegion(name, start, end, str(path))


def sidecar_regions(path: Path) -> tuple[list[OperandRegion], str]:
    try:
        with path.open() as file:
            data = json.load(file)
    except json.JSONDecodeError as error:
        raise SystemExit(f'{path}: invalid JSON: {error}') from error

    if not isinstance(data, dict):
        raise SystemExit(f'{path}: top-level JSON value must be an object')
    address_field = str(data.get('address_field', 'source_addr'))
    if address_field not in ADDRESS_FIELDS:
        joined = ', '.join(ADDRESS_FIELDS)
        raise SystemExit(f'{path}: address_field must be one of {joined}')

    raw_regions = data.get('regions')
    if raw_regions is None:
        raise SystemExit(f'{path}: missing regions')

    regions: list[OperandRegion] = []
    if isinstance(raw_regions, dict):
        for name, raw in raw_regions.items():
            regions.append(_sidecar_region(str(name), raw, path))
    elif isinstance(raw_regions, list):
        for index, raw in enumerate(raw_regions):
            if not isinstance(raw, dict):
                raise SystemExit(f'{path}: regions[{index}] must be an object')
            name = raw.get('name')
            if not name:
                raise SystemExit(f'{path}: regions[{index}] needs name')
            regions.append(_sidecar_region(str(name), raw, path))
    else:
        raise SystemExit(f'{path}: regions must be an object or list')

    if not regions:
        raise SystemExit(f'{path}: no operand regions configured')
    return regions, address_field


def transcript_regions(result_dir: Path | None) -> list[OperandRegion]:
    if result_dir is None:
        return []
    transcript = result_dir / 'transcript'
    if not transcript.is_file():
        return []
    regions: list[OperandRegion] = []
    with transcript.open(errors='replace') as file:
        for line in file:
            match = REGION_LINE_RE.search(line)
            if match is None:
                continue
            name = match.group('name')
            base = parse_number(match.group('base'))
            end = parse_number(match.group('end'))
            size = parse_number(match.group('size'))
            if size > 0 and end != base + size - 1:
                end = base + size - 1
            regions.append(OperandRegion(name, base, end, str(transcript)))
    return regions


def _upsert(regions_by_name: dict[str, OperandRegion], region: OperandRegion) -> None:
    regions_by_name[region.name] = region


def _has_cli_regions(args: argparse.Namespace | None) -> bool:
    if args is None:
        return False
    return any(getattr(args, attr, None) for attr in ('a_range', 'b_range', 'c_range')) or bool(
        getattr(args, 'operand_region', []) or []
    )


def load_operand_regions(graph_dir: Path, args: argparse.Namespace | None = None) -> OperandRegionMap:
    regions_by_name: dict[str, OperandRegion] = {}
    sources: list[str] = []
    address_field = getattr(args, 'operand_address_field', None) or 'source_addr'
    sidecar_path: Path | None = None
    explicit_sidecar = getattr(args, 'operand_regions_json', None) if args is not None else None
    if explicit_sidecar and not Path(explicit_sidecar).is_file():
        raise SystemExit(f'Operand region sidecar not found: {explicit_sidecar}')

    for candidate in sidecar_candidates(graph_dir, args):
        if candidate.is_file():
            sidecar, sidecar_address_field = sidecar_regions(candidate)
            for region in sidecar:
                _upsert(regions_by_name, region)
            sources.append('sidecar')
            address_field = sidecar_address_field
            sidecar_path = candidate
            break

    if not regions_by_name:
        transcript = transcript_regions(result_dir_from_graph_dir(graph_dir))
        if transcript:
            for region in transcript:
                _upsert(regions_by_name, region)
            sources.append('transcript')

    if args is not None:
        cli_regions = _has_cli_regions(args)
        if cli_regions and getattr(args, 'operand_address_field', None):
            address_field = args.operand_address_field
        for name, attr in (('A', 'a_range'), ('B', 'b_range'), ('C', 'c_range')):
            value = getattr(args, attr, None)
            if value:
                _upsert(regions_by_name, parse_range(value, name, 'cli'))
                sources.append(f'cli-{name}')
        for region in getattr(args, 'operand_region', []) or []:
            _upsert(regions_by_name, region)
            sources.append(f'cli-{region.name}')

    allow_legacy = bool(getattr(args, 'allow_legacy_route_operands', False))
    if getattr(args, 'no_legacy_operand_fallback', False):
        allow_legacy = False
    if not regions_by_name and allow_legacy:
        for region in LEGACY_ROUTE_REGIONS:
            _upsert(regions_by_name, region)
        sources.append('legacy-route')
        address_field = 'addr'

    regions = tuple(sorted(regions_by_name.values(), key=lambda region: (region.start, region.end, region.name)))
    source = '+'.join(dict.fromkeys(sources)) if sources else 'none'
    operand_regions = OperandRegionMap(regions, source, address_field, sidecar_path)
    if getattr(args, 'require_exact_operands', False):
        require_exact_operand_regions(operand_regions)
    return operand_regions


def operand_regions_are_exact(regions: OperandRegionMap) -> bool:
    source_parts = set(part for part in regions.source.split('+') if part)
    return bool(regions.regions) and regions.address_field == 'source_addr' and not (
        'legacy-route' in source_parts or 'none' in source_parts
    )


def require_exact_operand_regions(regions: OperandRegionMap, context: str = 'operand classification') -> None:
    if operand_regions_are_exact(regions):
        return
    raise SystemExit(
        f'{context} requires exact operand metadata. Provide {SIDECAR_FILENAME} '
        'or --a-range/--b-range/--operand-region ranges using source_addr. '
        'Use --allow-legacy-route-operands only for legacy/debug route-address labels.'
    )


def classify_operand(addr: str | None, regions: OperandRegionMap) -> str:
    return regions.classify(addr)


def operand_address_from_row(row: dict[str, str], regions: OperandRegionMap | None = None) -> str:
    if regions is not None and regions.address_field == 'addr':
        return row.get('addr', '')
    if regions is not None and regions.address_field == 'source_addr':
        return row.get('source_addr', '')
    return row.get('source_addr') or row.get('addr', '')


def operand_classification_address_field(regions: OperandRegionMap) -> str:
    return regions.address_field


def format_operand_regions(regions: OperandRegionMap) -> str:
    if not regions.regions:
        return 'none'
    return ';'.join(f'{region.name}:0x{region.start:x}-0x{region.end:x}' for region in regions.regions)


def classifier_summary_path(classified_csv: Path) -> Path:
    suffixes = (
        '_source_target_matrix.csv',
        '_target_tile_in_group.csv',
        '_tile_cycles.csv',
        '_details.csv',
    )
    for suffix in suffixes:
        if classified_csv.name.endswith(suffix):
            prefix = classified_csv.name[: -len(suffix)]
            return classified_csv.with_name(f'{prefix}_summary.csv')
    return classified_csv.with_name('port0_source_target_summary.csv')


def read_classifier_summary(classified_csv: Path) -> dict[str, str]:
    summary_path = classifier_summary_path(classified_csv)
    if not summary_path.is_file():
        return {}
    with summary_path.open(newline='') as file:
        return {row.get('metric', ''): row.get('value', '') for row in csv.DictReader(file)}


def classifier_summary_has_exact_operands(summary: dict[str, str]) -> bool:
    source_parts = set(part for part in summary.get('operand_region_source', '').split('+') if part)
    address_field = summary.get('operand_address_field', '')
    return bool(source_parts) and address_field == 'source_addr' and not (
        'legacy-route' in source_parts or 'none' in source_parts
    )


def validate_classified_operand_provenance(
    classified_csvs: list[Path],
    allow_legacy_route_operands: bool = False,
) -> None:
    if allow_legacy_route_operands:
        return
    bad: list[str] = []
    for classified_csv in classified_csvs:
        summary = read_classifier_summary(classified_csv)
        if not summary:
            bad.append(f'{classified_csv}: missing {classifier_summary_path(classified_csv).name}')
            continue
        if not classifier_summary_has_exact_operands(summary):
            source = summary.get('operand_region_source', 'missing')
            address_field = summary.get('operand_address_field', 'missing')
            bad.append(f'{classified_csv}: source={source}, address_field={address_field}')
    if bad:
        joined = '\n  '.join(bad)
        raise SystemExit(
            'Refusing to plot operand-labeled classifier CSVs without exact operand provenance.\n'
            f'  {joined}\n'
            'Regenerate classification with operand_regions.json or explicit source-address ranges, '
            'or pass --allow-legacy-route-operands for legacy/debug plots.'
        )

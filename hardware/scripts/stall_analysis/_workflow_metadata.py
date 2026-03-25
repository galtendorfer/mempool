#!/usr/bin/env python3
"""Helpers for loading and validating workflow metadata."""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = REPO_ROOT / 'config'
TOPOLOGY_KEYS = (
    'NUM_CORES',
    'NUM_GROUPS',
    'NUM_CORES_PER_TILE',
    'SEQ_MEM_SIZE',
    'BANKING_FACTOR',
    'L1_BANK_SIZE',
    'NUM_SUB_GROUPS_PER_GROUP',
)
KNOWN_TOPOLOGIES = {
    'mempool': {
        'NUM_CORES': 256,
        'NUM_GROUPS': 4,
        'NUM_CORES_PER_TILE': 4,
        'SEQ_MEM_SIZE': 512,
        'BANKING_FACTOR': 4,
        'L1_BANK_SIZE': 1024,
        'NUM_SUB_GROUPS_PER_GROUP': 1,
    },
    'terapool': {
        'NUM_CORES': 1024,
        'NUM_GROUPS': 4,
        'NUM_CORES_PER_TILE': 8,
        'SEQ_MEM_SIZE': 512,
        'BANKING_FACTOR': 4,
        'L1_BANK_SIZE': 1024,
        'NUM_SUB_GROUPS_PER_GROUP': 4,
    },
    'minpool': {
        'NUM_CORES': 16,
        'NUM_GROUPS': 4,
        'NUM_CORES_PER_TILE': 4,
        'SEQ_MEM_SIZE': 512,
        'BANKING_FACTOR': 4,
        'L1_BANK_SIZE': 1024,
        'NUM_SUB_GROUPS_PER_GROUP': 1,
    },
    'systolic': {
        'NUM_CORES': 256,
        'NUM_GROUPS': 4,
        'NUM_CORES_PER_TILE': 4,
        'SEQ_MEM_SIZE': 1024,
        'BANKING_FACTOR': 4,
        'L1_BANK_SIZE': 1024,
        'NUM_SUB_GROUPS_PER_GROUP': 1,
    },
}


def _strip_comment(line: str) -> str:
    return line.split('#', 1)[0].strip()


def _parse_value(raw: str):
    value = raw.strip().strip('"').strip("'")
    try:
        return int(value, 0)
    except ValueError:
        return value


def parse_kv_file(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = _strip_comment(line)
        if not line or '=' not in line:
            continue
        key, value = line.split('=', 1)
        values[key.strip()] = _parse_value(value)
    return values


def parse_make_vars(path: Path) -> dict:
    values = {}
    if not path.is_file():
        return values
    pattern = re.compile(r'^([A-Za-z_][A-Za-z0-9_]*)\s*(\?=|:=|=)\s*(.*?)\s*$')
    for line in path.read_text().splitlines():
        line = _strip_comment(line)
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            continue
        key, _, value = match.groups()
        values[key] = _parse_value(value)
    return values


def load_config_topology(config_name: str) -> dict | None:
    config_path = CONFIG_DIR / f'{config_name}.mk'
    if not config_path.is_file():
        return None
    base = parse_make_vars(CONFIG_DIR / 'config.mk')
    specific = parse_make_vars(config_path)
    merged = {**base, **specific}
    return {
        'config': config_name,
        'NUM_CORES': int(merged['num_cores']),
        'NUM_GROUPS': int(merged['num_groups']),
        'NUM_CORES_PER_TILE': int(merged['num_cores_per_tile']),
        'SEQ_MEM_SIZE': int(merged['seq_mem_size']),
        'BANKING_FACTOR': int(merged['banking_factor']),
        'L1_BANK_SIZE': int(merged['l1_bank_size']),
        'NUM_SUB_GROUPS_PER_GROUP': int(merged['num_sub_groups_per_group']),
        'source': f'config/{config_name}.mk',
    }


def infer_named_topology(metadata: dict) -> str | None:
    config_name = metadata.get('config')
    if isinstance(config_name, str) and config_name in ('mempool', 'terapool'):
        return config_name
    for name, values in KNOWN_TOPOLOGIES.items():
        if name not in ('mempool', 'terapool'):
            continue
        if all(int(metadata[key]) == expected for key, expected in values.items()):
            return name
    return None


def find_result_dir(*paths) -> Path | None:
    candidates = []
    for raw_path in paths:
        if not raw_path:
            continue
        path = Path(raw_path).resolve()
        if path.name in ('traces', 'data'):
            candidates.append(path.parent)
        elif path.parent.name in ('traces', 'data'):
            candidates.append(path.parent.parent)
    for candidate in candidates:
        if (candidate / 'env').is_file() or (candidate / 'topology.env').is_file():
            return candidate
    return candidates[0] if candidates else None


def load_result_dir_topology(result_dir: Path | None) -> dict | None:
    if result_dir is None:
        return None

    topology_file = result_dir / 'topology.env'
    topology = parse_kv_file(topology_file)
    if topology and all(key in topology for key in TOPOLOGY_KEYS):
        topology['source'] = str(topology_file)
        if 'CONFIG' in topology and 'config' not in topology:
            topology['config'] = topology['CONFIG']
        return topology

    env_data = parse_kv_file(result_dir / 'env')
    config_name = env_data.get('config') or env_data.get('CONFIG')
    if isinstance(config_name, str):
        topology = load_config_topology(config_name)
        if topology is not None:
            topology['source'] = f'{result_dir / "env"} -> {topology["source"]}'
            return topology
    return None


def load_env_topology(environ: dict) -> dict | None:
    values = {}
    for key in TOPOLOGY_KEYS:
        value = environ.get(key)
        if value is None:
            continue
        values[key] = int(value)
    if not values:
        return None
    missing = [key for key in TOPOLOGY_KEYS if key not in values]
    if missing:
        raise ValueError('Partial topology env detected; missing: ' + ', '.join(missing))
    values['source'] = 'environment'
    return values


def format_topology(metadata: dict) -> str:
    config_name = metadata.get('config')
    config_part = f'config={config_name}, ' if config_name else ''
    return (
        f'{config_part}cores={metadata["NUM_CORES"]}, '
        f'groups={metadata["NUM_GROUPS"]}, '
        f'cores/tile={metadata["NUM_CORES_PER_TILE"]}, '
        f'seq_mem_size={metadata["SEQ_MEM_SIZE"]}, '
        f'banking_factor={metadata["BANKING_FACTOR"]}, '
        f'l1_bank_size={metadata["L1_BANK_SIZE"]}, '
        f'subgroups/group={metadata["NUM_SUB_GROUPS_PER_GROUP"]}'
    )


def validate_topology_consistency(expected: dict, actual: dict):
    mismatches = []
    for key in TOPOLOGY_KEYS:
        if int(expected[key]) != int(actual[key]):
            mismatches.append(f'{key}: expected {expected[key]}, got {actual[key]}')
    if mismatches:
        raise ValueError('; '.join(mismatches))
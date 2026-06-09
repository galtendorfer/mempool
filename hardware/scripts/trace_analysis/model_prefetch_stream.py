#!/usr/bin/env python3
"""Trace-replay model for tile-local prefetch experiments.

The model is deliberately conservative: demand traffic always wins, prefetches
issue only into idle source ports, and the output is coverage/opportunity
metrics rather than an IPC estimate.  New algorithms can be added by subclassing
PrefetchAlgorithm in prefetch_algorithms.py and registering the class there.
"""

from __future__ import annotations

import argparse
import csv
import heapq
from collections import Counter, OrderedDict, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from prefetch_algorithms import ALGORITHMS, Prediction, PrefetchAlgorithm, create_prefetch_algorithm


TARGET_OPERAND = 'B'
ADDRESS_FIELDS = ('operand_addr_int', 'operand_addr', 'source_addr', 'route_addr')
STREAM_KEYS = ('core', 'core-port', 'core-domain')
BufferTag = int | tuple[int, int]


@dataclass(frozen=True)
class DemandEvent:
    cycle: int
    tile: int
    tile_core: int
    global_core: int
    port: int
    prefetch_domain: str
    remote_lane: int | None
    addr: int
    operand: str
    valid: int
    ready: int
    fire: int
    blocked: int


@dataclass(frozen=True)
class Route:
    port: int
    prefetch_domain: str
    remote_lane: int | None


@dataclass(frozen=True)
class PrefetchRequest:
    tile: int
    addr: int
    route: Route
    generated_cycle: int
    source_core: int
    reason: str


class TileBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.entries: OrderedDict[BufferTag, int] = OrderedDict()
        self.evictions = 0

    def contains(self, tag: BufferTag) -> bool:
        if tag not in self.entries:
            return False
        ready_cycle = self.entries.pop(tag)
        self.entries[tag] = ready_cycle
        return True

    def insert(self, tag: BufferTag, ready_cycle: int) -> bool:
        if self.capacity <= 0:
            return False
        replaced = tag in self.entries
        if replaced:
            self.entries.pop(tag)
        elif self.capacity > 0 and len(self.entries) >= self.capacity:
            self.entries.popitem(last=False)
            self.evictions += 1
        if self.capacity > 0:
            self.entries[tag] = ready_cycle
        return not replaced

    def __len__(self) -> int:
        return len(self.entries)


def parse_int(value: str | None, default: int | None = None) -> int | None:
    if value in (None, ''):
        return default
    return int(value, 0)


def parse_address(row: dict[str, str], address_field: str) -> int | None:
    value = row.get(address_field)
    if value in (None, ''):
        return None
    if address_field == 'operand_addr_int':
        return int(value, 0)
    return int(value, 16)


def read_events(path: Path, address_field: str) -> list[DemandEvent]:
    events: list[DemandEvent] = []
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            addr = parse_address(row, address_field)
            if addr is None:
                continue
            remote_lane = parse_int(row.get('remote_lane'))
            events.append(
                DemandEvent(
                    cycle=int(row['cycle']),
                    tile=int(row['source_tile']),
                    tile_core=int(row['source_tile_core']),
                    global_core=int(row['source_global_core']),
                    port=int(row['port']),
                    prefetch_domain=row.get('prefetch_domain', ''),
                    remote_lane=remote_lane,
                    addr=addr,
                    operand=row.get('operand', ''),
                    valid=int(row['valid']),
                    ready=int(row['ready']),
                    fire=int(row['fire']),
                    blocked=int(row['blocked']),
                )
            )
    if not events:
        raise SystemExit(f'No events loaded from {path}')
    events.sort(key=lambda event: (event.cycle, event.tile, event.port, event.tile_core, event.addr))
    return events


def build_route_map(events: Iterable[DemandEvent]) -> dict[tuple[int, int], Route]:
    votes: defaultdict[tuple[int, int], Counter[Route]] = defaultdict(Counter)
    for event in events:
        votes[(event.tile, event.addr)][Route(event.port, event.prefetch_domain, event.remote_lane)] += 1
    return {key: counter.most_common(1)[0][0] for key, counter in votes.items()}


def group_by_cycle(events: Iterable[DemandEvent]) -> dict[int, list[DemandEvent]]:
    by_cycle: defaultdict[int, list[DemandEvent]] = defaultdict(list)
    for event in events:
        by_cycle[event.cycle].append(event)
    return dict(by_cycle)


def queue_key(route: Route) -> tuple[str, int]:
    if route.prefetch_domain == 'remote_port':
        lane = -1 if route.remote_lane is None else route.remote_lane
        return ('remote', lane)
    return ('local', 0)


class PrefetchReplay:
    def __init__(
        self,
        events: list[DemandEvent],
        algorithm: PrefetchAlgorithm,
        target_operand: str,
        local_depth: int,
        remote_depth: int,
        buffer_entries: int,
        latency: int,
        route_tag: bool,
    ):
        self.events = events
        self.algorithm = algorithm
        self.target_operand = target_operand
        self.local_depth = local_depth
        self.remote_depth = remote_depth
        self.buffer_entries = buffer_entries
        self.latency = latency
        self.route_tag = route_tag
        self.route_map = build_route_map(events)
        self.by_cycle = group_by_cycle(events)
        self.buffers: defaultdict[int, TileBuffer] = defaultdict(lambda: TileBuffer(buffer_entries))
        self.queues: defaultdict[tuple[int, str, int], deque[PrefetchRequest]] = defaultdict(deque)
        self.inflight: list[tuple[int, int, PrefetchRequest]] = []
        self.next_inflight_id = 0
        self.queued_or_inflight: set[tuple[int, BufferTag]] = set()
        self.stats: Counter[str] = Counter()
        self.tile_stats: defaultdict[int, Counter[str]] = defaultdict(Counter)
        self.domain_stats: defaultdict[str, Counter[str]] = defaultdict(Counter)

    def run(self) -> None:
        min_cycle = min(self.by_cycle)
        max_cycle = max(self.by_cycle)
        for cycle in range(min_cycle, max_cycle + 1):
            events = self.by_cycle.get(cycle, [])
            busy_ports = {(event.tile, event.port) for event in events if event.valid}

            self._retire_prefetches(cycle)
            self._issue_prefetches(cycle, busy_ports)
            self._process_demands(cycle, events)

        self._retire_prefetches(max_cycle + self.latency + 1)
        self.stats['buffer_evictions'] = sum(buffer.evictions for buffer in self.buffers.values())
        self.stats['buffer_final_entries'] = sum(len(buffer) for buffer in self.buffers.values())

    def _retire_prefetches(self, cycle: int) -> None:
        while self.inflight and self.inflight[0][0] <= cycle:
            ready_cycle, _, request = heapq.heappop(self.inflight)
            tag = self._tag(request.route, request.addr)
            self.queued_or_inflight.discard((request.tile, tag))
            inserted = self.buffers[request.tile].insert(tag, ready_cycle)
            if inserted:
                self.stats['prefetch_fills'] += 1
                self.tile_stats[request.tile]['prefetch_fills'] += 1
                self.domain_stats[request.route.prefetch_domain]['prefetch_fills'] += 1

    def _issue_prefetches(self, cycle: int, busy_ports: set[tuple[int, int]]) -> None:
        for queue_id, queue in list(self.queues.items()):
            if not queue:
                continue
            tile, _, _ = queue_id
            request = queue[0]
            if (tile, request.route.port) in busy_ports:
                self.stats['issue_blocked_by_demand'] += 1
                continue
            queue.popleft()
            ready_cycle = cycle + self.latency
            heapq.heappush(self.inflight, (ready_cycle, self.next_inflight_id, request))
            self.next_inflight_id += 1
            self.stats['prefetch_issued'] += 1
            self.tile_stats[tile]['prefetch_issued'] += 1
            self.domain_stats[request.route.prefetch_domain]['prefetch_issued'] += 1

    def _process_demands(self, cycle: int, events: list[DemandEvent]) -> None:
        for event in events:
            self.stats['valid_rows'] += int(bool(event.valid))
            if event.operand != self.target_operand:
                continue

            self.stats['target_valid_rows'] += int(bool(event.valid))
            self.tile_stats[event.tile]['target_valid_rows'] += int(bool(event.valid))
            self.domain_stats[event.prefetch_domain]['target_valid_rows'] += int(bool(event.valid))

            hit = self.buffers[event.tile].contains(self._tag(self._event_route(event), event.addr))
            if event.blocked:
                self.stats['target_blocked_rows'] += 1
                self.tile_stats[event.tile]['target_blocked_rows'] += 1
                self.domain_stats[event.prefetch_domain]['target_blocked_rows'] += 1
                if hit:
                    self.stats['blocked_rows_with_ready_prefetch'] += 1
                    self.tile_stats[event.tile]['blocked_rows_with_ready_prefetch'] += 1
                    self.domain_stats[event.prefetch_domain]['blocked_rows_with_ready_prefetch'] += 1

            if not event.fire:
                continue

            self.stats['target_fire_rows'] += 1
            self.tile_stats[event.tile]['target_fire_rows'] += 1
            self.domain_stats[event.prefetch_domain]['target_fire_rows'] += 1
            if hit:
                self.stats['fire_hits'] += 1
                self.tile_stats[event.tile]['fire_hits'] += 1
                self.domain_stats[event.prefetch_domain]['fire_hits'] += 1
            else:
                self.stats['fire_misses'] += 1

            for prediction in self.algorithm.observe(event):
                self._enqueue_prediction(cycle, event, prediction)

    def _enqueue_prediction(self, cycle: int, event: DemandEvent, prediction: Prediction) -> None:
        self.stats['predictions'] += 1
        if prediction.addr == event.addr:
            self.stats['prediction_same_addr'] += 1
            return

        route = self._prediction_route(event, prediction)
        if route is None:
            self.stats['prediction_unknown_route'] += 1
            return

        tag = self._tag(route, prediction.addr)
        key = (event.tile, tag)
        if self.buffers[event.tile].contains(tag):
            self.stats['prediction_already_buffered'] += 1
            return
        if key in self.queued_or_inflight:
            self.stats['prediction_duplicate_queued_or_inflight'] += 1
            return

        domain, lane = queue_key(route)
        queue = self.queues[(event.tile, domain, lane)]
        depth = self.remote_depth if domain == 'remote' else self.local_depth
        if depth <= 0:
            self.stats[f'prediction_{domain}_disabled'] += 1
            return
        if len(queue) >= depth:
            self.stats[f'prediction_{domain}_queue_full'] += 1
            return

        request = PrefetchRequest(
            tile=event.tile,
            addr=prediction.addr,
            route=route,
            generated_cycle=cycle,
            source_core=event.tile_core,
            reason=prediction.reason,
        )
        queue.append(request)
        self.queued_or_inflight.add(key)
        self.stats['prefetch_enqueued'] += 1
        self.tile_stats[event.tile]['prefetch_enqueued'] += 1
        self.domain_stats[route.prefetch_domain]['prefetch_enqueued'] += 1

    def _tag(self, route: Route, addr: int) -> BufferTag:
        if self.route_tag:
            return (route.port, addr)
        return addr

    def _event_route(self, event: DemandEvent) -> Route:
        return Route(event.port, event.prefetch_domain, event.remote_lane)

    def _prediction_route(self, event: DemandEvent, prediction: Prediction) -> Route | None:
        if self.route_tag:
            return self._event_route(event)
        return self.route_map.get((event.tile, prediction.addr))


def pct(part: int, total: int) -> str:
    if total == 0:
        return ''
    return f'{100.0 * part / total:.6f}'


def write_summary(path: Path, replay: PrefetchReplay, args: argparse.Namespace) -> None:
    fire_hits = replay.stats['fire_hits']
    fire_rows = replay.stats['target_fire_rows']
    blocked_hits = replay.stats['blocked_rows_with_ready_prefetch']
    blocked_rows = replay.stats['target_blocked_rows']
    with path.open('w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('metric', 'value'))
        rows: list[tuple[str, object]] = [
            ('input', args.input),
            ('algorithm', args.algorithm),
            ('target_operand', args.target_operand),
            ('address_field', args.address_field),
            ('stream_key', args.stream_key),
            ('degree', args.degree),
            ('context_length', args.context_length),
            ('block_words', args.block_words),
            ('inner_delta', args.inner_delta),
            ('route_tag', int(args.route_tag)),
            ('latency', args.latency),
            ('buffer_entries_per_tile', args.buffer_entries),
            ('local_queue_depth', args.local_depth),
            ('remote_lane_depth', args.remote_depth),
            ('fire_hit_rate_pct', pct(fire_hits, fire_rows)),
            ('blocked_row_ready_prefetch_rate_pct', pct(blocked_hits, blocked_rows)),
        ]
        rows.extend(sorted(replay.stats.items()))
        for key, value in rows:
            writer.writerow((key, value))


def write_breakdown(path: Path, rows: dict[object, Counter[str]], key_name: str) -> None:
    fields = (
        key_name,
        'target_valid_rows',
        'target_fire_rows',
        'fire_hits',
        'fire_hit_rate_pct',
        'target_blocked_rows',
        'blocked_rows_with_ready_prefetch',
        'blocked_row_ready_prefetch_rate_pct',
        'prefetch_enqueued',
        'prefetch_issued',
        'prefetch_fills',
    )
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for key, counts in sorted(rows.items(), key=lambda item: item[0]):
            writer.writerow({
                key_name: key,
                'target_valid_rows': counts['target_valid_rows'],
                'target_fire_rows': counts['target_fire_rows'],
                'fire_hits': counts['fire_hits'],
                'fire_hit_rate_pct': pct(counts['fire_hits'], counts['target_fire_rows']),
                'target_blocked_rows': counts['target_blocked_rows'],
                'blocked_rows_with_ready_prefetch': counts['blocked_rows_with_ready_prefetch'],
                'blocked_row_ready_prefetch_rate_pct': pct(
                    counts['blocked_rows_with_ready_prefetch'],
                    counts['target_blocked_rows'],
                ),
                'prefetch_enqueued': counts['prefetch_enqueued'],
                'prefetch_issued': counts['prefetch_issued'],
                'prefetch_fills': counts['prefetch_fills'],
            })


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Replay a load-address stream with a pluggable prefetch algorithm.')
    parser.add_argument('input', type=Path, help='load-address stream CSV from export_load_address_stream.py')
    parser.add_argument(
        '--algorithm',
        choices=sorted(ALGORITHMS),
        default='stride',
        help='prefetch algorithm to use',
    )
    parser.add_argument('--target-operand', default=TARGET_OPERAND, help='operand to prefetch/evaluate')
    parser.add_argument(
        '--address-field',
        choices=ADDRESS_FIELDS,
        default='operand_addr_int',
        help='CSV address column used by the predictor and prefetch buffer',
    )
    parser.add_argument(
        '--stream-key',
        choices=STREAM_KEYS,
        default='core',
        help='demand-stream identity used by per-stream prefetch algorithms',
    )
    parser.add_argument('--degree', type=int, default=2, help='number of predictions per observed demand')
    parser.add_argument('--latency', type=int, default=4, help='prefetch fill latency in cycles')
    parser.add_argument('--buffer-entries', type=int, default=16, help='shared prefetch buffer entries per tile')
    parser.add_argument('--local-depth', type=int, default=2, help='local-port queue depth per tile')
    parser.add_argument('--remote-depth', type=int, default=4, help='remote-lane queue depth per tile')
    parser.add_argument('--min-confidence', type=int, default=2, help='stride confidence threshold')
    parser.add_argument('--min-transition-count', type=int, default=2, help='delta-transition count threshold')
    parser.add_argument('--context-length', type=int, default=3, help='delta-context history length')
    parser.add_argument('--block-words', type=int, default=4, help='block-stride words per contiguous block')
    parser.add_argument('--inner-delta', type=int, default=4, help='block-stride byte delta inside a block')
    parser.add_argument(
        '--route-tag',
        action='store_true',
        help='tag prefetch-buffer entries by route port as well as address',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        help='output directory; defaults to <input-dir>/prefetch_model/<algorithm>',
    )
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    events = read_events(args.input, args.address_field)
    algorithm = create_prefetch_algorithm(
        args.algorithm,
        degree=args.degree,
        min_confidence=args.min_confidence,
        min_count=args.min_transition_count,
        context_length=args.context_length,
        block_words=args.block_words,
        inner_delta=args.inner_delta,
        stream_key=args.stream_key,
    )
    replay = PrefetchReplay(
        events=events,
        algorithm=algorithm,
        target_operand=args.target_operand,
        local_depth=args.local_depth,
        remote_depth=args.remote_depth,
        buffer_entries=args.buffer_entries,
        latency=args.latency,
        route_tag=args.route_tag,
    )
    replay.run()

    output_dir = args.output_dir or args.input.parent / 'prefetch_model' / args.algorithm
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / 'summary.csv'
    tile_path = output_dir / 'by_tile.csv'
    domain_path = output_dir / 'by_domain.csv'
    write_summary(summary_path, replay, args)
    write_breakdown(tile_path, replay.tile_stats, 'tile')
    write_breakdown(domain_path, replay.domain_stats, 'prefetch_domain')

    print(f'Wrote summary to {summary_path}')
    print(f'Wrote tile breakdown to {tile_path}')
    print(f'Wrote domain breakdown to {domain_path}')
    print(
        'fire_hit_rate='
        f'{pct(replay.stats["fire_hits"], replay.stats["target_fire_rows"])}% '
        f'({replay.stats["fire_hits"]}/{replay.stats["target_fire_rows"]})'
    )


if __name__ == '__main__':
    main()

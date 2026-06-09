#!/usr/bin/env python3
"""Prefetch algorithms for model_prefetch_stream.py.

Add new algorithms here by subclassing PrefetchAlgorithm and registering the
class in ALGORITHMS.  The replay harness passes DemandEvent-like objects to
observe(); algorithms should only depend on stable fields such as tile,
tile_core, addr, cycle, port, and prefetch_domain.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Protocol


StreamKey = tuple[object, ...]


class DemandLike(Protocol):
    cycle: int
    tile: int
    tile_core: int
    global_core: int
    port: int
    prefetch_domain: str
    addr: int


@dataclass(frozen=True)
class Prediction:
    addr: int
    reason: str


class PrefetchAlgorithm:
    name = 'base'

    def __init__(self, degree: int, stream_key: str = 'core', **_: object):
        self.degree = degree
        self.stream_key = stream_key

    def event_key(self, event: DemandLike) -> StreamKey:
        if self.stream_key == 'core-port':
            return (event.tile, event.tile_core, event.port)
        if self.stream_key == 'core-domain':
            return (event.tile, event.tile_core, event.prefetch_domain)
        return (event.tile, event.tile_core)

    def observe(self, event: DemandLike) -> list[Prediction]:
        raise NotImplementedError


class NoPrefetch(PrefetchAlgorithm):
    name = 'none'

    def observe(self, event: DemandLike) -> list[Prediction]:
        return []


class NextLinePrefetch(PrefetchAlgorithm):
    name = 'next-line'

    def __init__(self, degree: int, line_bytes: int = 4, **kwargs: object):
        super().__init__(degree, **kwargs)
        self.line_bytes = line_bytes

    def observe(self, event: DemandLike) -> list[Prediction]:
        return [
            Prediction(event.addr + self.line_bytes * distance, f'next-line:{distance}')
            for distance in range(1, self.degree + 1)
        ]


class PerCoreStridePrefetch(PrefetchAlgorithm):
    name = 'stride'

    def __init__(self, degree: int, min_confidence: int = 2, **kwargs: object):
        super().__init__(degree, **kwargs)
        self.min_confidence = min_confidence
        self.last_addr: dict[StreamKey, int] = {}
        self.last_delta: dict[StreamKey, int] = {}
        self.confidence: Counter[StreamKey] = Counter()

    def observe(self, event: DemandLike) -> list[Prediction]:
        key = self.event_key(event)
        previous_addr = self.last_addr.get(key)
        if previous_addr is None:
            self.last_addr[key] = event.addr
            return []

        delta = event.addr - previous_addr
        if self.last_delta.get(key) == delta:
            self.confidence[key] += 1
        else:
            self.last_delta[key] = delta
            self.confidence[key] = 1
        self.last_addr[key] = event.addr

        if self.confidence[key] < self.min_confidence or delta == 0:
            return []
        return [
            Prediction(event.addr + delta * distance, f'stride:{delta}:{distance}')
            for distance in range(1, self.degree + 1)
        ]


class PerCoreDeltaTransitionPrefetch(PrefetchAlgorithm):
    """Predict the next delta from per-core delta transition history."""

    name = 'delta-transition'

    def __init__(self, degree: int, min_count: int = 2, **kwargs: object):
        super().__init__(degree, **kwargs)
        self.min_count = min_count
        self.last_addr: dict[StreamKey, int] = {}
        self.last_delta: dict[StreamKey, int] = {}
        self.transitions: defaultdict[StreamKey, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

    def observe(self, event: DemandLike) -> list[Prediction]:
        key = self.event_key(event)
        previous_addr = self.last_addr.get(key)
        if previous_addr is None:
            self.last_addr[key] = event.addr
            return []

        delta = event.addr - previous_addr
        previous_delta = self.last_delta.get(key)
        if previous_delta is not None:
            self.transitions[key][previous_delta][delta] += 1
        self.last_delta[key] = delta
        self.last_addr[key] = event.addr

        predictions: list[Prediction] = []
        predicted_addr = event.addr
        state_delta = delta
        for distance in range(1, self.degree + 1):
            next_delta = self._best_next_delta(key, state_delta)
            if next_delta is None or next_delta == 0:
                break
            predicted_addr += next_delta
            predictions.append(Prediction(predicted_addr, f'delta-transition:{state_delta}->{next_delta}:{distance}'))
            state_delta = next_delta
        return predictions

    def _best_next_delta(self, key: StreamKey, delta: int) -> int | None:
        candidates = self.transitions[key].get(delta)
        if not candidates:
            return None
        next_delta, count = candidates.most_common(1)[0]
        if count < self.min_count:
            return None
        return next_delta


class PerCoreDeltaContextPrefetch(PrefetchAlgorithm):
    """Predict the next delta from a short per-core delta-context history."""

    name = 'delta-context'

    def __init__(self, degree: int, context_length: int = 3, min_count: int = 2, **kwargs: object):
        super().__init__(degree, **kwargs)
        self.context_length = max(1, context_length)
        self.min_count = min_count
        self.last_addr: dict[StreamKey, int] = {}
        self.delta_history: defaultdict[StreamKey, deque[int]] = defaultdict(
            lambda: deque(maxlen=self.context_length)
        )
        self.transitions: defaultdict[StreamKey, dict[tuple[int, ...], Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

    def observe(self, event: DemandLike) -> list[Prediction]:
        key = self.event_key(event)
        previous_addr = self.last_addr.get(key)
        if previous_addr is None:
            self.last_addr[key] = event.addr
            return []

        delta = event.addr - previous_addr
        history = self.delta_history[key]
        if len(history) == self.context_length:
            self.transitions[key][tuple(history)][delta] += 1
        history.append(delta)
        self.last_addr[key] = event.addr

        if len(history) < self.context_length:
            return []

        predictions: list[Prediction] = []
        predicted_addr = event.addr
        context = tuple(history)
        for distance in range(1, self.degree + 1):
            next_delta = self._best_next_delta(key, context)
            if next_delta is None or next_delta == 0:
                break
            predicted_addr += next_delta
            predictions.append(Prediction(predicted_addr, f'delta-context:{context}->{next_delta}:{distance}'))
            context = (*context[1:], next_delta)
        return predictions

    def _best_next_delta(self, key: StreamKey, context: tuple[int, ...]) -> int | None:
        candidates = self.transitions[key].get(context)
        if not candidates:
            return None
        next_delta, count = candidates.most_common(1)[0]
        if count < self.min_count:
            return None
        return next_delta


class PerCoreBlockStridePrefetch(PrefetchAlgorithm):
    """Hardware-shaped block-stride predictor for fixed-width load blocks."""

    name = 'block-stride'

    def __init__(
        self,
        degree: int,
        block_words: int = 4,
        inner_delta: int = 4,
        min_count: int = 2,
        **kwargs: object,
    ):
        super().__init__(degree, **kwargs)
        self.block_words = max(2, block_words)
        self.inner_delta = inner_delta
        self.min_count = min_count
        self.last_addr: dict[StreamKey, int] = {}
        self.phase: defaultdict[StreamKey, int] = defaultdict(int)
        self.candidate_jump: dict[StreamKey, int] = {}
        self.jump_confidence: Counter[StreamKey] = Counter()
        self.row_jump: dict[StreamKey, int] = {}

    def observe(self, event: DemandLike) -> list[Prediction]:
        key = self.event_key(event)
        previous_addr = self.last_addr.get(key)
        if previous_addr is None:
            self.last_addr[key] = event.addr
            return []

        delta = event.addr - previous_addr
        old_phase = self.phase[key]
        if delta == self.inner_delta:
            self.phase[key] = min(old_phase + 1, self.block_words - 1)
        elif delta != 0 and old_phase >= self.block_words - 1:
            self._observe_row_jump(key, delta)
            self.phase[key] = 0
        else:
            self.phase[key] = 0

        self.last_addr[key] = event.addr
        jump = self.row_jump.get(key)
        if jump is None:
            return []

        return self._predict(event.addr, self.phase[key], jump)

    def _observe_row_jump(self, key: StreamKey, delta: int) -> None:
        if self.candidate_jump.get(key) == delta:
            self.jump_confidence[key] += 1
        else:
            self.candidate_jump[key] = delta
            self.jump_confidence[key] = 1
        if self.jump_confidence[key] >= self.min_count:
            self.row_jump[key] = delta

    def _predict(self, addr: int, phase: int, jump: int) -> list[Prediction]:
        predictions: list[Prediction] = []
        predicted_addr = addr
        predicted_phase = phase
        for distance in range(1, self.degree + 1):
            if predicted_phase < self.block_words - 1:
                predicted_addr += self.inner_delta
                predicted_phase += 1
            else:
                predicted_addr += jump
                predicted_phase = 0
            predictions.append(Prediction(
                predicted_addr,
                f'block-stride:phase{phase}:jump{jump}:d{distance}',
            ))
        return predictions


class PerCoreBlockJumpCyclePrefetch(PrefetchAlgorithm):
    """Block predictor that learns the jump sequence between fixed-width blocks."""

    name = 'block-jump-cycle'

    def __init__(
        self,
        degree: int,
        block_words: int = 4,
        inner_delta: int = 4,
        min_count: int = 2,
        **kwargs: object,
    ):
        super().__init__(degree, **kwargs)
        self.block_words = max(2, block_words)
        self.inner_delta = inner_delta
        self.min_count = min_count
        self.last_addr: dict[StreamKey, int] = {}
        self.phase: defaultdict[StreamKey, int] = defaultdict(int)
        self.last_jump: dict[StreamKey, int] = {}
        self.jump_transitions: defaultdict[StreamKey, dict[int, Counter[int]]] = defaultdict(
            lambda: defaultdict(Counter)
        )

    def observe(self, event: DemandLike) -> list[Prediction]:
        key = self.event_key(event)
        previous_addr = self.last_addr.get(key)
        if previous_addr is None:
            self.last_addr[key] = event.addr
            return []

        delta = event.addr - previous_addr
        old_phase = self.phase[key]
        if delta == self.inner_delta:
            self.phase[key] = min(old_phase + 1, self.block_words - 1)
        elif delta != 0 and old_phase >= self.block_words - 1:
            previous_jump = self.last_jump.get(key)
            if previous_jump is not None:
                self.jump_transitions[key][previous_jump][delta] += 1
            self.last_jump[key] = delta
            self.phase[key] = 0
        else:
            self.phase[key] = 0

        self.last_addr[key] = event.addr
        return self._predict(event.addr, self.phase[key], self.last_jump.get(key), key)

    def _predict(
        self,
        addr: int,
        phase: int,
        last_jump: int | None,
        key: StreamKey,
    ) -> list[Prediction]:
        predictions: list[Prediction] = []
        predicted_addr = addr
        predicted_phase = phase
        predicted_jump = last_jump
        for distance in range(1, self.degree + 1):
            if predicted_phase < self.block_words - 1:
                predicted_addr += self.inner_delta
                predicted_phase += 1
                reason = f'block-jump-cycle:inner:{distance}'
            else:
                next_jump = self._best_next_jump(key, predicted_jump)
                if next_jump is None:
                    break
                predicted_addr += next_jump
                predicted_phase = 0
                predicted_jump = next_jump
                reason = f'block-jump-cycle:jump{next_jump}:{distance}'
            predictions.append(Prediction(predicted_addr, reason))
        return predictions

    def _best_next_jump(self, key: StreamKey, last_jump: int | None) -> int | None:
        if last_jump is None:
            return None
        candidates = self.jump_transitions[key].get(last_jump)
        if not candidates:
            return None
        next_jump, count = candidates.most_common(1)[0]
        if count < self.min_count:
            return None
        return next_jump


ALGORITHMS: dict[str, type[PrefetchAlgorithm]] = {
    algorithm.name: algorithm
    for algorithm in (
        NoPrefetch,
        NextLinePrefetch,
        PerCoreStridePrefetch,
        PerCoreDeltaTransitionPrefetch,
        PerCoreDeltaContextPrefetch,
        PerCoreBlockStridePrefetch,
        PerCoreBlockJumpCyclePrefetch,
    )
}


def create_prefetch_algorithm(name: str, **kwargs: object) -> PrefetchAlgorithm:
    try:
        algorithm_cls = ALGORITHMS[name]
    except KeyError as error:
        choices = ', '.join(sorted(ALGORITHMS))
        raise ValueError(f'Unknown prefetch algorithm {name!r}; choices: {choices}') from error
    return algorithm_cls(**kwargs)

"""Residual-guided, bounded anomaly harvesting for the ARC-R2 primitive DSL.

The engine is opt-in. It never reads ARC test outputs, never mutates the base
primitive vocabulary, and retains a discovered program only after exact
validation against every supplied training pair.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import sys
from typing import Iterable

from .primitives import FitResult, Grid, Primitive, Program, as_grid, fit_program, generate_primitives


@dataclass(frozen=True)
class ResidualTrace:
    """A training-pair discrepancy left by the best available single primitive."""
    task_id: str
    pair_index: int
    input_grid: Grid
    predicted_grid: Grid
    target_grid: Grid
    residual_mask: tuple[tuple[int, ...], ...]
    residual_norm: int
    failed_op: str


def compute_residual(predicted: Grid, target: Grid) -> tuple[tuple[tuple[int, ...], ...], int]:
    """Return a target-shaped mismatch mask and deterministic residual score."""
    if len(predicted) != len(target) or any(len(a) != len(b) for a, b in zip(predicted, target)):
        mask = tuple(tuple(1 for _ in row) for row in target)
        return mask, sum(len(row) for row in target) + 1
    mask = tuple(tuple(int(a != b) for a, b in zip(pr, tr)) for pr, tr in zip(predicted, target))
    return mask, sum(sum(row) for row in mask)


class BoundedDSLRegistry:
    """Bounded cache of validated programs; byte accounting is an estimate."""
    def __init__(self, max_memory_bytes: int = 512 * 1024) -> None:
        if max_memory_bytes < 1:
            raise ValueError("max_memory_bytes must be positive")
        self.max_memory_bytes = max_memory_bytes
        self._programs: OrderedDict[str, tuple[Program, int, int]] = OrderedDict()
        self._current_bytes = 0

    @staticmethod
    def _estimate_bytes(program: Program) -> int:
        return sys.getsizeof(program) + sys.getsizeof(program.operations) + sum(
            sys.getsizeof(op) + sys.getsizeof(op.name) + sys.getsizeof(op.args)
            for op in program.operations
        )

    def register(self, program: Program) -> bool:
        key = str(program)
        cost = self._estimate_bytes(program)
        if cost > self.max_memory_bytes:
            return False
        old = self._programs.pop(key, None)
        if old:
            self._current_bytes -= old[1]
        while self._current_bytes + cost > self.max_memory_bytes and self._programs:
            evict_key = min(
                self._programs,
                key=lambda item: (self._programs[item][2], -self._programs[item][0].complexity),
            )
            _, old_cost, _ = self._programs.pop(evict_key)
            self._current_bytes -= old_cost
        self._programs[key] = (program, cost, 0)
        self._current_bytes += cost
        return True

    def record_hit(self, program: Program) -> None:
        key = str(program)
        if key in self._programs:
            value, cost, hits = self._programs.pop(key)
            self._programs[key] = (value, cost, hits + 1)

    def get_all(self) -> tuple[Program, ...]:
        return tuple(value[0] for value in self._programs.values())

    @property
    def memory_usage(self) -> int:
        return self._current_bytes


@dataclass(frozen=True)
class HarvestResult:
    baseline: FitResult
    program: Program | None
    traces_collected: int
    probes_tested: int
    programs_retained: int
    memory_bytes: int

    @property
    def exact_fit(self) -> bool:
        return self.program is not None


class AnomalyHarvestingEngine:
    """Probe bounded two-primitive compositions using training residuals."""
    def __init__(self, memory_budget_bytes: int = 512 * 1024,
                 max_probes_per_cycle: int = 64, top_ops_per_trace: int = 8) -> None:
        if max_probes_per_cycle < 1 or top_ops_per_trace < 1:
            raise ValueError("probe and per-trace limits must be positive")
        self.registry = BoundedDSLRegistry(memory_budget_bytes)
        self.max_probes_per_cycle = max_probes_per_cycle
        self.top_ops_per_trace = top_ops_per_trace
        self.harvested_traces: list[ResidualTrace] = []

    @staticmethod
    def _score(program: Program, pairs: tuple[tuple[Grid, Grid], ...]) -> tuple[int, bool]:
        total, exact = 0, True
        for source, target in pairs:
            try:
                _, residual = compute_residual(program.apply(source), target)
            except (ValueError, IndexError, TypeError):
                return 10**12, False
            total += residual
            exact = exact and residual == 0
        return total, exact

    def _collect_traces(self, task_id: str, pairs: tuple[tuple[Grid, Grid], ...],
                        primitives: tuple[Primitive, ...]) -> list[list[Primitive]]:
        ranked_per_pair = []
        for pair_index, (source, target) in enumerate(pairs):
            ranked = []
            for primitive in primitives:
                try:
                    predicted = Program((primitive,)).apply(source)
                    _, norm = compute_residual(predicted, target)
                except (ValueError, IndexError, TypeError):
                    continue
                ranked.append((norm, str(primitive), primitive, predicted))
            ranked.sort(key=lambda item: (item[0], item[1]))
            if ranked:
                norm, _, best_op, predicted = ranked[0]
                mask, _ = compute_residual(predicted, target)
                self.harvested_traces.append(ResidualTrace(
                    task_id, pair_index, source, predicted, target, mask, norm, str(best_op)
                ))
            ranked_per_pair.append([item[2] for item in ranked[:self.top_ops_per_trace]])
        return ranked_per_pair

    def solve(self, task_id: str, examples: Iterable[tuple[Grid, Grid]],
              candidate_cap: int = 5000) -> HarvestResult:
        pairs = tuple((as_grid(a), as_grid(b)) for a, b in examples)
        if not pairs:
            raise ValueError("at least one training pair is required")
        baseline = fit_program(pairs, candidate_cap=candidate_cap, max_depth=1)
        if baseline.exact_fit:
            return HarvestResult(baseline, baseline.program, 0, 0, 0, 0)

        primitives = generate_primitives(
            tuple(source for source, _ in pairs), tuple(target for _, target in pairs)
        )
        ranked_per_pair = self._collect_traces(task_id, pairs, primitives)
        relevant = sorted({op for group in ranked_per_pair for op in group}, key=str)

        candidates = []
        seen = set()
        for first in relevant:
            for second in relevant:
                candidate = Program((first, second))
                key = str(candidate)
                if key in seen:
                    continue
                seen.add(key)
                score, exact = self._score(candidate, pairs)
                if exact:
                    candidates.append((score, key, candidate))
        candidates.sort(key=lambda item: (item[0], item[1]))

        tested = retained = 0
        for _, _, candidate in candidates:
            if tested >= self.max_probes_per_cycle:
                break
            tested += 1
            _, exact = self._score(candidate, pairs)
            if exact and self.registry.register(candidate):
                retained += 1
                self.registry.record_hit(candidate)
        programs = self.registry.get_all()
        winner = min(programs, key=lambda p: (p.complexity, str(p))) if programs else None
        return HarvestResult(baseline, winner, len(self.harvested_traces), tested,
                             retained, self.registry.memory_usage)


def fit_program_with_harvesting(
    examples: tuple[tuple[Grid, Grid], ...], candidate_cap: int = 5000,
    max_probes_per_cycle: int = 64, memory_budget_bytes: int = 512 * 1024,
    task_id: str = "arc-task",
) -> HarvestResult:
    """Opt-in wrapper; the established fit_program implementation is unchanged."""
    engine = AnomalyHarvestingEngine(memory_budget_bytes, max_probes_per_cycle)
    return engine.solve(task_id, examples, candidate_cap)

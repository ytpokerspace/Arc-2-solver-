"""Bounded beam and UCT search over a shared finite operation vocabulary."""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Hashable
from .dsl import Program
from .transforms import OPERATIONS

Score = Callable[[Program], float]

def _operations(limits: SearchLimits) -> tuple[str, ...]:
    return OPERATIONS if limits.include_prte else tuple(op for op in OPERATIONS if op != "PRTE.POLARITY.REVERSE")

@dataclass
class SearchLimits:
    beam_width: int = 20
    max_depth: int = 4
    candidate_cap: int = 2000
    mcts_iterations: int = 250
    exploration: float = 1.414
    seed: int = 0
    include_prte: bool = True
    max_seconds: float = 10.0

    def __post_init__(self):
        if self.beam_width < 1 or self.max_depth < 0 or self.candidate_cap < 0 or self.mcts_iterations < 0:
            raise ValueError("search limits are outside valid bounds")
        if self.max_seconds < 0:
            raise ValueError("max_seconds cannot be negative")

@dataclass
class SearchStats:
    evaluated: int = 0
    expanded: int = 0
    beam_rounds: int = 0
    mcts_iterations: int = 0
    stopped_by_cap: bool = False
    stopped_by_time: bool = False

def beam_search(score: Score, limits: SearchLimits, stats: SearchStats | None = None, state_key: Callable[[Program], Hashable] | None = None) -> list[tuple[float, Program]]:
    stats = stats if stats is not None else SearchStats()
    started = time.monotonic()
    frontier = [Program()]
    seen = {()}
    score_cache: dict[Hashable, float] = {}
    ranked: list[tuple[float, Program]] = []
    for _depth in range(1, limits.max_depth + 1):
        next_items = []
        for program in frontier:
            for op in _operations(limits):
                if time.monotonic() - started >= limits.max_seconds:
                    stats.stopped_by_time = True
                    break
                if stats.evaluated >= limits.candidate_cap:
                    stats.stopped_by_cap = True
                    break
                child = program.then(op)
                key = child.canonical()
                if key in seen: continue
                seen.add(key)
                fingerprint = state_key(child) if state_key is not None else key
                if fingerprint in score_cache:
                    value = score_cache[fingerprint]
                else:
                    if stats.evaluated >= limits.candidate_cap:
                        stats.stopped_by_cap = True
                        break
                    value = score(child)
                    score_cache[fingerprint] = value
                    stats.evaluated += 1
                stats.expanded += 1
                next_items.append((value, child))
            if stats.stopped_by_cap: break
        if not next_items: break
        next_items.sort(key=lambda item: (item[0] - 0.002 * item[1].complexity, -item[1].complexity), reverse=True)
        ranked.extend(next_items)
        frontier = [p for _, p in next_items[:limits.beam_width]]
        stats.beam_rounds += 1
        if stats.stopped_by_cap or stats.stopped_by_time: break
    unique = {p.canonical(): (s, p) for s, p in ranked}
    return sorted(unique.values(), key=lambda item: item[0] - 0.002 * item[1].complexity, reverse=True)

@dataclass
class _Node:
    program: Program
    parent: "_Node | None" = None
    visits: int = 0
    total: float = 0.0
    children: list["_Node"] = field(default_factory=list)
    untried: list[str] = field(default_factory=list)

    def uct(self, parent_visits: int, c: float) -> float:
        if self.visits == 0: return float("inf")
        return self.total / self.visits + c * math.sqrt(math.log(max(1, parent_visits)) / self.visits)

def mcts_search(score: Score, limits: SearchLimits, stats: SearchStats | None = None, state_key: Callable[[Program], Hashable] | None = None) -> list[tuple[float, Program]]:
    stats = stats if stats is not None else SearchStats()
    started = time.monotonic()
    rng = random.Random(limits.seed)
    operations = list(_operations(limits))
    root = _Node(Program(), untried=operations.copy())
    best: dict[tuple[str, ...], tuple[float, Program]] = {}
    score_cache: dict[Hashable, float] = {}
    for _ in range(limits.mcts_iterations):
        if time.monotonic() - started >= limits.max_seconds:
            stats.stopped_by_time = True
            break
        node = root
        while not node.untried and node.children and node.program.complexity < limits.max_depth:
            node = max(node.children, key=lambda child: child.uct(node.visits, limits.exploration))
        if node.untried and node.program.complexity < limits.max_depth:
            op = rng.choice(node.untried)
            node.untried.remove(op)
            child = _Node(node.program.then(op), parent=node, untried=operations.copy())
            node.children.append(child)
            node = child
        fingerprint = state_key(node.program) if state_key is not None else node.program.canonical()
        if fingerprint not in score_cache:
            if stats.evaluated >= limits.candidate_cap:
                stats.stopped_by_cap = True
                break
            score_cache[fingerprint] = score(node.program)
            stats.evaluated += 1
        value = score_cache[fingerprint]
        stats.mcts_iterations += 1
        best[node.program.canonical()] = (value, node.program)
        reward = max(0.0, min(1.0, value))
        while node is not None:
            node.visits += 1
            node.total += reward
            node = node.parent
    return sorted(best.values(), key=lambda item: item[0] - 0.002 * item[1].complexity, reverse=True)

"""Bounded beam and UCT search over a shared finite operation vocabulary."""
from __future__ import annotations
import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable
from .dsl import Program
from .transforms import OPERATIONS

Score = Callable[[Program], float]

@dataclass
class SearchLimits:
    beam_width: int = 20
    max_depth: int = 4
    candidate_cap: int = 2000
    mcts_iterations: int = 250
    exploration: float = 1.414
    seed: int = 0

@dataclass
class SearchStats:
    evaluated: int = 0
    expanded: int = 0
    beam_rounds: int = 0
    mcts_iterations: int = 0
    stopped_by_cap: bool = False

def beam_search(score: Score, limits: SearchLimits, stats: SearchStats | None = None) -> list[tuple[float, Program]]:
    stats = stats if stats is not None else SearchStats()
    frontier = [Program()]
    seen = {()}
    ranked: list[tuple[float, Program]] = []
    for _depth in range(1, limits.max_depth + 1):
        next_items = []
        for program in frontier:
            for op in OPERATIONS:
                if stats.evaluated >= limits.candidate_cap:
                    stats.stopped_by_cap = True
                    break
                child = program.then(op)
                key = child.canonical()
                if key in seen: continue
                seen.add(key)
                value = score(child)
                stats.evaluated += 1
                stats.expanded += 1
                next_items.append((value, child))
            if stats.stopped_by_cap: break
        if not next_items: break
        next_items.sort(key=lambda item: (item[0] - 0.002 * item[1].complexity, -item[1].complexity), reverse=True)
        ranked.extend(next_items)
        frontier = [p for _, p in next_items[:limits.beam_width]]
        stats.beam_rounds += 1
        if stats.stopped_by_cap: break
    unique = {p.canonical(): (s, p) for s, p in ranked}
    return sorted(unique.values(), key=lambda item: item[0] - 0.002 * item[1].complexity, reverse=True)

@dataclass
class _Node:
    program: Program
    parent: "_Node | None" = None
    visits: int = 0
    total: float = 0.0
    children: list["_Node"] = field(default_factory=list)
    untried: list[str] = field(default_factory=lambda: list(OPERATIONS))

    def uct(self, parent_visits: int, c: float) -> float:
        if self.visits == 0: return float("inf")
        return self.total / self.visits + c * math.sqrt(math.log(max(1, parent_visits)) / self.visits)

def mcts_search(score: Score, limits: SearchLimits, stats: SearchStats | None = None) -> list[tuple[float, Program]]:
    stats = stats if stats is not None else SearchStats()
    rng = random.Random(limits.seed)
    root = _Node(Program())
    best: dict[tuple[str, ...], tuple[float, Program]] = {}
    for _ in range(limits.mcts_iterations):
        if stats.evaluated >= limits.candidate_cap:
            stats.stopped_by_cap = True
            break
        node = root
        while not node.untried and node.children and node.program.complexity < limits.max_depth:
            node = max(node.children, key=lambda child: child.uct(node.visits, limits.exploration))
        if node.untried and node.program.complexity < limits.max_depth:
            op = rng.choice(node.untried)
            node.untried.remove(op)
            child = _Node(node.program.then(op), parent=node)
            node.children.append(child)
            node = child
        value = score(node.program)
        stats.evaluated += 1
        stats.mcts_iterations += 1
        best[node.program.canonical()] = (value, node.program)
        reward = max(0.0, min(1.0, value))
        while node is not None:
            node.visits += 1
            node.total += reward
            node = node.parent
    return sorted(best.values(), key=lambda item: item[0] - 0.002 * item[1].complexity, reverse=True)

"""QBS-guided, dual-track, progressively widened MCTS research scaffold.

QBS ideas are supplied as explicit policies; this module does not invent or
claim to have generated the user's ten QBS ideas.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .grid import Grid
from .primitive_engine import Primitive, PrimitiveRegistry

ScoreFn = Callable[[Grid], float]


@dataclass(frozen=True)
class SeedPolicy:
    seed_id: str
    idea: str
    preferred_families: tuple[str, ...]
    compatibility_tags: tuple[str, ...] = ()
    weight: float = 1.0


def hybridize(policies: Iterable[SeedPolicy], limit: int = 10) -> tuple[SeedPolicy, ...]:
    """Create deterministic pairwise hybrids only when compatibility tags overlap."""
    source = tuple(policies)
    hybrids: list[SeedPolicy] = []
    for i, left in enumerate(source):
        for right in source[i + 1:]:
            shared = tuple(sorted(set(left.compatibility_tags) & set(right.compatibility_tags)))
            if not shared:
                continue
            families = tuple(dict.fromkeys(left.preferred_families + right.preferred_families))
            hybrids.append(SeedPolicy(
                seed_id=f"HYBRID:{left.seed_id}+{right.seed_id}",
                idea=f"Hybrid of {left.idea} AND {right.idea}",
                preferred_families=families,
                compatibility_tags=shared,
                weight=(left.weight + right.weight) / 2,
            ))
            if len(hybrids) >= limit:
                return tuple(hybrids)
    return tuple(hybrids)


@dataclass
class SearchLimits:
    iterations: int = 1000
    max_depth: int = 5
    candidate_cap: int = 2500
    initial_width: int = 3
    widening_k: float = 2.0
    widening_alpha: float = 0.5
    exploration: float = 1.414
    seed: int = 0


@dataclass
class Candidate:
    score: float
    program: tuple[str, ...]
    grid: Grid


@dataclass
class _Node:
    grid: Grid
    program: tuple[str, ...]
    parent: "_Node | None" = None
    visits: int = 0
    total: float = 0.0
    children: list["_Node"] = field(default_factory=list)
    next_action: int = 0

    def uct(self, parent_visits: int, exploration: float) -> float:
        if self.visits == 0:
            return float("inf")
        return self.total / self.visits + exploration * math.sqrt(
            math.log(max(1, parent_visits)) / self.visits
        )


@dataclass
class TrackResult:
    track: str
    candidates: tuple[Candidate, ...]
    evaluated: int
    expanded: int
    pruned: int


def _rank_actions(actions: tuple[Primitive, ...], policy: SeedPolicy) -> tuple[Primitive, ...]:
    priority = {name: i for i, name in enumerate(policy.preferred_families)}
    return tuple(sorted(actions, key=lambda p: (
        priority.get(p.spec.family, len(priority) + 1),
        p.spec.cost,
        p.spec.name,
    )))


def run_mcts_track(initial: Grid, registry: PrimitiveRegistry,
                   policies: tuple[SeedPolicy, ...], score: ScoreFn,
                   limits: SearchLimits, track: str) -> TrackResult:
    if not policies:
        raise ValueError("AT_LEAST_ONE_SEED_POLICY_REQUIRED")
    if limits.iterations <= 0 or limits.max_depth < 1 or limits.candidate_cap < 1:
        raise ValueError("INVALID_SEARCH_LIMITS")

    actions = registry.enabled()
    if not actions:
        raise ValueError("EMPTY_PRIMITIVE_REGISTRY")
    # The ten QBS seeds are policy priors, not ten extra grid operators.
    family_order: list[str] = []
    for policy in policies:
        for family in policy.preferred_families:
            if family not in family_order:
                family_order.append(family)
    policy = SeedPolicy(
        seed_id=f"{track}:aggregate",
        idea="Aggregate policy prior over supplied QBS seeds",
        preferred_families=tuple(family_order),
        compatibility_tags=tuple(sorted({t for p in policies for t in p.compatibility_tags})),
        weight=sum(p.weight for p in policies) / len(policies),
    )
    ordered = _rank_actions(actions, policy)
    rng = random.Random(limits.seed)
    root = _Node(initial, ())
    cache: dict[Grid, float] = {initial: score(initial)}
    best: dict[tuple[str, ...], Candidate] = {
        (): Candidate(cache[initial], (), initial)
    }
    expanded = pruned = 0

    for _ in range(limits.iterations):
        node = root
        # Selection
        while node.children and len(node.program) < limits.max_depth:
            node = max(node.children, key=lambda n: n.uct(node.visits, limits.exploration))
            if len(node.program) >= limits.max_depth:
                break

        if len(node.program) < limits.max_depth:
            allowed = min(
                len(ordered),
                max(limits.initial_width,
                    int(limits.widening_k * max(1, node.visits) ** limits.widening_alpha)),
            )
            # Progressive widening: expose only the next ranked actions.
            if node.next_action < allowed:
                primitive = ordered[node.next_action]
                node.next_action += 1
                try:
                    child_grid = primitive.apply(node.grid)
                    if child_grid == node.grid:
                        pruned += 1
                    else:
                        child = _Node(child_grid, node.program + (primitive.spec.name,), node)
                        node.children.append(child)
                        expanded += 1
                        node = child
                except (ValueError, IndexError, TypeError):
                    pruned += 1

        if node.grid not in cache:
            if len(cache) >= limits.candidate_cap:
                break
            cache[node.grid] = score(node.grid)
        value = cache[node.grid]
        best[node.program] = Candidate(value, node.program, node.grid)
        current: _Node | None = node
        reward = max(0.0, min(1.0, value))
        while current is not None:
            current.visits += 1
            current.total += reward
            current = current.parent

    ranked = sorted(best.values(), key=lambda c: (c.score, -len(c.program), c.program), reverse=True)
    return TrackResult(track, tuple(ranked), len(cache), expanded, pruned)


def run_dual_tracks(initial: Grid, registry: PrimitiveRegistry,
                    seeds: tuple[SeedPolicy, ...], hybrids: tuple[SeedPolicy, ...],
                    score: ScoreFn, limits: SearchLimits) -> tuple[TrackResult, TrackResult]:
    if len(seeds) != 10:
        raise ValueError(f"EXPECTED_TEN_QBS_SEEDS:actual={len(seeds)}")
    if not hybrids:
        raise ValueError("NO_COMPATIBLE_QBS_HYBRIDS")
    # Independent RNG seeds and budgets keep the tracks experimentally separable.
    a_limits = SearchLimits(**{**limits.__dict__, "seed": limits.seed})
    b_limits = SearchLimits(**{**limits.__dict__, "seed": limits.seed + 1})
    seed_track = run_mcts_track(initial, registry, seeds, score, a_limits, "seed")
    hybrid_track = run_mcts_track(initial, registry, hybrids, score, b_limits, "hybrid")
    return seed_track, hybrid_track

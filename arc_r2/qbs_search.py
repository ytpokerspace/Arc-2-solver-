"""QBS-guided, dual-track, progressively widened MCTS research scaffold.

Search states contain every training input so a candidate is evaluated
consistently across the complete task, not just one example.
"""
from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Iterable

from .grid import Grid
from .primitive_engine import Primitive, PrimitiveRegistry

ScoreFn = Callable[["TaskState"], float]
StateGuard = Callable[["TaskState"], bool]


@dataclass(frozen=True)
class TaskState:
    grids: tuple[Grid, ...]

    def __post_init__(self) -> None:
        if not self.grids:
            raise ValueError("TASK_STATE_REQUIRES_AT_LEAST_ONE_GRID")
        if any(not g or not g[0] for g in self.grids):
            raise ValueError("EMPTY_GRID_IN_TASK_STATE")
        if any(any(len(row) != len(g[0]) for row in g) for g in self.grids):
            raise ValueError("NON_RECTANGULAR_GRID_IN_TASK_STATE")

    def apply(self, primitive: Primitive) -> "TaskState":
        return TaskState(tuple(primitive.apply(grid) for grid in self.grids))


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
    state: TaskState


@dataclass
class _Node:
    state: TaskState
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


def _rank_actions(actions: tuple[Primitive, ...], policies: tuple[SeedPolicy, ...]) -> tuple[Primitive, ...]:
    family_votes: dict[str, float] = defaultdict(float)
    first_seen: dict[str, int] = {}
    for policy in policies:
        for rank, family in enumerate(policy.preferred_families):
            family_votes[family] += policy.weight / (rank + 1)
            first_seen.setdefault(family, len(first_seen))
    return tuple(sorted(actions, key=lambda p: (
        -family_votes.get(p.spec.family, 0.0),
        first_seen.get(p.spec.family, len(first_seen) + 1),
        p.spec.cost,
        p.spec.name,
    )))


def run_mcts_track(initial: TaskState, registry: PrimitiveRegistry,
                   policies: tuple[SeedPolicy, ...], score: ScoreFn,
                   limits: SearchLimits, track: str,
                   state_guard: StateGuard | None = None) -> TrackResult:
    if not policies:
        raise ValueError("AT_LEAST_ONE_SEED_POLICY_REQUIRED")
    if limits.iterations <= 0 or limits.max_depth < 1 or limits.candidate_cap < 1:
        raise ValueError("INVALID_SEARCH_LIMITS")

    actions = registry.enabled()
    if not actions:
        raise ValueError("EMPTY_PRIMITIVE_REGISTRY")
    ordered = _rank_actions(actions, policies)
    rng = random.Random(limits.seed)
    root = _Node(initial, ())
    cache: dict[TaskState, float] = {initial: score(initial)}
    best: dict[tuple[str, ...], Candidate] = {
        (): Candidate(cache[initial], (), initial)
    }
    expanded = pruned = 0

    for _ in range(limits.iterations):
        node = root
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
            if node.next_action < allowed:
                primitive = ordered[node.next_action]
                node.next_action += 1
                try:
                    if not all(primitive.precondition is None or primitive.precondition(g)
                               for g in node.state.grids):
                        pruned += 1
                    else:
                        child_state = node.state.apply(primitive)
                        if child_state == node.state or (state_guard and not state_guard(child_state)):
                            pruned += 1
                        elif child_state in cache:
                            # Equivalent full-task states need not create duplicate subtrees.
                            pruned += 1
                        else:
                            child = _Node(child_state, node.program + (primitive.spec.name,), node)
                            node.children.append(child)
                            expanded += 1
                            node = child
                except (ValueError, IndexError, TypeError):
                    pruned += 1

        if node.state not in cache:
            if len(cache) >= limits.candidate_cap:
                break
            cache[node.state] = score(node.state)
        value = cache[node.state]
        best[node.program] = Candidate(value, node.program, node.state)
        current: _Node | None = node
        reward = max(0.0, min(1.0, value))
        while current is not None:
            current.visits += 1
            current.total += reward
            current = current.parent

    ranked = sorted(best.values(), key=lambda c: (c.score, -len(c.program), c.program), reverse=True)
    return TrackResult(track, tuple(ranked), len(cache), expanded, pruned)


def run_dual_tracks(initial: TaskState, registry: PrimitiveRegistry,
                    seeds: tuple[SeedPolicy, ...], hybrids: tuple[SeedPolicy, ...],
                    score: ScoreFn, limits: SearchLimits,
                    state_guard: StateGuard | None = None) -> tuple[TrackResult, TrackResult]:
    if len(seeds) != 10:
        raise ValueError(f"EXPECTED_TEN_QBS_SEEDS:actual={len(seeds)}")
    if not hybrids:
        raise ValueError("NO_COMPATIBLE_QBS_HYBRIDS")
    a_limits = SearchLimits(**{**limits.__dict__, "seed": limits.seed})
    b_limits = SearchLimits(**{**limits.__dict__, "seed": limits.seed + 1})
    seed_track = run_mcts_track(initial, registry, seeds, score, a_limits, "seed", state_guard)
    hybrid_track = run_mcts_track(initial, registry, hybrids, score, b_limits, "hybrid", state_guard)
    return seed_track, hybrid_track


def beam_rerank(results: tuple[TrackResult, ...], registry: PrimitiveRegistry,
                beam_width: int = 32) -> tuple[Candidate, ...]:
    """Merge both tracks, deduplicate equivalent task states, preserve family diversity."""
    if beam_width < 1:
        raise ValueError("BEAM_WIDTH_MUST_BE_POSITIVE")
    unique: dict[TaskState, Candidate] = {}
    for result in results:
        for candidate in result.candidates:
            prior = unique.get(candidate.state)
            if prior is None or (candidate.score, -len(candidate.program)) > (
                prior.score, -len(prior.program)
            ):
                unique[candidate.state] = candidate

    ranked = sorted(
        unique.values(),
        key=lambda c: (c.score, -len(c.program), c.program),
        reverse=True,
    )
    selected: list[Candidate] = []
    seen_families: set[str] = set()
    deferred: list[Candidate] = []
    for candidate in ranked:
        family = registry.get(candidate.program[0]).spec.family if candidate.program else "identity"
        if family not in seen_families:
            selected.append(candidate)
            seen_families.add(family)
        else:
            deferred.append(candidate)
        if len(selected) == beam_width:
            return tuple(selected)
    selected.extend(deferred[:max(0, beam_width - len(selected))])
    return tuple(selected)

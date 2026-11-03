"""QBS-guided predictive pair-matching MCTS for ARC-R2.

The search first scores ten primitive hypotheses against every training pair,
then performs one bounded hybrid round from those seeds. MCTS continues from
that informed tree. Exact training matches are preserved as resolved evidence.
All probability values are smoothed training-fit estimates, not calibrated
probabilities of unseen-task success.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path

from .primitives import (
    Grid, Primitive, Program, as_grid, generate_primitives,
)
from .single_stage_memory import SingleStageMemory


@dataclass
class SearchNode:
    operations: tuple[Primitive, ...]
    predictions: tuple[Grid, ...]
    solved: frozenset[int]
    score: float
    probability_estimate: float
    feasibility: float
    training_pair_success_rate: float
    alignment_score: float
    parent: "SearchNode | None" = None
    children: list["SearchNode"] = field(default_factory=list)
    visits: int = 0
    value_sum: float = 0.0
    expanded: bool = False

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else self.score


@dataclass(frozen=True)
class PredictiveSearchResult:
    program: Program | None
    exact_fit: bool
    solved_pairs: int
    total_pairs: int
    iterations: int
    nodes_created: int
    first_hypotheses: tuple[str, ...]
    hybrid_candidates: tuple[str, ...]
    memory_hit: bool = False
    memory_saved: bool = False
    hypotheses_scored: int = 0
    hybrid_seeds: int = 0
    hybrid_nodes_created: int = 0
    best_probability_estimate: float = 0.0
    best_feasibility: float = 0.0
    best_training_pair_success_rate: float = 0.0
    best_alignment_score: float = 0.0
    duplicate_states_pruned: int = 0


def _distance(a: Grid, b: Grid) -> int:
    """Cell mismatch count, including any grid-size difference."""
    height = max(len(a), len(b))
    width = max(len(a[0]), len(b[0]))
    mismatch = 0
    for r in range(height):
        for c in range(width):
            av = a[r][c] if r < len(a) and c < len(a[0]) else None
            bv = b[r][c] if r < len(b) and c < len(b[0]) else None
            mismatch += av != bv
    return mismatch


def _fingerprint(pairs: tuple[tuple[Grid, Grid], ...]) -> str:
    payload = [[[list(row) for row in source], [list(row) for row in target]]
               for source, target in pairs]
    return hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _node_rank(node: SearchNode) -> tuple:
    """Exact pair coverage first; soft signals only break ties."""
    return (
        len(node.solved),
        node.probability_estimate,
        node.feasibility,
        node.alignment_score,
        -len(node.operations),
        tuple(str(op) for op in node.operations),
    )


def _evaluate(
    operations: tuple[Primitive, ...],
    pairs: tuple[tuple[Grid, Grid], ...],
    previously_solved: frozenset[int],
) -> SearchNode | None:
    predictions = []
    solved = set()
    alignments = []
    for index, (source, target) in enumerate(pairs):
        try:
            predicted = Program(operations).apply(source)
        except (ValueError, IndexError, TypeError):
            # A candidate that cannot execute on every training input is infeasible.
            return None
        if index in previously_solved and predicted != target:
            # A later hybrid step may never destroy a solved training pair.
            return None
        predictions.append(predicted)
        if predicted == target:
            solved.add(index)
        denom = max(
            1,
            max(len(predicted) * len(predicted[0]), len(target) * len(target[0])),
        )
        alignments.append(max(0.0, 1.0 - (_distance(predicted, target) / denom)))

    success_rate = len(solved) / len(pairs)
    # Beta(1,1) smoothing of observed exact training-pair matches.
    # This is not a calibrated probability of unseen-task success.
    probability_estimate = (len(solved) + 1) / (len(pairs) + 2)
    feasibility = 1.0  # The candidate executed on every supplied training input.
    alignment_score = sum(alignments) / len(alignments)
    # Exact pair success is the strongest signal; alignment helps order partial fits.
    score = (
        0.40 * success_rate
        + 0.25 * probability_estimate
        + 0.20 * alignment_score
        + 0.15 * feasibility
    )
    return SearchNode(
        operations, tuple(predictions), frozenset(solved), score,
        probability_estimate, feasibility, success_rate, alignment_score,
    )


def _expand(
    node: SearchNode,
    pairs: tuple[tuple[Grid, Grid], ...],
    branch_width: int,
    max_depth: int,
) -> tuple[int, int]:
    node.expanded = True
    unresolved = [i for i in range(len(pairs)) if i not in node.solved]
    if not unresolved or len(node.operations) >= max_depth:
        return 0, 0

    current_grids = tuple(node.predictions[i] for i in unresolved)
    targets = tuple(pairs[i][1] for i in unresolved)
    primitives = generate_primitives(current_grids, targets)
    ranked: list[SearchNode] = []
    seen_predictions: set[tuple[Grid, ...]] = set()
    duplicates_pruned = 0

    for operation in primitives:
        operations = node.operations + (operation,)
        child = _evaluate(operations, pairs, node.solved)
        if child is None:
            continue
        # Equivalent predictions on all training pairs are one search state.
        if child.predictions in seen_predictions:
            duplicates_pruned += 1
            continue
        seen_predictions.add(child.predictions)
        child.parent = node
        ranked.append(child)

    ranked.sort(key=lambda item: _node_rank(item), reverse=True)
    node.children = ranked[:branch_width]
    return len(node.children), duplicates_pruned


def _select(node: SearchNode, exploration: float) -> SearchNode:
    current = node
    while current.expanded and current.children:
        parent_visits = max(1, current.visits)
        current = max(
            current.children,
            key=lambda child: (
                child.mean_value
                + exploration * math.sqrt(
                    math.log(parent_visits + 1) / (child.visits + 1)
                ),
                child.score,
                tuple(str(op) for op in child.operations),
            ),
        )
    return current


def _backpropagate(node: SearchNode, reward: float) -> None:
    current: SearchNode | None = node
    while current is not None:
        current.visits += 1
        current.value_sum += reward
        current = current.parent


def search(
    examples: tuple[tuple[Grid, Grid], ...],
    *,
    iterations: int = 200,
    branch_width: int = 10,
    max_depth: int = 3,
    exploration: float = 1.1,
    memory_path: str | Path | None = None,
    hypothesis_count: int = 10,
    hybrid_branch_width: int = 5,
) -> PredictiveSearchResult:
    """Run ten-hypothesis scoring, one hybrid seed round, then predictive MCTS."""
    if not examples:
        raise ValueError("at least one training pair is required")
    if iterations < 1 or branch_width < 1 or max_depth < 1:
        raise ValueError("iterations, branch_width, and max_depth must be positive")
    if max_depth > 4:
        raise ValueError("max_depth must be <= 4")
    if hypothesis_count < 2 or hybrid_branch_width < 1:
        raise ValueError("hypothesis_count must be >= 2 and hybrid_branch_width positive")

    pairs = tuple((as_grid(a), as_grid(b)) for a, b in examples)
    memory = SingleStageMemory(memory_path) if memory_path else None
    fingerprint = _fingerprint(pairs)

    if memory is not None:
        cached = memory.load(fingerprint)
        if cached is not None:
            try:
                remembered = Program(tuple(Primitive(name, args) for name, args in cached))
                if all(remembered.apply(source) == target for source, target in pairs):
                    return PredictiveSearchResult(
                        remembered, True, len(pairs), len(pairs), 0, 0,
                        (), (), memory_hit=True,
                        best_probability_estimate=(len(pairs) + 1) / (len(pairs) + 2),
                        best_feasibility=1.0,
                        best_training_pair_success_rate=1.0,
                        best_alignment_score=1.0,
                    )
            except (ValueError, IndexError, TypeError):
                pass

    root = _evaluate((), pairs, frozenset())
    assert root is not None
    nodes_created = 1
    hypotheses_scored = 1
    hybrid_seeds_used = 0
    hybrid_nodes_created = 0
    duplicate_states_pruned = 0
    first_hypotheses: list[str] = []
    hybrid_candidates: list[str] = []
    best: SearchNode | None = None
    used_iterations = 0
    hybrid_seeded = False

    def consider(candidate: SearchNode) -> None:
        nonlocal best
        if best is None or _node_rank(candidate) > _node_rank(best):
            best = candidate

    for _ in range(iterations):
        if best is not None and len(best.solved) == len(pairs):
            break

        leaf = _select(root, exploration)
        created, pruned = _expand(leaf, pairs, branch_width, max_depth)
        nodes_created += created
        hypotheses_scored += created
        duplicate_states_pruned += pruned

        if leaf is root and not hybrid_seeded:
            hybrid_seeded = True
            seeds = leaf.children[:hypothesis_count]
            hybrid_seeds_used = len(seeds)
            first_hypotheses = [
                (
                    f"{child.operations[-1]} | score={child.score:.3f} "
                    f"p_est={child.probability_estimate:.3f} "
                    f"feasible={child.feasibility:.1f} "
                    f"train_exact_rate={child.training_pair_success_rate:.3f} "
                    f"alignment={child.alignment_score:.3f}"
                )
                for child in seeds
            ]
            # One QBS-style hybrid round: expand each of the top ten ideas once.
            for seed in seeds:
                if len(seed.operations) >= max_depth:
                    continue
                hybrid_created, hybrid_pruned = _expand(
                    seed, pairs, hybrid_branch_width, max_depth
                )
                hybrid_nodes_created += hybrid_created
                nodes_created += hybrid_created
                hypotheses_scored += hybrid_created
                duplicate_states_pruned += hybrid_pruned
                for child in seed.children:
                    label = (
                        f"{' -> '.join(str(op) for op in child.operations)} "
                        f"| score={child.score:.3f} "
                        f"p_est={child.probability_estimate:.3f} "
                        f"feasible={child.feasibility:.1f} "
                        f"train_exact_rate={child.training_pair_success_rate:.3f} "
                        f"alignment={child.alignment_score:.3f}"
                    )
                    if label not in hybrid_candidates:
                        hybrid_candidates.append(label)
                    consider(child)

        if leaf.children:
            for child in leaf.children:
                _backpropagate(child, child.score)
                consider(child)
                if len(child.solved) == len(pairs):
                    break
        else:
            _backpropagate(leaf, leaf.score)

        used_iterations += 1
        if best is not None and len(best.solved) == len(pairs):
            break

    winner = Program(best.operations) if best is not None else None
    exact = winner is not None and all(
        winner.apply(source) == target for source, target in pairs
    )
    memory_saved = False
    if exact and memory is not None and winner is not None:
        memory.save(fingerprint, tuple((op.name, op.args) for op in winner.operations))
        memory_saved = True

    return PredictiveSearchResult(
        winner if exact else None, bool(exact),
        len(pairs) if exact else (len(best.solved) if best else 0),
        len(pairs), used_iterations, nodes_created,
        tuple(first_hypotheses), tuple(hybrid_candidates[:100]),
        memory_saved=memory_saved,
        hypotheses_scored=hypotheses_scored,
        hybrid_seeds=hybrid_seeds_used,
        hybrid_nodes_created=hybrid_nodes_created,
        best_probability_estimate=best.probability_estimate if best else 0.0,
        best_feasibility=best.feasibility if best else 0.0,
        best_training_pair_success_rate=best.training_pair_success_rate if best else 0.0,
        best_alignment_score=best.alignment_score if best else 0.0,
        duplicate_states_pruned=duplicate_states_pruned,
    )

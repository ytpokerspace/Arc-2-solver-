"""Predictive pair-matching MCTS for ARC-R2.

Every node is an executable primitive chain. Exact training-pair matches become
resolved evidence: later operations must preserve them, and their distance no
longer contributes to search scoring. A complete solution is accepted only when
the final program reproduces every training output exactly.
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
from .grid_compression import encode_grid


@dataclass
class SearchNode:
    operations: tuple[Primitive, ...]
    predictions: tuple[Grid, ...]
    solved: frozenset[int]
    score: float
    probability_estimate: float
    feasibility: float
    training_pair_success_rate: float
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


def _evaluate(
    operations: tuple[Primitive, ...],
    pairs: tuple[tuple[Grid, Grid], ...],
    previously_solved: frozenset[int],
) -> SearchNode | None:
    predictions = []
    solved = set()
    for index, (source, target) in enumerate(pairs):
        try:
            predicted = Program(operations).apply(source)
        except (ValueError, IndexError, TypeError):
            return None
        # Once a pair is solved, a later hybrid step may not break that match.
        if index in previously_solved and predicted != target:
            return None
        predictions.append(predicted)
        if predicted == target:
            solved.add(index)

    unresolved = [i for i in range(len(pairs)) if i not in solved]
    if unresolved:
        similarities = []
        for i in unresolved:
            target = pairs[i][1]
            denom = max(1, max(len(predictions[i]) * len(predictions[i][0]),
                               len(target) * len(target[0])))
            similarities.append(1.0 - (_distance(predictions[i], target) / denom))
        progress = sum(similarities) / len(similarities)
    else:
        progress = 1.0
    # These are transparent search heuristics, not calibrated generalization probabilities.
    probability_estimate = max(0.0, min(1.0, progress))
    feasibility = 1.0  # Every operation in this node executed successfully.
    training_pair_success_rate = len(solved) / len(pairs)
    score = (0.40 * probability_estimate + 0.35 * feasibility
             + 0.25 * training_pair_success_rate)
    return SearchNode(
        operations, tuple(predictions), frozenset(solved), score,
        probability_estimate, feasibility, training_pair_success_rate,
    )


def _expand(node: SearchNode, pairs: tuple[tuple[Grid, Grid], ...],
            branch_width: int) -> int:
    node.expanded = True
    unresolved = [i for i in range(len(pairs)) if i not in node.solved]
    if not unresolved:
        return 0
    current_grids = tuple(node.predictions[i] for i in unresolved)
    targets = tuple(pairs[i][1] for i in unresolved)
    primitives = generate_primitives(current_grids, targets)
    ranked: list[tuple[float, str, SearchNode]] = []
    for operation in primitives:
        operations = node.operations + (operation,)
        child = _evaluate(operations, pairs, node.solved)
        if child is None:
            continue
        # Keep temporarily unhelpful steps: some valid transformations need
        # an intermediate operation that only becomes useful after hybridization.
        child.parent = node
        ranked.append((child.score, str(operation), child))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    # Collapse chains that produce identical predictions on every training pair.
    # The compact codec is lossless and is used only as an exact state key.
    unique = []
    seen_states = set()
    duplicates_pruned = 0
    for item in ranked:
        child = item[2]
        state_key = tuple(encode_grid(grid) for grid in child.predictions)
        if state_key in seen_states:
            duplicates_pruned += 1
            continue
        seen_states.add(state_key)
        unique.append(child)
        if len(unique) >= branch_width:
            break
    node.children = unique
    return len(node.children), duplicates_pruned


def _select(node: SearchNode, exploration: float) -> SearchNode:
    current = node
    while current.expanded and current.children:
        parent_visits = max(1, current.visits)
        current = max(
            current.children,
            key=lambda child: (
                child.mean_value
                + exploration * math.sqrt(math.log(parent_visits + 1) / (child.visits + 1)),
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
) -> PredictiveSearchResult:
    """Search executable hybrid programs; exact pair matches are terminal evidence."""
    if not examples:
        raise ValueError("at least one training pair is required")
    if iterations < 1 or branch_width < 1 or max_depth < 1:
        raise ValueError("iterations, branch_width, and max_depth must be positive")
    if max_depth > 4:
        raise ValueError("max_depth must be <= 4")
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
                    )
            except (ValueError, IndexError, TypeError):
                pass

    root = _evaluate((), pairs, frozenset())
    assert root is not None
    nodes_created = 1
    duplicate_states_pruned = 0
    first_hypotheses: list[str] = []
    hybrid_candidates: list[str] = []
    best: SearchNode | None = None
    used_iterations = 0

    for step in range(iterations):
        if best is not None and len(best.solved) == len(pairs):
            break
        leaf = _select(root, exploration)
        if len(leaf.operations) >= max_depth or len(leaf.solved) == len(pairs):
            _backpropagate(leaf, leaf.score)
            used_iterations += 1
            continue
        created, pruned = _expand(leaf, pairs, branch_width)
        nodes_created += created
        duplicate_states_pruned += pruned
        if leaf is root:
            first_hypotheses = [
                (f"{child.operations[0]} | priority={child.score:.3f} "
                 f"p_est={child.probability_estimate:.3f} "
                 f"feasible={child.feasibility:.1f} "
                 f"train_exact_rate={child.training_pair_success_rate:.3f}")
                for child in leaf.children[:10]
            ]
        else:
            for child in leaf.children:
                label = (
                    f"{' -> '.join(str(op) for op in child.operations)} "
                    f"| priority={child.score:.3f} "
                    f"p_est={child.probability_estimate:.3f} "
                    f"feasible={child.feasibility:.1f} "
                    f"train_exact_rate={child.training_pair_success_rate:.3f}"
                )
                if label not in hybrid_candidates:
                    hybrid_candidates.append(label)
        if leaf.children:
            for child in leaf.children:
                _backpropagate(child, child.score)
                if best is None or (len(child.solved), child.score,
                                    -len(child.operations)) > (
                                        len(best.solved), best.score, -len(best.operations)
                                    ):
                    best = child
                if len(child.solved) == len(pairs):
                    best = child
                    break
        else:
            _backpropagate(leaf, leaf.score)
        used_iterations += 1
        if best is not None and len(best.solved) == len(pairs):
            break

    winner = Program(best.operations) if best is not None else None
    exact = winner is not None and all(winner.apply(source) == target for source, target in pairs)
    memory_saved = False
    if exact and memory is not None and winner is not None:
        memory.save(fingerprint, tuple((op.name, op.args) for op in winner.operations))
        memory_saved = True
    return PredictiveSearchResult(
        winner if exact else None, bool(exact),
        len(pairs) if exact else (len(best.solved) if best else 0),
        len(pairs), used_iterations, nodes_created,
        tuple(first_hypotheses), tuple(hybrid_candidates[:100]),
        memory_saved=memory_saved, duplicate_states_pruned=duplicate_states_pruned,
    )

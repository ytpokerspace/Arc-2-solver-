"""Hybrid candidate coordinator and training-example evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from .dsl import Program
from .grid import Grid
from .objects import detect_objects
from .relations import relation_graph
from .search import SearchLimits, SearchStats, beam_search, mcts_search
from .quantum_brainstorming import quantum_brainstorm_search

@dataclass(frozen=True)
class Example:
    input: Grid
    output: Grid

@dataclass
class SolveResult:
    program: Program
    score: float
    exact_examples: int
    total_examples: int
    candidates_evaluated: int
    beam_candidates: int
    mcts_candidates: int
    qbs_rounds: int = 0
    qbs_probability_estimate: float = 0.0
    qbs_feasibility: float = 0.0

def score_program(program: Program, examples: tuple[Example, ...], use_object_signal: bool = True) -> float:
    """Score only executable candidates; invalid programs receive zero."""
    if not examples:
        return 0.0
    predictions = []
    try:
        predictions = [program.apply(ex.input) for ex in examples]
    except Exception:
        return 0.0

    exact = sum(predicted == ex.output for predicted, ex in zip(predictions, examples))
    # Exact match is the primary objective; a small cell agreement signal helps rank near misses.
    cell_total = 0
    cell_match = 0
    object_total = 0
    object_match = 0
    relation_scores = []
    for ex, predicted in zip(examples, predictions):
        if len(predicted) != len(ex.output) or len(predicted[0]) != len(ex.output[0]):
            continue
        for pr, tr in zip(predicted, ex.output):
            cell_total += len(tr)
            cell_match += sum(a == b for a, b in zip(pr, tr))
        predicted_objects = detect_objects(predicted)
        target_objects = detect_objects(ex.output)
        object_total += 1
        object_match += (len(predicted_objects) == len(target_objects))
        predicted_rel = [label for _, _, labels in relation_graph(predicted_objects) for label in labels]
        target_rel = [label for _, _, labels in relation_graph(target_objects) for label in labels]
        from collections import Counter
        pc, tc = Counter(predicted_rel), Counter(target_rel)
        union = sum((pc | tc).values())
        relation_scores.append(sum((pc & tc).values()) / union if union else 1.0)
    cell_score = cell_match / cell_total if cell_total else 0.0
    object_score = object_match / object_total if object_total else 0.0
    if not use_object_signal:
        return 0.80 * exact / len(examples) + 0.20 * cell_score
    relation_score = sum(relation_scores) / len(relation_scores) if relation_scores else 0.0
    return (0.80 * exact / len(examples) + 0.12 * cell_score
            + 0.05 * object_score + 0.03 * relation_score)

class HybridSolver:
    def __init__(self, limits: SearchLimits | None = None, strategy: str = "hybrid", use_object_signal: bool = True):
        if strategy not in {"hybrid", "beam", "mcts", "identity", "qbs"}:
            raise ValueError("strategy must be hybrid, beam, mcts, identity, or qbs")
        self.limits = limits or SearchLimits()
        self.strategy = strategy
        self.use_object_signal = use_object_signal

    def solve(self, examples: tuple[Example, ...]) -> SolveResult:
        if not examples:
            raise ValueError("at least one training example is required")
        beam_stats, mcts_stats = SearchStats(), SearchStats()
        scorer = lambda p: score_program(p, examples, self.use_object_signal)
        state_key = lambda p: tuple(p.apply(ex.input) for ex in examples)
        beam = beam_search(scorer, self.limits, beam_stats, state_key=state_key) if self.strategy in {"hybrid", "beam"} else []
        if self.strategy == "qbs":
            beam = quantum_brainstorm_search(
                examples, self.limits, scorer, beam_stats, rounds=10, elite_size=4
            )
        remaining = max(0, self.limits.candidate_cap - beam_stats.evaluated)
        mlimits = SearchLimits(
            beam_width=self.limits.beam_width, max_depth=self.limits.max_depth,
            candidate_cap=remaining, mcts_iterations=self.limits.mcts_iterations,
            exploration=self.limits.exploration, seed=self.limits.seed,
            include_prte=self.limits.include_prte,
        )
        mcts = mcts_search(scorer, mlimits, mcts_stats, state_key=state_key) if remaining and self.strategy in {"hybrid", "mcts"} else []
        if self.strategy == "identity":
            beam = [(scorer(Program()), Program())]
            beam_stats.evaluated = 1
        combined = {p.canonical(): (s, p) for s, p in beam + mcts}
        if not combined:
            combined[()] = (scorer(Program()), Program())
        score, program = max(
            combined.values(),
            key=lambda item: (
                sum(item[1].apply(ex.input) == ex.output for ex in examples),
                item[0],
                -item[1].complexity,
            ),
        )
        exact = sum(program.apply(ex.input) == ex.output for ex in examples)
        return SolveResult(program, score, exact, len(examples),
                           beam_stats.evaluated + mcts_stats.evaluated,
                           len(beam), len(mcts), beam_stats.qbs_rounds,
                           (exact + 1) / (len(examples) + 2) if self.strategy == "qbs" else 0.0,
                           1.0 if self.strategy == "qbs" else 0.0)

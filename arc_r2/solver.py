"""Hybrid candidate coordinator and training-example evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from .dsl import Program
from .grid import Grid
from .objects import detect_objects
from .search import SearchLimits, SearchStats, beam_search, mcts_search

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

def score_program(program: Program, examples: tuple[Example, ...], use_object_signal: bool = True) -> float:
    if not examples: return 0.0
    exact = sum(program.apply(ex.input) == ex.output for ex in examples)
    # Exact match is the primary objective; a small cell agreement signal helps rank near misses.
    cell_total = 0
    cell_match = 0
    object_total = 0
    object_match = 0
    for ex in examples:
        try: predicted = program.apply(ex.input)
        except Exception: continue
        if len(predicted) != len(ex.output) or len(predicted[0]) != len(ex.output[0]):
            continue
        for pr, tr in zip(predicted, ex.output):
            cell_total += len(tr)
            cell_match += sum(a == b for a, b in zip(pr, tr))
        object_total += 1
        object_match += (len(detect_objects(predicted)) == len(detect_objects(ex.output)))
    cell_score = cell_match / cell_total if cell_total else 0.0
    object_score = object_match / object_total if object_total else 0.0
    if not use_object_signal:
        return 0.80 * exact / len(examples) + 0.20 * cell_score
    return (0.80 * exact / len(examples) + 0.15 * cell_score
            + 0.05 * object_score)

class HybridSolver:
    def __init__(self, limits: SearchLimits | None = None, strategy: str = "hybrid", use_object_signal: bool = True):
        if strategy not in {"hybrid", "beam", "mcts", "identity"}:
            raise ValueError("strategy must be hybrid, beam, mcts, or identity")
        self.limits = limits or SearchLimits()
        self.strategy = strategy
        self.use_object_signal = use_object_signal

    def solve(self, examples: tuple[Example, ...]) -> SolveResult:
        if not examples:
            raise ValueError("at least one training example is required")
        beam_stats, mcts_stats = SearchStats(), SearchStats()
        scorer = lambda p: score_program(p, examples, self.use_object_signal)
        beam = beam_search(scorer, self.limits, beam_stats) if self.strategy in {"hybrid", "beam"} else []
        remaining = max(0, self.limits.candidate_cap - beam_stats.evaluated)
        mlimits = SearchLimits(
            beam_width=self.limits.beam_width, max_depth=self.limits.max_depth,
            candidate_cap=remaining, mcts_iterations=self.limits.mcts_iterations,
            exploration=self.limits.exploration, seed=self.limits.seed,
        )
        mcts = mcts_search(scorer, mlimits, mcts_stats) if remaining and self.strategy in {"hybrid", "mcts"} else []
        if self.strategy == "identity":
            beam = [(scorer(Program()), Program())]
        combined = {p.canonical(): (s, p) for s, p in beam + mcts}
        if not combined:
            combined[()] = (scorer(Program()), Program())
        score, program = max(combined.values(), key=lambda item: (item[0], -item[1].complexity))
        exact = sum(program.apply(ex.input) == ex.output for ex in examples)
        return SolveResult(program, score, exact, len(examples),
                           beam_stats.evaluated + mcts_stats.evaluated,
                           len(beam), len(mcts))

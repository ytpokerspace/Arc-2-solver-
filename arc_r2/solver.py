"""Hybrid candidate coordinator and training-example evaluator."""
from __future__ import annotations
from dataclasses import dataclass
from .dsl import Program
from .grid import Grid
from .objects import detect_objects
from .relations import relation_graph
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
    hian_report: dict | None = None

def score_program(program: Program, examples: tuple[Example, ...], use_object_signal: bool = True) -> float:
    if not examples: return 0.0
    exact = sum(program.apply(ex.input) == ex.output for ex in examples)
    # Exact match is the primary objective; a small cell agreement signal helps rank near misses.
    cell_total = 0
    cell_match = 0
    object_total = 0
    object_match = 0
    relation_scores = []
    for ex in examples:
        try: predicted = program.apply(ex.input)
        except Exception: continue
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
    def __init__(self, limits: SearchLimits | None = None, strategy: str = "hybrid",
                 use_object_signal: bool = True, use_hian: bool = False):
        if strategy not in {"hybrid", "beam", "mcts", "identity"}:
            raise ValueError("strategy must be hybrid, beam, mcts, or identity")
        self.limits = limits or SearchLimits()
        self.strategy = strategy
        self.use_object_signal = use_object_signal
        self.use_hian = use_hian

    def solve(self, examples: tuple[Example, ...]) -> SolveResult:
        if not examples:
            raise ValueError("at least one training example is required")
        beam_stats, mcts_stats = SearchStats(), SearchStats()
        scorer = lambda p: score_program(p, examples, self.use_object_signal)
        state_key = lambda p: tuple(p.apply(ex.input) for ex in examples)
        beam = beam_search(scorer, self.limits, beam_stats, state_key=state_key) if self.strategy in {"hybrid", "beam"} else []
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

        hian_report = None
        if self.use_hian:
            from .hian import analyze_hypotheses
            evidence_examples = tuple(
                (f"train-{index}", example.input, example.output)
                for index, example in enumerate(examples)
            )
            hian_report = analyze_hypotheses(
                (program for _, program in combined.values()),
                evidence_examples,
            )
            selected = tuple(hian_report["selected_hypothesis"]["program"])
            score, program = combined[selected]
        else:
            score, program = max(
                combined.values(),
                key=lambda item: (item[0], -item[1].complexity),
            )

        exact = sum(program.apply(ex.input) == ex.output for ex in examples)
        return SolveResult(program, score, exact, len(examples),
                           beam_stats.evaluated + mcts_stats.evaluated,
                           len(beam), len(mcts), hian_report)

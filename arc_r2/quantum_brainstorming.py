"""Quantum Brainstorming search over ARC-R2 DSL programs.

The search keeps a diverse elite set, creates ordered hybrids, and iterates.
Correctness on supplied training examples is ranked before heuristic score,
feasibility is checked before scoring, and candidate/round budgets are bounded.

The reported fit rate is an observed training-set fraction, NOT a calibrated
probability of solving unseen ARC tasks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .dsl import Program
from .search import SearchLimits, SearchStats
from .transforms import OPERATIONS


@dataclass(frozen=True)
class Hypothesis:
    program: Program
    fit_rate: float
    estimated_probability: float
    feasibility: float
    heuristic_score: float

    @property
    def rank(self) -> tuple[float, float, float, float]:
        # Exact training fit and executable feasibility outrank soft signals.
        return (
            float(self.fit_rate == 1.0),
            self.estimated_probability,
            self.feasibility,
            self.heuristic_score - 0.002 * self.program.complexity,
        )


def _join(left: Program, right: Program) -> Program:
    return Program(left.instructions + right.instructions)


def quantum_brainstorm_search(
    examples: tuple,
    limits: SearchLimits,
    score: Callable[[Program], float],
    stats: SearchStats | None = None,
    *,
    rounds: int = 10,
    elite_size: int = 12,
) -> list[tuple[float, Program]]:
    """Run bounded hypothesis generation, hybridization, and iterative selection.

    Each round combines elite programs in both orders. Candidates are evaluated
    only on the supplied training examples. No held-out/test target is accepted
    by this interface, preventing accidental test-label leakage.
    """
    stats = stats if stats is not None else SearchStats()
    if not examples:
        raise ValueError("at least one training example is required")
    if rounds < 1 or elite_size < 2:
        raise ValueError("rounds must be positive and elite_size at least 2")

    cache: dict[tuple[str, ...], Hypothesis] = {}

    def evaluate(program: Program) -> Hypothesis | None:
        key = program.canonical()
        if key in cache:
            return cache[key]
        if stats.evaluated >= limits.candidate_cap:
            stats.stopped_by_cap = True
            return None
        predictions = []
        try:
            for example in examples:
                predictions.append(program.apply(example.input))
        except Exception:
            stats.evaluated += 1
            stats.expanded += 1
            return None

        exact_count = sum(
            prediction == example.output
            for prediction, example in zip(predictions, examples)
        )
        fit_rate = exact_count / len(examples)
        # Beta(1, 1) smoothing: an engineering estimate from observed fit,
        # not a calibrated estimate of unseen ARC-task success.
        estimated_probability = (exact_count + 1) / (len(examples) + 2)
        try:
            heuristic = float(score(program))
        except Exception:
            heuristic = 0.0
        hypothesis = Hypothesis(program, fit_rate, estimated_probability, 1.0, heuristic)
        cache[key] = hypothesis
        stats.evaluated += 1
        stats.expanded += 1
        return hypothesis

    # Round zero: independently test every primitive and identity.
    evaluate(Program())
    operations = OPERATIONS if limits.include_prte else tuple(
        operation for operation in OPERATIONS if operation != "PRTE.POLARITY.REVERSE"
    )
    for operation in operations:
        evaluate(Program().then(operation))

    elite = sorted(cache.values(), key=lambda item: item.rank, reverse=True)[:elite_size]

    # Iteratively hybridize the best hypotheses in both operation orders.
    for _ in range(rounds):
        if stats.stopped_by_cap or len(elite) < 2:
            break
        stats.qbs_rounds += 1
        proposals: dict[tuple[str, ...], Program] = {}
        for i, left in enumerate(elite):
            for right in elite[i + 1:]:
                for hybrid in (_join(left.program, right.program),
                               _join(right.program, left.program)):
                    if hybrid.complexity <= limits.max_depth:
                        proposals.setdefault(hybrid.canonical(), hybrid)
        if not proposals:
            break

        for program in proposals.values():
            if stats.evaluated >= limits.candidate_cap:
                stats.stopped_by_cap = True
                break
            evaluate(program)

        if not cache:
            break
        next_elite = sorted(cache.values(), key=lambda item: item.rank, reverse=True)[:elite_size]
        if [h.program.canonical() for h in next_elite] == [
            h.program.canonical() for h in elite
        ]:
            # No new better hypotheses: remaining rounds would repeat work.
            elite = next_elite
            break
        elite = next_elite

    ranked = sorted(cache.values(), key=lambda item: item.rank, reverse=True)
    return [(hypothesis.heuristic_score, hypothesis.program) for hypothesis in ranked]

"""Quantum Brainstorming search over the parameterized object DSL.

This is a separate typed search lane from the grid DSL: object instructions
carry parameters (object index, offsets, axes, scale, placement), so they cannot
safely be concatenated with grid-operation strings. QBS keeps the two program
types explicit and evaluates only supplied training examples.
"""
from __future__ import annotations

from dataclasses import dataclass

from .object_dsl import (
    ObjectProgram, generate_object_instructions,
)


@dataclass(frozen=True)
class ObjectHypothesis:
    program: ObjectProgram
    exact_count: int
    fit_rate: float
    probability_estimate: float
    feasibility: float
    cell_accuracy: float

    @property
    def rank(self) -> tuple[float, float, float, float, int]:
        # Correctness is lexicographically first; complexity only breaks ties.
        return (
            float(self.exact_count == self._total),
            self.fit_rate,
            self.feasibility,
            self.cell_accuracy,
            -self.program.complexity,
        )

    _total: int = 1


@dataclass(frozen=True)
class ObjectQBSResult:
    program: ObjectProgram
    exact_examples: int
    total_examples: int
    candidates_evaluated: int
    rounds_completed: int
    stopped_by_cap: bool
    fit_rate: float
    probability_estimate: float
    feasibility: float


def _join(left: ObjectProgram, right: ObjectProgram) -> ObjectProgram:
    return ObjectProgram(left.instructions + right.instructions)


def object_quantum_brainstorm_search(
    examples: tuple,
    *,
    max_depth: int = 2,
    candidate_cap: int = 1200,
    parameter_cap: int = 1200,
    max_offset: int = 2,
    rounds: int = 10,
    elite_size: int = 4,
) -> ObjectQBSResult:
    """Generate, rank, and hybridize object-DSL hypotheses deterministically."""
    if not examples:
        raise ValueError("at least one training example is required")
    if not 1 <= max_depth <= 4:
        raise ValueError("max_depth must be in 1..4")
    if not 1 <= candidate_cap <= 10000:
        raise ValueError("candidate_cap must be in 1..10000")
    if rounds < 1 or elite_size < 2:
        raise ValueError("rounds must be positive and elite_size at least 2")

    instructions = generate_object_instructions(
        tuple(example.input for example in examples),
        parameter_cap=parameter_cap,
        max_offset=max_offset,
    )
    cache: dict[tuple, ObjectHypothesis] = {}
    stopped = False

    def evaluate(program: ObjectProgram) -> ObjectHypothesis | None:
        nonlocal stopped
        key = program.canonical()
        if key in cache:
            return cache[key]
        if len(cache) >= candidate_cap:
            stopped = True
            return None
        predictions = []
        try:
            predictions = [program.apply(example.input) for example in examples]
        except (ValueError, TypeError, IndexError):
            return None

        exact = sum(
            prediction == example.output
            for prediction, example in zip(predictions, examples)
        )
        matches = total_cells = 0
        for prediction, example in zip(predictions, examples):
            target = example.output
            if len(prediction) == len(target) and len(prediction[0]) == len(target[0]):
                for predicted_row, target_row in zip(prediction, target):
                    matches += sum(a == b for a, b in zip(predicted_row, target_row))
                    total_cells += len(target_row)
        cell_accuracy = matches / total_cells if total_cells else 0.0
        fit = exact / len(examples)
        # Smoothed observed training fit, not a calibrated unseen-task probability.
        probability = (exact + 1) / (len(examples) + 2)
        hypothesis = ObjectHypothesis(
            program, exact, fit, probability, 1.0, cell_accuracy, len(examples)
        )
        cache[key] = hypothesis
        return hypothesis

    evaluate(ObjectProgram())
    for instruction in instructions:
        if len(cache) >= candidate_cap:
            stopped = True
            break
        evaluate(ObjectProgram().then(instruction))

    elite = sorted(cache.values(), key=lambda item: item.rank, reverse=True)[:elite_size]
    rounds_completed = 0
    for _ in range(rounds):
        if stopped or len(elite) < 2:
            break
        proposals: dict[tuple, ObjectProgram] = {}
        for index, left in enumerate(elite):
            for right in elite[index + 1:]:
                for hybrid in (_join(left.program, right.program),
                               _join(right.program, left.program)):
                    if hybrid.complexity <= max_depth:
                        proposals.setdefault(hybrid.canonical(), hybrid)
        if not proposals:
            break
        before = len(cache)
        for program in proposals.values():
            if len(cache) >= candidate_cap:
                stopped = True
                break
            evaluate(program)
        rounds_completed += 1
        elite = sorted(cache.values(), key=lambda item: item.rank, reverse=True)[:elite_size]
        if len(cache) == before:
            break

    best = max(cache.values(), key=lambda item: item.rank)
    return ObjectQBSResult(
        best.program, best.exact_count, len(examples), len(cache),
        rounds_completed, stopped, best.fit_rate,
        best.probability_estimate, best.feasibility,
    )

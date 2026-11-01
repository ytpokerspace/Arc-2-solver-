"""Bounded beam search over parameterized object-level ARC programs."""
from __future__ import annotations

from dataclasses import dataclass

from .grid import Grid, as_grid
from .object_dsl import ObjectProgram, generate_object_instructions


@dataclass(frozen=True)
class ObjectExample:
    input: Grid
    output: Grid


@dataclass(frozen=True)
class ObjectSolveResult:
    program: ObjectProgram
    score: float
    exact_examples: int
    total_examples: int
    candidates_evaluated: int
    stopped_by_cap: bool


def _score(program: ObjectProgram, examples: tuple[ObjectExample, ...]) -> tuple[float, int]:
    exact = 0
    matches = 0
    total = 0
    for example in examples:
        # Count every target cell in the denominator. A failed or wrong-sized
        # prediction must not improve its score by disappearing from evaluation.
        total += sum(len(row) for row in example.output)
        try:
            predicted = program.apply(example.input)
        except (ValueError, TypeError, IndexError):
            # An instruction sequence that cannot be applied to every training
            # example is invalid as a task-level hypothesis.
            return -1.0, exact
        if predicted == example.output:
            exact += 1
        if len(predicted) == len(example.output) and len(predicted[0]) == len(example.output[0]):
            for predicted_row, target_row in zip(predicted, example.output):
                matches += sum(a == b for a, b in zip(predicted_row, target_row))
    cell_score = matches / total if total else 0.0
    return 0.90 * exact / len(examples) + 0.10 * cell_score, exact


class ObjectHybridSolver:
    """Train-example-only bounded search for object-level transforms.

    The initial implementation uses a deterministic beam. It never inspects
    test outputs, and all candidate parameters are generated within explicit caps.
    """
    def __init__(self, beam_width: int = 20, max_depth: int = 2,
                 candidate_cap: int = 1200, parameter_cap: int = 1200,
                 max_offset: int = 2):
        if not 1 <= beam_width <= 100:
            raise ValueError("beam_width must be in 1..100")
        if not 1 <= max_depth <= 4:
            raise ValueError("max_depth must be in 1..4")
        if not 1 <= candidate_cap <= 10000:
            raise ValueError("candidate_cap must be in 1..10000")
        self.beam_width = beam_width
        self.max_depth = max_depth
        self.candidate_cap = candidate_cap
        self.parameter_cap = parameter_cap
        self.max_offset = max_offset

    def solve(self, examples: tuple[ObjectExample, ...]) -> ObjectSolveResult:
        if not examples:
            raise ValueError("at least one training example is required")
        normalized = tuple(ObjectExample(as_grid(e.input), as_grid(e.output)) for e in examples)
        instructions = generate_object_instructions(
            tuple(e.input for e in normalized), self.parameter_cap, self.max_offset
        )
        identity = ObjectProgram()
        identity_score, identity_exact = _score(identity, normalized)
        best = (identity_score, identity)
        frontier = [identity]
        seen = {identity.canonical()}
        evaluated = 1
        stopped = False

        for _ in range(self.max_depth):
            next_ranked = []
            for parent in frontier:
                for instruction in instructions:
                    child = parent.then(instruction)
                    key = child.canonical()
                    if key in seen:
                        continue
                    seen.add(key)
                    if evaluated >= self.candidate_cap:
                        stopped = True
                        break
                    score, exact = _score(child, normalized)
                    evaluated += 1
                    if score >= 0.0:
                        next_ranked.append((score - 0.002 * child.complexity, score, child, exact))
                    if (score, -child.complexity) > (best[0], -best[1].complexity):
                        best = (score, child)
                    if exact == len(normalized):
                        best = (score, child)
                        break
                if stopped or best[0] >= 1.0:
                    break
            if stopped or best[0] >= 1.0 or not next_ranked:
                break
            next_ranked.sort(key=lambda item: (item[0], -item[2].complexity), reverse=True)
            frontier = [item[2] for item in next_ranked[:self.beam_width]]

        final_score, final_exact = _score(best[1], normalized)
        return ObjectSolveResult(best[1], final_score, final_exact, len(normalized), evaluated, stopped)

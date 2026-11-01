"""Unified QBS search over grid and parameterized object DSL primitives.

Programs are typed instruction streams. Grid operations and object operations
can be hybridized in either order without coercing one DSL into the other.
Only supplied training examples are scored; no MCTS is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from .grid import Grid, as_grid
from .transforms import OPERATIONS, apply_operation
from .object_dsl import ObjectInstruction, apply_object_instruction, generate_object_instructions


@dataclass(frozen=True)
class UnifiedInstruction:
    kind: str
    value: str | ObjectInstruction

    def canonical(self) -> tuple:
        if self.kind == "grid":
            return ("grid", self.value)
        return ("object", self.value.canonical())


@dataclass(frozen=True)
class UnifiedProgram:
    instructions: tuple[UnifiedInstruction, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.instructions)

    def canonical(self) -> tuple:
        return tuple(item.canonical() for item in self.instructions)

    def then(self, instruction: UnifiedInstruction) -> "UnifiedProgram":
        return UnifiedProgram(self.instructions + (instruction,))

    def apply(self, grid: Grid) -> Grid:
        result = as_grid(grid)
        for instruction in self.instructions:
            if instruction.kind == "grid":
                result = apply_operation(result, instruction.value)
            else:
                result = apply_object_instruction(result, instruction.value)
        return result

    def __str__(self) -> str:
        if not self.instructions:
            return "IDENTITY"
        return " -> ".join(
            str(item.value) if item.kind == "grid" else str(item.value)
            for item in self.instructions
        )


@dataclass(frozen=True)
class UnifiedHypothesis:
    program: UnifiedProgram
    exact: int
    total: int
    fit_rate: float
    probability_estimate: float
    feasibility: float
    cell_accuracy: float

    @property
    def rank(self) -> tuple:
        return (
            float(self.exact == self.total),
            self.fit_rate,
            self.feasibility,
            self.cell_accuracy,
            -self.program.complexity,
        )


@dataclass(frozen=True)
class UnifiedQBSResult:
    program: UnifiedProgram
    exact_examples: int
    total_examples: int
    candidates_evaluated: int
    rounds_completed: int
    stopped_by_cap: bool
    fit_rate: float
    probability_estimate: float
    feasibility: float
    retained_hypotheses: tuple[UnifiedProgram, ...]


def unified_qbs_search(
    examples: tuple,
    *,
    max_depth: int = 3,
    candidate_cap: int = 2500,
    parameter_cap: int = 1200,
    max_offset: int = 2,
    rounds: int = 10,
    elite_size: int = 4,
) -> UnifiedQBSResult:
    if not examples:
        raise ValueError("at least one training example is required")
    if not 1 <= max_depth <= 5:
        raise ValueError("max_depth must be in 1..5")
    if not 1 <= candidate_cap <= 20000:
        raise ValueError("candidate_cap must be in 1..20000")
    if elite_size != 4:
        raise ValueError("the retained superposition is the top four hypotheses")
    if rounds < 1:
        raise ValueError("rounds must be positive")

    inputs = tuple(as_grid(example.input) for example in examples)
    object_seeds = generate_object_instructions(
        inputs, parameter_cap=parameter_cap, max_offset=max_offset
    )
    seeds = [UnifiedInstruction("grid", op) for op in OPERATIONS]
    seeds.extend(UnifiedInstruction("object", op) for op in object_seeds)

    cache: dict[tuple, UnifiedHypothesis] = {}
    stopped = False

    def evaluate(program: UnifiedProgram):
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

        exact = sum(pred == example.output for pred, example in zip(predictions, examples))
        matched = total_cells = 0
        for pred, example in zip(predictions, examples):
            target = example.output
            if len(pred) == len(target) and len(pred[0]) == len(target[0]):
                for prow, trow in zip(pred, target):
                    matched += sum(a == b for a, b in zip(prow, trow))
                    total_cells += len(trow)
        fit = exact / len(examples)
        probability = (exact + 1) / (len(examples) + 2)
        hypothesis = UnifiedHypothesis(
            program, exact, len(examples), fit, probability, 1.0,
            matched / total_cells if total_cells else 0.0,
        )
        cache[key] = hypothesis
        return hypothesis

    evaluate(UnifiedProgram())
    for seed in seeds:
        if len(cache) >= candidate_cap:
            stopped = True
            break
        evaluate(UnifiedProgram().then(seed))

    elite = sorted(cache.values(), key=lambda h: h.rank, reverse=True)[:elite_size]
    completed = 0
    for _ in range(rounds):
        if stopped or len(elite) < 2:
            break
        proposals = {}
        for i, left in enumerate(elite):
            for right in elite[i + 1:]:
                for program in (
                    UnifiedProgram(left.program.instructions + right.program.instructions),
                    UnifiedProgram(right.program.instructions + left.program.instructions),
                ):
                    if program.complexity <= max_depth:
                        proposals.setdefault(program.canonical(), program)
        if not proposals:
            break
        before = len(cache)
        for program in proposals.values():
            if len(cache) >= candidate_cap:
                stopped = True
                break
            evaluate(program)
        completed += 1
        elite = sorted(cache.values(), key=lambda h: h.rank, reverse=True)[:elite_size]
        if len(cache) == before:
            break

    best = max(cache.values(), key=lambda h: h.rank)
    retained = tuple(h.program for h in sorted(cache.values(), key=lambda h: h.rank, reverse=True)[:elite_size])
    return UnifiedQBSResult(
        best.program, best.exact, best.total, len(cache), completed, stopped,
        best.fit_rate, best.probability_estimate, best.feasibility, retained,
    )

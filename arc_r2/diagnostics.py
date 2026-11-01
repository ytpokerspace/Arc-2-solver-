"""R2 diagnostics: candidate coverage, train fit, safe rejection, and deduplication."""
from __future__ import annotations

from itertools import product
from typing import Any

from .dsl import Program
from .object_dsl import ObjectInstruction, ObjectProgram, generate_object_instructions
from .transforms import OPERATIONS


def _candidate_programs(engine: str, training_pairs: tuple[dict, ...],
                        max_depth: int, candidate_cap: int):
    if engine == "baseline":
        vocabulary = tuple(OPERATIONS)
        yield Program()
        count = 1
        for depth in range(1, max_depth + 1):
            for operations in product(vocabulary, repeat=depth):
                if count >= candidate_cap:
                    return
                program = Program()
                for operation in operations:
                    program = program.then(operation)
                yield program
                count += 1
        return

    inputs = tuple(pair["input"] for pair in training_pairs)
    vocabulary = generate_object_instructions(inputs, candidate_cap)
    yield ObjectProgram()
    count = 1
    for depth in range(1, max_depth + 1):
        for instructions in product(vocabulary, repeat=depth):
            if count >= candidate_cap:
                return
            yield ObjectProgram(tuple(instructions))
            count += 1


def diagnose_candidates(engine: str, training_pairs: tuple[dict, ...],
                         held_out: dict, max_depth: int = 1,
                         candidate_cap: int = 120) -> dict[str, Any]:
    """Measure expressivity separately from solver ranking.

    Deduplication keys include outputs on every fit input and the held-out input,
    so candidates are merged only when they are behaviorally equivalent on this fold.
    """
    from .grid import as_grid

    if engine not in {"baseline", "object"}:
        raise ValueError("engine must be baseline or object")
    fit_inputs = tuple(as_grid(pair["input"]) for pair in training_pairs)
    fit_targets = tuple(as_grid(pair["output"]) for pair in training_pairs)
    held_input = as_grid(held_out["input"])
    held_target = as_grid(held_out["output"])

    seen_outputs: dict[tuple, tuple[int, str]] = {}
    rejected = 0
    generated = 0
    train_fit = 0
    held_out_coverage = 0
    best_fit_count = 0

    for program in _candidate_programs(engine, training_pairs, max_depth, candidate_cap):
        generated += 1
        try:
            fit_predictions = tuple(program.apply(grid) for grid in fit_inputs)
            held_prediction = program.apply(held_input)
        except (ValueError, TypeError, IndexError, ZeroDivisionError):
            rejected += 1
            continue

        behavior_key = fit_predictions + (held_prediction,)
        description = str(program)
        complexity = program.complexity
        previous = seen_outputs.get(behavior_key)
        if previous is not None:
            # Generation is depth-ascending, but retain the explicit shorter rule.
            if complexity < previous[0]:
                seen_outputs[behavior_key] = (complexity, description)
            continue
        seen_outputs[behavior_key] = (complexity, description)

        exact_fit = all(pred == target for pred, target in zip(fit_predictions, fit_targets))
        exact_held = held_prediction == held_target
        fit_count = sum(pred == target for pred, target in zip(fit_predictions, fit_targets))
        best_fit_count = max(best_fit_count, fit_count)
        train_fit += int(exact_fit)
        held_out_coverage += int(exact_held)

    return {
        "generated_candidates": generated,
        "unique_behaviors": len(seen_outputs),
        "duplicate_behaviors_removed": generated - rejected - len(seen_outputs),
        "rejected_invalid_candidates": rejected,
        "train_fit_candidates": train_fit,
        "held_out_coverage_candidates": held_out_coverage,
        "best_fit_pair_matches": best_fit_count,
        "fit_pair_count": len(training_pairs),
        "candidate_cap": candidate_cap,
        "max_depth": max_depth,
    }

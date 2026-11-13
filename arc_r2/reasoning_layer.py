"""Optional coordinator that records ARC solver runs in ReasoningMemory.

The coordinator is deliberately additive: the existing solver remains the
source of predictions, and the memory layer records evidence without changing
candidate generation or selecting a prediction.
"""
from __future__ import annotations

from typing import Any, Iterable

from .reasoning_memory import ReasoningMemory


def solve_and_remember(
    solver: Any,
    examples: tuple,
    memory: ReasoningMemory,
    *,
    record_id: str,
    intent: str,
    title: str = "ARC solver experiment",
    requirements: Iterable[str] = (),
    assumptions: Iterable[str] = (),
    unknowns: Iterable[str] = (),
    representations: Iterable[str] = (),
    hypotheses: Iterable[str] = (),
    sources: Iterable[str] = (),
    tags: Iterable[str] = ("arc2",),
):
    """Run the existing solver and persist an evidence-labelled record.

    A perfect fit on visible training examples is recorded as tested, not
    validated: held-out evaluation is required before validation status.
    """
    result = solver.solve(examples)
    exact = int(result.exact_examples)
    total = int(result.total_examples)
    status = "tested" if total > 0 and exact == total else "implemented"
    outcome = {
        "exact_training_examples": exact,
        "total_training_examples": total,
        "training_fit_complete": bool(total > 0 and exact == total),
        "score": float(result.score),
        "candidates_evaluated": int(result.candidates_evaluated),
        "program": str(result.program),
        "evidence_scope": "visible training examples only; not held-out validation",
    }
    record = memory.put({
        "id": record_id,
        "title": title,
        "intent": intent,
        "status": status,
        "requirements": list(requirements),
        "assumptions": list(assumptions),
        "unknowns": list(unknowns),
        "representations": list(representations),
        "hypotheses": list(hypotheses),
        "evidence": [outcome],
        "tests": ["existing solver training-example fit"],
        "sources": list(sources),
        "tags": list(tags),
        "outcome": outcome,
    })
    return result, record

def find_applicable_verified_pattern(memory: ReasoningMemory, examples: tuple, *, tags=()):
    """Return a validated baseline program only when it exactly fits every current example.

    This is deliberately conservative: unvalidated records, object-engine records,
    parse failures, and any imperfect fit are ignored.
    """
    from types import SimpleNamespace
    from .dsl import parse_program

    for record in memory.verified_patterns(tags):
        try:
            program = parse_program(str((record.get("outcome") or {}).get("program", "")))
            if not examples or not all(program.apply(example.input) == example.output
                                       for example in examples):
                continue
            return SimpleNamespace(
                program=program,
                score=1.0,
                exact_examples=len(examples),
                total_examples=len(examples),
                candidates_evaluated=0,
                reused_memory_record=record["id"],
            ), record
        except (KeyError, TypeError, ValueError, IndexError):
            continue
    return None, None

def validate_and_promote_pattern(memory: ReasoningMemory, record: dict, full_result: Any,
                                 examples: tuple, solver_factory: Any):
    """Promote a baseline record only when one identical program survives every LOO fold.

    This is within-task validation, not independent-task or official ARC validation.
    """
    if len(examples) < 2 or not record or record.get("status") not in {"tested", "implemented"}:
        return record
    full_program = full_result.program
    fold_evidence = []
    for heldout_index, heldout in enumerate(examples):
        visible = tuple(item for index, item in enumerate(examples) if index != heldout_index)
        try:
            fold_result = solver_factory().solve(visible)
            same_program = str(fold_result.program) == str(full_program)
            exact_heldout = fold_result.program.apply(heldout.input) == heldout.output
        except (ValueError, TypeError, IndexError):
            return record
        fold_evidence.append({
            "heldout_index": heldout_index,
            "same_program_as_full_fit": same_program,
            "heldout_exact": exact_heldout,
        })
        if not same_program or not exact_heldout:
            return record
    promoted = dict(record)
    promoted["status"] = "validated"
    outcome = dict(promoted.get("outcome") or {})
    outcome["validation_scope"] = "leave-one-training-pair-out within-task validation"
    outcome["validation_folds"] = fold_evidence
    outcome["validation_exact"] = len(fold_evidence)
    outcome["validation_total"] = len(fold_evidence)
    outcome["evidence_scope"] = (
        "same selected program reproduced on every leave-one-training-pair-out fold; "
        "not independent-task generalization or official ARC evaluation"
    )
    promoted["outcome"] = outcome
    promoted["evidence"] = list(promoted.get("evidence") or []) + [{
        "validation_scope": outcome["validation_scope"],
        "folds": fold_evidence,
    }]
    return memory.put(promoted)

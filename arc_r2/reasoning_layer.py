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

"""HIAN: evidence-backed hypothesis ranking for ARC-R2.

This module is deliberately deterministic and offline. It does not invent
transformations: it organizes and audits candidate Programs supplied by the
existing search engines. Confidence is a smoothed training-fit statistic,
not a calibrated estimate of unseen-task accuracy.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable

from .dsl import Program
from .grid import Grid
from .emoji_alchemy import apply_emoji_program, generate_alchemy_candidates, semantic_similarity
from .emoji_codec import decode_grid


@dataclass(frozen=True)
class Evidence:
    source_id: str
    exact_match: bool
    cell_accuracy: float
    semantic_accuracy: float
    predicted_shape: tuple[int, int] | None
    expected_shape: tuple[int, int]


@dataclass(frozen=True)
class Hypothesis:
    hypothesis_id: str
    program: tuple[str, ...]
    exact_matches: int
    total_examples: int
    cell_accuracy: float
    semantic_accuracy: float
    confidence_estimate: float
    supporting_sources: tuple[str, ...]
    contradicting_sources: tuple[str, ...]
    complexity: int

    @property
    def first_concept(self) -> str:
        return self.program[0] if self.program else "IDENTITY"


def _shape(grid: Grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]))


def _cell_accuracy(predicted: Grid, expected: Grid) -> float:
    if _shape(predicted) != _shape(expected):
        return 0.0
    total = sum(len(row) for row in expected)
    if total == 0:
        return 1.0
    matched = sum(
        left == right
        for predicted_row, expected_row in zip(predicted, expected)
        for left, right in zip(predicted_row, expected_row)
    )
    return matched / total


def evaluate_hypothesis(
    program: Program,
    examples: Iterable[tuple[str, Grid, Grid]],
) -> tuple[Hypothesis, tuple[Evidence, ...]]:
    """Evaluate one candidate against named training examples."""
    evidence: list[Evidence] = []
    for source_id, input_grid, expected in examples:
        predicted: Grid | None
        try:
            symbolic_prediction = apply_emoji_program(input_grid, program)
            predicted = decode_grid(symbolic_prediction)
            exact = predicted == expected
            accuracy = _cell_accuracy(predicted, expected)
            semantic = semantic_similarity(predicted, expected)
            predicted_shape = _shape(predicted)
        except Exception:
            exact = False
            accuracy = 0.0
            semantic = 0.0
            predicted_shape = None
        evidence.append(Evidence(
            source_id=str(source_id),
            exact_match=exact,
            cell_accuracy=accuracy,
            semantic_accuracy=semantic,
            predicted_shape=predicted_shape,
            expected_shape=_shape(expected),
        ))

    total = len(evidence)
    exact_count = sum(item.exact_match for item in evidence)
    mean_cells = (
        sum(item.cell_accuracy for item in evidence) / total if total else 0.0
    )
    mean_semantic = (
        sum(item.semantic_accuracy for item in evidence) / total if total else 0.0
    )
    # Beta(1, 1) smoothing: transparent training-fit estimate only.
    confidence = (exact_count + 1) / (total + 2) if total else 0.5
    canonical = program.canonical()
    hypothesis = Hypothesis(
        hypothesis_id="hian:" + ("|".join(canonical) if canonical else "IDENTITY"),
        program=canonical,
        exact_matches=exact_count,
        total_examples=total,
        cell_accuracy=mean_cells,
        semantic_accuracy=mean_semantic,
        confidence_estimate=confidence,
        supporting_sources=tuple(e.source_id for e in evidence if e.exact_match),
        contradicting_sources=tuple(e.source_id for e in evidence if not e.exact_match),
        complexity=program.complexity,
    )
    return hypothesis, tuple(evidence)


def analyze_hypotheses(
    candidates: Iterable[Program],
    examples: Iterable[tuple[str, Grid, Grid]],
    top_k: int = 20,
) -> dict[str, Any]:
    """Build a hierarchy of candidate concepts and rank by observed evidence."""
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    example_rows = tuple(examples)
    candidate_rows = tuple(candidates)
    if not candidate_rows:
        raise ValueError("at least one candidate hypothesis is required")
    unique: dict[tuple[str, ...], Program] = {}
    for candidate in generate_alchemy_candidates(candidate_rows, max_depth=2):
        unique.setdefault(candidate.canonical(), candidate)
    if not unique:
        raise ValueError("at least one candidate hypothesis is required")

    evaluated = [
        evaluate_hypothesis(program, example_rows)
        for program in unique.values()
    ]
    ranked = sorted(
        (item[0] for item in evaluated),
        key=lambda h: (
            h.exact_matches,
            h.semantic_accuracy,
            h.cell_accuracy,
            h.confidence_estimate,
            -h.complexity,
            h.program,
        ),
        reverse=True,
    )
    hierarchy: dict[str, list[str]] = {}
    for hypothesis in ranked:
        hierarchy.setdefault(hypothesis.first_concept, []).append(
            hypothesis.hypothesis_id
        )

    evidence_by_id = {
        hypothesis.hypothesis_id: [asdict(row) for row in evidence]
        for hypothesis, evidence in evaluated
    }
    return {
        "system": "HIAN",
        "version": "1.1-emoji-alchemy",
        "semantic_method": "lossless emoji-token execution with color, object-shape, spatial-relation, coarse-location, and dimension similarity",
        "candidate_generation": "existing search candidates plus all DSL compositions up to depth two",
        "candidate_count": len(ranked),
        "training_example_count": len(example_rows),
        "confidence_semantics": (
            "Beta(1,1)-smoothed exact training-fit estimate; not calibrated "
            "unseen-task probability."
        ),
        "hierarchy": hierarchy,
        "ranked_hypotheses": [asdict(item) for item in ranked[:top_k]],
        "evidence": {
            hypothesis_id: evidence_by_id[hypothesis_id]
            for hypothesis_id in [item.hypothesis_id for item in ranked[:top_k]]
        },
        "selected_hypothesis": asdict(ranked[0]),
    }

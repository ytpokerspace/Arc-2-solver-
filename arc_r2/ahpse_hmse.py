"""Explicit opt-in adapter that runs AHPSE through the existing HMSE engine.

The adapter is intentionally separate from the default solver entry point.
It accepts only visible TrainingExample objects and returns declarative
hypotheses plus HMSE execution evidence.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Any

from .ahpse import SynthesisResult, TrainingExample, synthesize
from .hmse import (
    ExecutionResult,
    MethodRegistry,
    MethodSpec,
    Procedure,
    SearchPartition,
    Stage,
    ValidationResult,
    execute_procedure,
)


def register_ahpse_methods(registry: MethodRegistry) -> None:
    """Register AHPSE's deterministic synthesis method and exact-fit validator."""
    def propose(
        state: Mapping[str, Any], parameters: Mapping[str, Any]
    ) -> SynthesisResult:
        examples = state.get("training_examples")
        if not isinstance(examples, (tuple, list)) or not examples:
            raise ValueError("training_examples must be a non-empty tuple or list")
        if not all(isinstance(item, TrainingExample) for item in examples):
            raise TypeError("training_examples must contain TrainingExample values")
        return synthesize(
            examples,
            max_depth=int(parameters.get("max_depth", 2)),
            candidate_cap=int(parameters.get("candidate_cap", 500)),
        )

    def exact_fit(
        stage: Stage,
        state: Mapping[str, Any],
        output: Any,
        spec: MethodSpec,
    ) -> ValidationResult:
        if not isinstance(output, SynthesisResult):
            return ValidationResult(False, 0.0, {"reason": "invalid_result_type"})
        evidence = {
            "candidates_generated": output.candidates_generated,
            "candidates_checked": output.candidates_checked,
            "exact_fit_count": output.exact_fit_count,
            "abstained": output.abstained,
            "reason": output.reason,
        }
        # This score means exact consistency with the supplied training pairs,
        # not a probability of generalizing to an unseen ARC task.
        return ValidationResult(
            accepted=not output.abstained and output.exact_fit_count > 0,
            score=1.0 if not output.abstained and output.exact_fit_count > 0 else 0.0,
            evidence=evidence,
        )

    registry.register_method("arc.ahpse.synthesize", propose)
    registry.register_validator("arc.ahpse.exact_fit", exact_fit)


def build_ahpse_procedure(
    *, max_depth: int = 2, candidate_cap: int = 500
) -> Procedure:
    """Build a one-stage HMSE procedure; no default solver wiring is changed."""
    if max_depth < 0:
        raise ValueError("max_depth must be non-negative")
    if candidate_cap < 1:
        raise ValueError("candidate_cap must be at least 1")
    return Procedure(
        procedure_id="arc.ahpse.opt-in.v1",
        stages=(
            Stage(
                stage_id="generate-and-validate-hypotheses",
                output_key="ahpse_result",
                partitions=(
                    SearchPartition(
                        partition_id="bounded-symbolic-synthesis",
                        methods=(
                            MethodSpec(
                                method_id="arc.ahpse.synthesize",
                                parameters={
                                    "max_depth": max_depth,
                                    "candidate_cap": candidate_cap,
                                },
                                requires=("training_examples",),
                            ),
                        ),
                    ),
                ),
                requires=("training_examples",),
                validator_id="arc.ahpse.exact_fit",
                max_evaluations=1,
                stop_score=1.0,
            ),
        ),
        description=(
            "Explicit opt-in AHPSE synthesis executed and recorded by HMSE; "
            "exact training fit is not a generalization guarantee."
        ),
        input_keys=("training_examples",),
    )


def run_ahpse_with_hmse(
    examples: Iterable[TrainingExample],
    *,
    max_depth: int = 2,
    candidate_cap: int = 500,
) -> tuple[SynthesisResult, ExecutionResult]:
    """Run AHPSE through HMSE using a fresh, isolated method registry."""
    training_examples = tuple(examples)
    if not training_examples:
        raise ValueError("at least one training example is required")
    registry = MethodRegistry()
    register_ahpse_methods(registry)
    procedure = build_ahpse_procedure(
        max_depth=max_depth, candidate_cap=candidate_cap
    )
    execution = execute_procedure(
        procedure, {"training_examples": training_examples}, registry
    )
    result = execution.state["ahpse_result"]
    return result, execution

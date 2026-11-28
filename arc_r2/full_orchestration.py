"""Full opt-in ARC orchestration: perception, synthesis, verification, consensus.

The default solver is untouched. All candidate families receive only visible
training pairs. Test outputs are never accepted by this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Mapping

from .ahpse import Hypothesis, TrainingExample, synthesize
from .hmse import (
    ExecutionResult, MethodRegistry, MethodSpec, Procedure, SearchPartition,
    Stage, ValidationResult, execute_procedure,
)
from .object_solver import ObjectExample, ObjectHybridSolver
from .primitives import Program, fit_program
from .shape_radar import scan_grid
from .grid import Grid, as_grid
from .hmse_rule_selector import propose_rule_candidates


@dataclass(frozen=True)
class Candidate:
    family: str
    program: Any
    complexity: int

    def apply(self, grid: Grid) -> Grid:
        return as_grid(self.program.apply(grid))


@dataclass(frozen=True)
class CandidatePool:
    candidates: tuple[Candidate, ...]
    family_evidence: Mapping[str, Any]


@dataclass(frozen=True)
class Decision:
    predictions: tuple[Grid | None, ...]
    ambiguous: tuple[bool, ...]
    candidate_count: int
    abstentions: int
    reason: str


@dataclass(frozen=True)
class FullOrchestrationResult:
    decision: Decision
    execution: ExecutionResult
    radar_reports: Mapping[str, Any]
    candidate_pool: CandidatePool


def _completed(stage: Stage, state: Mapping[str, Any], output: Any,
               spec: MethodSpec) -> ValidationResult:
    """Accept a completed stage record, including an explicit abstention."""
    return ValidationResult(True, 1.0, {"recorded": output is not None})


def _complexity(program: Any) -> int:
    value = getattr(program, "complexity", 10_000)
    return int(value() if callable(value) else value)


def _scan(state: Mapping[str, Any], parameters: Mapping[str, Any]):
    examples = state["training_examples"]
    tests = state["test_inputs"]
    input_reports = tuple(scan_grid(item.input) for item in examples)
    output_reports = tuple(scan_grid(item.output) for item in examples)
    test_reports = tuple(scan_grid(grid) for grid in tests)
    pair_deltas = tuple({
        "input_objects": source["scan"]["primary_object_count"],
        "output_objects": target["scan"]["primary_object_count"],
        "input_relations": len(source["relationships"]),
        "output_relations": len(target["relationships"]),
    } for source, target in zip(input_reports, output_reports))
    return {
        "schema": "arc-full-perception/v1",
        "training_inputs": input_reports,
        "training_outputs": output_reports,
        "test_inputs": test_reports,
        "training_pair_deltas": pair_deltas,
    }


def _synthesize(state: Mapping[str, Any], parameters: Mapping[str, Any]) -> CandidatePool:
    examples: tuple[TrainingExample, ...] = state["training_examples"]
    radar: Mapping[str, Any] = state["radar_reports"]
    pairs = tuple((item.input, item.output) for item in examples)
    cap = int(parameters["candidate_cap"])
    depth = int(parameters["max_depth"])
    candidates: list[Candidate] = []
    evidence: dict[str, Any] = {
        "perception": {
            "schema": radar["schema"],
            "training_pairs_scanned": len(radar["training_inputs"]),
            "input_object_counts": [item["scan"]["primary_object_count"] for item in radar["training_inputs"]],
            "output_object_counts": [item["scan"]["primary_object_count"] for item in radar["training_outputs"]],
            "pair_relation_deltas": list(radar["training_pair_deltas"]),
        }
    }

    family_timing: dict[str, float] = {}

    family_start = perf_counter()
    symbolic = synthesize(examples, max_depth=depth, candidate_cap=cap)
    family_timing["ahpse"] = perf_counter() - family_start
    ahpse_start_count = len(candidates)
    candidates.extend(Candidate("ahpse", item, item.complexity)
                      for item in symbolic.hypotheses)
    evidence["ahpse"] = {
        "generated": symbolic.candidates_generated,
        "checked": symbolic.candidates_checked,
        "exact_fits": symbolic.exact_fit_count,
        "candidate_count": len(candidates) - ahpse_start_count,
        "runtime_seconds": round(family_timing["ahpse"], 6),
        "constraint_checks": symbolic.constraint_checks,
        "constraint_rejections": symbolic.constraint_rejections,
        "abstained": symbolic.abstained,
    }

    family_start = perf_counter()
    primitive = fit_program(pairs, candidate_cap=cap, max_depth=min(depth, 2),
                             memory_path=None)
    family_timing["primitive"] = perf_counter() - family_start
    primitive_start_count = len(candidates)
    if primitive.exact_fit and primitive.program is not None:
        candidates.append(Candidate("primitive", primitive.program,
                                    _complexity(primitive.program)))
    evidence["primitive"] = {
        "candidates_evaluated": primitive.candidates_evaluated,
        "exact_fit": primitive.exact_fit,
        "fit_program_count": primitive.fit_program_count,
        "candidate_count": len(candidates) - primitive_start_count,
        "runtime_seconds": round(family_timing["primitive"], 6),
        "memory_enabled": False,
    }

    family_start = perf_counter()
    registered_rules = propose_rule_candidates(pairs, radar["training_inputs"])
    family_timing["registered_rules"] = perf_counter() - family_start
    rules_start_count = len(candidates)
    for rule in registered_rules:
        if rule.accepted and rule.program is not None:
            candidates.append(Candidate(
                "rule:" + rule.rule_id, rule.program, _complexity(rule.program)
            ))
    evidence["registered_rules"] = [
        {"rule_id": rule.rule_id, "accepted": rule.accepted,
         "reason": rule.reason, "training_pairs_checked": rule.training_pairs_checked}
        for rule in registered_rules
    ]
    evidence["registered_rule_family"] = {
        "candidate_count": len(candidates) - rules_start_count,
        "runtime_seconds": round(family_timing["registered_rules"], 6),
    }

    object_signal_present = any(
        report["scan"]["primary_object_count"] > 0
        for report in radar["training_inputs"]
    )
    # Under unanimous consensus, an additional candidate cannot improve a
    # decision once another family has supplied any exact-fit candidate:
    # it can only preserve agreement or introduce disagreement. Reserve the
    # expensive object search for folds where simpler families found none.
    if depth >= 1 and object_signal_present and not candidates:
        family_start = perf_counter()
        object_examples = tuple(ObjectExample(item.input, item.output) for item in examples)
        object_result = ObjectHybridSolver(
            max_depth=min(depth, 4),
            candidate_cap=min(cap, 10_000),
            parameter_cap=min(cap, 10_000),
        ).solve(object_examples)
        if object_result.exact_examples == len(examples):
            candidates.append(Candidate("object", object_result.program,
                                        _complexity(object_result.program)))
        family_timing["object"] = perf_counter() - family_start
        evidence["object"] = {
            "candidates_evaluated": object_result.candidates_evaluated,
            "candidate_count": int(object_result.exact_examples == len(examples)),
            "runtime_seconds": round(family_timing["object"], 6),
            "exact_examples": object_result.exact_examples,
            "total_examples": object_result.total_examples,
            "exact_fit": object_result.exact_examples == len(examples),
            "enabled_by_shape_radar": True,
        }
    else:
        family_timing["object"] = 0.0
        if depth == 0:
            object_skip_reason = "max_depth=0 disables object search"
        elif not object_signal_present:
            object_skip_reason = "Shape Radar found no foreground objects in training inputs"
        else:
            object_skip_reason = (
                "skipped because another family already supplied an exact-fit candidate; "
                "additional candidates cannot improve unanimous consensus"
            )
        evidence["object"] = {
            "candidates_evaluated": 0,
            "candidate_count": 0,
            "runtime_seconds": 0.0,
            "exact_examples": 0,
            "total_examples": len(examples),
            "exact_fit": False,
            "skipped": object_skip_reason,
            "enabled_by_shape_radar": False,
        }
    # Keep provenance-distinct candidates; consensus is evaluated on each
    # requested input, not by trusting a family name or training score.
    evidence["family_timing_seconds"] = {
        name: round(seconds, 6) for name, seconds in family_timing.items()
    }
    evidence["family_candidate_counts"] = {
        "ahpse": evidence["ahpse"]["candidate_count"],
        "primitive": evidence["primitive"]["candidate_count"],
        "registered_rules": evidence["registered_rule_family"]["candidate_count"],
        "object": evidence["object"]["candidate_count"],
    }
    return CandidatePool(tuple(candidates), evidence)


def _decide(state: Mapping[str, Any], parameters: Mapping[str, Any]) -> Decision:
    pool: CandidatePool = state["candidate_pool"]
    tests: tuple[Grid, ...] = state["test_inputs"]
    predictions: list[Grid | None] = []
    ambiguous: list[bool] = []
    abstentions = 0
    for test_input in tests:
        outputs: dict[Grid, list[str]] = {}
        for candidate in pool.candidates:
            try:
                output = candidate.apply(test_input)
            except (ValueError, TypeError, IndexError):
                continue
            outputs.setdefault(output, []).append(candidate.family)
        if len(outputs) == 1:
            predictions.append(next(iter(outputs)))
            ambiguous.append(False)
        else:
            predictions.append(None)
            ambiguous.append(len(outputs) > 1)
            abstentions += 1
    if not pool.candidates:
        reason = "no exact-fit candidate from any registered family"
    elif abstentions and any(ambiguous):
        reason = "candidate families disagree; abstained rather than guessing"
    elif abstentions:
        reason = "all candidates failed to produce a valid prediction"
    else:
        reason = "all candidate families with exact-fit programs agree"
    return Decision(tuple(predictions), tuple(ambiguous), len(pool.candidates),
                    abstentions, reason)


def build_full_procedure(*, max_depth: int = 2,
                         candidate_cap: int = 500) -> Procedure:
    if not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be in 0..2")
    if not 1 <= candidate_cap <= 10_000:
        raise ValueError("candidate_cap must be in 1..10000")
    return Procedure(
        procedure_id="arc.full-orchestration.v1",
        description=("Shape Radar perception -> AHPSE, primitive and object "
                     "candidate families -> exact-fit consensus with abstention."),
        input_keys=("training_examples", "test_inputs"),
        stages=(
            Stage("perception", "radar_reports", (
                SearchPartition("shape-radar", (
                    MethodSpec("arc.full.scan", requires=("training_examples", "test_inputs")),
                )),
            ), requires=("training_examples", "test_inputs"),
                validator_id="arc.full.completed", max_evaluations=1),
            Stage("candidate-synthesis", "candidate_pool", (
                SearchPartition("symbolic-and-dsl-families", (
                    MethodSpec("arc.full.synthesize", {
                        "max_depth": max_depth, "candidate_cap": candidate_cap,
                    }, ("training_examples", "radar_reports")),
                )),
            ), requires=("training_examples", "radar_reports"),
                validator_id="arc.full.completed", max_evaluations=1),
            Stage("consensus-and-abstention", "decision", (
                SearchPartition("independent-validation", (
                    MethodSpec("arc.full.decide", requires=("candidate_pool", "test_inputs")),
                )),
            ), requires=("candidate_pool", "test_inputs"),
                validator_id="arc.full.completed", max_evaluations=1),
        ),
    )


def run_full_orchestration(examples: Iterable[TrainingExample],
                           test_inputs: Iterable[Grid], *,
                           max_depth: int = 2,
                           candidate_cap: int = 500) -> FullOrchestrationResult:
    """Run all stages using train-only labels and label-free prediction inputs."""
    training = tuple(
        TrainingExample.from_values(item.input, item.output)
        if isinstance(item, TrainingExample)
        else TrainingExample.from_values(item["input"], item["output"])
        for item in examples
    )
    tests = tuple(as_grid(grid) for grid in test_inputs)
    if not training:
        raise ValueError("at least one visible training example is required")
    registry = MethodRegistry()
    registry.register_method("arc.full.scan", _scan)
    registry.register_method("arc.full.synthesize", _synthesize)
    registry.register_method("arc.full.decide", _decide)
    registry.register_validator("arc.full.completed", _completed)
    procedure = build_full_procedure(max_depth=max_depth, candidate_cap=candidate_cap)
    execution = execute_procedure(
        procedure,
        {"training_examples": training, "test_inputs": tests},
        registry,
    )
    return FullOrchestrationResult(
        decision=execution.state["decision"],
        execution=execution,
        radar_reports=execution.state["radar_reports"],
        candidate_pool=execution.state["candidate_pool"],
    )

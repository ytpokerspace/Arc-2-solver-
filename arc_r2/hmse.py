"""Hierarchical Method Synthesis Engine (HMSE).

HMSE orchestrates reusable, domain-independent methods. It executes a saved
procedure as dependent stages, partitions each stage's candidate methods,
enforces a deterministic evaluation budget, validates candidates, and records
an execution trace. It does not infer ARC rules by itself: perception and rule
formulation supply the candidate method specifications.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping


@dataclass(frozen=True)
class MethodSpec:
    method_id: str
    parameters: Mapping[str, Any] = field(default_factory=dict)
    requires: tuple[str, ...] = ()


@dataclass(frozen=True)
class SearchPartition:
    partition_id: str
    methods: tuple[MethodSpec, ...]


@dataclass(frozen=True)
class Stage:
    stage_id: str
    output_key: str
    partitions: tuple[SearchPartition, ...]
    requires: tuple[str, ...] = ()
    validator_id: str = "accept_non_none"
    max_evaluations: int = 100
    stop_score: float = 1.0


@dataclass(frozen=True)
class Procedure:
    procedure_id: str
    stages: tuple[Stage, ...]
    version: int = 1
    description: str = ""

    @property
    def estimated_steps(self) -> int:
        return len(self.stages)

    @property
    def search_budget(self) -> int:
        return sum(min(stage.max_evaluations,
                       sum(len(partition.methods) for partition in stage.partitions))
                   for stage in self.stages)


@dataclass(frozen=True)
class ValidationResult:
    accepted: bool
    score: float
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CandidateResult:
    partition_id: str
    method_id: str
    accepted: bool
    score: float
    evidence: Mapping[str, Any]


@dataclass(frozen=True)
class StageResult:
    stage_id: str
    output_key: str
    selected_method: str
    selected_partition: str
    evaluations: int
    candidates: tuple[CandidateResult, ...]


@dataclass(frozen=True)
class ExecutionResult:
    procedure_id: str
    state: Mapping[str, Any]
    stages: tuple[StageResult, ...]

    @property
    def total_evaluations(self) -> int:
        return sum(stage.evaluations for stage in self.stages)


class ProcedureError(RuntimeError):
    """Raised when a procedure is malformed or cannot complete a stage."""


Method = Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
Validator = Callable[[Stage, Mapping[str, Any], Any, MethodSpec], ValidationResult]


class MethodRegistry:
    """Runtime implementations; saved procedures contain IDs, never code."""

    def __init__(self) -> None:
        self._methods: dict[str, Method] = {}
        self._validators: dict[str, Validator] = {
            "accept_non_none": self._accept_non_none,
        }

    @staticmethod
    def _accept_non_none(
        stage: Stage, state: Mapping[str, Any], output: Any, spec: MethodSpec
    ) -> ValidationResult:
        return ValidationResult(output is not None, 1.0 if output is not None else 0.0)

    def register_method(self, method_id: str, method: Method) -> None:
        if not method_id or method_id in self._methods:
            raise ValueError(f"method ID is empty or already registered: {method_id!r}")
        self._methods[method_id] = method

    def register_validator(self, validator_id: str, validator: Validator) -> None:
        if not validator_id or validator_id in self._validators:
            raise ValueError(f"validator ID is empty or already registered: {validator_id!r}")
        self._validators[validator_id] = validator

    def method(self, method_id: str) -> Method:
        try:
            return self._methods[method_id]
        except KeyError as exc:
            raise ProcedureError(f"unregistered method: {method_id}") from exc

    def validator(self, validator_id: str) -> Validator:
        try:
            return self._validators[validator_id]
        except KeyError as exc:
            raise ProcedureError(f"unregistered validator: {validator_id}") from exc


def validate_procedure(procedure: Procedure) -> None:
    if not procedure.procedure_id:
        raise ValueError("procedure_id cannot be empty")
    if procedure.version < 1:
        raise ValueError("procedure version must be positive")
    stage_ids: set[str] = set()
    available: set[str] = set()
    for stage in procedure.stages:
        if not stage.stage_id or stage.stage_id in stage_ids:
            raise ValueError(f"duplicate or empty stage ID: {stage.stage_id!r}")
        stage_ids.add(stage.stage_id)
        if not stage.output_key:
            raise ValueError(f"stage {stage.stage_id} has no output_key")
        missing = set(stage.requires) - available
        if missing:
            raise ValueError(
                f"stage {stage.stage_id} has unsatisfied dependencies: {sorted(missing)}"
            )
        if stage.max_evaluations < 1:
            raise ValueError(f"stage {stage.stage_id} must have a positive search budget")
        if not 0.0 <= stage.stop_score <= 1.0:
            raise ValueError(f"stage {stage.stage_id} stop_score must be in [0, 1]")
        partition_ids: set[str] = set()
        for partition in stage.partitions:
            if not partition.partition_id or partition.partition_id in partition_ids:
                raise ValueError(f"duplicate or empty partition ID in {stage.stage_id}")
            partition_ids.add(partition.partition_id)
            for spec in partition.methods:
                if not spec.method_id:
                    raise ValueError(f"empty method ID in stage {stage.stage_id}")
                if not isinstance(spec.parameters, Mapping):
                    raise ValueError(f"parameters for {spec.method_id} must be a mapping")
        if not stage.partitions or not any(p.methods for p in stage.partitions):
            raise ValueError(f"stage {stage.stage_id} has no candidate methods")
        available.add(stage.output_key)


def _round_robin_candidates(
    partitions: tuple[SearchPartition, ...], limit: int
):
    """Yield (partition, method) fairly, preserving authored order."""
    positions = [0] * len(partitions)
    yielded = 0
    while yielded < limit:
        progressed = False
        for index, partition in enumerate(partitions):
            if positions[index] < len(partition.methods):
                spec = partition.methods[positions[index]]
                positions[index] += 1
                yielded += 1
                progressed = True
                yield partition, spec
                if yielded >= limit:
                    break
        if not progressed:
            break


def execute_procedure(
    procedure: Procedure,
    initial_state: Mapping[str, Any],
    registry: MethodRegistry,
) -> ExecutionResult:
    """Execute the best validated candidate at each stage under its own budget.

    Candidate search is round-robin across partitions, so one large partition
    cannot consume the entire stage budget before alternatives are considered.
    The highest-scoring accepted candidate wins; ties preserve search order.
    """
    validate_procedure(procedure)
    state = dict(initial_state)
    stage_results: list[StageResult] = []

    for stage in procedure.stages:
        missing = set(stage.requires) - state.keys()
        if missing:
            raise ProcedureError(
                f"stage {stage.stage_id} missing runtime inputs: {sorted(missing)}"
            )
        validator = registry.validator(stage.validator_id)
        best: tuple[float, int, SearchPartition, MethodSpec, Any, ValidationResult] | None = None
        candidate_results: list[CandidateResult] = []
        evaluations = 0

        for partition, spec in _round_robin_candidates(
            stage.partitions, stage.max_evaluations
        ):
            evaluations += 1
            missing_method_inputs = set(spec.requires) - state.keys()
            if missing_method_inputs:
                result = ValidationResult(
                    False, 0.0,
                    {"reason": "missing_method_inputs",
                     "missing": sorted(missing_method_inputs)},
                )
                candidate_results.append(CandidateResult(
                    partition.partition_id, spec.method_id, False, 0.0, result.evidence
                ))
                continue

            try:
                output = registry.method(spec.method_id)(dict(state), dict(spec.parameters))
                verdict = validator(stage, state, output, spec)
            except (ValueError, TypeError, IndexError, KeyError) as exc:
                output = None
                verdict = ValidationResult(
                    False, 0.0, {"reason": "candidate_error", "error": str(exc)}
                )
            if not 0.0 <= verdict.score <= 1.0:
                raise ProcedureError(
                    f"validator {stage.validator_id} returned score outside [0, 1]"
                )
            candidate_results.append(CandidateResult(
                partition.partition_id, spec.method_id, verdict.accepted,
                verdict.score, dict(verdict.evidence)
            ))
            if verdict.accepted and (
                best is None or verdict.score > best[0]
            ):
                best = (verdict.score, len(candidate_results) - 1,
                        partition, spec, output, verdict)
            if verdict.accepted and verdict.score >= stage.stop_score:
                break

        if best is None:
            raise ProcedureError(
                f"stage {stage.stage_id} found no validated method "
                f"within {stage.max_evaluations} evaluations"
            )
        _, _, chosen_partition, chosen_spec, chosen_output, _ = best
        state[stage.output_key] = chosen_output
        stage_results.append(StageResult(
            stage.stage_id, stage.output_key, chosen_spec.method_id,
            chosen_partition.partition_id, evaluations, tuple(candidate_results)
        ))

    return ExecutionResult(procedure.procedure_id, state, tuple(stage_results))


def compose_procedures(
    procedure_id: str, *procedures: Procedure, description: str = ""
) -> Procedure:
    """Compose procedures in order; stage dependencies remain explicit."""
    stages = tuple(stage for procedure in procedures for stage in procedure.stages)
    composed = Procedure(procedure_id, stages, 1, description)
    validate_procedure(composed)
    return composed


def replace_stage(procedure: Procedure, stage_id: str, replacement: Stage) -> Procedure:
    """Replace one stage without changing the rest of the saved procedure."""
    if not any(stage.stage_id == stage_id for stage in procedure.stages):
        raise KeyError(f"unknown stage: {stage_id}")
    updated = Procedure(
        procedure.procedure_id,
        tuple(replacement if stage.stage_id == stage_id else stage
              for stage in procedure.stages),
        procedure.version + 1,
        procedure.description,
    )
    validate_procedure(updated)
    return updated


class MethodLibrary:
    """JSON library of declarative procedures; executable code stays registered."""

    @staticmethod
    def save(procedure: Procedure, path: str | Path) -> None:
        validate_procedure(procedure)
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(procedure), indent=2, sort_keys=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(payload + "\n", encoding="utf-8")
        temporary.replace(target)

    @staticmethod
    def load(path: str | Path) -> Procedure:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        stages = []
        for raw_stage in payload["stages"]:
            partitions = tuple(
                SearchPartition(
                    partition_id=raw_partition["partition_id"],
                    methods=tuple(MethodSpec(
                        method_id=raw_method["method_id"],
                        parameters=raw_method.get("parameters", {}),
                        requires=tuple(raw_method.get("requires", ())),
                    ) for raw_method in raw_partition["methods"]),
                ) for raw_partition in raw_stage["partitions"]
            )
            stages.append(Stage(
                stage_id=raw_stage["stage_id"],
                output_key=raw_stage["output_key"],
                partitions=partitions,
                requires=tuple(raw_stage.get("requires", ())),
                validator_id=raw_stage.get("validator_id", "accept_non_none"),
                max_evaluations=raw_stage.get("max_evaluations", 100),
                stop_score=raw_stage.get("stop_score", 1.0),
            ))
        procedure = Procedure(
            procedure_id=payload["procedure_id"],
            stages=tuple(stages),
            version=payload.get("version", 1),
            description=payload.get("description", ""),
        )
        validate_procedure(procedure)
        return procedure

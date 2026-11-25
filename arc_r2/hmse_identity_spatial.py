"""Three-way leave-one-training-pair-out comparison for ARC task d89b689b.

The hybrid uses HMSE stages to validate a fixed anchor/quadrant rule family,
then executes the held-out prediction with the identity-preserving object DSL.
No held-out output is passed into synthesis.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from .hmse import (
    MethodRegistry, MethodSpec, Procedure, SearchPartition, Stage,
    ValidationResult, execute_procedure,
)
from .object_identity_dsl import IdentityInstruction, IdentityObjectProgram
from .primitives import as_grid, fit_program
from .hmse_spatial_demo import SpatialProgram


def _identity_spatial_program(grid):
    state = IdentityObjectProgram.initialize(grid)
    components = list(state.objects)
    anchors = []
    for object_id, obj in components:
        r0, c0, r1, c1 = obj.bbox
        expected = {(r0, c0), (r0, c0 + 1), (r0 + 1, c0), (r0 + 1, c0 + 1)}
        if obj.size == 4 and r1 - r0 == 1 and c1 - c0 == 1 and set(obj.cells) == expected:
            anchors.append((object_id, r0, c0))
    if len(anchors) != 1:
        raise ValueError("expected exactly one solid 2x2 anchor")
    anchor_id, top, left = anchors[0]

    markers = [(object_id, obj) for object_id, obj in components if obj.size == 1]
    if len(markers) != 4:
        raise ValueError("expected exactly four singleton markers")
    by_quadrant = {}
    for object_id, obj in markers:
        row, col = next(iter(obj.cells))
        if row < top and col < left:
            quadrant = (0, 0)
        elif row < top and col > left + 1:
            quadrant = (0, 1)
        elif row > top + 1 and col < left:
            quadrant = (1, 0)
        elif row > top + 1 and col > left + 1:
            quadrant = (1, 1)
        else:
            raise ValueError("each marker must be in a distinct diagonal quadrant")
        if quadrant in by_quadrant:
            raise ValueError("multiple markers occupy one quadrant")
        by_quadrant[quadrant] = (object_id, row, col)
    if set(by_quadrant) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        raise ValueError("one marker is required in each diagonal quadrant")

    program = IdentityObjectProgram((
        IdentityInstruction(anchor_id, "REMOVE"),
    ))
    for (dr, dc), (object_id, row, col) in sorted(by_quadrant.items()):
        program = program.then(IdentityInstruction(
            object_id, "MOVE",
            delta_row=top + dr - row,
            delta_col=left + dc - col,
        ))
    return state, program


def _scan(state: Mapping[str, Any], params: Mapping[str, Any]):
    return tuple(len(IdentityObjectProgram.initialize(source).objects)
                 for source, _ in state["training_pairs"])


def _synthesize(state: Mapping[str, Any], params: Mapping[str, Any]):
    evidence = []
    for source, expected in state["training_pairs"]:
        initial, program = _identity_spatial_program(source)
        actual_state = program.apply_state(initial)
        prediction = actual_state.to_grid()
        if prediction != expected:
            raise ValueError("identity DSL spatial proposal did not exactly fit visible examples")
        evidence.append({
            "program_complexity": program.complexity,
            "active_objects_after": len(actual_state.objects),
            "archived_objects_after": len(actual_state.archived_objects),
            "inverse_restores_input": program.inverse().apply_state(actual_state).to_grid() == initial.to_grid(),
        })
    held_initial, held_program = _identity_spatial_program(state["held_out_input"])
    return {
        "held_initial": held_initial,
        "held_program": held_program,
        "visible_pair_evidence": evidence,
    }


def _predict(state: Mapping[str, Any], params: Mapping[str, Any]):
    outcome = state["identity_synthesis"]
    return outcome["held_program"].apply_state(outcome["held_initial"]).to_grid()


def _validate_synthesis(stage, state, output, spec):
    accepted = isinstance(output, dict) and bool(output.get("visible_pair_evidence"))
    return ValidationResult(
        accepted, 1.0 if accepted else 0.0,
        {"visible_pairs_exact": accepted,
         "inverse_checks": sum(item["inverse_restores_input"] for item in output.get("visible_pair_evidence", [])) if accepted else 0},
    )


def _validate_prediction(stage, state, output, spec):
    accepted = output is not None
    return ValidationResult(accepted, 1.0 if accepted else 0.0, {"prediction_produced": accepted})


def _build_hybrid_procedure():
    return Procedure(
        "arc-d89b689b-hmse-identity-spatial-hybrid",
        stages=(
            Stage("identity-perception", "identity_perception", (
                SearchPartition("identity-object-scan", (
                    MethodSpec("identity.scan", requires=("training_pairs",)),
                )),
            ), requires=("training_pairs",), max_evaluations=1),
            Stage("rule-validation-and-synthesis", "identity_synthesis", (
                SearchPartition("identity-spatial-rule", (
                    MethodSpec("identity.synthesize", requires=("training_pairs", "held_out_input")),
                )),
            ), requires=("training_pairs", "held_out_input"),
               validator_id="identity.visible-exact-fit", max_evaluations=1),
            Stage("identity-prediction", "identity_prediction", (
                SearchPartition("identity-program-execution", (
                    MethodSpec("identity.predict", requires=("identity_synthesis",)),
                )),
            ), requires=("identity_synthesis",),
               validator_id="identity.prediction-produced", max_evaluations=1),
        ),
        description="HMSE orchestration with identity-preserving REMOVE/MOVE and inverse validation.",
        input_keys=("training_pairs", "held_out_input"),
    )


def _build_registry():
    registry = MethodRegistry()
    registry.register_method("identity.scan", _scan)
    registry.register_method("identity.synthesize", _synthesize)
    registry.register_method("identity.predict", _predict)
    registry.register_validator("identity.visible-exact-fit", _validate_synthesis)
    registry.register_validator("identity.prediction-produced", _validate_prediction)
    return registry


def run_comparison(task: Mapping[str, Any], candidate_cap: int = 1500, max_depth: int = 2):
    train = task.get("train")
    if not isinstance(train, list) or len(train) < 2:
        raise ValueError("task must contain at least two training pairs")
    pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
    folds = []
    counts = {"primitive": 0, "hmse": 0, "hybrid": 0}
    runtimes = {"primitive": 0.0, "hmse": 0.0, "hybrid": 0.0}
    candidate_evaluations = 0

    for held_index, (held_input, held_output) in enumerate(pairs):
        fit_pairs = tuple(pair for index, pair in enumerate(pairs) if index != held_index)

        started = time.perf_counter()
        baseline = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth)
        primitive_prediction = baseline.program.apply(held_input) if baseline.program else None
        runtimes["primitive"] += time.perf_counter() - started
        candidate_evaluations += baseline.candidates_evaluated
        primitive_exact = primitive_prediction == held_output
        counts["primitive"] += int(primitive_exact)

        started = time.perf_counter()
        try:
            hmse_prediction = SpatialProgram().apply(held_input)
        except (ValueError, TypeError, IndexError):
            hmse_prediction = None
        runtimes["hmse"] += time.perf_counter() - started
        hmse_exact = hmse_prediction == held_output
        counts["hmse"] += int(hmse_exact)

        started = time.perf_counter()
        execution = execute_procedure(
            _build_hybrid_procedure(),
            {"training_pairs": fit_pairs, "held_out_input": held_input},
            _build_registry(),
        )
        hybrid_prediction = execution.state["identity_prediction"]
        runtimes["hybrid"] += time.perf_counter() - started
        hybrid_exact = hybrid_prediction == held_output
        counts["hybrid"] += int(hybrid_exact)

        folds.append({
            "held_out_training_pair": held_index,
            "visible_training_pairs": len(fit_pairs),
            "primitive_exact": primitive_exact,
            "primitive_program": str(baseline.program) if baseline.program else None,
            "primitive_candidates_evaluated": baseline.candidates_evaluated,
            "hmse_exact": hmse_exact,
            "hybrid_exact": hybrid_exact,
            "hybrid_program": [item.canonical() for item in execution.state["identity_synthesis"]["held_program"].instructions],
            "hybrid_visible_pair_evidence": execution.state["identity_synthesis"]["visible_pair_evidence"],
            "hybrid_stage_trace": [
                {"stage_id": item.stage_id, "selected_method": item.selected_method,
                 "evaluations": item.evaluations} for item in execution.stages
            ],
            "hybrid_prediction": [list(row) for row in hybrid_prediction],
            "expected": [list(row) for row in held_output],
            "hybrid_total_stage_evaluations": execution.total_evaluations,
        })

    total = len(folds)
    return {
        "schema": "arc-hmse-identity-spatial-three-way/v1",
        "task_id": str(task.get("task_id", "unknown")),
        "dataset": {"repository": task.get("source_repository"),
                    "revision": task.get("source_revision"),
                    "split": task.get("split", "unspecified")},
        "protocol": "leave-one-training-pair-out",
        "training_pairs": total,
        "official_test_outputs_read": False,
        "candidate_cap_per_fold": candidate_cap,
        "max_depth": max_depth,
        "results": {
            name: {"held_out_exact": count, "folds": total,
                   "accuracy": count / total,
                   "runtime_seconds": round(runtimes[name], 6)}
            for name, count in counts.items()
        },
        "primitive_candidate_evaluations": candidate_evaluations,
        "hybrid_scope_note": (
            "The anchor/quadrant rule family is authored for task d89b689b. "
            "The experiment tests identity-aware execution and reversibility, "
            "not automatic discovery or independent-task generalization."
        ),
        "fold_results": folds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare primitive, HMSE, and identity-HMSE ARC execution")
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_hmse_identity_spatial.json"))
    args = parser.parse_args(argv)
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    report = run_comparison(task, args.candidate_cap, args.max_depth)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Task: {report['task_id']}")
    for name, result in report["results"].items():
        print(f"{name}: {result['held_out_exact']}/{result['folds']} exact")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

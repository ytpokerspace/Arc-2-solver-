"""End-to-end HMSE demonstration on a pinned ARC-AGI-2 training task.

The demo keeps the held-out output outside the solver. Shape Radar detects
enclosed structure; a bounded rule proposer infers a hole-fill color from the
visible training-pair differences and verifies the candidate on every visible
pair. If that proposal does not fit, the existing primitive search is used.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from .hmse import (
    MethodRegistry, MethodSpec, Procedure, ProcedureError, SearchPartition,
    Stage, ValidationResult, execute_procedure,
)
from .primitives import (
    Primitive, Program, as_grid, background_color, fit_program,
)
from .shape_radar import scan_grid


@dataclass(frozen=True)
class SynthesisOutcome:
    program: Program | None
    exact_fit: bool
    strategy: str
    candidates_evaluated: int
    radar_hole_objects: int


def _scan_training_pairs(state: Mapping[str, Any], params: Mapping[str, Any]):
    connectivity = int(params.get("connectivity", 4))
    return tuple(
        scan_grid(source, connectivity=connectivity)
        for source, _ in state["training_pairs"]
    )


def _synthesize_from_evidence(
    state: Mapping[str, Any], params: Mapping[str, Any]
) -> SynthesisOutcome:
    pairs = tuple((as_grid(source), as_grid(target))
                  for source, target in state["training_pairs"])
    scans = state["perception"]
    hole_objects = sum(
        1 for scan in scans for obj in scan["objects"] if obj["holes"] > 0
    )

    # Propose a single fill color only when all observed changes add that
    # color into background cells and preserve every original foreground cell.
    added_colors: set[int] = set()
    compatible = len(scans) == len(pairs)
    for source, target in pairs:
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            compatible = False
            break
        bg = background_color(source)
        for r in range(len(source)):
            for c in range(len(source[0])):
                if source[r][c] == bg and target[r][c] != bg:
                    added_colors.add(target[r][c])
                elif source[r][c] != target[r][c]:
                    compatible = False
                    break
            if not compatible:
                break
        if not compatible:
            break

    if compatible and hole_objects > 0 and len(added_colors) == 1:
        color = next(iter(added_colors))
        candidate = Program((Primitive("FILL_HOLES", (color,)),))
        if all(candidate.apply(source) == target for source, target in pairs):
            return SynthesisOutcome(candidate, True, "shape-radar-hole-rule", 1, hole_objects)

    # Bounded fallback to the existing general primitive enumerator.
    result = fit_program(
        pairs,
        candidate_cap=int(params.get("candidate_cap", 1500)),
        max_depth=int(params.get("max_depth", 2)),
    )
    return SynthesisOutcome(
        result.program, result.exact_fit, "primitive-search-fallback",
        result.candidates_evaluated, hole_objects,
    )


def _apply_synthesized_program(state: Mapping[str, Any], params: Mapping[str, Any]):
    outcome: SynthesisOutcome = state["synthesis"]
    if outcome.program is None:
        return None
    return outcome.program.apply(as_grid(state["held_out_input"]))


def _validate_exact_fit(stage, state, output, spec) -> ValidationResult:
    accepted = (
        isinstance(output, SynthesisOutcome)
        and output.program is not None
        and output.exact_fit
    )
    return ValidationResult(
        accepted,
        1.0 if accepted else 0.0,
        {
            "exact_fit": bool(accepted),
            "strategy": output.strategy if isinstance(output, SynthesisOutcome) else None,
            "candidates_evaluated": (
                output.candidates_evaluated if isinstance(output, SynthesisOutcome) else 0
            ),
        },
    )


def build_demo_procedure(candidate_cap: int = 1500, max_depth: int = 2) -> Procedure:
    """Build a fixed, inspectable three-stage schedule for this demonstration."""
    return Procedure(
        "arc-00d62c1b-shape-radar-hmse-demo",
        stages=(
            Stage(
                "perception", "perception",
                (SearchPartition("shape-radar", (
                    MethodSpec("demo.scan-training-pairs", requires=("training_pairs",)),
                )),),
                requires=("training_pairs",),
                max_evaluations=1,
            ),
            Stage(
                "rule-synthesis", "synthesis",
                (SearchPartition("rule-proposals", (
                    MethodSpec(
                        "demo.synthesize-from-evidence",
                        {"candidate_cap": candidate_cap, "max_depth": max_depth},
                        ("training_pairs", "perception"),
                    ),
                )),),
                requires=("training_pairs", "perception"),
                validator_id="demo.exact-fit",
                max_evaluations=1,
            ),
            Stage(
                "prediction", "prediction",
                (SearchPartition("program-application", (
                    MethodSpec(
                        "demo.apply-program",
                        requires=("synthesis", "held_out_input"),
                    ),
                )),),
                requires=("synthesis", "held_out_input"),
                max_evaluations=1,
            ),
        ),
        description=(
            "Shape Radar perception, evidence-based rule proposal with bounded "
            "primitive-search fallback, then held-out prediction."
        ),
        input_keys=("training_pairs", "held_out_input"),
    )


def build_registry() -> MethodRegistry:
    registry = MethodRegistry()
    registry.register_method("demo.scan-training-pairs", _scan_training_pairs)
    registry.register_method("demo.synthesize-from-evidence", _synthesize_from_evidence)
    registry.register_method("demo.apply-program", _apply_synthesized_program)
    registry.register_validator("demo.exact-fit", _validate_exact_fit)
    return registry


def run_comparison(
    task: Mapping[str, Any], candidate_cap: int = 1500, max_depth: int = 2
) -> dict[str, Any]:
    train = task.get("train")
    if not isinstance(train, list) or len(train) < 2:
        raise ValueError("task must contain at least two training pairs")

    pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
    folds = []
    baseline_started = time.perf_counter()
    baseline_total_candidates = 0
    baseline_exact = 0
    hmse_started = time.perf_counter()
    hmse_total_candidates = 0
    hmse_exact = 0

    for held_index, (held_input, held_output) in enumerate(pairs):
        fit_pairs = tuple(pair for index, pair in enumerate(pairs) if index != held_index)

        baseline = fit_program(
            fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth
        )
        baseline_total_candidates += baseline.candidates_evaluated
        baseline_prediction = (
            baseline.program.apply(held_input) if baseline.program is not None else None
        )
        baseline_match = baseline_prediction == held_output
        baseline_exact += int(baseline_match)

        execution = execute_procedure(
            build_demo_procedure(candidate_cap, max_depth),
            {"training_pairs": fit_pairs, "held_out_input": held_input},
            build_registry(),
        )
        outcome: SynthesisOutcome = execution.state["synthesis"]
        prediction = execution.state["prediction"]
        hmse_total_candidates += outcome.candidates_evaluated
        hmse_match = prediction == held_output
        hmse_exact += int(hmse_match)

        folds.append({
            "held_out_training_pair": held_index,
            "visible_training_pairs": len(fit_pairs),
            "baseline_program": str(baseline.program) if baseline.program else None,
            "baseline_candidates_evaluated": baseline.candidates_evaluated,
            "baseline_exact": baseline_match,
            "hmse_program": str(outcome.program) if outcome.program else None,
            "hmse_strategy": outcome.strategy,
            "hmse_candidates_evaluated": outcome.candidates_evaluated,
            "hmse_radar_hole_objects": outcome.radar_hole_objects,
            "hmse_stage_trace": [
                {
                    "stage_id": stage.stage_id,
                    "selected_method": stage.selected_method,
                    "evaluations": stage.evaluations,
                }
                for stage in execution.stages
            ],
            "hmse_exact": hmse_match,
            "prediction": [list(row) for row in prediction] if prediction is not None else None,
            "expected": [list(row) for row in held_output],
            "hmse_total_stage_evaluations": execution.total_evaluations,
        })

    baseline_runtime = time.perf_counter() - baseline_started
    hmse_runtime = time.perf_counter() - hmse_started
    return {
        "schema": "arc-hmse-single-task-demo/v1",
        "task_id": str(task.get("task_id", "unknown")),
        "dataset": {
            "repository": task.get("source_repository"),
            "revision": task.get("source_revision"),
            "split": task.get("split", "unspecified"),
        },
        "protocol": "leave-one-training-pair-out",
        "training_pairs": len(pairs),
        "official_test_outputs_read": False,
        "candidate_cap_per_fold": candidate_cap,
        "max_depth": max_depth,
        "baseline": {
            "engine": "existing primitive fit_program",
            "held_out_exact": baseline_exact,
            "folds": len(folds),
            "accuracy": baseline_exact / len(folds),
            "candidate_evaluations": baseline_total_candidates,
            "runtime_seconds": round(baseline_runtime, 6),
        },
        "hmse": {
            "engine": "Shape Radar + fixed HMSE schedule + evidence-based hole rule",
            "held_out_exact": hmse_exact,
            "folds": len(folds),
            "accuracy": hmse_exact / len(folds),
            "candidate_evaluations": hmse_total_candidates,
            "runtime_seconds": round(hmse_runtime, 6),
            "scope_note": (
                "The three-stage schedule is authored for this demonstration; "
                "automatic schedule invention is not claimed."
            ),
        },
        "fold_results": folds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the single-task Shape Radar + HMSE demo")
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_hmse_demo.json"))
    args = parser.parse_args(argv)
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    report = run_comparison(task, args.candidate_cap, args.max_depth)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Task: {report['task_id']}")
    print(f"Baseline: {report['baseline']['held_out_exact']}/{report['baseline']['folds']} exact")
    print(f"Shape Radar + HMSE: {report['hmse']['held_out_exact']}/{report['hmse']['folds']} exact")
    print(f"HMSE rule strategies: {sorted({f['hmse_strategy'] for f in report['fold_results']})}")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

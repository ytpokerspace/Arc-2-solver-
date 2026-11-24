"""Evidence-driven selection between known ARC rule families.

The selector is not told which task family it is evaluating. It proposes every
registered family, checks each proposal against every visible training pair,
and selects only when exactly one family has an exact fit. Zero or multiple
fits are reported as unresolved rather than guessed.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .hmse import (
    MethodRegistry, MethodSpec, Procedure, SearchPartition, Stage,
    ValidationResult, execute_procedure,
)
from .hmse_demo import _background_color if False else None
from .primitives import Primitive, Program, as_grid, background_color, fit_program
from .shape_radar import scan_grid
from .hmse_spatial_demo import SpatialProgram


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    program: Any | None
    accepted: bool
    reason: str
    training_pairs_checked: int


@dataclass(frozen=True)
class SelectionOutcome:
    selected_rule: str | None
    selected_program: Any | None
    status: str
    candidates: tuple[RuleCandidate, ...]


@dataclass(frozen=True)
class PredictionOutcome:
    prediction: tuple[tuple[int, ...], ...] | None
    selected_rule: str | None
    status: str


def _hole_fill_candidate(pairs, scans) -> RuleCandidate:
    if len(scans) != len(pairs):
        return RuleCandidate("hole-fill", None, False, "perception-count-mismatch", len(pairs))
    hole_objects = sum(
        1 for scan in scans for obj in scan["objects"] if obj["holes"] > 0
    )
    if hole_objects == 0:
        return RuleCandidate("hole-fill", None, False, "no-enclosed-shapes-observed", len(pairs))

    added_colors = set()
    for source, target in pairs:
        if len(source) != len(target) or len(source[0]) != len(target[0]):
            return RuleCandidate("hole-fill", None, False, "grid-dimensions-changed", len(pairs))
        bg = background_color(source)
        for r in range(len(source)):
            for c in range(len(source[0])):
                if source[r][c] == bg and target[r][c] != bg:
                    added_colors.add(target[r][c])
                elif source[r][c] != target[r][c]:
                    return RuleCandidate("hole-fill", None, False, "existing-foreground-changed", len(pairs))
    if len(added_colors) != 1:
        return RuleCandidate("hole-fill", None, False, "added-color-not-unique", len(pairs))

    program = Program((Primitive("FILL_HOLES", (next(iter(added_colors)),)),))
    try:
        exact = all(program.apply(source) == target for source, target in pairs)
    except (ValueError, TypeError, IndexError):
        exact = False
    if not exact:
        return RuleCandidate("hole-fill", None, False, "proposal-failed-exact-fit", len(pairs))
    return RuleCandidate("hole-fill", program, True, "exact-fit", len(pairs))


def _spatial_candidate(pairs) -> RuleCandidate:
    program = SpatialProgram()
    try:
        exact = all(program.apply(source) == target for source, target in pairs)
    except (ValueError, TypeError, IndexError):
        exact = False
    if not exact:
        return RuleCandidate("anchor-quadrant-relocation", None, False,
                             "proposal-failed-exact-fit", len(pairs))
    return RuleCandidate("anchor-quadrant-relocation", program, True, "exact-fit", len(pairs))


def _scan_training_pairs(state: Mapping[str, Any], params: Mapping[str, Any]):
    return tuple(scan_grid(source, connectivity=4)
                 for source, _ in state["training_pairs"])


def _select_rule(state: Mapping[str, Any], params: Mapping[str, Any]):
    pairs = tuple((as_grid(source), as_grid(target))
                  for source, target in state["training_pairs"])
    candidates = (
        _hole_fill_candidate(pairs, state["perception"]),
        _spatial_candidate(pairs),
    )
    accepted = tuple(candidate for candidate in candidates if candidate.accepted)
    if len(accepted) == 1:
        chosen = accepted[0]
        return SelectionOutcome(chosen.rule_id, chosen.program, "selected-unique-exact-fit", candidates)
    if not accepted:
        return SelectionOutcome(None, None, "no-rule-family-exact-fit", candidates)
    return SelectionOutcome(None, None, "ambiguous-multiple-exact-fits", candidates)


def _predict(state: Mapping[str, Any], params: Mapping[str, Any]):
    selection: SelectionOutcome = state["selection"]
    if selection.selected_program is None:
        return PredictionOutcome(None, None, selection.status)
    try:
        prediction = selection.selected_program.apply(as_grid(state["held_out_input"]))
    except (ValueError, TypeError, IndexError):
        return PredictionOutcome(None, selection.selected_rule, "selected-rule-rejected-held-out-input")
    return PredictionOutcome(prediction, selection.selected_rule, "predicted")


def _accept_selection(stage, state, output, spec):
    accepted = isinstance(output, SelectionOutcome)
    return ValidationResult(accepted, 1.0 if accepted else 0.0,
                            {"status": output.status if accepted else "invalid-selection"})


def _accept_prediction(stage, state, output, spec):
    accepted = isinstance(output, PredictionOutcome)
    return ValidationResult(accepted, 1.0 if accepted else 0.0,
                            {"status": output.status if accepted else "invalid-prediction"})


def build_procedure():
    return Procedure(
        "arc-shared-rule-selector",
        stages=(
            Stage("perception", "perception", (
                SearchPartition("shape-radar", (
                    MethodSpec("selector.scan-training-pairs", requires=("training_pairs",)),
                )),
            ), requires=("training_pairs",), max_evaluations=1),
            Stage("rule-selection", "selection", (
                SearchPartition("registered-rule-families", (
                    MethodSpec("selector.evaluate-rule-families",
                               requires=("training_pairs", "perception")),
                )),
            ), requires=("training_pairs", "perception"),
               validator_id="selector.accept-selection", max_evaluations=1),
            Stage("prediction", "prediction", (
                SearchPartition("selected-rule-application", (
                    MethodSpec("selector.apply-selected-rule",
                               requires=("selection", "held_out_input")),
                )),
            ), requires=("selection", "held_out_input"),
               validator_id="selector.accept-prediction", max_evaluations=1),
        ),
        description="Perceive, evaluate all registered rule families, select a unique exact fit, predict.",
        input_keys=("training_pairs", "held_out_input"),
    )


def build_registry():
    registry = MethodRegistry()
    registry.register_method("selector.scan-training-pairs", _scan_training_pairs)
    registry.register_method("selector.evaluate-rule-families", _select_rule)
    registry.register_method("selector.apply-selected-rule", _predict)
    registry.register_validator("selector.accept-selection", _accept_selection)
    registry.register_validator("selector.accept-prediction", _accept_prediction)
    return registry


def run_task(task: Mapping[str, Any], candidate_cap: int = 1500, max_depth: int = 2):
    train = task.get("train")
    if not isinstance(train, list) or len(train) < 2:
        raise ValueError("task must contain at least two training pairs")
    pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
    folds = []
    baseline_exact = baseline_candidates = 0
    selected_exact = selected_count = ambiguous_count = unresolved_count = 0
    baseline_runtime = selector_runtime = 0.0

    for held_index, (held_input, held_output) in enumerate(pairs):
        fit_pairs = tuple(pair for i, pair in enumerate(pairs) if i != held_index)
        started = time.perf_counter()
        baseline = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth)
        baseline_prediction = baseline.program.apply(held_input) if baseline.program else None
        baseline_runtime += time.perf_counter() - started
        baseline_match = baseline_prediction == held_output
        baseline_exact += int(baseline_match)
        baseline_candidates += baseline.candidates_evaluated

        started = time.perf_counter()
        execution = execute_procedure(
            build_procedure(),
            {"training_pairs": fit_pairs, "held_out_input": held_input},
            build_registry(),
        )
        selector_runtime += time.perf_counter() - started
        selection: SelectionOutcome = execution.state["selection"]
        prediction: PredictionOutcome = execution.state["prediction"]
        if selection.selected_rule is not None:
            selected_count += 1
        if selection.status == "ambiguous-multiple-exact-fits":
            ambiguous_count += 1
        elif selection.status == "no-rule-family-exact-fit":
            unresolved_count += 1
        match = prediction.prediction == held_output
        selected_exact += int(match)
        folds.append({
            "held_out_training_pair": held_index,
            "visible_training_pairs": len(fit_pairs),
            "baseline_exact": baseline_match,
            "baseline_program": str(baseline.program) if baseline.program else None,
            "baseline_candidates_evaluated": baseline.candidates_evaluated,
            "selector_status": selection.status,
            "selected_rule": selection.selected_rule,
            "candidate_results": [
                {"rule_id": item.rule_id, "accepted": item.accepted,
                 "reason": item.reason, "training_pairs_checked": item.training_pairs_checked}
                for item in selection.candidates
            ],
            "prediction_status": prediction.status,
            "selector_exact": match,
            "prediction": [list(row) for row in prediction.prediction]
                          if prediction.prediction is not None else None,
            "expected": [list(row) for row in held_output],
            "stage_trace": [
                {"stage_id": stage.stage_id, "selected_method": stage.selected_method,
                 "evaluations": stage.evaluations}
                for stage in execution.stages
            ],
        })

    return {
        "task_id": str(task.get("task_id", "unknown")),
        "dataset": {"repository": task.get("source_repository"),
                    "revision": task.get("source_revision"),
                    "split": task.get("split", "unspecified")},
        "protocol": "leave-one-training-pair-out",
        "training_pairs": len(pairs),
        "held_out_outputs_visible_to_selector": False,
        "rule_families": ["hole-fill", "anchor-quadrant-relocation"],
        "baseline": {"exact": baseline_exact, "folds": len(folds),
                     "candidate_evaluations": baseline_candidates,
                     "runtime_seconds": round(baseline_runtime, 6)},
        "selector": {"exact": selected_exact, "folds": len(folds),
                     "selected": selected_count, "ambiguous": ambiguous_count,
                     "unresolved": unresolved_count,
                     "candidate_family_checks": 2 * len(folds),
                     "runtime_seconds": round(selector_runtime, 6)},
        "fold_results": folds,
    }


def run_suite(tasks, candidate_cap: int = 1500, max_depth: int = 2):
    reports = [run_task(task, candidate_cap, max_depth) for task in tasks]
    total_folds = sum(item["selector"]["folds"] for item in reports)
    total_exact = sum(item["selector"]["exact"] for item in reports)
    return {
        "schema": "arc-hmse-shared-rule-selector/v1",
        "protocol": "leave-one-training-pair-out",
        "task_count": len(reports),
        "total_folds": total_folds,
        "selector_exact": total_exact,
        "selector_accuracy": total_exact / total_folds if total_folds else 0.0,
        "tasks": reports,
        "interpretation_limits": [
            "Only two authored rule families are registered; this is not open-ended ARC rule discovery.",
            "The selector chooses only a unique exact fit; ambiguous or unsupported cases remain unresolved.",
            "Folds are correlated examples from two tasks, not independent benchmark tasks.",
            "Baseline and selector candidate counts measure different search spaces and are not an efficiency comparison.",
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run shared Shape Radar + HMSE rule selector")
    parser.add_argument("--task-json", action="append", required=True, type=Path,
                        help="Training-only task JSON; repeat to evaluate multiple tasks.")
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_hmse_selector_report.json"))
    args = parser.parse_args(argv)
    tasks = [json.loads(path.read_text(encoding="utf-8")) for path in args.task_json]
    report = run_suite(tasks, args.candidate_cap, args.max_depth)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Tasks: {report['task_count']}")
    print(f"Selector: {report['selector_exact']}/{report['total_folds']} exact")
    for item in report["tasks"]:
        print(f"  {item['task_id']}: {item['selector']['exact']}/{item['selector']['folds']} "
              f"exact; selected={item['selector']['selected']}; "
              f"ambiguous={item['selector']['ambiguous']}; unresolved={item['selector']['unresolved']}")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

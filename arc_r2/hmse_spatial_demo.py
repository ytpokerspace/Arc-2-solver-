"""Spatial-relocation HMSE demonstration on ARC-AGI-2 task d89b689b.

The solver sees only training pairs. A structural proposer detects a 2x2
anchor and four singleton markers, then maps marker quadrants to anchor cells.
The held-out output is reserved for the outer evaluator.
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
from .primitives import as_grid, fit_program
from .shape_radar import scan_grid


def _components(grid):
    h, w = len(grid), len(grid[0])
    seen = set()
    found = []
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                    if 0 <= ny < h and 0 <= nx < w and (ny, nx) not in seen and grid[ny][nx] == color:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            found.append((color, tuple(sorted(cells))))
    return tuple(found)


def _background(grid):
    counts = {}
    for row in grid:
        for color in row:
            counts[color] = counts.get(color, 0) + 1
    return 0 if 0 in counts else min(counts, key=lambda c: (-counts[c], c))


@dataclass(frozen=True)
class SpatialProgram:
    """Map one singleton from each quadrant into a central 2x2 anchor."""

    def apply(self, value):
        grid = as_grid(value)
        components = _components(grid)
        anchors = []
        for color, cells in components:
            if len(cells) != 4:
                continue
            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            if r1 - r0 == 1 and c1 - c0 == 1 and set(cells) == {
                (r0, c0), (r0, c1), (r1, c0), (r1, c1)
            }:
                anchors.append((color, r0, c0, r1, c1))
        if len(anchors) != 1:
            raise ValueError("expected exactly one solid 2x2 anchor")
        _, r0, c0, r1, c1 = anchors[0]
        markers = [(color, cells[0]) for color, cells in components
                   if len(cells) == 1]
        if len(markers) != 4:
            raise ValueError("expected exactly four singleton markers")

        positions = {}
        for color, (r, c) in markers:
            if r < r0 and c < c0:
                quadrant = (0, 0)
            elif r < r0 and c > c1:
                quadrant = (0, 1)
            elif r > r1 and c < c0:
                quadrant = (1, 0)
            elif r > r1 and c > c1:
                quadrant = (1, 1)
            else:
                raise ValueError("each marker must occupy a distinct diagonal quadrant")
            if quadrant in positions:
                raise ValueError("multiple markers occupy the same quadrant")
            positions[quadrant] = color
        if set(positions) != {(0, 0), (0, 1), (1, 0), (1, 1)}:
            raise ValueError("one marker is required in each diagonal quadrant")

        output = [list(row) for row in grid]
        bg = _background(grid)
        for _, (r, c) in markers:
            output[r][c] = bg
        for (dr, dc), color in positions.items():
            output[r0 + dr][c0 + dc] = color
        return tuple(tuple(row) for row in output)


@dataclass(frozen=True)
class SynthesisOutcome:
    program: SpatialProgram | None
    exact_fit: bool
    strategy: str
    candidates_evaluated: int
    radar_object_count: int


def _scan_training_pairs(state: Mapping[str, Any], params: Mapping[str, Any]):
    return tuple(scan_grid(source, connectivity=4)
                 for source, _ in state["training_pairs"])


def _synthesize_from_evidence(state: Mapping[str, Any], params: Mapping[str, Any]):
    pairs = tuple((as_grid(source), as_grid(target))
                  for source, target in state["training_pairs"])
    radar_objects = sum(len(scan["objects"]) for scan in state["perception"])
    candidate = SpatialProgram()
    try:
        exact = all(candidate.apply(source) == target for source, target in pairs)
    except (ValueError, TypeError, IndexError):
        exact = False
    if not exact:
        raise ValueError("spatial mapping proposal did not exactly fit visible examples")
    return SynthesisOutcome(candidate, True, "anchor-quadrant-relocation", 1, radar_objects)


def _apply_program(state: Mapping[str, Any], params: Mapping[str, Any]):
    outcome = state["synthesis"]
    if outcome.program is None:
        return None
    return outcome.program.apply(state["held_out_input"])


def _validate_fit(stage, state, output, spec):
    accepted = isinstance(output, SynthesisOutcome) and output.program is not None and output.exact_fit
    return ValidationResult(accepted, 1.0 if accepted else 0.0,
                           {"exact_fit": accepted,
                            "strategy": output.strategy if isinstance(output, SynthesisOutcome) else None})


def build_demo_procedure():
    return Procedure(
        "arc-d89b689b-spatial-hmse-demo",
        stages=(
            Stage("perception", "perception", (
                SearchPartition("shape-radar", (
                    MethodSpec("spatial.scan-training-pairs", requires=("training_pairs",)),
                )),
            ), requires=("training_pairs",), max_evaluations=1),
            Stage("rule-synthesis", "synthesis", (
                SearchPartition("spatial-rule-proposals", (
                    MethodSpec("spatial.synthesize-from-evidence",
                               requires=("training_pairs", "perception")),
                )),
            ), requires=("training_pairs", "perception"),
               validator_id="spatial.exact-fit", max_evaluations=1),
            Stage("prediction", "prediction", (
                SearchPartition("program-application", (
                    MethodSpec("spatial.apply-program",
                               requires=("synthesis", "held_out_input")),
                )),
            ), requires=("synthesis", "held_out_input"), max_evaluations=1),
        ),
        description="Shape Radar perception, anchor-relative spatial rule proposal, held-out prediction.",
        input_keys=("training_pairs", "held_out_input"),
    )


def build_registry():
    registry = MethodRegistry()
    registry.register_method("spatial.scan-training-pairs", _scan_training_pairs)
    registry.register_method("spatial.synthesize-from-evidence", _synthesize_from_evidence)
    registry.register_method("spatial.apply-program", _apply_program)
    registry.register_validator("spatial.exact-fit", _validate_fit)
    return registry


def run_comparison(task: Mapping[str, Any], candidate_cap: int = 1500, max_depth: int = 2):
    train = task.get("train")
    if not isinstance(train, list) or len(train) < 2:
        raise ValueError("task must contain at least two training pairs")
    pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
    folds = []
    baseline_runtime = hmse_runtime = 0.0
    baseline_candidates = baseline_exact = hmse_exact = 0
    for held_index, (held_input, held_output) in enumerate(pairs):
        fit_pairs = tuple(pair for i, pair in enumerate(pairs) if i != held_index)

        started = time.perf_counter()
        baseline = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth)
        baseline_prediction = baseline.program.apply(held_input) if baseline.program else None
        baseline_runtime += time.perf_counter() - started
        baseline_candidates += baseline.candidates_evaluated
        baseline_match = baseline_prediction == held_output
        baseline_exact += int(baseline_match)

        started = time.perf_counter()
        execution = execute_procedure(
            build_demo_procedure(),
            {"training_pairs": fit_pairs, "held_out_input": held_input},
            build_registry(),
        )
        hmse_runtime += time.perf_counter() - started
        outcome = execution.state["synthesis"]
        prediction = execution.state["prediction"]
        hmse_match = prediction == held_output
        hmse_exact += int(hmse_match)
        folds.append({
            "held_out_training_pair": held_index,
            "visible_training_pairs": len(fit_pairs),
            "baseline_program": str(baseline.program) if baseline.program else None,
            "baseline_candidates_evaluated": baseline.candidates_evaluated,
            "baseline_exact": baseline_match,
            "hmse_program": "ANCHOR_QUADRANT_RELOCATION",
            "hmse_strategy": outcome.strategy,
            "hmse_candidates_evaluated": outcome.candidates_evaluated,
            "hmse_radar_object_count": outcome.radar_object_count,
            "hmse_stage_trace": [
                {"stage_id": item.stage_id, "selected_method": item.selected_method,
                 "evaluations": item.evaluations} for item in execution.stages
            ],
            "hmse_exact": hmse_match,
            "prediction": [list(row) for row in prediction],
            "expected": [list(row) for row in held_output],
            "hmse_total_stage_evaluations": execution.total_evaluations,
        })
    return {
        "schema": "arc-hmse-single-task-demo/v1",
        "task_id": str(task.get("task_id", "unknown")),
        "dataset": {"repository": task.get("source_repository"),
                    "revision": task.get("source_revision"),
                    "split": task.get("split", "unspecified")},
        "protocol": "leave-one-training-pair-out",
        "training_pairs": len(pairs),
        "official_test_outputs_read": False,
        "candidate_cap_per_fold": candidate_cap,
        "max_depth": max_depth,
        "baseline": {"engine": "existing primitive fit_program",
                     "held_out_exact": baseline_exact, "folds": len(folds),
                     "accuracy": baseline_exact / len(folds),
                     "candidate_evaluations": baseline_candidates,
                     "runtime_seconds": round(baseline_runtime, 6)},
        "hmse": {"engine": "Shape Radar + fixed HMSE schedule + anchor-quadrant rule",
                 "held_out_exact": hmse_exact, "folds": len(folds),
                 "accuracy": hmse_exact / len(folds),
                 "candidate_evaluations": len(folds),
                 "runtime_seconds": round(hmse_runtime, 6),
                 "scope_note": "The stage schedule and spatial rule family are authored for this task; automatic schedule invention and general rule induction are not claimed."},
        "fold_results": folds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run spatial Shape Radar + HMSE demonstration")
    parser.add_argument("--task-json", required=True, type=Path)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_hmse_spatial_demo.json"))
    args = parser.parse_args(argv)
    task = json.loads(args.task_json.read_text(encoding="utf-8"))
    report = run_comparison(task, args.candidate_cap, args.max_depth)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Task: {report['task_id']}")
    print(f"Baseline: {report['baseline']['held_out_exact']}/{report['baseline']['folds']} exact")
    print(f"Shape Radar + HMSE: {report['hmse']['held_out_exact']}/{report['hmse']['folds']} exact")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

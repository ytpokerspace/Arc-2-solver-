"""Deterministic stress matrix for semantic object identity matching.

All scenarios are synthetic component-association tests. They do not use ARC
test outputs and do not measure ARC task-solving accuracy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grid import as_grid
from .objects import GridObject, detect_objects
from .object_identity_dsl import IdentityObjectProgram
from .semantic_identity import match_identities


def _placed(color: int, shape: tuple[tuple[int, int], ...], top: int, left: int,
            turns: int = 0) -> GridObject:
    points = list(shape)
    for _ in range(turns % 4):
        points = [(c, -r) for r, c in points]
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    cells = frozenset((top + r - min_r, left + c - min_c) for r, c in points)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return GridObject(color, cells, (min(rows), min(cols), max(rows), max(cols)))


def _grid(objects: tuple[GridObject, ...], height: int = 20, width: int = 20):
    cells = [[0 for _ in range(width)] for _ in range(height)]
    for obj in objects:
        for r, c in obj.cells:
            if cells[r][c] != 0:
                raise ValueError("stress fixture objects overlap")
            cells[r][c] = obj.color
    return as_grid(cells)


def _case(reference_objects: tuple[GridObject, ...], target_objects: tuple[GridObject, ...],
          expected_colors: dict[int, int], *, expected_ambiguous: int = 0) -> dict:
    reference_grid = _grid(reference_objects)
    target_grid = _grid(target_objects)
    state = IdentityObjectProgram.initialize(reference_grid)
    result = match_identities(state, target_grid)
    refs_by_color = {}
    for object_id, obj in state.objects:
        refs_by_color.setdefault(obj.color, []).append(object_id)
    target_by_color = {obj.color: i for i, obj in enumerate(detect_objects(target_grid))}
    expected_ids = {
        object_id: target_by_color[color]
        for color, index in expected_colors.items()
        for object_id in refs_by_color[color]
    }
    predicted = {match.object_id: match.target_index for match in result.matches}
    correct = sum(predicted.get(oid) == index for oid, index in expected_ids.items())
    incorrect = sum(oid in predicted and predicted[oid] != index for oid, index in expected_ids.items())
    return {
        "reference_objects": len(state.objects),
        "target_objects": len(detect_objects(target_grid)),
        "expected_matches": len(expected_ids),
        "correct": correct,
        "incorrect_assignments": incorrect,
        "assigned": len(result.matches),
        "abstained": len(result.unmatched_reference),
        "ambiguous_reference": list(result.ambiguous_reference),
        "unmatched_reference": list(result.unmatched_reference),
        "unmatched_target": list(result.unmatched_target),
        "expected_ambiguous": expected_ambiguous,
        "expected_unmatched_reference": len(state.objects) - len(expected_ids) - expected_ambiguous,
    }


def _scenario_passed(name: str, case: dict) -> bool:
    checks = {
        "far_relocation_rotation_order_change": (
            case["correct"] == 3 and case["incorrect_assignments"] == 0
            and case["abstained"] == 0
        ),
        "repeated_identical_objects": (
            case["assigned"] == 0 and len(case["ambiguous_reference"]) == 2
            and case["incorrect_assignments"] == 0
        ),
        "one_missing_object": (
            case["correct"] == 2 and case["expected_unmatched_reference"] == 1
            and case["incorrect_assignments"] == 0
        ),
        "one_new_object": (
            case["correct"] == 2 and len(case["unmatched_target"]) == 1
            and case["incorrect_assignments"] == 0
        ),
    }
    return checks[name]


def _render_summary(report: dict) -> str:
    lines = [
        "# Semantic identity stress matrix",
        "",
        f"**Overall:** {'PASS' if report['passed'] else 'FAIL'}",
        "",
        "| Scenario | Result | Correct | Incorrect | Abstained |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, case in report["scenarios"].items():
        lines.append(
            f"| {name} | {'PASS' if case['passed'] else 'FAIL'} | "
            f"{case['correct']} | {case['incorrect_assignments']} | {case['abstained']} |"
        )
    lines.extend([
        "",
        f"Failed scenarios: {', '.join(report['failed_scenarios']) or 'none'}",
        "",
        "> Synthetic object-association tests only. This report does not measure ARC task accuracy or generalization.",
        "",
    ])
    return "\n".join(lines)


def run_stress_matrix() -> dict:
    l_shape = ((0, 0), (0, 1), (1, 0))
    domino = ((0, 0), (0, 1))
    square = ((0, 0), (0, 1), (1, 0), (1, 1))
    single = ((0, 0),)

    cases = {
        "far_relocation_rotation_order_change": _case(
            (_placed(1, l_shape, 1, 1), _placed(2, domino, 2, 8), _placed(3, square, 8, 2)),
            (_placed(3, square, 1, 15), _placed(2, domino, 15, 15, 1), _placed(1, l_shape, 15, 1, 1)),
            {1: 1, 2: 2, 3: 0},
        ),
        "repeated_identical_objects": _case(
            (_placed(4, single, 2, 2), _placed(4, single, 2, 12)),
            (_placed(4, single, 15, 2), _placed(4, single, 15, 12)),
            {},
            expected_ambiguous=2,
        ),
        "one_missing_object": _case(
            (_placed(1, single, 1, 1), _placed(2, domino, 2, 8), _placed(3, square, 8, 2)),
            (_placed(3, square, 15, 15), _placed(1, single, 15, 1)),
            {1: 1, 3: 0},
        ),
        "one_new_object": _case(
            (_placed(1, single, 1, 1), _placed(2, domino, 2, 8)),
            (_placed(3, l_shape, 15, 15), _placed(2, domino, 15, 8), _placed(1, single, 15, 1)),
            {1: 2, 2: 1},
        ),
    }
    for name, case in cases.items():
        case["passed"] = _scenario_passed(name, case)
    failed = [name for name, case in cases.items() if not case["passed"]]
    report = {
        "benchmark_type": "synthetic_semantic_identity_stress_matrix",
        "arc_task_accuracy_claim": False,
        "scenario_count": len(cases),
        "passed": not failed,
        "failed_scenarios": failed,
        "scenarios": cases,
        "pass_criteria": {
            "far_relocation_rotation_order_change": "3 correct; 0 incorrect; 0 abstained",
            "repeated_identical_objects": "0 assignments; 2 ambiguous abstentions; 0 incorrect",
            "one_missing_object": "2 correct; 1 unmatched reference; 0 incorrect",
            "one_new_object": "2 correct; 1 unmatched target; 0 incorrect",
        },
        "note": "Synthetic object association only; no ARC task accuracy or generalization claim.",
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report = run_stress_matrix()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if args.summary:
        args.summary.write_text(_render_summary(report), encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

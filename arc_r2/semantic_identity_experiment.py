"""Controlled preflight experiment for semantic object identity matching.

This is a synthetic component-level benchmark, not an ARC task accuracy score.
Ground-truth identities are known from construction and are independent of
object detection order. No test outputs or external services are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .grid import as_grid
from .objects import GridObject, detect_objects
from .object_identity_dsl import IdentityState
from .semantic_identity import match_identities


def _object(color: int, cells: set[tuple[int, int]]) -> GridObject:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return GridObject(color, frozenset(cells),
                      (min(rows), min(cols), max(rows), max(cols)))


def _placed(color: int, shape: tuple[tuple[int, int], ...],
            top: int, left: int, rotate: bool = False) -> GridObject:
    points = []
    for r, c in shape:
        if rotate:
            r, c = c, -r
        points.append((top + r, left + c))
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return _object(color, {(r - min_r, c - min_c) for r, c in points})


def _grid(objects: tuple[GridObject, ...], height: int = 20,
          width: int = 20):
    cells = [[0 for _ in range(width)] for _ in range(height)]
    for obj in objects:
        for r, c in obj.cells:
            if cells[r][c] != 0:
                raise ValueError("synthetic objects overlap")
            cells[r][c] = obj.color
    return as_grid(cells)


def run_experiment() -> dict:
    # Distinct colors and shapes provide unambiguous construction-time labels.
    shapes = {
        "obj-0000": (1, ((0, 0), (0, 1), (1, 0))),       # L triomino
        "obj-0001": (2, ((0, 0), (0, 1))),               # horizontal domino
        "obj-0002": (3, ((0, 0), (0, 1), (1, 0), (1, 1))),# square
    }
    reference_objects = tuple(
        (oid, _placed(color, shape, top, left))
        for (oid, (color, shape)), (top, left) in zip(
            shapes.items(), ((1, 1), (2, 8), (8, 2)))
    )
    reference = IdentityState(20, 20, 0, reference_objects)

    # Move all objects to new regions, reverse their detection order, and
    # rotate the domino. Their stable IDs are retained only as ground truth.
    target_objects = (
        _placed(3, shapes["obj-0002"][1], 1, 15),
        _placed(2, shapes["obj-0001"][1], 15, 15, rotate=True),
        _placed(1, shapes["obj-0000"][1], 15, 1, rotate=True),
    )
    target_grid = _grid(target_objects)
    detected = detect_objects(target_grid)
    expected_by_color = {obj.color: index for index, obj in enumerate(detected)}
    expected = {
        oid: expected_by_color[color]
        for oid, (color, _) in shapes.items()
    }

    semantic = match_identities(reference, target_grid)
    semantic_map = {match.object_id: match.target_index
                    for match in semantic.matches}
    baseline = {
        oid: index for index, oid in enumerate(reference.object_ids())
    }

    def metrics(predicted: dict[str, int]) -> dict:
        correct = sum(predicted.get(oid) == index
                      for oid, index in expected.items())
        assigned = len(predicted)
        return {
            "correct": correct,
            "total": len(expected),
            "accuracy": correct / len(expected),
            "assigned": assigned,
            "abstained": len(expected) - assigned,
            "incorrect_assignments": sum(
                predicted.get(oid) is not None and
                predicted.get(oid) != index
                for oid, index in expected.items()),
        }

    return {
        "benchmark_type": "synthetic_component_identity_preflight",
        "arc_task_accuracy_claim": False,
        "scenario_count": 1,
        "objects_per_scenario": len(expected),
        "baseline_index_order": metrics(baseline),
        "semantic_identity": metrics(semantic_map),
        "semantic_ambiguous": list(semantic.ambiguous_reference),
        "semantic_unmatched_reference": list(semantic.unmatched_reference),
        "ground_truth_target_indices": expected,
        "semantic_matches": semantic_map,
        "note": (
            "Synthetic controlled test only. It checks identity association "
            "under relocation, detection-order reversal, and rotation; it does "
            "not measure ARC task solving or generalization."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_experiment()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

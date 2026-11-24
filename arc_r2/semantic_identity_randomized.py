"""Seeded randomized stress testing for semantic object identity matching.

Synthetic component-association tests only. No ARC test outputs are used and
no ARC task-solving accuracy or generalization claim is made.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

from .grid import as_grid
from .objects import GridObject, detect_objects
from .object_identity_dsl import IdentityObjectProgram
from .semantic_identity import match_identities

GRID_SIZE = 30
SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (1, 1), (1, 2)),
)


def _transform(shape: tuple[tuple[int, int], ...], turns: int):
    points = list(shape)
    for _ in range(turns % 4):
        points = [(c, -r) for r, c in points]
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple((r - min_r, c - min_c) for r, c in points)


def _placed(color: int, shape: tuple[tuple[int, int], ...], top: int, left: int) -> GridObject:
    cells = frozenset((top + r, left + c) for r, c in shape)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return GridObject(color, cells, (min(rows), min(cols), max(rows), max(cols)))


def _random_layout(rng: random.Random, colors: list[int], shapes: list[tuple[tuple[int, int], ...]]):
    occupied: set[tuple[int, int]] = set()
    objects = []
    for color, shape in zip(colors, shapes):
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        placed = None
        for _ in range(500):
            top = rng.randrange(GRID_SIZE - height + 1)
            left = rng.randrange(GRID_SIZE - width + 1)
            cells = {(top + r, left + c) for r, c in shape}
            if not cells & occupied:
                placed = _placed(color, shape, top, left)
                occupied.update(cells)
                break
        if placed is None:
            raise RuntimeError("could not place non-overlapping randomized objects")
        objects.append(placed)
    return tuple(objects)


def _grid(objects: tuple[GridObject, ...]):
    cells = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for obj in objects:
        for r, c in obj.cells:
            if cells[r][c] != 0:
                raise ValueError("randomized fixture objects overlap")
            cells[r][c] = obj.color
    return as_grid(cells)


def run_randomized_stress(*, seed: int = 20261124, trials: int = 250) -> dict:
    if trials < 1 or trials > 10000:
        raise ValueError("trials must be between 1 and 10000")
    rng = random.Random(seed)
    total_expected = total_correct = total_incorrect = total_abstained = 0
    failure_examples = []
    trial_records = []

    for trial in range(trials):
        object_count = rng.randint(2, 6)
        colors = rng.sample(range(1, 10), object_count)
        reference_shapes = [rng.choice(SHAPES) for _ in colors]
        target_shapes = [
            _transform(shape, rng.randrange(4))
            for shape in reference_shapes
        ]
        reference_objects = _random_layout(rng, colors, reference_shapes)
        target_order = list(range(object_count))
        rng.shuffle(target_order)
        target_colors = [colors[index] for index in target_order]
        target_ordered_shapes = [target_shapes[index] for index in target_order]
        target_objects = _random_layout(rng, target_colors, target_ordered_shapes)

        reference_grid = _grid(reference_objects)
        target_grid = _grid(target_objects)
        state = IdentityObjectProgram.initialize(reference_grid)
        result = match_identities(state, target_grid)

        target_index_by_color = {
            obj.color: index for index, obj in enumerate(detect_objects(target_grid))
        }
        expected = {
            object_id: target_index_by_color[obj.color]
            for object_id, obj in state.objects
        }
        predicted = {match.object_id: match.target_index for match in result.matches}
        correct = sum(predicted.get(object_id) == target_index
                      for object_id, target_index in expected.items())
        incorrect = sum(object_id in predicted and predicted[object_id] != target_index
                        for object_id, target_index in expected.items())
        abstained = sum(object_id not in predicted for object_id in expected)
        total_expected += len(expected)
        total_correct += correct
        total_incorrect += incorrect
        total_abstained += abstained

        record = {
            "trial": trial,
            "objects": object_count,
            "expected": len(expected),
            "correct": correct,
            "incorrect": incorrect,
            "abstained": abstained,
            "ambiguous": len(result.ambiguous_reference),
        }
        trial_records.append(record)
        if (incorrect or abstained) and len(failure_examples) < 10:
            failure_examples.append(record)

    passed = total_incorrect == 0 and total_abstained == 0
    result_core = {
        "benchmark_type": "seeded_randomized_semantic_identity_stress",
        "arc_task_accuracy_claim": False,
        "seed": seed,
        "trials": trials,
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "shape_catalog_size": len(SHAPES),
        "objects_per_trial": {"minimum": 2, "maximum": 6},
        "expected_matches": total_expected,
        "correct_matches": total_correct,
        "incorrect_assignments": total_incorrect,
        "abstentions": total_abstained,
        "match_rate": round(total_correct / total_expected, 6) if total_expected else 0.0,
        "passed": passed,
        "failure_examples": failure_examples,
        "trial_records": trial_records,
        "acceptance": "zero incorrect assignments and zero abstentions for uniquely colored, shape-preserving objects",
        "note": "Synthetic object association only; this is not an ARC task-accuracy or generalization benchmark.",
    }
    fingerprint_payload = json.dumps(result_core, sort_keys=True, separators=(",", ":"))
    result_core["result_sha256"] = hashlib.sha256(fingerprint_payload.encode("utf-8")).hexdigest()
    return result_core


def render_summary(report: dict) -> str:
    return "\n".join([
        "# Seeded randomized semantic identity stress",
        "",
        f"**Result:** {'PASS' if report['passed'] else 'FAIL'}",
        f"- Seed: {report['seed']}",
        f"- Trials: {report['trials']}",
        f"- Expected object matches: {report['expected_matches']}",
        f"- Correct: {report['correct_matches']}",
        f"- Incorrect assignments: {report['incorrect_assignments']}",
        f"- Abstentions: {report['abstentions']}",
        f"- Match rate: {report['match_rate']:.2%}",
        f"- Result SHA-256: `{report['result_sha256']}`",
        "",
        "## Failure examples",
        "",
        *([f"- Trial {item['trial']}: {item['correct']}/{item['expected']} correct, "
           f"{item['incorrect']} incorrect, {item['abstained']} abstained"
           for item in report["failure_examples"]] or ["- None"]),
        "",
        "> Synthetic component association only. No ARC task-solving accuracy claim.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261124)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report = run_randomized_stress(seed=args.seed, trials=args.trials)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    if args.summary:
        args.summary.write_text(render_summary(report), encoding="utf-8")
    print(rendered)
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

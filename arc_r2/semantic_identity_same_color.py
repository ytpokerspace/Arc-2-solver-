"""Same-color randomized stress for shape-based object identity matching.

Every object has the same color. Ground-truth correspondence is derived from
unique shape classes preserved under rotation and reflection, while positions
and detection order change. Objects are kept 4-disconnected so the component detector sees
the intended objects. Synthetic association only; not ARC task accuracy.
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
from .semantic_identity_randomized import GRID_SIZE, SHAPES

# These five shapes are distinct under rotation/reflection and have enough
# pairwise shape separation for the matcher's configured ambiguity margin.
SHAPE_INDICES = (0, 1, 4, 5, 9)
SAME_COLOR_SHAPES = tuple(SHAPES[index] for index in SHAPE_INDICES)
OBJECT_COLOR = 1


def _transform(shape, turns: int, reflected: bool):
    points = list(shape)
    if reflected:
        points = [(r, -c) for r, c in points]
    for _ in range(turns % 4):
        points = [(c, -r) for r, c in points]
    min_r = min(r for r, _ in points)
    min_c = min(c for _, c in points)
    return tuple((r - min_r, c - min_c) for r, c in points)


def _shape_key(obj: GridObject):
    points = [(r - obj.bbox[0], c - obj.bbox[1]) for r, c in obj.cells]
    variants = []
    for reflected in (False, True):
        for turns in range(4):
            transformed = []
            for r, c in points:
                if reflected:
                    c = -c
                for _ in range(turns):
                    r, c = c, -r
                transformed.append((r, c))
            min_r = min(r for r, _ in transformed)
            min_c = min(c for _, c in transformed)
            shifted = [(r - min_r, c - min_c) for r, c in transformed]
            span = max(
                max(r for r, _ in shifted),
                max(c for _, c in shifted),
                1,
            )
            variants.append(tuple(sorted(
                (round(r / span, 3), round(c / span, 3))
                for r, c in shifted
            )))
    return min(variants)


def _placed(shape, top: int, left: int) -> GridObject:
    cells = frozenset((top + r, left + c) for r, c in shape)
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return GridObject(
        OBJECT_COLOR,
        cells,
        (min(rows), min(cols), max(rows), max(cols)),
    )


def _random_layout(rng: random.Random, shapes):
    occupied: set[tuple[int, int]] = set()
    objects = []
    for shape in shapes:
        height = max(r for r, _ in shape) + 1
        width = max(c for _, c in shape) + 1
        placed = None
        for _ in range(2000):
            top = rng.randrange(GRID_SIZE - height + 1)
            left = rng.randrange(GRID_SIZE - width + 1)
            cells = {(top + r, left + c) for r, c in shape}
            halo = set(cells)
            for row, col in cells:
                halo.update((
                    (row - 1, col), (row + 1, col),
                    (row, col - 1), (row, col + 1),
                ))
            if not halo & occupied:
                placed = _placed(shape, top, left)
                occupied.update(cells)
                break
        if placed is None:
            raise RuntimeError("could not place separated same-color objects")
        objects.append(placed)
    return tuple(objects)


def _grid(objects):
    cells = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    for obj in objects:
        for row, col in obj.cells:
            if cells[row][col] != 0:
                raise ValueError("same-color randomized objects overlap")
            cells[row][col] = obj.color
    return as_grid(cells)


def run_same_color_stress(*, seed: int = 20261125, trials: int = 250) -> dict:
    if trials < 1 or trials > 10000:
        raise ValueError("trials must be between 1 and 10000")
    rng = random.Random(seed)
    total_expected = total_correct = total_incorrect = total_abstained = 0
    failure_examples = []
    trial_records = []

    for trial in range(trials):
        object_count = rng.randint(2, len(SAME_COLOR_SHAPES))
        reference_shapes = rng.sample(SAME_COLOR_SHAPES, object_count)
        target_shapes = [
            _transform(shape, rng.randrange(4), rng.choice((False, True)))
            for shape in reference_shapes
        ]
        reference_grid = _grid(_random_layout(rng, reference_shapes))
        target_grid = _grid(_random_layout(rng, target_shapes))

        state = IdentityObjectProgram.initialize(reference_grid)
        targets = detect_objects(target_grid)
        target_index_by_shape = {
            _shape_key(obj): index for index, obj in enumerate(targets)
        }
        if len(target_index_by_shape) != object_count:
            raise AssertionError("target shape classes are not unique")
        expected = {
            object_id: target_index_by_shape[_shape_key(obj)]
            for object_id, obj in state.objects
        }

        result = match_identities(state, target_grid)
        predicted = {
            match.object_id: match.target_index for match in result.matches
        }
        correct = sum(
            predicted.get(object_id) == target_index
            for object_id, target_index in expected.items()
        )
        incorrect = sum(
            object_id in predicted and predicted[object_id] != target_index
            for object_id, target_index in expected.items()
        )
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
    core = {
        "benchmark_type": "same_color_shape_based_identity_stress",
        "arc_task_accuracy_claim": False,
        "seed": seed,
        "trials": trials,
        "grid_size": [GRID_SIZE, GRID_SIZE],
        "object_color": OBJECT_COLOR,
        "shape_catalog_indices": list(SHAPE_INDICES),
        "shape_catalog_size": len(SAME_COLOR_SHAPES),
        "objects_per_trial": {"minimum": 2, "maximum": len(SAME_COLOR_SHAPES)},
        "expected_matches": total_expected,
        "correct_matches": total_correct,
        "incorrect_assignments": total_incorrect,
        "abstentions": total_abstained,
        "match_rate": round(total_correct / total_expected, 6),
        "passed": passed,
        "failure_examples": failure_examples,
        "trial_records": trial_records,
        "acceptance": (
            "zero incorrect assignments and zero abstentions for same-color "
            "objects with distinct rotation/reflection-invariant shapes"
        ),
        "note": (
            "Synthetic component association only. All objects share one "
            "color; shapes are distinct and objects remain 4-disconnected. "
            "This is not an ARC task-accuracy or generalization benchmark."
        ),
    }
    fingerprint_payload = json.dumps(core, sort_keys=True, separators=(",", ":"))
    core["result_sha256"] = hashlib.sha256(
        fingerprint_payload.encode("utf-8")
    ).hexdigest()
    return core


def render_summary(report: dict) -> str:
    return "\n".join([
        "# Same-color shape-based identity stress",
        "",
        f"**Result:** {'PASS' if report['passed'] else 'FAIL'}",
        f"- Seed: {report['seed']}",
        f"- Trials: {report['trials']}",
        f"- Expected matches: {report['expected_matches']}",
        f"- Correct: {report['correct_matches']}",
        f"- Incorrect assignments: {report['incorrect_assignments']}",
        f"- Abstentions: {report['abstentions']}",
        f"- Match rate: {report['match_rate']:.2%}",
        f"- Result SHA-256: `{report['result_sha256']}`",
        "",
        "> Synthetic component association only; no ARC task-accuracy claim.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261125)
    parser.add_argument("--trials", type=int, default=250)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    report = run_same_color_stress(seed=args.seed, trials=args.trials)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        args.summary.write_text(render_summary(report), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

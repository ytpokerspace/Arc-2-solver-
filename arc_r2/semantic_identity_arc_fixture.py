"""Training-pair identity-association check on ARC-AGI-2 task d89b689b.

This uses only the task's published training input/output pairs. The four
single-cell marker objects retain unique colors while moving into a 2x2
destination; the 2x2 anchor itself is expected to be unmatched. This measures
component association only, not ARC task solving or held-out generalization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .grid import as_grid
from .object_identity_dsl import IdentityObjectProgram
from .objects import detect_objects
from .semantic_identity import match_identities

DEFAULT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "tests" / "fixtures" / "arc_agi2_d89b689b_train.json"
)
ANCHOR_COLOR = 8


def run_training_pair_experiment(task_json: Path = DEFAULT_FIXTURE) -> dict:
    task = json.loads(Path(task_json).read_text(encoding="utf-8"))
    if task.get("split") != "training":
        raise ValueError("only training-split fixtures are permitted")
    if task.get("task_id") != "d89b689b":
        raise ValueError("this experiment is scoped to task d89b689b")

    pair_reports = []
    total_expected = total_correct = total_incorrect = total_abstained = 0
    total_unmatched_anchor = 0

    for pair_index, pair in enumerate(task["train"]):
        reference = IdentityObjectProgram.initialize(as_grid(pair["input"]))
        target_grid = as_grid(pair["output"])
        targets = detect_objects(target_grid)
        target_by_color: dict[int, list[int]] = {}
        for target_index, target in enumerate(targets):
            target_by_color.setdefault(target.color, []).append(target_index)

        markers = [
            (object_id, obj)
            for object_id, obj in reference.objects
            if obj.color != ANCHOR_COLOR
        ]
        anchors = [
            object_id for object_id, obj in reference.objects
            if obj.color == ANCHOR_COLOR and obj.size == 4
        ]
        if len(markers) != 4 or len(anchors) != 1 or len(targets) != 4:
            raise AssertionError(
                f"pair {pair_index}: expected four markers, one 2x2 anchor, "
                "and four output marker components"
            )

        expected: dict[str, int] = {}
        for object_id, obj in markers:
            indices = target_by_color.get(obj.color, [])
            if len(indices) != 1:
                raise AssertionError(
                    f"pair {pair_index}: marker color {obj.color} is not unique "
                    "in the output"
                )
            expected[object_id] = indices[0]

        if set(target_by_color) != {obj.color for _, obj in markers}:
            raise AssertionError(
                f"pair {pair_index}: output colors do not exactly preserve "
                "the four marker colors"
            )

        result = match_identities(reference, target_grid)
        predicted = {
            match.object_id: match.target_index for match in result.matches
        }
        correct = sum(predicted.get(oid) == index for oid, index in expected.items())
        incorrect = sum(
            oid in predicted and predicted[oid] != index
            for oid, index in expected.items()
        )
        abstained = len(expected) - len(predicted)
        unmatched_anchor = sum(oid in result.unmatched_reference for oid in anchors)

        total_expected += len(expected)
        total_correct += correct
        total_incorrect += incorrect
        total_abstained += abstained
        total_unmatched_anchor += unmatched_anchor
        pair_reports.append({
            "pair_index": pair_index,
            "expected_matches": len(expected),
            "correct_matches": correct,
            "incorrect_assignments": incorrect,
            "abstentions": abstained,
            "anchor_unmatched": unmatched_anchor,
            "ambiguous_reference": list(result.ambiguous_reference),
        })

    payload = {
        "benchmark_type": "arc_agi2_training_pair_identity_association",
        "task_id": task["task_id"],
        "source_repository": task.get("source_repository"),
        "source_revision": task.get("source_revision"),
        "split": task["split"],
        "training_pairs": len(pair_reports),
        "expected_matches": total_expected,
        "correct_matches": total_correct,
        "incorrect_assignments": total_incorrect,
        "abstentions": total_abstained,
        "unmatched_anchor_objects": total_unmatched_anchor,
        "match_rate": total_correct / total_expected if total_expected else 0.0,
        "passed": (
            total_expected == 12
            and total_correct == total_expected
            and total_incorrect == 0
            and total_abstained == 0
            and total_unmatched_anchor == len(pair_reports)
        ),
        "ground_truth_rule": (
            "Each of the four single-cell markers has a unique color preserved "
            "from input to output; the output component of that color is its "
            "training-pair correspondence. The 2x2 color-8 anchor is not a "
            "persistent output object."
        ),
        "arc_task_accuracy_claim": False,
        "held_out_test_data_used": False,
        "note": (
            "Real ARC-AGI-2 training-pair component association only. This does "
            "not evaluate output-grid prediction, test split performance, or "
            "independent-task generalization."
        ),
        "pair_reports": pair_reports,
    }
    fingerprint = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {**payload, "result_sha256": fingerprint}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-json", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    report = run_training_pair_experiment(args.task_json)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if args.summary:
        lines = [
            "# ARC-AGI-2 training-pair identity association",
            "",
            f"- Task: `{report['task_id']}`",
            f"- Training pairs: {report['training_pairs']}",
            f"- Correct associations: {report['correct_matches']}/{report['expected_matches']}",
            f"- Incorrect assignments: {report['incorrect_assignments']}",
            f"- Abstentions: {report['abstentions']}",
            f"- Unmatched anchor objects: {report['unmatched_anchor_objects']}",
            f"- Result: {'PASS' if report['passed'] else 'FAIL'}",
            "",
            report["note"],
            "",
        ]
        args.summary.write_text("\n".join(lines), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

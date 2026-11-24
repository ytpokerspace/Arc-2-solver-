"""Compare index-order and semantic identity on ARC-AGI-2 training pairs.

Uses only the published training split for task d89b689b. Ground-truth
correspondence is established by unique marker colors preserved from each
input to its output. This measures component association, not ARC grid-solving
accuracy or held-out generalization.
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


def _metrics(predicted: dict[str, int], expected: dict[str, int]) -> dict:
    correct = sum(predicted.get(object_id) == index
                  for object_id, index in expected.items())
    incorrect = sum(
        object_id in predicted and predicted[object_id] != index
        for object_id, index in expected.items()
    )
    assigned = len(predicted)
    return {
        "expected_matches": len(expected),
        "assigned": assigned,
        "correct_matches": correct,
        "incorrect_assignments": incorrect,
        "abstentions": len(expected) - assigned,
        "match_rate": correct / len(expected) if expected else 0.0,
    }


def run_training_pair_experiment(task_json: Path = DEFAULT_FIXTURE) -> dict:
    task = json.loads(Path(task_json).read_text(encoding="utf-8"))
    if task.get("split") != "training":
        raise ValueError("only training-split fixtures are permitted")
    if task.get("task_id") != "d89b689b":
        raise ValueError("this experiment is scoped to task d89b689b")

    pair_reports = []
    totals = {
        "baseline_index_order": {
            "expected_matches": 0, "assigned": 0, "correct_matches": 0,
            "incorrect_assignments": 0, "abstentions": 0,
        },
        "semantic_identity": {
            "expected_matches": 0, "assigned": 0, "correct_matches": 0,
            "incorrect_assignments": 0, "abstentions": 0,
        },
    }
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

        # Baseline control: preserve the original deterministic detection
        # indices and assume index i in the output is the same object as index
        # i in the input. If the output has fewer components, missing indices
        # are abstentions rather than fabricated matches.
        baseline_predicted = {
            object_id: reference_index
            for reference_index, (object_id, obj) in enumerate(reference.objects)
            if obj.color != ANCHOR_COLOR and reference_index < len(targets)
        }
        baseline_metrics = _metrics(baseline_predicted, expected)

        semantic_result = match_identities(reference, target_grid)
        semantic_predicted = {
            match.object_id: match.target_index
            for match in semantic_result.matches
        }
        semantic_metrics = _metrics(semantic_predicted, expected)
        unmatched_anchor = sum(
            object_id in semantic_result.unmatched_reference for object_id in anchors
        )

        for name, metrics in (
            ("baseline_index_order", baseline_metrics),
            ("semantic_identity", semantic_metrics),
        ):
            for key in totals[name]:
                totals[name][key] += metrics[key]

        total_unmatched_anchor += unmatched_anchor
        pair_reports.append({
            "pair_index": pair_index,
            "baseline_index_order": baseline_metrics,
            "semantic_identity": semantic_metrics,
            "semantic_ambiguous_reference": list(semantic_result.ambiguous_reference),
            "semantic_unmatched_reference": list(semantic_result.unmatched_reference),
            "semantic_unmatched_target": list(semantic_result.unmatched_target),
            "anchor_unmatched": unmatched_anchor,
        })

    for metrics in totals.values():
        metrics["match_rate"] = (
            metrics["correct_matches"] / metrics["expected_matches"]
            if metrics["expected_matches"] else 0.0
        )

    semantic = totals["semantic_identity"]
    baseline = totals["baseline_index_order"]
    payload = {
        "benchmark_type": "arc_agi2_training_pair_index_vs_semantic_identity",
        "task_id": task["task_id"],
        "source_repository": task.get("source_repository"),
        "source_revision": task.get("source_revision"),
        "split": task["split"],
        "training_pairs": len(pair_reports),
        "comparison": totals,
        # Backward-compatible semantic summary fields.
        "expected_matches": semantic["expected_matches"],
        "correct_matches": semantic["correct_matches"],
        "incorrect_assignments": semantic["incorrect_assignments"],
        "abstentions": semantic["abstentions"],
        "unmatched_anchor_objects": total_unmatched_anchor,
        "match_rate": semantic["match_rate"],
        "passed": (
            semantic["expected_matches"] == 12
            and semantic["correct_matches"] == semantic["expected_matches"]
            and semantic["incorrect_assignments"] == 0
            and semantic["abstentions"] == 0
            and total_unmatched_anchor == len(pair_reports)
        ),
        "baseline_index_order": baseline,
        "semantic_identity": semantic,
        "ground_truth_rule": (
            "Each of the four single-cell markers has a unique color preserved "
            "from input to output; the output component of that color is its "
            "training-pair correspondence. The 2x2 color-8 anchor is not a "
            "persistent output object."
        ),
        "baseline_rule": (
            "Assign each input marker the same detection index in the output; "
            "indices beyond the output component count are counted as abstentions."
        ),
        "arc_task_accuracy_claim": False,
        "held_out_test_data_used": False,
        "note": (
            "Same three ARC-AGI-2 training pairs are evaluated by both index-order "
            "and semantic identity association. This measures component matching "
            "only, not output-grid prediction, held-out test performance, or "
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
        baseline = report["baseline_index_order"]
        semantic = report["semantic_identity"]
        lines = [
            "# ARC-AGI-2 training-pair identity comparison",
            "",
            f"- Task: `{report['task_id']}`",
            f"- Training pairs: {report['training_pairs']}",
            "",
            "| Method | Correct | Incorrect | Abstentions | Match rate |",
            "|---|---:|---:|---:|---:|",
            f"| Index-order baseline | {baseline['correct_matches']}/{baseline['expected_matches']} | "
            f"{baseline['incorrect_assignments']} | {baseline['abstentions']} | {baseline['match_rate']:.2%} |",
            f"| Semantic identity | {semantic['correct_matches']}/{semantic['expected_matches']} | "
            f"{semantic['incorrect_assignments']} | {semantic['abstentions']} | {semantic['match_rate']:.2%} |",
            "",
            f"- Unmatched anchor objects under semantic identity: {report['unmatched_anchor_objects']}",
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

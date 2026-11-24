"""Blind ARC-AGI-2 evaluation-input runner.

This module reads only task training examples and evaluation test inputs. It never
reads test outputs, never writes solver memory, and never reports test accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .primitives import as_grid, fit_program


def load_evaluation_tasks(dataset: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load evaluation task JSON files in stable filename order."""
    if not dataset.is_dir():
        raise ValueError("evaluation dataset must be a directory")
    files = sorted(dataset.glob("*.json"))
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        files = files[:limit]
    tasks: list[dict[str, Any]] = []
    for file in files:
        task = json.loads(file.read_text(encoding="utf-8"))
        if not isinstance(task, dict):
            raise ValueError(f"{file.name} must contain a task object")
        tasks.append({"task_id": file.stem, **task})
    return tasks


def run_evaluation(dataset: Path, *, limit: int | None = None,
                   candidate_cap: int = 1500, max_depth: int = 2) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit on each task's complete train split and predict its test inputs."""
    tasks = load_evaluation_tasks(dataset, limit)
    predictions: dict[str, Any] = {}
    task_reports: list[dict[str, Any]] = []
    started = time.perf_counter()
    total_tests = 0
    predicted_tests = 0
    abstained_tests = 0
    exact_fit_tasks = 0
    total_candidates = 0

    for task_index, task in enumerate(tasks, start=1):
        task_id = str(task["task_id"])
        print(f"[EVAL] task={task_index}/{len(tasks)} id={task_id}", flush=True)
        task_started = time.perf_counter()
        train = task.get("train")
        tests = task.get("test")
        if not isinstance(train, list) or not train:
            raise ValueError(f"{task_id}: train must be a non-empty list")
        if not isinstance(tests, list) or not tests:
            raise ValueError(f"{task_id}: test must be a non-empty list")

        fit_pairs = tuple(
            (as_grid(pair["input"]), as_grid(pair["output"]))
            for pair in train
        )
        test_inputs = [as_grid(item["input"]) for item in tests]
        fit_started = time.perf_counter()
        fit = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth)
        fit_seconds = time.perf_counter() - fit_started
        if fit.exact_fit:
            exact_fit_tasks += 1
        total_candidates += fit.candidates_evaluated

        task_predictions: list[Any] = []
        test_results: list[dict[str, Any]] = []
        for test_index, test_input in enumerate(test_inputs):
            prediction = None
            error = None
            prediction_started = time.perf_counter()
            if fit.program is not None:
                try:
                    candidate = fit.program.apply(test_input)
                    prediction = as_grid(candidate)
                except (ValueError, TypeError, IndexError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            prediction_seconds = time.perf_counter() - prediction_started
            if prediction is None:
                abstained_tests += 1
            else:
                predicted_tests += 1
            total_tests += 1
            task_predictions.append([list(row) for row in prediction] if prediction is not None else None)
            test_results.append({
                "test_index": test_index,
                "prediction_produced": prediction is not None,
                "prediction_shape": [len(prediction), len(prediction[0])] if prediction is not None else None,
                "prediction_seconds": round(prediction_seconds, 6),
                "error": error,
            })

        predictions[task_id] = task_predictions
        task_reports.append({
            "task_id": task_id,
            "train_pairs": len(fit_pairs),
            "test_inputs": len(test_inputs),
            "fit_exact": fit.exact_fit,
            "fit_program": str(fit.program) if fit.program is not None else None,
            "fit_program_count": fit.fit_program_count,
            "candidates_evaluated": fit.candidates_evaluated,
            "fit_seconds": round(fit_seconds, 6),
            "task_seconds": round(time.perf_counter() - task_started, 6),
            "test_results": test_results,
        })

    elapsed = time.perf_counter() - started
    report = {
        "benchmark": "ARC-AGI-2 official evaluation inputs",
        "dataset": str(dataset),
        "task_count": len(tasks),
        "test_input_count": total_tests,
        "tasks_with_exact_training_fit": exact_fit_tasks,
        "predictions_produced": predicted_tests,
        "abstentions": abstained_tests,
        "prediction_coverage": predicted_tests / total_tests if total_tests else 0.0,
        "candidate_cap_per_task": candidate_cap,
        "max_depth": max_depth,
        "candidates_evaluated_total": total_candidates,
        "runtime_seconds": round(elapsed, 6),
        "test_accuracy": None,
        "test_outputs_read": False,
        "solver_memory_enabled": False,
        "semantic_identity_integrated": False,
        "semantic_identity_note": "The current semantic identity matcher is not part of primitive program fitting or prediction; this run is the primitive baseline only.",
        "scope": "Fit each task on all provided train pairs; predict every provided evaluation test input. Test output labels are never accessed.",
        "task_results": task_reports,
    }
    return report, predictions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Predict on official ARC-AGI-2 evaluation inputs without reading test labels")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None, help="optional task limit; omit to process every JSON task")
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_evaluation_predictions.json"))
    parser.add_argument("--report", type=Path, default=Path("arc_r2_evaluation_report.json"))
    args = parser.parse_args(argv)
    if not args.dataset.is_dir():
        parser.error("dataset must be an existing directory")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not 1 <= args.candidate_cap <= 50000:
        parser.error("candidate-cap must be in 1..50000")

    report, predictions = run_evaluation(
        args.dataset, limit=args.limit,
        candidate_cap=args.candidate_cap, max_depth=args.max_depth,
    )
    args.output.write_text(json.dumps(predictions, indent=2) + "\n", encoding="utf-8")
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-AGI-2 OFFICIAL EVALUATION INPUTS | TEST LABELS: NOT READ")
    print(f"Tasks: {report['task_count']} | Test inputs: {report['test_input_count']}")
    print(f"Predictions: {report['predictions_produced']} | Abstentions: {report['abstentions']}")
    print(f"Coverage: {report['prediction_coverage']:.3%} | Test accuracy: NOT AVAILABLE")
    print(f"Runtime: {report['runtime_seconds']:.3f}s")
    print(f"[SAVED] predictions={args.output} report={args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

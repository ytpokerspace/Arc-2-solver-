"""Controlled leave-one-training-pair-out comparison of full orchestration."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .ahpse import TrainingExample
from .full_orchestration import run_full_orchestration
from .grid import as_grid
from .primitives import fit_program


def load_tasks(dataset: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if dataset.is_dir():
        files = sorted(dataset.glob("*.json"))
        if limit is not None:
            files = files[:limit]
        tasks = []
        for file in files:
            raw = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError(f"{file.name} must contain a task object")
            tasks.append({"task_id": file.stem, **raw})
        return tasks
    raw = json.loads(dataset.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        if "train" in raw:
            tasks = [raw]
        else:
            if not all(isinstance(value, dict) for value in raw.values()):
                raise ValueError("task mapping values must be task objects")
            tasks = [{"task_id": key, **value} for key, value in sorted(raw.items())]
    elif isinstance(raw, list):
        tasks = raw
    else:
        raise ValueError("dataset must be a task directory, list, or task mapping")
    return tasks if limit is None else tasks[:limit]


def run_benchmark(dataset: Path, *, limit: int | None = None,
                  candidate_cap: int = 500, max_depth: int = 2) -> dict[str, Any]:
    tasks = load_tasks(dataset, limit)
    folds: list[dict[str, Any]] = []
    skipped: list[str] = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id", task_index))
        train = task.get("train")
        if not isinstance(train, list) or len(train) < 2:
            skipped.append(task_id)
            continue
        pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
        for held_index, (held_input, held_output) in enumerate(pairs):
            fit_pairs = tuple(pair for index, pair in enumerate(pairs) if index != held_index)
            examples = tuple(TrainingExample(source, target) for source, target in fit_pairs)

            full_start = time.perf_counter()
            full = run_full_orchestration(
                examples, (held_input,), max_depth=max_depth, candidate_cap=candidate_cap,
            )
            full_seconds = time.perf_counter() - full_start
            full_prediction = full.decision.predictions[0]

            baseline_start = time.perf_counter()
            baseline = fit_program(
                fit_pairs, candidate_cap=candidate_cap, max_depth=min(max_depth, 2),
                memory_path=None,
            )
            baseline_prediction = None
            if baseline.program is not None:
                try:
                    baseline_prediction = as_grid(baseline.program.apply(held_input))
                except (ValueError, TypeError, IndexError):
                    baseline_prediction = None
            baseline_seconds = time.perf_counter() - baseline_start

            folds.append({
                "task_id": task_id,
                "held_out_training_pair": held_index,
                "fit_pairs": len(fit_pairs),
                "full_prediction": [list(row) for row in full_prediction] if full_prediction is not None else None,
                "full_exact": full_prediction == held_output,
                "full_abstained": full_prediction is None,
                "full_ambiguous": full.decision.ambiguous[0],
                "full_candidate_count": full.decision.candidate_count,
                "full_stage_evaluations": full.execution.total_evaluations,
                "full_runtime_seconds": round(full_seconds, 6),
                "full_family_evidence": dict(full.candidate_pool.family_evidence),
                "baseline_prediction": [list(row) for row in baseline_prediction] if baseline_prediction is not None else None,
                "baseline_exact": baseline_prediction == held_output,
                "baseline_abstained": baseline_prediction is None,
                "baseline_candidates_evaluated": baseline.candidates_evaluated,
                "baseline_runtime_seconds": round(baseline_seconds, 6),
                "expected_held_out": [list(row) for row in held_output],
            })

    count = len(folds)
    full_correct = sum(row["full_exact"] for row in folds)
    base_correct = sum(row["baseline_exact"] for row in folds)
    return {
        "benchmark": "ARC-R2 full orchestration vs primitive baseline",
        "dataset": dataset.name,
        "tasks_considered": len(tasks),
        "eligible_tasks": len({row["task_id"] for row in folds}),
        "skipped_tasks": skipped,
        "folds": count,
        "full_orchestration": {
            "held_out_exact": full_correct,
            "held_out_accuracy": full_correct / count if count else 0.0,
            "abstentions": sum(row["full_abstained"] for row in folds),
            "ambiguous_abstentions": sum(row["full_ambiguous"] for row in folds),
            "candidate_count_total": sum(row["full_candidate_count"] for row in folds),
            "runtime_seconds_total": round(sum(row["full_runtime_seconds"] for row in folds), 6),
        },
        "primitive_baseline": {
            "held_out_exact": base_correct,
            "held_out_accuracy": base_correct / count if count else 0.0,
            "abstentions": sum(row["baseline_abstained"] for row in folds),
            "candidates_evaluated_total": sum(row["baseline_candidates_evaluated"] for row in folds),
            "runtime_seconds_total": round(sum(row["baseline_runtime_seconds"] for row in folds), 6),
        },
        "accuracy_delta_full_minus_baseline": (full_correct - base_correct) / count if count else 0.0,
        "candidate_cap_per_family_per_fold": candidate_cap,
        "max_depth": max_depth,
        "runtime_seconds_total": round(time.perf_counter() - started, 6),
        "memory_enabled": False,
        "hidden_test_outputs_read": False,
        "protocol": (
            "Leave one visible training pair out. Both engines receive identical "
            "remaining pairs and the held-out input. The held-out output is scored "
            "only after prediction; no persistent memory is used."
        ),
        "fold_results": folds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare full orchestration with primitive baseline")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--candidate-cap", type=int, default=500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_full_orchestration_benchmark.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists():
        parser.error("dataset path does not exist")
    if args.limit is not None and args.limit < 1:
        parser.error("limit must be positive")
    if not 1 <= args.candidate_cap <= 10_000:
        parser.error("candidate-cap must be in 1..10000")
    report = run_benchmark(args.dataset, limit=args.limit,
                           candidate_cap=args.candidate_cap, max_depth=args.max_depth)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "tasks_considered": report["tasks_considered"],
        "folds": report["folds"],
        "full_orchestration": report["full_orchestration"],
        "primitive_baseline": report["primitive_baseline"],
        "accuracy_delta_full_minus_baseline": report["accuracy_delta_full_minus_baseline"],
        "hidden_test_outputs_read": report["hidden_test_outputs_read"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

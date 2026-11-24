"""Task-disjoint HCM-32 transfer benchmark for public ARC training pairs.

Source tasks alone populate a frozen HCM-32 memory. Distinct target tasks are
then evaluated leave-one-training-pair-out against a primitive-only baseline.
Official test arrays are never read, and target tasks never write to memory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from .hcm32_bridge import HCM32Bridge
from .primitive_benchmark import load_tasks
from .primitives import as_grid, fit_program


def _training_pairs(task: dict[str, Any]):
    train = task.get("train")
    if not isinstance(train, list) or len(train) < 2:
        return None
    pairs = []
    for pair in train:
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise ValueError(f"task {task.get('task_id', '?')} has a malformed training pair")
        pairs.append((as_grid(pair["input"]), as_grid(pair["output"])))
    return tuple(pairs)


def _evaluate(tasks, candidate_cap: int, max_depth: int, bridge: HCM32Bridge | None):
    folds = []
    skipped = []
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id", task_index))
        pairs = _training_pairs(task)
        if pairs is None:
            skipped.append(task_id)
            continue
        for held_index, (held_input, held_output) in enumerate(pairs):
            fit_pairs = tuple(pair for index, pair in enumerate(pairs) if index != held_index)
            match = bridge.find_program(fit_pairs) if bridge is not None else None
            if match is not None:
                program = match.program
                candidates = 0
                hcm_hit = True
                fit_exact = True
            else:
                result = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth)
                program = result.program
                candidates = result.candidates_evaluated
                hcm_hit = False
                fit_exact = result.exact_fit
            prediction = None
            exact = False
            error = None
            if program is not None:
                try:
                    prediction = program.apply(held_input)
                    exact = prediction == held_output
                except (ValueError, TypeError, IndexError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            folds.append({
                "task_id": task_id,
                "held_out_training_pair": held_index,
                "fit_pairs": len(fit_pairs),
                "program": str(program) if program else None,
                "fit_exact": fit_exact,
                "hcm32_memory_hit": hcm_hit,
                "candidates_evaluated": candidates,
                "held_out_exact": exact,
                "prediction": [list(row) for row in prediction] if prediction is not None else None,
                "expected": [list(row) for row in held_output],
                "error": error,
            })
    return folds, skipped


def run_transfer_benchmark(
    dataset: Path,
    task_limit: int = 1000,
    candidate_cap: int = 1500,
    max_depth: int = 1,
    memory_path: Path = Path("hcm32_transfer_memory.json"),
) -> dict[str, Any]:
    """Learn on even-indexed tasks, freeze memory, and evaluate odd-indexed tasks."""
    if not 2 <= task_limit <= 1000:
        raise ValueError("task_limit must be in 2..1000")
    if not 1 <= candidate_cap <= 50000:
        raise ValueError("candidate_cap must be in 1..50000")
    if not 0 <= max_depth <= 2:
        raise ValueError("max_depth must be in 0..2")

    tasks = load_tasks(dataset, task_limit)
    source_tasks = tasks[::2]
    target_tasks = tasks[1::2]

    memory_path.parent.mkdir(parents=True, exist_ok=True)
    # A run always starts from an empty memory: no previous target can leak in.
    memory_path.unlink(missing_ok=True)
    learner = HCM32Bridge(memory_path)
    source_records = []
    source_skipped = []
    started = time.perf_counter()

    for task_index, task in enumerate(source_tasks):
        task_id = str(task.get("task_id", task_index))
        pairs = _training_pairs(task)
        if pairs is None:
            source_skipped.append(task_id)
            continue
        result = fit_program(pairs, candidate_cap=candidate_cap, max_depth=max_depth)
        procedure_id = None
        if result.exact_fit and result.program is not None:
            procedure_id = learner.store_program(result.program, pairs)
        source_records.append({
            "task_id": task_id,
            "training_pairs": len(pairs),
            "fit_exact": result.exact_fit,
            "program": str(result.program) if result.program else None,
            "candidates_evaluated": result.candidates_evaluated,
            "hcm32_procedure_id": procedure_id,
        })

    if not memory_path.exists():
        # HCM32Bridge creates an empty in-memory model; persist it so the frozen
        # snapshot has a concrete, hashable artifact even when no rule was learned.
        learner.memory.save()
    memory_digest_before = hashlib.sha256(memory_path.read_bytes()).hexdigest()

    baseline_folds, baseline_skipped = _evaluate(
        target_tasks, candidate_cap, max_depth, bridge=None
    )
    transfer_folds, transfer_skipped = _evaluate(
        target_tasks, candidate_cap, max_depth, bridge=HCM32Bridge(memory_path)
    )
    memory_digest_after = hashlib.sha256(memory_path.read_bytes()).hexdigest()
    if memory_digest_before != memory_digest_after:
        raise RuntimeError("frozen source memory changed during target evaluation")

    baseline_exact = sum(row["held_out_exact"] for row in baseline_folds)
    transfer_exact = sum(row["held_out_exact"] for row in transfer_folds)
    hits = sum(row["hcm32_memory_hit"] for row in transfer_folds)
    baseline_candidates = sum(row["candidates_evaluated"] for row in baseline_folds)
    transfer_candidates = sum(row["candidates_evaluated"] for row in transfer_folds)
    folds = len(transfer_folds)
    elapsed = time.perf_counter() - started

    return {
        "engine": "task-disjoint-hcm32-transfer-vs-primitive-baseline",
        "dataset": dataset.name,
        "task_limit": task_limit,
        "split": "deterministic sorted-task order; even indices are source, odd indices are target",
        "source_task_ids": [str(task.get("task_id", i)) for i, task in enumerate(source_tasks)],
        "target_task_ids": [str(task.get("task_id", i)) for i, task in enumerate(target_tasks)],
        "source_tasks_with_fit_records": len(source_records),
        "source_tasks_skipped": source_skipped,
        "source_fit_records": source_records,
        "learned_procedure_count": len(learner.memory.data["procedures"]),
        "target_tasks_skipped_baseline": baseline_skipped,
        "target_tasks_skipped_transfer": transfer_skipped,
        "target_folds": folds,
        "baseline_held_out_exact": baseline_exact,
        "baseline_held_out_accuracy": baseline_exact / folds if folds else 0.0,
        "hcm32_transfer_held_out_exact": transfer_exact,
        "hcm32_transfer_held_out_accuracy": transfer_exact / folds if folds else 0.0,
        "hcm32_retrieval_hits": hits,
        "hcm32_correct_retrievals": sum(
            row["hcm32_memory_hit"] and row["held_out_exact"] for row in transfer_folds
        ),
        "hcm32_incorrect_retrievals": sum(
            row["hcm32_memory_hit"] and not row["held_out_exact"] for row in transfer_folds
        ),
        "baseline_candidates_evaluated": baseline_candidates,
        "hcm32_path_candidates_evaluated": transfer_candidates,
        "candidate_evaluations_avoided": baseline_candidates - transfer_candidates,
        "source_memory_sha256_before_target_evaluation": memory_digest_before,
        "source_memory_sha256_after_target_evaluation": memory_digest_after,
        "source_memory_unchanged_during_target_evaluation": memory_digest_before == memory_digest_after,
        "candidate_cap_per_fold": candidate_cap,
        "max_depth": max_depth,
        "runtime_seconds": round(elapsed, 6),
        "scope": "task-disjoint source-to-target transfer; target leave-one-training-pair-out; official test arrays are never read; target tasks never modify source memory",
        "baseline_fold_results": baseline_folds,
        "hcm32_transfer_fold_results": transfer_folds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Measure task-disjoint HCM-32 transfer on public ARC training pairs")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--candidate-cap", type=int, default=1500)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--memory", type=Path, default=Path("hcm32_transfer_memory.json"))
    parser.add_argument("--output", type=Path, default=Path("arc_r2_hcm32_transfer_benchmark.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists() or not (args.dataset.is_file() or args.dataset.is_dir()):
        parser.error("dataset file or directory does not exist")
    report = run_transfer_benchmark(
        args.dataset, args.limit, args.candidate_cap, args.max_depth, args.memory
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 TASK-DISJOINT HCM-32 TRANSFER | TEST ARRAYS: UNUSED")
    print(f"Source tasks: {len(report['source_task_ids'])}; target tasks: {len(report['target_task_ids'])}")
    print(f"Learned procedures: {report['learned_procedure_count']}")
    print(f"Baseline exact: {report['baseline_held_out_exact']}/{report['target_folds']}")
    print(f"HCM-32 exact: {report['hcm32_transfer_held_out_exact']}/{report['target_folds']}")
    print(f"HCM-32 retrieval hits: {report['hcm32_retrieval_hits']}")
    print(f"Candidate evaluations avoided: {report['candidate_evaluations_avoided']}")
    print(f"Source memory unchanged: {report['source_memory_unchanged_during_target_evaluation']}")
    print(f"Runtime: {report['runtime_seconds']:.3f}s")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

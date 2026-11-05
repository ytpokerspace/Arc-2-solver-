"""Leave-one-training-pair-out benchmark for QBS-guided predictive MCTS."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .primitives import as_grid
from .predictive_mcts import search


def load_tasks(path: Path, limit: int) -> list[dict[str, Any]]:
    if path.is_dir():
        tasks = []
        for file in sorted(path.glob("*.json"))[:limit]:
            item = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(item, dict):
                raise ValueError(f"{file.name} must contain a task object")
            tasks.append(dict(item, task_id=file.stem))
        return tasks
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [dict(value, task_id=key) for key, value in sorted(raw.items())]
    if not isinstance(raw, list):
        raise ValueError("dataset must be a task list, task mapping, or directory")
    return raw[:limit]


def run_benchmark(
    dataset: Path,
    limit: int = 20,
    iterations: int = 40,
    branch_width: int = 10,
    max_depth: int = 3,
    memory_path: Path | None = None,
) -> dict[str, Any]:
    tasks = load_tasks(dataset, limit)
    folds: list[dict[str, Any]] = []
    skipped: list[str] = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id", task_index))
        print(f"[TASK] {task_index + 1}/{len(tasks)} id={task_id}", flush=True)
        train = task.get("train")
        if not isinstance(train, list) or len(train) < 2:
            skipped.append(task_id)
            continue
        pairs = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train)
        for held_index, (held_input, held_output) in enumerate(pairs):
            print(f"[FOLD] task={task_id} held_out={held_index + 1}/{len(pairs)}", flush=True)
            fit_pairs = tuple(pair for index, pair in enumerate(pairs) if index != held_index)
            result = search(
                fit_pairs, iterations=iterations, branch_width=branch_width,
                max_depth=max_depth, memory_path=memory_path,
            )
            prediction = None
            exact = False
            error = None
            if result.program is not None:
                try:
                    prediction = result.program.apply(held_input)
                    exact = prediction == held_output
                except (ValueError, TypeError, IndexError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            folds.append({
                "task_id": task_id,
                "held_out_training_pair": held_index,
                "fit_pairs": len(fit_pairs),
                "program": str(result.program) if result.program else None,
                "fit_exact": result.exact_fit,
                "solved_pairs": result.solved_pairs,
                "iterations": result.iterations,
                "nodes_created": result.nodes_created,
                "hypotheses_scored": result.hypotheses_scored,
                "hybrid_seeds": result.hybrid_seeds,
                "hybrid_nodes_created": result.hybrid_nodes_created,
                "best_probability_estimate": result.best_probability_estimate,
                "best_feasibility": result.best_feasibility,
                "best_training_pair_success_rate": result.best_training_pair_success_rate,
                "best_alignment_score": result.best_alignment_score,
                "duplicate_states_pruned": result.duplicate_states_pruned,
                "first_hypotheses": list(result.first_hypotheses),
                "hybrid_candidates": list(result.hybrid_candidates),
                "memory_hit": result.memory_hit,
                "memory_saved": result.memory_saved,
                "held_out_exact": exact,
                "prediction": [list(row) for row in prediction] if prediction is not None else None,
                "expected": [list(row) for row in held_output],
                "error": error,
            })
    elapsed = time.perf_counter() - started
    exact = sum(row["held_out_exact"] for row in folds)
    fit = sum(row["fit_exact"] for row in folds)
    count = max(1, len(folds))
    return {
        "engine": "qbs-guided-predictive-pair-matching-mcts",
        "search_methods": [
            "score ten first-step hypotheses on every training pair",
            "one bounded hybrid expansion from the top hypotheses",
            "MCTS UCT selection and backpropagation",
            "preserve solved training pairs",
            "accept only exact fits on every supplied training pair",
        ],
        "probability_score_calibrated": False,
        "probability_definition": "Beta(1,1)-smoothed exact training-pair match rate; not unseen-task probability",
        "feasibility_definition": "1.0 only when the candidate executes on every supplied training input; invalid candidates are rejected",
        "alignment_definition": "mean normalized cell agreement across supplied training pairs",
        "mcts_used": True,
        "beam_search_used": False,
        "dataset": dataset.name,
        "tasks_considered": len(tasks),
        "eligible_tasks": len({row["task_id"] for row in folds}),
        "skipped_tasks": skipped,
        "folds": len(folds),
        "fit_folds_with_exact_program": fit,
        "held_out_exact": exact,
        "held_out_accuracy": exact / count if folds else 0.0,
        "mean_hypotheses_scored": sum(row["hypotheses_scored"] for row in folds) / count,
        "mean_hybrid_seeds": sum(row["hybrid_seeds"] for row in folds) / count,
        "mean_hybrid_nodes_created": sum(row["hybrid_nodes_created"] for row in folds) / count,
        "mean_duplicate_states_pruned": sum(row["duplicate_states_pruned"] for row in folds) / count,
        "mean_probability_estimate": sum(row["best_probability_estimate"] for row in folds) / count,
        "mean_feasibility": sum(row["best_feasibility"] for row in folds) / count,
        "mean_training_pair_success_rate": sum(row["best_training_pair_success_rate"] for row in folds) / count,
        "iterations_per_fold": iterations,
        "branch_width": branch_width,
        "max_depth": max_depth,
        "memory_hits": sum(row["memory_hit"] for row in folds),
        "memory_saves": sum(row["memory_saved"] for row in folds),
        "runtime_seconds": round(elapsed, 6),
        "scope": "leave-one-training-pair-out cross-validation; official test arrays are never read",
        "fold_results": folds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark QBS-guided predictive ARC-R2 MCTS")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--branch-width", type=int, default=10)
    parser.add_argument("--max-depth", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--memory", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_predictive_mcts_benchmark.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists() or not (args.dataset.is_file() or args.dataset.is_dir()):
        parser.error("dataset file or directory does not exist")
    if not 1 <= args.limit <= 500:
        parser.error("limit must be in 1..500")
    if not 1 <= args.iterations <= 1000 or not 1 <= args.branch_width <= 50:
        parser.error("iterations must be 1..1000 and branch-width 1..50")
    report = run_benchmark(args.dataset, args.limit, args.iterations,
                           args.branch_width, args.max_depth, args.memory)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 QBS-GUIDED PREDICTIVE MCTS | BEAM: OFF | TEST ANSWERS: UNUSED")
    print(f"Exact held-out training pairs: {report['held_out_exact']}/{report['folds']} "
          f"({report['held_out_accuracy']:.3%})")
    print(f"Exact training fits: {report['fit_folds_with_exact_program']}/{report['folds']}")
    print(f"Mean hybrid seeds: {report['mean_hybrid_seeds']:.2f}; "
          f"mean hybrid nodes: {report['mean_hybrid_nodes_created']:.2f}")
    print(f"Memory: {report['memory_hits']} hits, {report['memory_saves']} saves")
    print(f"Runtime: {report['runtime_seconds']:.3f}s")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

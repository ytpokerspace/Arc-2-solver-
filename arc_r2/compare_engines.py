"""Leave-one-out comparison of baseline, HIAN, and object ARC engines.

Only supplied training pairs are used. Each fold holds one training pair out,
fits on the remaining pairs, and scores the held-out pair. Test arrays are
deliberately never read. This is a cross-validation diagnostic, not an official
ARC-AGI-2 leaderboard score.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .diagnostics import diagnose_candidates
from .grid import as_grid
from .object_solver import ObjectExample, ObjectHybridSolver
from .search import SearchLimits
from .solver import Example, HybridSolver
from .robust_solver import solve_two_attempts


def load_tasks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if limit is not None:
            files = files[:limit]
        raw = []
        for file_path in files:
            task = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(task, dict):
                raise ValueError(f"{file_path.name} must contain a task object")
            raw.append(dict(task, task_id=file_path.stem))
    else:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(raw, dict):
            raw = [dict(value, task_id=key) for key, value in sorted(raw.items())]
        if not isinstance(raw, list):
            raise ValueError("dataset must be a task list, task-ID mapping, or directory of task JSON files")
    for index, task in enumerate(raw):
        if not isinstance(task, dict) or not isinstance(task.get("train"), list):
            raise ValueError(f"task {index} must contain a train list")
        for pair_index, pair in enumerate(task["train"]):
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise ValueError(f"task {index} training pair {pair_index} needs input and output")
            as_grid(pair["input"])
            as_grid(pair["output"])
    return raw


def _predict(engine: str, training_pairs: tuple[dict, ...], held_out: dict,
             max_depth: int, candidate_cap: int, mcts_iterations: int) -> dict[str, Any]:
    started = time.perf_counter()
    candidates = 0
    try:
        if engine in {"baseline", "hian"}:
            examples = tuple(
                Example(as_grid(pair["input"]), as_grid(pair["output"]))
                for pair in training_pairs
            )
            limits = SearchLimits(
                max_depth=max_depth,
                candidate_cap=candidate_cap,
                mcts_iterations=mcts_iterations,
            )
            result = HybridSolver(
                limits, strategy="hybrid", use_hian=(engine == "hian")
            ).solve(examples)
            candidates = result.candidates_evaluated
            prediction = result.program.apply(as_grid(held_out["input"]))
            hian_report = result.hian_report
        elif engine == "robust_tta":
            exact_examples = tuple(
                (as_grid(pair["input"]), as_grid(pair["output"]))
                for pair in training_pairs
            )
            report = solve_two_attempts(
                exact_examples, as_grid(held_out["input"]), max_depth=1
            )
            candidates = report["exact_fit_candidates_seen"]
            attempts = report["attempts"]
            expected = as_grid(held_out["output"])
            attempt_grids = tuple(as_grid(item["output"]) for item in attempts)
            exact = bool(attempt_grids and attempt_grids[0] == expected)
            within_two = any(grid == expected for grid in attempt_grids)
            return {
                "exact": exact,
                "within_two_exact": within_two,
                "attempts": attempts,
                "tta_views": report["views"],
                "views_with_exact_fit": report["views_with_exact_fit"],
                "candidates": candidates,
                "runtime_seconds": time.perf_counter() - started,
                "error": None,
                "hian_report": None,
            }
        else:
            examples = tuple(
                ObjectExample(as_grid(pair["input"]), as_grid(pair["output"]))
                for pair in training_pairs
            )
            result = ObjectHybridSolver(
                max_depth=min(max_depth, 4),
                candidate_cap=candidate_cap,
                parameter_cap=candidate_cap,
            ).solve(examples)
            candidates = result.candidates_evaluated
            prediction = result.program.apply(as_grid(held_out["input"]))
            hian_report = None
        exact = prediction == as_grid(held_out["output"])
        return {
            "exact": exact,
            "within_two_exact": exact,
            "candidates": candidates,
            "runtime_seconds": time.perf_counter() - started,
            "error": None,
            "hian_report": hian_report,
        }
    except Exception as exc:
        return {
            "exact": False,
            "within_two_exact": False,
            "candidates": candidates,
            "runtime_seconds": time.perf_counter() - started,
            "error": f"{type(exc).__name__}: {exc}",
            "hian_report": None,
        }


def compare_engines(dataset: Path, limit: int = 20, max_depth: int = 2,
                    candidate_cap: int = 1200, mcts_iterations: int = 100) -> dict[str, Any]:
    tasks = load_tasks(dataset, limit)
    eligible = [
        (task.get("task_id", str(index)), task)
        for index, task in enumerate(tasks)
        if len(task["train"]) >= 2
    ]
    skipped = [
        task.get("task_id", str(index))
        for index, task in enumerate(tasks)
        if len(task["train"]) < 2
    ]
    if not eligible:
        raise ValueError("no eligible tasks: leave-one-out comparison needs at least two train pairs per task")

    engines = ("baseline", "hian", "object", "robust_tta")
    totals = {
        engine: {"folds": 0, "exact": 0, "within_two_exact": 0, "candidates": 0,
                 "runtime_seconds": 0.0, "failures": 0}
        for engine in engines
    }
    task_rows = []
    fold_rows = []

    for task_id, task in eligible:
        task_exact = {engine: 0 for engine in engines}
        task_within_two = {engine: 0 for engine in engines}
        task_folds = len(task["train"])
        for held_index, held_out in enumerate(task["train"]):
            training_pairs = tuple(
                pair for index, pair in enumerate(task["train"]) if index != held_index
            )
            fold = {
                "task_id": str(task_id),
                "held_out_train_pair": held_index,
                "fit_pair_count": len(training_pairs),
                "engines": {},
                "diagnostics": {},
            }
            for engine in engines:
                result = _predict(
                    engine, training_pairs, held_out,
                    max_depth, candidate_cap, mcts_iterations,
                )
                fold["engines"][engine] = result
                # HIAN and baseline share the same candidate generator; the
                # diagnostic coverage audit therefore uses the baseline path.
                if engine == "robust_tta":
                    fold["diagnostics"][engine] = {
                        "method": "exact-fit TTA; only full-training-fit candidates are retained"
                    }
                else:
                    diagnostic_engine = "baseline" if engine == "hian" else engine
                    fold["diagnostics"][engine] = diagnose_candidates(
                        diagnostic_engine, training_pairs, held_out,
                        max_depth=max_depth, candidate_cap=candidate_cap,
                    )
                totals[engine]["folds"] += 1
                totals[engine]["exact"] += int(result["exact"])
                totals[engine]["within_two_exact"] += int(result.get("within_two_exact", result["exact"]))
                totals[engine]["candidates"] += result["candidates"]
                totals[engine]["runtime_seconds"] += result["runtime_seconds"]
                totals[engine]["failures"] += int(result["error"] is not None)
                task_exact[engine] += int(result["exact"])
                task_within_two[engine] += int(result.get("within_two_exact", result["exact"]))
            fold_rows.append(fold)
        task_rows.append({
            "task_id": str(task_id),
            "held_out_pairs": task_folds,
            **{f"{engine}_exact": task_exact[engine] for engine in engines},
            **{f"{engine}_within_two": task_within_two[engine] for engine in engines},
        })

    summary = {}
    for engine, values in totals.items():
        folds = values["folds"]
        summary[engine] = {
            **values,
            "held_out_exact_match_accuracy": values["exact"] / folds if folds else 0.0,
            "mean_candidates_per_fold": values["candidates"] / folds if folds else 0.0,
            "mean_runtime_seconds_per_fold": values["runtime_seconds"] / folds if folds else 0.0,
            "within_two_exact_match_accuracy": values["within_two_exact"] / folds if folds else 0.0,
            "fully_generalized_tasks": sum(
                row[f"{engine}_exact"] == row["held_out_pairs"] for row in task_rows
            ),
        }

    return {
        "experiment": "leave_one_training_pair_out",
        "dataset_file": dataset.name,
        "tasks_considered": len(tasks),
        "eligible_tasks": len(eligible),
        "skipped_tasks_with_fewer_than_two_train_pairs": skipped,
        "max_depth": max_depth,
        "candidate_cap_per_fold": candidate_cap,
        "mcts_iterations_per_baseline_and_hian_fold": mcts_iterations,
        "data_boundary": (
            "Only train input/output pairs are used. Each held-out train pair is "
            "scored after fitting on the other train pairs. The task test arrays "
            "are never read. This is cross-validation on provided training data, "
            "not official test accuracy."
        ),
        "diagnostic_definitions": {
            "train_fit_candidates": "Unique candidate behaviors that exactly reproduce every fit pair.",
            "held_out_coverage_candidates": "Unique candidate behaviors that exactly reproduce the held-out output, regardless of solver ranking.",
            "rejected_invalid_candidates": "Candidates safely discarded because an operation could not be applied.",
            "duplicate_behaviors_removed": "Programs with identical outputs on all fit and held-out inputs; shortest program retained.",
        },
        "robust_tta_method": (
            "Each held-out fold uses D4 geometry views and deterministic palette relabelings. "
            "Only programs exactly fitting every remaining training pair survive. Predictions "
            "are inverse-transformed and voted. This pilot caps robust program depth at one; "
            "top-1 and within-two-attempt exact rates are reported separately."
        ),
        "hian_method": (
            "HIAN reranks the same bounded baseline candidate pool using exact "
            "training evidence first, then mean cell agreement, a Beta(1,1)-"
            "smoothed training-fit estimate, and program simplicity. Its estimate "
            "is not a calibrated unseen-task probability."
        ),
        "summary": summary,
        "per_task": task_rows,
        "folds": fold_rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare ARC-R2 baseline, HIAN, object, and exact-fit TTA engines with leave-one-out CV"
    )
    parser.add_argument("--dataset", required=True, type=Path, help="local ARC JSON file")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--candidate-cap", type=int, default=1200)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_engine_comparison.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists() or not (args.dataset.is_file() or args.dataset.is_dir()):
        parser.error("dataset file or directory does not exist")
    if not 1 <= args.limit <= 500:
        parser.error("limit must be in 1..500")
    if not 1 <= args.max_depth <= 4:
        parser.error("max-depth must be in 1..4 so both grid engines use a comparable depth")
    if not 1 <= args.candidate_cap <= 10000:
        parser.error("candidate-cap must be in 1..10000")
    if not 0 <= args.mcts_iterations <= 5000:
        parser.error("mcts-iterations must be in 0..5000")

    report = compare_engines(
        args.dataset, args.limit, args.max_depth,
        args.candidate_cap, args.mcts_iterations,
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 ENGINE COMPARISON | LOCAL FILE ONLY | TEST ANSWERS UNUSED")
    print(f"Eligible tasks: {report['eligible_tasks']} / {report['tasks_considered']}")
    for engine, result in report["summary"].items():
        print(
            f"{engine}: {result['exact']}/{result['folds']} held-out exact "
            f"({result['held_out_exact_match_accuracy']:.3%}); "
            f"{result['candidates']} candidates; "
            f"{result['runtime_seconds']:.3f}s; "
            f"{result['failures']} failed folds"
        )
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

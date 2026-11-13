"""Unified ARC-AGI-2 solver: deterministic primitives plus bounded predictive MCTS.

Training pairs are the only source of target outputs during fitting. Test cases
are read for their input grids only. A program is emitted only when it exactly
reproduces every supplied training pair; otherwise the solver abstains.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .primitives import Program, as_grid, fit_program
from .predictive_mcts import search as predictive_mcts_search

Grid = tuple[tuple[int, ...], ...]
Example = tuple[Grid, Grid]


@dataclass(frozen=True)
class UnifiedConfig:
    primitive_candidate_cap: int = 5000
    primitive_max_depth: int = 2
    mcts_iterations: int = 40
    mcts_branch_width: int = 10
    mcts_max_depth: int = 3
    primitive_memory_path: str | None = None
    mcts_memory_path: str | None = None

    def __post_init__(self) -> None:
        if not 1 <= self.primitive_candidate_cap <= 50000:
            raise ValueError("primitive_candidate_cap must be in 1..50000")
        if not 0 <= self.primitive_max_depth <= 2:
            raise ValueError("primitive_max_depth must be in 0..2")
        if not 1 <= self.mcts_iterations <= 1000:
            raise ValueError("mcts_iterations must be in 1..1000")
        if not 1 <= self.mcts_branch_width <= 50:
            raise ValueError("mcts_branch_width must be in 1..50")
        if not 1 <= self.mcts_max_depth <= 4:
            raise ValueError("mcts_max_depth must be in 1..4")


@dataclass(frozen=True)
class CandidateEvidence:
    source: str
    program: Program
    exact_training_pairs: int
    total_training_pairs: int
    candidates_evaluated: int


@dataclass(frozen=True)
class UnifiedResult:
    program: Program | None
    exact_fit: bool
    source: str | None
    training_pairs: int
    candidates_evaluated: int
    primitive_candidates: int
    mcts_nodes_created: int
    mcts_iterations: int
    candidate_evidence: tuple[CandidateEvidence, ...]


def _program_key(program: Program) -> tuple:
    return (
        program.complexity,
        tuple((operation.name, repr(operation.args)) for operation in program.operations),
    )


def solve_training_pairs(
    examples: Iterable[tuple[Any, Any]],
    config: UnifiedConfig | None = None,
) -> UnifiedResult:
    """Run both ARC-2 search paths and select the simplest exact-fit program."""
    config = config or UnifiedConfig()
    pairs: tuple[Example, ...] = tuple(
        (as_grid(source), as_grid(target)) for source, target in examples
    )
    if not pairs:
        raise ValueError("at least one training pair is required")

    candidates: list[CandidateEvidence] = []
    primitive_result = fit_program(
        pairs,
        candidate_cap=config.primitive_candidate_cap,
        max_depth=config.primitive_max_depth,
        memory_path=config.primitive_memory_path,
    )
    if primitive_result.program is not None and primitive_result.exact_fit:
        candidates.append(CandidateEvidence(
            "primitive-enumeration", primitive_result.program,
            len(pairs), len(pairs), primitive_result.candidates_evaluated,
        ))

    mcts_result = predictive_mcts_search(
        pairs,
        iterations=config.mcts_iterations,
        branch_width=config.mcts_branch_width,
        max_depth=config.mcts_max_depth,
        memory_path=config.mcts_memory_path,
    )
    if mcts_result.program is not None and mcts_result.exact_fit:
        candidates.append(CandidateEvidence(
            "predictive-mcts", mcts_result.program,
            mcts_result.solved_pairs, mcts_result.total_pairs,
            mcts_result.iterations,
        ))

    # Both engines are allowed to propose candidates. Exact agreement on every
    # training pair is a hard gate; complexity and canonical order break ties.
    unique: dict[tuple, CandidateEvidence] = {}
    for evidence in candidates:
        key = _program_key(evidence.program)
        previous = unique.get(key)
        if previous is None or evidence.candidates_evaluated < previous.candidates_evaluated:
            unique[key] = evidence

    if not unique:
        return UnifiedResult(
            None, False, None, len(pairs),
            primitive_result.candidates_evaluated + mcts_result.iterations,
            primitive_result.candidates_evaluated, mcts_result.nodes_created,
            mcts_result.iterations, (),
        )

    winner = min(unique.values(), key=lambda item: _program_key(item.program))
    # Final independent exact-fit gate before the selected program can escape.
    try:
        exact = all(winner.program.apply(source) == target for source, target in pairs)
    except (ValueError, TypeError, IndexError):
        exact = False
    if not exact:
        return UnifiedResult(
            None, False, None, len(pairs),
            primitive_result.candidates_evaluated + mcts_result.iterations,
            primitive_result.candidates_evaluated, mcts_result.nodes_created,
            mcts_result.iterations, (),
        )
    return UnifiedResult(
        winner.program, True, winner.source, len(pairs),
        primitive_result.candidates_evaluated + mcts_result.iterations,
        primitive_result.candidates_evaluated, mcts_result.nodes_created,
        mcts_result.iterations,
        tuple(sorted(unique.values(), key=lambda item: _program_key(item.program))),
    )


def load_tasks(path: Path, limit: int = 20) -> list[dict[str, Any]]:
    """Load ARC task JSON without interpreting hidden test outputs."""
    if limit < 1:
        raise ValueError("limit must be positive")
    if path.is_dir():
        tasks = []
        for file in sorted(path.glob("*.json"))[:limit]:
            value = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"{file.name} must contain a task object")
            tasks.append(dict(value, task_id=file.stem))
        return tasks
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [dict(value, task_id=key) for key, value in sorted(raw.items())]
    if not isinstance(raw, list):
        raise ValueError("dataset must be a task list, task mapping, or directory")
    return raw[:limit]


def predict_task(task: dict[str, Any], config: UnifiedConfig | None = None) -> dict[str, Any]:
    """Fit all visible training pairs and predict test inputs; never read test outputs."""
    train = task.get("train")
    if not isinstance(train, list) or not train:
        raise ValueError("task must contain at least one training pair")
    examples = tuple(
        (as_grid(pair["input"]), as_grid(pair["output"])) for pair in train
    )
    result = solve_training_pairs(examples, config)
    predictions = []
    test_cases = task.get("test", [])
    if not isinstance(test_cases, list):
        raise ValueError("task test field must be a list")
    for index, case in enumerate(test_cases):
        # Deliberately access only case["input"]; test output labels are never read.
        test_input = as_grid(case["input"])
        predicted = None
        error = None
        if result.program is not None:
            try:
                predicted = result.program.apply(test_input)
            except (ValueError, TypeError, IndexError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        predictions.append({
            "test_index": index,
            "input": [list(row) for row in test_input],
            "prediction": [list(row) for row in predicted] if predicted is not None else None,
            "error": error,
        })
    return {
        "task_id": str(task.get("task_id", "unknown")),
        "status": "predicted" if result.program is not None else "abstained",
        "program": str(result.program) if result.program is not None else None,
        "selected_engine": result.source,
        "training_exact": result.exact_fit,
        "training_pairs": result.training_pairs,
        "candidates_evaluated": result.candidates_evaluated,
        "primitive_candidates": result.primitive_candidates,
        "mcts_nodes_created": result.mcts_nodes_created,
        "mcts_iterations": result.mcts_iterations,
        "test_predictions": predictions,
    }


def run_prediction(dataset: Path, limit: int = 20,
                   config: UnifiedConfig | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    tasks = load_tasks(dataset, limit)
    results = [predict_task(task, config) for task in tasks]
    return {
        "schema_version": 1,
        "engine": "arc2-unified-primitive-mcts",
        "dataset": dataset.name,
        "tasks_considered": len(tasks),
        "tasks_predicted": sum(item["status"] == "predicted" for item in results),
        "tasks_abstained": sum(item["status"] == "abstained" for item in results),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "scope": "fit on all visible training pairs; predict test inputs only; test outputs are not read",
        "results": results,
    }


def run_leave_one_out(dataset: Path, limit: int = 20,
                      config: UnifiedConfig | None = None) -> dict[str, Any]:
    """Within-task validation: hide one training pair, fit on the rest, test the holdout."""
    config = config or UnifiedConfig()
    tasks = load_tasks(dataset, limit)
    folds = []
    skipped = []
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        pairs = task.get("train", [])
        task_id = str(task.get("task_id", task_index))
        if not isinstance(pairs, list) or len(pairs) < 2:
            skipped.append(task_id)
            continue
        checked = tuple(
            (as_grid(pair["input"]), as_grid(pair["output"])) for pair in pairs
        )
        for held_index, (held_input, held_output) in enumerate(checked):
            fit_pairs = tuple(pair for index, pair in enumerate(checked) if index != held_index)
            result = solve_training_pairs(fit_pairs, config)
            prediction = None
            error = None
            if result.program is not None:
                try:
                    prediction = result.program.apply(held_input)
                except (ValueError, TypeError, IndexError) as exc:
                    error = f"{type(exc).__name__}: {exc}"
            exact = prediction == held_output if prediction is not None else False
            folds.append({
                "task_id": task_id,
                "held_out_training_pair": held_index,
                "fit_pairs": len(fit_pairs),
                "fit_exact": result.exact_fit,
                "selected_engine": result.source,
                "program": str(result.program) if result.program is not None else None,
                "held_out_exact": exact,
                "prediction": [list(row) for row in prediction] if prediction is not None else None,
                "expected": [list(row) for row in held_output],
                "candidates_evaluated": result.candidates_evaluated,
                "error": error,
            })
    exact_count = sum(row["held_out_exact"] for row in folds)
    return {
        "schema_version": 1,
        "protocol": "leave_one_training_pair_out",
        "engine": "arc2-unified-primitive-mcts",
        "dataset": dataset.name,
        "tasks_considered": len(tasks),
        "eligible_tasks": len({row["task_id"] for row in folds}),
        "skipped_tasks": skipped,
        "folds": len(folds),
        "held_out_exact": exact_count,
        "held_out_accuracy": exact_count / len(folds) if folds else None,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "scope": "within-task training-pair validation; not official hidden-test accuracy",
        "fold_results": folds,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified ARC-AGI-2 solver and training-pair validator")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--mode", choices=("predict", "validate"), default="predict")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--primitive-candidate-cap", type=int, default=5000)
    parser.add_argument("--primitive-max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--mcts-iterations", type=int, default=40)
    parser.add_argument("--mcts-branch-width", type=int, default=10)
    parser.add_argument("--mcts-max-depth", type=int, choices=(1, 2, 3, 4), default=3)
    parser.add_argument("--memory-dir", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("arc2_unified_report.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists() or not (args.dataset.is_file() or args.dataset.is_dir()):
        parser.error("dataset file or directory does not exist")
    memory_dir = args.memory_dir
    config = UnifiedConfig(
        primitive_candidate_cap=args.primitive_candidate_cap,
        primitive_max_depth=args.primitive_max_depth,
        mcts_iterations=args.mcts_iterations,
        mcts_branch_width=args.mcts_branch_width,
        mcts_max_depth=args.mcts_max_depth,
        primitive_memory_path=str(memory_dir / "primitive-memory.json") if memory_dir else None,
        mcts_memory_path=str(memory_dir / "mcts-memory.json") if memory_dir else None,
    )
    report = (
        run_prediction(args.dataset, args.limit, config)
        if args.mode == "predict"
        else run_leave_one_out(args.dataset, args.limit, config)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[ARC2] mode={args.mode} tasks={report.get('tasks_considered', 0)} "
          f"scope={report['scope']}")
    if args.mode == "validate":
        print(f"[VALIDATION] held_out={report['held_out_exact']}/{report['folds']} "
              f"accuracy={report['held_out_accuracy']}")
    else:
        print(f"[PREDICTION] predicted={report['tasks_predicted']} "
              f"abstained={report['tasks_abstained']}")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

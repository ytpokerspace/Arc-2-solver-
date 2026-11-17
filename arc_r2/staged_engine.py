"""Sequential ARC solver: exact-fit primitives, then beam, then MCTS.

Only training pairs are passed to search. Test inputs are transformed only after
a stage has produced a program that exactly matches every training pair.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import time
from typing import Any, Iterable

from .grid import Grid, as_grid, to_lists
from .primitives import fit_program
from .dsl import Program as DSLProgram
from .search import SearchLimits, SearchStats, beam_search, mcts_search
from .solver import Example, score_program


@dataclass
class StageResult:
    stage: str
    status: str
    candidates_evaluated: int
    program: str | None = None


@dataclass
class StagedResult:
    status: str
    stage: str | None
    program: Any | None
    predictions: list[Grid]
    candidates_evaluated: int
    stages: list[StageResult]
    elapsed_seconds: float

    def report(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "stage": self.stage,
            "program": str(self.program) if self.program is not None else None,
            "predictions": [to_lists(grid) for grid in self.predictions],
            "candidates_evaluated": self.candidates_evaluated,
            "stages": [asdict(stage) for stage in self.stages],
            "elapsed_seconds": round(self.elapsed_seconds, 6),
        }


def _exact(program: Any, examples: tuple[Example, ...]) -> bool:
    try:
        return all(program.apply(ex.input) == ex.output for ex in examples)
    except (ValueError, TypeError, IndexError):
        return False


def solve_staged(
    train_pairs: Iterable[tuple[Any, Any]],
    test_inputs: Iterable[Any] = (),
    *,
    total_candidate_budget: int = 10000,
    stage_a_cap: int = 5000,
    limits: SearchLimits | None = None,
) -> StagedResult:
    """Run A -> B -> C with one candidate-evaluation budget.

    A: parameterized primitive exact-fit enumeration.
    B: DSL beam search.
    C: DSL MCTS, only if B has no exact training fit.

    The test inputs are never used to score, rank, or select a program.
    """
    started = time.perf_counter()
    pairs = tuple((as_grid(a), as_grid(b)) for a, b in train_pairs)
    tests = tuple(as_grid(grid) for grid in test_inputs)
    if not pairs:
        raise ValueError("at least one training pair is required")
    if total_candidate_budget < 1:
        raise ValueError("total_candidate_budget must be at least 1")
    if stage_a_cap < 1:
        raise ValueError("stage_a_cap must be at least 1")

    remaining = total_candidate_budget
    stages: list[StageResult] = []

    # A — exact-fit primitive engine.
    a_cap = min(stage_a_cap, remaining, 50000)
    a_result = fit_program(pairs, candidate_cap=a_cap, max_depth=2)
    remaining -= a_result.candidates_evaluated
    stages.append(StageResult(
        "A", "exact_fit" if a_result.exact_fit else "no_exact_fit",
        a_result.candidates_evaluated,
        str(a_result.program) if a_result.exact_fit else None,
    ))
    if a_result.exact_fit and a_result.program is not None and _exact(a_result.program, tuple(
        Example(source, target) for source, target in pairs
    )):
        predictions = [a_result.program.apply(grid) for grid in tests]
        return StagedResult("solved_on_training", "A", a_result.program, predictions,
                            total_candidate_budget - remaining, stages,
                            time.perf_counter() - started)

    examples = tuple(Example(source, target) for source, target in pairs)
    base_limits = limits or SearchLimits()
    def state_key(program: DSLProgram):
        try:
            return tuple(program.apply(example.input) for example in examples)
        except (ValueError, TypeError, IndexError):
            # Invalid operations must not crash search or collide with valid states.
            return ("__INVALID__", program.canonical())
    scorer = lambda program: score_program(program, examples)

    # B — beam search. The ranked list is checked with the exact oracle.
    if remaining > 0:
        b_stats = SearchStats()
        b_limits = SearchLimits(
            beam_width=base_limits.beam_width,
            max_depth=base_limits.max_depth,
            candidate_cap=remaining,
            mcts_iterations=base_limits.mcts_iterations,
            exploration=base_limits.exploration,
            seed=base_limits.seed,
            include_prte=base_limits.include_prte,
        )
        b_candidates = beam_search(scorer, b_limits, b_stats, state_key=state_key)
        remaining -= b_stats.evaluated
        b_program = next((program for _, program in b_candidates if _exact(program, examples)), None)
        stages.append(StageResult(
            "B", "exact_fit" if b_program is not None else
            ("budget_exhausted" if b_stats.stopped_by_cap else "no_exact_fit"),
            b_stats.evaluated, str(b_program) if b_program is not None else None,
        ))
        if b_program is not None:
            predictions = [b_program.apply(grid) for grid in tests]
            return StagedResult("solved_on_training", "B", b_program, predictions,
                                total_candidate_budget - remaining, stages,
                                time.perf_counter() - started)
    else:
        stages.append(StageResult("B", "skipped_no_budget", 0))

    # C — MCTS runs only after B fails and only with remaining budget.
    if remaining > 0:
        c_stats = SearchStats()
        c_limits = SearchLimits(
            beam_width=base_limits.beam_width,
            max_depth=base_limits.max_depth,
            candidate_cap=remaining,
            mcts_iterations=base_limits.mcts_iterations,
            exploration=base_limits.exploration,
            seed=base_limits.seed,
            include_prte=base_limits.include_prte,
        )
        c_candidates = mcts_search(scorer, c_limits, c_stats, state_key=state_key)
        remaining -= c_stats.evaluated
        c_program = next((program for _, program in c_candidates if _exact(program, examples)), None)
        stages.append(StageResult(
            "C", "exact_fit" if c_program is not None else
            ("budget_exhausted" if c_stats.stopped_by_cap else "no_exact_fit"),
            c_stats.evaluated, str(c_program) if c_program is not None else None,
        ))
        if c_program is not None:
            predictions = [c_program.apply(grid) for grid in tests]
            return StagedResult("solved_on_training", "C", c_program, predictions,
                                total_candidate_budget - remaining, stages,
                                time.perf_counter() - started)
    else:
        stages.append(StageResult("C", "skipped_no_budget", 0))

    return StagedResult("abstained", None, None, [], total_candidate_budget - remaining,
                        stages, time.perf_counter() - started)


def load_tasks(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if path.is_dir():
        files = sorted(path.glob("*.json"))
        if limit is not None:
            files = files[:limit]
        return [dict(json.loads(file.read_text(encoding="utf-8")), task_id=file.stem)
                for file in files]
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        tasks = [dict(value, task_id=key) for key, value in sorted(raw.items())]
    elif isinstance(raw, list):
        tasks = raw
    else:
        raise ValueError("dataset must be a task mapping, task list, or directory")
    return tasks if limit is None else tasks[:limit]


def run_dataset(path: Path, limit: int | None = None, **kwargs: Any) -> dict[str, Any]:
    """Produce predictions using train pairs only; test outputs are not accessed."""
    reports = []
    for index, task in enumerate(load_tasks(path, limit)):
        task_id = str(task.get("task_id", index))
        train = tuple((pair["input"], pair["output"]) for pair in task.get("train", []))
        tests = tuple(pair["input"] for pair in task.get("test", []))
        result = solve_staged(train, tests, **kwargs)
        reports.append({"task_id": task_id, **result.report()})
    return {
        "engine": "sequential-A-B-C",
        "stage_order": ["A:primitive_exact_fit", "B:DSL_beam", "C:DSL_MCTS"],
        "test_outputs_used_for_search": False,
        "tasks": len(reports),
        "solved_on_training": sum(row["status"] == "solved_on_training" for row in reports),
        "abstained": sum(row["status"] == "abstained" for row in reports),
        "results": reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Sequential ARC engine: primitive -> beam -> MCTS")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--budget", type=int, default=10000)
    parser.add_argument("--stage-a-cap", type=int, default=5000)
    parser.add_argument("--beam-width", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--mcts-iterations", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_staged_results.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists():
        parser.error("dataset path does not exist")
    limits = SearchLimits(beam_width=args.beam_width, max_depth=args.max_depth,
                          candidate_cap=args.budget, mcts_iterations=args.mcts_iterations,
                          seed=args.seed)
    report = run_dataset(args.dataset, args.limit, total_candidate_budget=args.budget,
                         stage_a_cap=args.stage_a_cap, limits=limits)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 STAGED ENGINE | A -> B -> C | TEST OUTPUTS UNUSED")
    print(f"Tasks: {report['tasks']} | exact training fit: {report['solved_on_training']} "
          f"| abstained: {report['abstained']}")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

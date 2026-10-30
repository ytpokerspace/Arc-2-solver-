"""Local training-fit comparison and operator ablation harness.

This measures fit to each task's provided training examples only. It is not
held-out ARC test accuracy and does not establish generalization.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from .grid import as_grid
from .search import SearchLimits
from .solver import Example, HybridSolver

VARIANTS = (
    ("identity", "identity", True, True),
    ("beam", "beam", True, True),
    ("mcts", "mcts", True, True),
    ("hybrid", "hybrid", True, True),
    ("hybrid_without_prte", "hybrid", False, True),
    ("hybrid_without_object_signal", "hybrid", True, False),
)

def load_tasks(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [dict(value, task_id=key) for key, value in sorted(raw.items())]
    if not isinstance(raw, list):
        raise ValueError("dataset must be a task list or task-id mapping")
    return raw

def run_benchmark(dataset: Path, limit: int, max_depth: int, candidate_cap: int, mcts_iterations: int):
    tasks = load_tasks(dataset)[:limit]
    rows = []
    print("ARC-R2 BENCHMARK | LOCAL FILE ONLY | NETWORK DISABLED | PAID CALLS: 0")
    for variant, strategy, include_prte, use_objects in VARIANTS:
        started = time.perf_counter()
        exact = total = candidates = solved_tasks = 0
        for index, task in enumerate(tasks, 1):
            examples = tuple(Example(as_grid(pair["input"]), as_grid(pair["output"]))
                             for pair in task["train"])
            limits = SearchLimits(max_depth=max_depth, candidate_cap=candidate_cap,
                                  mcts_iterations=mcts_iterations, include_prte=include_prte)
            result = HybridSolver(limits, strategy=strategy,
                                  use_object_signal=use_objects).solve(examples)
            exact += result.exact_examples
            total += result.total_examples
            candidates += result.candidates_evaluated
            solved_tasks += result.exact_examples == result.total_examples
            print(f"[{variant}] task {index}/{len(tasks)}: "
                  f"{result.exact_examples}/{result.total_examples} exact")
        elapsed = time.perf_counter() - started
        rows.append({
            "variant": variant, "tasks": len(tasks), "train_examples": total,
            "exact_train_examples": exact,
            "train_example_accuracy": exact / total if total else 0.0,
            "fully_fit_tasks": solved_tasks, "candidate_evaluations": candidates,
            "runtime_seconds": round(elapsed, 6),
            "scope": "training-example fit only; not held-out test accuracy",
        })
        print(f"[SUMMARY] {variant}: {exact}/{total} training examples; "
              f"{candidates} candidates; {elapsed:.4f}s")
    return rows

def main(argv=None):
    parser = argparse.ArgumentParser(description="ARC-R2 local search and ablation comparison")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--candidate-cap", type=int, default=1200)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_benchmark.json"))
    args = parser.parse_args(argv)
    if not args.dataset.is_file(): parser.error("dataset file not found")
    if not 1 <= args.limit <= 500 or not 1 <= args.max_depth <= 8:
        parser.error("limit must be 1..500 and max-depth 1..8")
    if not 1 <= args.candidate_cap <= 10000 or not 0 <= args.mcts_iterations <= 5000:
        parser.error("candidate-cap must be 1..10000 and mcts-iterations 0..5000")
    rows = run_benchmark(args.dataset, args.limit, args.max_depth,
                         args.candidate_cap, args.mcts_iterations)
    args.output.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"[SAVED] {args.output}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())

"""Local-only command line runner with bounded execution and visible status."""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from .grid import as_grid, to_lists
from .search import SearchLimits
from .solver import Example, HybridSolver
from .object_solver import ObjectExample, ObjectHybridSolver


def _task_examples(task):
    return tuple(Example(as_grid(pair["input"]), as_grid(pair["output"])) for pair in task["train"])


def main(argv=None):
    parser = argparse.ArgumentParser(description="ARC-R2 local symbolic solver")
    parser.add_argument("--demo", action="store_true", help="run built-in synthetic demonstration")
    parser.add_argument("--dataset", type=Path, help="local ARC-format JSON file; never downloaded")
    parser.add_argument("--engine", choices=("baseline", "object", "robust"), default="baseline",
                        help="baseline grid DSL, object DSL, or exact-fit TTA two-attempt solver")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--candidate-cap", type=int, default=1200)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--output", type=Path, default=Path("arc_r2_runs"))
    args = parser.parse_args(argv)
    if args.limit < 1 or args.limit > 500 or args.max_depth < 1 or args.max_depth > 8:
        parser.error("limit must be 1..500 and max-depth 1..8")
    if args.candidate_cap < 1 or args.candidate_cap > 10000 or args.mcts_iterations < 0 or args.mcts_iterations > 5000:
        parser.error("candidate-cap must be 1..10000 and mcts-iterations 0..5000")
    if not args.demo and not args.dataset:
        parser.error("choose --demo or --dataset")
    if args.dataset and not args.dataset.is_file():
        parser.error("dataset path does not exist or is not a file")
    if args.demo:
        raw = [{"train": [
            {"input": [[0,1,0],[0,0,1]], "output": [[0,0,1],[1,0,0]]},
            {"input": [[1,0,0],[0,1,0]], "output": [[0,0,1],[0,1,0]]}
        ], "test": []}]
    else:
        raw = json.loads(args.dataset.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = [dict(value, task_id=key) for key, value in sorted(raw.items())]
    if not isinstance(raw, list):
        parser.error("dataset must be a JSON list or ARC task-id mapping")
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / f"{args.engine}_run.jsonl"
    limits = SearchLimits(max_depth=args.max_depth, candidate_cap=args.candidate_cap,
                          mcts_iterations=args.mcts_iterations)
    if args.engine == "robust":
        from .robust_solver import solve_two_attempts
    started = time.perf_counter()
    print("ARC-R2 | SAFETY: ACTIVE | NETWORK: DISABLED | PAID CALLS: 0")
    print("[PASS] Local input selected; no network or API integration")
    print(f"[ENGINE] {args.engine}")
    results = []
    try:
        for index, task in enumerate(raw[:args.limit], 1):
            print(f"[RUNNING] Task {index}/{min(len(raw), args.limit)}")
            if args.engine == "robust":
                training_pairs = tuple(
                    (as_grid(pair["input"]), as_grid(pair["output"]))
                    for pair in task["train"]
                )
                predictions = [
                    solve_two_attempts(training_pairs, as_grid(case["input"]),
                                        max_depth=min(args.max_depth, 2))
                    for case in task.get("test", [])
                ]
                item = {"task": index, "task_id": task.get("task_id"), "engine": args.engine,
                        "program": "EXACT_FIT_TTA_VOTING",
                        "train_exact": None, "train_total": len(training_pairs),
                        "score": None,
                        "candidates": sum(row["exact_fit_candidates_seen"] for row in predictions),
                        "test_predictions": predictions}
                results.append(item)
                available = sum(len(row["attempts"]) for row in predictions)
                print(f"[RESULT] robust test cases={len(predictions)}; "
                      f"attempts={available}; views={sum(row['views'] for row in predictions)}")
            else:
                if args.engine == "object":
                    examples = tuple(ObjectExample(as_grid(pair["input"]), as_grid(pair["output"]))
                                     for pair in task["train"])
                    result = ObjectHybridSolver(
                        max_depth=min(args.max_depth, 4),
                        candidate_cap=args.candidate_cap,
                        parameter_cap=args.candidate_cap,
                    ).solve(examples)
                    apply_program = result.program.apply
                else:
                    examples = _task_examples(task)
                    result = HybridSolver(limits).solve(examples)
                    apply_program = result.program.apply
                predictions = [to_lists(apply_program(as_grid(case["input"])))
                               for case in task.get("test", [])]
                item = {"task": index, "task_id": task.get("task_id"), "engine": args.engine,
                        "program": str(result.program),
                        "train_exact": result.exact_examples, "train_total": result.total_examples,
                        "score": round(result.score, 6), "candidates": result.candidates_evaluated,
                        "test_predictions": predictions}
                results.append(item)
                print(f"[RESULT] train exact {result.exact_examples}/{result.total_examples}; "
                      f"program={result.program}; candidates={result.candidates_evaluated}")
            with log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(item, sort_keys=True) + "\n")
    except KeyboardInterrupt:
        print("[STOPPED] User cancellation received; partial log preserved.")
        return 130
    elapsed = time.perf_counter() - started
    print(f"[COMPLETE] tasks={len(results)} elapsed_seconds={elapsed:.4f}")
    print(f"[LOG] {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

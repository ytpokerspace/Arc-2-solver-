"""Leakage-resistant leave-one-training-pair-out evaluation for ARC-R2.

This is an internal validation protocol, not an official ARC score. Each
training pair is withheld in turn; the solver sees only the remaining pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

from .benchmark import VARIANTS, load_tasks
from .grid import as_grid
from .search import SearchLimits
from .solver import Example, HybridSolver


def leave_one_out_folds(pairs):
    """Yield (heldout_index, visible_training_pairs, heldout_pair)."""
    if len(pairs) < 2:
        return
    for heldout_index, heldout in enumerate(pairs):
        visible = tuple(pair for index, pair in enumerate(pairs)
                        if index != heldout_index)
        yield heldout_index, visible, heldout


def evaluate_variant(tasks, variant, max_depth=3, candidate_cap=1200,
                    mcts_iterations=100):
    """Evaluate one existing solver configuration with within-task holdouts."""
    name, strategy, include_prte, use_objects = variant
    started = time.perf_counter()
    fold_rows = []
    eligible_tasks = skipped_tasks = fully_correct_tasks = 0
    heldout_exact = heldout_total = 0
    train_exact = train_total = candidate_evaluations = 0

    for task_index, task in enumerate(tasks, 1):
        pairs = task.get("train", [])
        if len(pairs) < 2:
            skipped_tasks += 1
            continue

        eligible_tasks += 1
        task_correct = True
        task_id = task.get("task_id", str(task_index))

        for heldout_index, visible_pairs, heldout in leave_one_out_folds(pairs):
            examples = tuple(
                Example(as_grid(pair["input"]), as_grid(pair["output"]))
                for pair in visible_pairs
            )
            limits = SearchLimits(
                max_depth=max_depth,
                candidate_cap=candidate_cap,
                mcts_iterations=mcts_iterations,
                seed=0,
                include_prte=include_prte,
            )
            result = HybridSolver(
                limits,
                strategy=strategy,
                use_object_signal=use_objects,
            ).solve(examples)

            prediction = result.program.apply(as_grid(heldout["input"]))
            target = as_grid(heldout["output"])
            exact = prediction == target

            heldout_total += 1
            heldout_exact += int(exact)
            train_exact += result.exact_examples
            train_total += result.total_examples
            candidate_evaluations += result.candidates_evaluated
            task_correct = task_correct and exact

            fold_rows.append({
                "task_id": task_id,
                "heldout_pair_index": heldout_index,
                "visible_training_pairs": len(visible_pairs),
                "training_exact": result.exact_examples,
                "training_total": result.total_examples,
                "heldout_exact": exact,
                "selected_program": str(result.program),
                "candidate_evaluations": result.candidates_evaluated,
            })

        fully_correct_tasks += int(task_correct)

    elapsed = time.perf_counter() - started
    return {
        "variant": name,
        "tasks_eligible": eligible_tasks,
        "tasks_skipped_fewer_than_two_pairs": skipped_tasks,
        "tasks_all_holdouts_exact": fully_correct_tasks,
        "heldout_exact": heldout_exact,
        "heldout_total": heldout_total,
        "heldout_accuracy": heldout_exact / heldout_total if heldout_total else None,
        "training_exact_across_folds": train_exact,
        "training_total_across_folds": train_total,
        "training_fit_accuracy_across_folds": (
            train_exact / train_total if train_total else None
        ),
        "candidate_evaluations": candidate_evaluations,
        "runtime_seconds": round(elapsed, 6),
        "folds": fold_rows,
        "scope": (
            "leave-one-training-pair-out within-task validation; "
            "not official ARC evaluation accuracy"
        ),
    }


def run_leave_one_out(dataset: Path, limit=20, max_depth=3,
                      candidate_cap=1200, mcts_iterations=100,
                      selected_variants=None):
    raw_bytes = dataset.read_bytes()
    tasks = load_tasks(dataset)[:limit]
    variants = tuple(
        variant for variant in VARIANTS
        if selected_variants is None or variant[0] in selected_variants
    )
    if not variants:
        raise ValueError("no benchmark variants selected")

    results = [
        evaluate_variant(
            tasks, variant, max_depth=max_depth,
            candidate_cap=candidate_cap,
            mcts_iterations=mcts_iterations,
        )
        for variant in variants
    ]
    return {
        "schema_version": 1,
        "protocol": "leave_one_training_pair_out",
        "dataset": {
            "filename": dataset.name,
            "sha256": hashlib.sha256(raw_bytes).hexdigest(),
            "tasks_loaded": len(tasks),
            "task_limit": limit,
        },
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "git_commit": os.environ.get("GITHUB_SHA", "not-provided"),
        },
        "limits_per_fold": {
            "max_depth": max_depth,
            "candidate_cap": candidate_cap,
            "mcts_iterations": mcts_iterations,
            "random_seed": 0,
        },
        "variants": results,
        "interpretation": (
            "Held-out pairs are withheld from solver fitting. This estimates "
            "within-task rule consistency; it is not independent-task "
            "generalization and is not an official ARC score."
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="ARC-R2 leave-one-training-pair-out validation"
    )
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--candidate-cap", type=int, default=1200)
    parser.add_argument("--mcts-iterations", type=int, default=100)
    parser.add_argument("--variant", action="append", choices=[v[0] for v in VARIANTS])
    parser.add_argument("--output", type=Path, default=Path("arc_r2_heldout.json"))
    args = parser.parse_args(argv)

    if not args.dataset.is_file():
        parser.error("dataset file not found")
    if not 1 <= args.limit <= 500 or not 1 <= args.max_depth <= 8:
        parser.error("limit must be 1..500 and max-depth 1..8")
    if not 1 <= args.candidate_cap <= 10000 or not 0 <= args.mcts_iterations <= 5000:
        parser.error("candidate-cap must be 1..10000 and mcts-iterations 0..5000")

    report = run_leave_one_out(
        args.dataset,
        limit=args.limit,
        max_depth=args.max_depth,
        candidate_cap=args.candidate_cap,
        mcts_iterations=args.mcts_iterations,
        selected_variants=set(args.variant) if args.variant else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 HELD-OUT VALIDATION | LOCAL DATA ONLY | PAID CALLS: 0")
    print(f"[DATASET] tasks={report['dataset']['tasks_loaded']} "
          f"sha256={report['dataset']['sha256']}")
    for row in report["variants"]:
        print(
            f"[{row['variant']}] held-out={row['heldout_exact']}/"
            f"{row['heldout_total']} "
            f"tasks={row['tasks_all_holdouts_exact']}/"
            f"{row['tasks_eligible']} candidates={row['candidate_evaluations']} "
            f"time={row['runtime_seconds']:.4f}s"
        )
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Primitive-only ARC training harness. No MCTS, beam search, or hidden-test access."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .hcm32_bridge import HCM32Bridge
from .primitives import as_grid, fit_program


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


def run_benchmark(dataset: Path, limit: int = 20, candidate_cap: int = 5000,
                  max_depth: int = 2, memory_path: Path | None = None,
                  hcm32_memory_path: Path | None = None) -> dict[str, Any]:
    tasks = load_tasks(dataset, limit)
    folds = []
    skipped = []
    pending_hcm32 = []
    hcm32 = HCM32Bridge(hcm32_memory_path) if hcm32_memory_path else None
    started = time.perf_counter()
    for task_index, task in enumerate(tasks):
        task_id = str(task.get("task_id", task_index))
        print(f"[TASK] {task_index + 1}/{len(tasks)} id={task_id}", flush=True)
        train = task.get("train")
        if not isinstance(train, list) or len(train) < 2:
            skipped.append(task_id)
            continue
        checked = []
        for pair in train:
            checked.append((as_grid(pair["input"]), as_grid(pair["output"])))
        for held_index, (held_input, held_output) in enumerate(checked):
            print(f"[FOLD] task={task_id} held_out={held_index + 1}/{len(checked)}", flush=True)
            fit_pairs = tuple(pair for i, pair in enumerate(checked) if i != held_index)
            hcm_match = hcm32.find_program(fit_pairs) if hcm32 else None
            if hcm_match is not None:
                program = hcm_match.program
                fit_exact = True
                fit_program_count = 1
                candidates_evaluated = 0
                single_stage_memory_hit = False
                single_stage_memory_saved = False
                hcm32_memory_hit = True
            else:
                result = fit_program(fit_pairs, candidate_cap=candidate_cap, max_depth=max_depth,
                                     memory_path=str(memory_path) if memory_path else None)
                program = result.program
                fit_exact = result.exact_fit
                fit_program_count = result.fit_program_count
                candidates_evaluated = result.candidates_evaluated
                single_stage_memory_hit = result.memory_hit
                single_stage_memory_saved = result.memory_saved
                hcm32_memory_hit = False
                if hcm32 and result.exact_fit and result.program is not None:
                    # Defer writes until every fold has been evaluated. A rule
                    # learned in one fold must not leak into another fold.
                    pending_hcm32.append((result.program, fit_pairs))
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
                "fit_program_count": fit_program_count,
                "candidates_evaluated": candidates_evaluated,
                "single_stage_memory_hit": single_stage_memory_hit,
                "single_stage_memory_saved": single_stage_memory_saved,
                "hcm32_memory_hit": hcm32_memory_hit,
                "held_out_exact": exact,
                "prediction": [list(row) for row in prediction] if prediction is not None else None,
                "expected": [list(row) for row in held_output],
                "error": error,
            })
    # Persist newly learned procedures only after all predictions are fixed.
    # This prevents cross-validation folds from training on one another's holdouts.
    hcm32_saved = 0
    if hcm32:
        seen = set()
        for program, examples in pending_hcm32:
            procedure_id = hcm32.store_program(program, examples)
            if procedure_id is not None and procedure_id not in seen:
                seen.add(procedure_id)
                hcm32_saved += 1
    elapsed = time.perf_counter() - started
    exact = sum(row["held_out_exact"] for row in folds)
    fit = sum(row["fit_exact"] for row in folds)
    return {
        "engine": "primitive-first-exhaustive+hcm32" if hcm32 else "primitive-first-exhaustive",
        "search_methods": (["verified HCM-32 exact-fit retrieval", "deterministic primitive enumeration",
                            "one- and two-operation composition"] if hcm32 else
                           ["deterministic primitive enumeration", "one- and two-operation composition"]),
        "mcts_used": False,
        "beam_search_used": False,
        "dataset": dataset.name,
        "tasks_considered": len(tasks),
        "eligible_tasks": len({row["task_id"] for row in folds}),
        "skipped_tasks": skipped,
        "folds": len(folds),
        "fit_folds_with_exact_program": fit,
        "held_out_exact": exact,
        "held_out_accuracy": exact / len(folds) if folds else 0.0,
        "candidate_cap_per_fold": candidate_cap,
        "single_stage_memory_path": str(memory_path) if memory_path else None,
        "single_stage_memory_hits": sum(row["single_stage_memory_hit"] for row in folds),
        "single_stage_memory_saves": sum(row["single_stage_memory_saved"] for row in folds),
        "hcm32_memory_path": str(hcm32_memory_path) if hcm32_memory_path else None,
        "hcm32_memory_hits": sum(row["hcm32_memory_hit"] for row in folds),
        "hcm32_memory_saves": hcm32_saved,
        "max_depth": max_depth,
        "runtime_seconds": round(elapsed, 6),
        "scope": "leave-one-training-pair-out cross-validation; official test arrays are never read",
        "fold_results": folds,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run the primitive-only ARC training benchmark")
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--candidate-cap", type=int, default=5000)
    parser.add_argument("--max-depth", type=int, choices=(0, 1, 2), default=2)
    parser.add_argument("--memory", type=Path, default=None,
                        help="optional persistent single-stage exact-fit memory JSON file")
    parser.add_argument("--hcm32-memory", type=Path, default=None,
                        help="optional HCM-32 procedural memory JSON file (separate from --memory)")
    parser.add_argument("--output", type=Path, default=Path("arc_r2_primitive_benchmark.json"))
    args = parser.parse_args(argv)
    if not args.dataset.exists() or not (args.dataset.is_file() or args.dataset.is_dir()):
        parser.error("dataset file or directory does not exist")
    if not 1 <= args.limit <= 500:
        parser.error("limit must be in 1..500")
    if not 1 <= args.candidate_cap <= 50000:
        parser.error("candidate-cap must be in 1..50000")
    report = run_benchmark(args.dataset, args.limit, args.candidate_cap, args.max_depth,
                           args.memory, args.hcm32_memory)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ARC-R2 PRIMITIVE BENCHMARK | MCTS: OFF | BEAM: OFF | TEST ANSWERS: UNUSED")
    print(f"Tasks: {report['eligible_tasks']}/{report['tasks_considered']} eligible")
    print(f"Exact held-out training pairs: {report['held_out_exact']}/{report['folds']} "
          f"({report['held_out_accuracy']:.3%})")
    print(f"Folds with an exact fit program: {report['fit_folds_with_exact_program']}/{report['folds']}")
    print(f"Single-stage memory: {report['single_stage_memory_hits']} hits, "
          f"{report['single_stage_memory_saves']} saves")
    print(f"HCM-32 memory: {report['hcm32_memory_hits']} hits, "
          f"{report['hcm32_memory_saves']} saved procedures")
    print(f"Runtime: {report['runtime_seconds']:.3f}s")
    print(f"[SAVED] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

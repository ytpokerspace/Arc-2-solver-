# ARC-R2 reliability and learning systems

This note documents the engineering layer added in `arc_r2/research_systems.py`.
It is experimental infrastructure, not a claim that ARC-AGI-2 is solved.

## Components

| System | Entry point | Contract |
|---|---|---|
| Task/data isolation | `TaskRecord`, `TaskDataManager` | Loads training input/output pairs and test inputs only. Rejects test entries containing an output. Duplicate task IDs are rejected. |
| Typed, bounded execution | `SafeDSLExecutor` | Executes only approved DSL operations; enforces program depth and grid-side limits; fails closed. |
| Candidate provenance | `CandidateLedger` | Stable SHA-256-derived candidate IDs, parent/hybrid metadata, source, search scores, exact-fit counts and rejection reasons. |
| Search budgets | `SearchBudget`, `SearchLimits.max_seconds` | Caps expansions, executions, program depth, wall-clock time, hybrids and estimated memory. Beam and MCTS share the solver's remaining time budget. |
| Independent verification | `IndependentVerifier` | Replays each instruction through the safe executor and requires exact integer-grid equality on every training pair. The main solver now returns verification status and counterexample indices. |
| Counterexample-guided refinement | `cegis_refine` | Rejects partial-fit candidates, reports failing training-pair indices, and optionally asks a bounded callback for refinements. It never receives hidden test outputs. |
| Spatial scene graph | `build_scene_graph` | Adds object color, pixel count, bounding box, dimensions, normalized shape, horizontal/vertical symmetry, directed relative-position, containment and adjacency edges. |
| Uncertainty/calibration | `disagreement_report`, `calibration_metrics` | Reports candidate disagreement and computes Brier score/ECE from labeled evaluation outcomes. These metrics do not prove correctness. |
| Abstraction library | `AbstractionLibrary` | Accepts a routine only after a minimum number of unique validation task IDs meet the configured exact-fit rate. Tracks reuse. |
| Telemetry | `JSONLTelemetry` | Emits deterministic JSONL events suitable for benchmark artifacts. |

## Main solver integration

`HybridSolver.solve()` now:
1. applies one wall-clock budget across Beam Search and MCTS;
2. records generated candidates and their origin;
3. independently checks candidates against every visible training pair;
4. records exact-fit counts and rejection reasons;
5. marks the selected program with `verified_exact_fit`;
6. returns verification details, candidate provenance, and search-budget telemetry.

A high score is not a solve. If the selected candidate fails exact verification, the result explicitly reports `verified_exact_fit: false`.

HIAN is limited to its top ten reported hypotheses in the integrated solver. Its smoothed training-fit statistic remains descriptive, not a calibrated probability. Calibration must be measured on held-out tasks that were not used to tune the solver.

## CEGIS and hybridization

The CEGIS runner is exposed as a reusable orchestration function. It cannot invent a domain-specific refinement by itself: a caller must provide a `refine(program, counterexample_indices)` callback. Each proposed hybrid can be checked with `compatible_hybrid`, which rejects over-depth compositions and requires exact training verification before returning a program.

This separation is intentional. A failure report can identify which training examples contradict a program, but the system must not pretend it knows the semantic cause unless a diagnostic or refiner actually establishes it.

## Resource accounting

Beam width, candidate caps, MCTS iterations, maximum depth and wall-clock time are enforced by the search layer. `SearchBudget.consume_memory` tracks an estimated allocation supplied by the caller; it is **not** an operating-system RSS measurement or a hard process-memory sandbox. The current solver does not yet pass live process-memory measurements into that estimate.

## Current limitations / next engineering steps

- CEGIS refinement requires a task-aware callback; automatic repair of object/relationship assumptions is not yet implemented.
- Abstraction storage is gated by unique validation task IDs, but cross-task extraction and automatic routine substitution are future work.
- Candidate disagreement and calibration utilities exist; confidence calibration still requires a separate held-out evaluation dataset.
- The current scene graph is a deterministic geometric description, not a learned relational reasoner.
- The reliability layer does not yet integrate CSC-LCH context hydration. That should be a later, separately ablated component.
- Benchmark telemetry utilities are available; the public training pilot must be rerun and its artifact inspected before claiming any accuracy or runtime improvement.

## Validation

Run:

```bash
python -m compileall -q arc_r2 tests
python -m unittest discover -s tests -v
python -m arc_r2.cli --demo
python -m arc_r2.cli --demo --engine object --candidate-cap 300
```

All claims about solved tasks must identify dataset, split, number of tasks/folds, and exact-match criteria. Training-pair fit, leave-one-training-pair-out validation, and official hidden-test accuracy are distinct measurements.

# ARC-2 Solver (ARC-R2)

A dependency-free, Python-first research solver for ARC-AGI-2-style grid tasks. The initial implementation is deliberately conservative: it provides explicit grid operators, a small typed transformation DSL, bounded Beam Search, a UCT-guided Monte Carlo Tree Search (MCTS), cross-example evaluation, and reproducible local tests.

**Research status:** prototype / baseline implementation. No ARC-AGI-2 training-set score is claimed. Synthetic tests are not official benchmark results.

## Requirements
- Python 3.10+
- Standard library only
- No network access, credentials, API keys, or paid services required

## Run
```bash
python -m unittest discover -s tests -v
python -m arc_r2.cli --demo
```

For a local ARC JSON file (list of tasks with `train` and `test` arrays):
```bash
python -m arc_r2.cli --dataset path/to/tasks.json --limit 20
```
The runner reads only the supplied local file. It does not download datasets.

## Architecture
- `grid.py`: immutable grid validation and helpers.
- `transforms.py`: distinct spatial, color/polarity, and inverse operators.
- `dsl.py`: typed operation/program representation.
- `search.py`: bounded Beam Search and UCT-guided MCTS.
- `solver.py`: shared candidate evaluation and hybrid coordination.
- `cli.py`: visible progress, local dataset runner, and safety limits.

## Safety and reproducibility
Search depth, candidate count, MCTS iterations, and task count are bounded. The program performs no network operations and has no API integration. It never modifies input datasets. JSONL logs are written only to the selected output directory (default: current directory's `arc_r2_runs/`). Use Ctrl-C to stop. An interrupted run records its state where possible.

## Test gates
1. Code integrity (syntax/imports)
2. Unit tests
3. Integration tests
4. Synthetic training tasks
5. User-supplied ARC training tasks
6. Formal benchmarks after prior gates pass
7. Release verification

A task's training examples are used to select a program; test examples are predictions only and are never used for fitting. Reported accuracy must identify the dataset and split.

## Limitations
The initial DSL is intentionally small and does not solve arbitrary ARC tasks. Object-centric parsing, richer relational operators, official dataset ingestion validation, and evidence-based operator ablations remain future work. MCTS currently explores bounded operator sequences and shares the same evaluator as Beam Search; it is not a claim of optimal search.

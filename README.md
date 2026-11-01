# ARC-2 Solver (ARC-R2)

A Python-first research solver for ARC-AGI-2-style grid tasks. The initial implementation is deliberately conservative: it provides explicit grid operators, a small typed transformation DSL, bounded Beam Search, a UCT-guided Monte Carlo Tree Search (MCTS), cross-example evaluation, and reproducible local tests.

**Research status:** prototype / experimental solver. No ARC-AGI-2 leaderboard score is claimed. Synthetic tests are not official benchmark results.

## Requirements
- Python 3.10+
- NumPy and SciPy (`python -m pip install -r requirements.txt`)
- No network access, credentials, API keys, or paid services required

## Run
Use the single-file entry point:
```bash
python arc_solver.py --demo
python arc_solver.py --dataset path/to/tasks.json --limit 20

# Try the parameterized object-centric search engine
python arc_solver.py --engine object --dataset path/to/tasks.json --limit 20
```

The package CLI is also available:
```bash
python -m unittest discover -s tests -v
python -m arc_r2.cli --demo

# Exact-fit filtering, D4/color-permutation TTA, and up to two distinct attempts
python -m arc_r2.cli --engine robust --dataset path/to/tasks.json --limit 20 --max-depth 2
```

For a local ARC JSON file (either a task list or the standard task-ID-to-task mapping, each task containing `train` and `test` arrays):
```bash
python -m arc_r2.cli --dataset path/to/tasks.json --limit 20
python -m arc_r2.benchmark --dataset path/to/tasks.json --limit 20

# Compare baseline, HIAN, object, and robust-TTA engines using leave-one-training-pair-out validation
python -m arc_r2.compare_engines --dataset path/to/tasks.json --limit 20 --max-depth 2
```
The runner reads only the supplied local file. It does not download datasets. Predictions for each supplied test input are written to the JSONL run log; hidden test answers are not needed or inferred.

## Architecture
- `arc_solver.py`: direct Python-script entry point.
- `grid.py`: immutable grid validation and helpers.
- `objects.py` and `relations.py`: four-connected object extraction and spatial relation graph.
- `object_ops.py`: deterministic geometry transforms and shape/size descriptors.
- `object_dsl.py`: parameterized object instructions, bounded candidate generation, and collision/bounds checks.
- `object_solver.py`: bounded beam search over object-centric programs.
- `prte.py`: experimental dominant-background polarity reversal.
- `transforms.py`: distinct spatial, color/polarity, and inverse operators.
- `dsl.py`: typed operation/program representation.
- `search.py`: bounded Beam Search and UCT-guided MCTS.
- `solver.py`: shared candidate evaluation and selectable identity/Beam/MCTS/hybrid strategies, with optional HIAN reranking.
- `hian.py`: evidence-backed hypothesis ranking, source-level support/contradiction tracking, and a hierarchy grouped by first transformation.
- `emoji_codec.py`: reversible ARC color-to-emoji tokens and lossless emoji object graphs.
- `emoji_alchemy.py`: symbolic DSL execution through emoji tokens, semantic signatures, and bounded candidate synthesis.
- `robust_solver.py`: exact-fit candidate filtering, int8/30x30 NumPy batching, D4 plus color-permutation test-time augmentation, inverse voting, and structurally distinct two-attempt selection.
- `benchmark.py`: local baseline comparison and PRTE/object-signal ablations.
- `compare_engines.py`: paired leave-one-training-pair-out comparison of baseline, HIAN, object, and robust-TTA engines; reports top-one and within-two accuracy.
- `cli.py`: visible progress, local dataset runner, and safety limits.

## HIAN experiment
HIAN is an optional meta-reasoning layer. In the Emoji Alchemy experiment, HIAN supplements the existing search candidates with all compositions of the current DSL operators up to depth two. Programs are evaluated through a reversible emoji-token representation and compared using color distributions, object shapes, object colors, spatial relations, coarse object locations, and dimensions. It records exact-match support and contradictions, groups hypotheses by first transformation, and ranks exact training matches first, then semantic similarity, cell agreement, a smoothed training-fit estimate, and simplicity. Semantic similarity is a ranking signal, never proof of correctness; only exact integer-grid reconstruction counts as a solved example. The confidence estimate is descriptive training evidence, not a calibrated probability of unseen-task success. Baseline behavior remains unchanged when `use_hian=False` (the default). The comparison runner reports baseline, HIAN, and object engines on identical leave-one-training-pair-out folds.

## Exact-fit TTA experiment
The optional `robust` engine checks both the bounded grid DSL and bounded object DSL, retaining a candidate only when its integer-grid output exactly matches **every** supplied training output. Survivors are deduplicated by their behavior on the full training set and ranked by shortest program length. It then solves rotated, reflected, and deterministically color-permuted versions of the task, inverts each augmentation on its prediction, and votes over reconstructed outputs. It emits at most two attempts: attempt 1 is the highest-voted output; attempt 2 must be a different output supported by a different transformation-family signature. If no structurally distinct alternative survives, only one attempt is returned. This is a bounded experiment, not a guarantee that an exact-fit program generalizes.

Grid candidates are represented as `int8` tensors padded to 30x30 with `-1` as a sentinel. Geometry and color operations are batched with NumPy; repeated operation results are memoized by grid shape, bytes, and operation. Object extraction uses `scipy.ndimage.label` with four-connectivity when SciPy is installed (the portable Python fallback remains available).

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
The baseline grid DSL is intentionally small. The experimental object DSL currently searches a bounded set of one- and two-step scale, rotate, reflect, move, and place programs. Object indices are based on deterministic connected-component ordering at each instruction step; this can be ambiguous when objects change order. Parameter generation is deliberately capped and does not cover every possible ARC transformation. Official dataset ingestion validation and evidence-based operator conclusions remain future work. The benchmark harnesses exist, but no ARC-AGI-2 dataset results are claimed. The `ARC-R2 public training comparison` GitHub Actions workflow runs a small, pinned public-training pilot and saves its JSON report as an artifact. It is a smoke-scale research measurement, not an official leaderboard result. `compare_engines.py` uses only the supplied training pairs: it holds one pair out, fits on the others, and evaluates the held-out pair. Tasks with fewer than two training pairs are skipped. This cross-validation diagnostic is not official test accuracy. MCTS currently explores bounded operator sequences and shares the same evaluator as Beam Search; it is not a claim of optimal search.

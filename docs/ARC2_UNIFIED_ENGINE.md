# ARC-AGI-2 Unified Solver

## Scope

This engine is exclusively for the static ARC-AGI-2 input/output task format. It does not implement interactive ARC environments, action selection, game-state control, or ARC-AGI-3 mechanics.

A task contains:

- `train`: known input/output grid pairs used to infer a transformation.
- `test`: unseen input grids for which the solver must produce predictions. Test output labels, when present in a local file, are not read by the prediction path.

## Unified search

`arc_r2.unified_engine` combines two existing ARC-2 search paths that share the same immutable grid and program representation:

1. **Deterministic primitive enumeration** — searches zero-, one-, and bounded two-operation programs using the explicit primitive vocabulary.
2. **Predictive pair-matching MCTS** — explores bounded compositions and uses intermediate training-pair agreement to guide search.

Both are candidate generators, not authorities. A candidate is eligible only if it exactly reproduces every training pair supplied to the fit. The coordinator deduplicates candidates and selects the shortest exact-fit program found, using canonical operation order for deterministic tie-breaking. This is the best candidate found under configured limits, not a proof of global optimality or hidden-test correctness.

If neither search path finds an exact fit, the solver abstains.

## Commands

Run the unit suite:

```bash
python -m compileall -q arc_r2 tests
python -m unittest discover -s tests -v
```

Predict hidden test inputs using all visible training pairs:

```bash
python -m arc_r2.unified_engine \
  --mode predict \
  --dataset path/to/ARC-AGI-2/data/evaluation \
  --limit 20 \
  --output arc2_predictions.json
```

Run within-task validation on public training tasks:

```bash
python -m arc_r2.unified_engine \
  --mode validate \
  --dataset path/to/ARC-AGI-2/data/training \
  --limit 20 \
  --output arc2_unified_validation.json
```

Tune search limits explicitly with `--primitive-candidate-cap`, `--primitive-max-depth`, `--mcts-iterations`, `--mcts-branch-width`, and `--mcts-max-depth`. Persistent memory is optional and is split by search engine under `--memory-dir`.

## Correct evaluation protocol

### Prediction

Fit using **all** visible training pairs. Apply the selected program to each test input. Do not inspect test outputs, use them for search, or rank candidates against them.

### Validation

Use leave-one-training-pair-out cross-validation:

1. Hide one known training pair.
2. Fit on all remaining training pairs.
3. Apply the selected program to the held-out input.
4. Compare the prediction with the held-out known output.
5. Repeat for every training pair.

This measures within-task consistency. It is not official hidden-test accuracy and does not establish generalization to unseen tasks.

## Emoji engine

`arc_r2.emoji_engine` maps human-readable symbols to stable canonical stage IDs. Emoji are trace labels only; canonical IDs and typed Python functions define behavior.

Canonical path:

`📥 INPUT → 👁️ PERCEPTION → 🕸️ SCENE_GRAPH → ⚙️ DSL → 🧩 PRIMITIVE_ENUMERATION + 🌳 PREDICTIVE_MCTS → ✅ EXACT_FIT → ⚖️ SELECT → 📤 TEST_PREDICTION`

## Evidence and limitations

- Exact training fit is necessary, not sufficient, for a correct test prediction.
- The selected program is optimal only among candidates actually generated within the configured search budgets, under the stated simplicity ordering.
- Search heuristic scores are not calibrated probabilities of hidden-test success.
- A unit test verifies software behavior, not ARC benchmark performance.
- Keep public training-pair validation, actual test predictions, and official evaluation results as separate evidence categories.

# ARC-2 Solver (ARC-R2)

A research solver scoped to **ARC-AGI-2 static grid-transformation tasks**. The unified path combines deterministic primitive enumeration with bounded predictive MCTS over the shared grid/program representation. It is not an interactive game agent and does not implement ARC-AGI-3 action loops.

## What the task data means

- **Training pairs** (`train`): known input grids and their correct output grids. The solver infers a transformation from these pairs.
- **Test questions** (`test`): unseen input grids. The solver applies the inferred transformation to each test input and emits a predicted output grid.
- **Hidden test outputs**: never read by the prediction path and never used for candidate generation or selection.

A program is accepted only when it exactly reproduces every training pair supplied to the fit. If no exact-fit program is found within the configured search budgets, the solver abstains instead of returning a partial-fit guess.

## Unified engine

- `arc_r2/unified_engine.py`: coordinates the deterministic primitive and predictive-MCTS search paths; deduplicates exact-fit candidates; selects the simplest candidate found; predicts test inputs; and runs leave-one-training-pair-out validation.
- `arc_r2/primitives.py`: grid operations, object extraction, deterministic program execution, and bounded primitive enumeration.
- `arc_r2/predictive_mcts.py`: bounded predictive pair-matching MCTS.
- `arc_r2/emoji_engine.py`: maps human-readable emoji labels to stable canonical ARC-2 stage IDs. Emojis are trace labels, not executable commands.
- `docs/ARC2_UNIFIED_ENGINE.md`: architecture, commands, and evaluation boundaries.

The selected program is the simplest exact-fit candidate found by the configured searches. This is not a proof of global optimality or hidden-test correctness. Search scores are not calibrated probabilities.

## Run tests

Python 3.10+; standard library only.

```bash
python -m compileall -q arc_r2 tests
python -m unittest discover -s tests -v
```

## Predict test inputs

Fit on all visible training pairs and emit predictions for test inputs:

```bash
python -m arc_r2.unified_engine \
  --mode predict \
  --dataset path/to/ARC-AGI-2/data/evaluation \
  --limit 20 \
  --output arc2_predictions.json
```

The prediction runner reads each test case's `input` field only. It does not compare against or serialize a test answer.

## Validate using training pairs

Run leave-one-training-pair-out validation on public training tasks:

```bash
python -m arc_r2.unified_engine \
  --mode validate \
  --dataset path/to/ARC-AGI-2/data/training \
  --limit 20 \
  --output arc2_unified_validation.json
```

For each fold, one training pair is hidden from synthesis, the remaining pairs are used to infer a program, and the program is applied to the held-out input. The known held-out output is used only to score that validation fold. This measures within-task rule consistency; it is not official hidden-test accuracy or independent-task generalization.

Search budgets can be set with `--primitive-candidate-cap`, `--primitive-max-depth`, `--mcts-iterations`, `--mcts-branch-width`, and `--mcts-max-depth`. Optional persistent memory is split by engine using `--memory-dir`.

## Evidence boundaries

- Unit tests establish software behavior, not benchmark accuracy.
- Exact fit on training pairs is necessary, not sufficient, for a correct test prediction.
- Catalog size, executable operator count, candidate count, and solved-task count are different measurements.
- Public training-pair validation, test-input predictions, and official evaluation results must be reported separately.
- No claim is made that this prototype solves ARC-AGI-2.

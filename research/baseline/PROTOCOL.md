# Baseline measurement protocol

## Current control implementation

The control is the unmodified ARC-R2 solver on the main-branch commit
`faa2d507bd4ec354b64ee7af01d9404c2b94336b`.

The existing `arc_r2.benchmark` command compares identity, Beam Search,
MCTS, hybrid search, and two hybrid ablations. Its original score is
training-example fit only. Do not describe that score as held-out accuracy.

## New validation protocol

Use leave-one-training-pair-out validation. For every eligible task, with
at least two training pairs:

1. Hide one training pair.
2. Fit the existing solver using only the remaining training pairs.
3. Apply the selected program to the hidden pair's input.
4. Compare the prediction with the hidden pair's output.
5. Repeat until each training pair has been held out once.

Tasks with fewer than two training pairs are counted as skipped. The held-out
output is used only for scoring after the program has been selected.

Run:

```bash
python -m unittest discover -s tests -v
python -m arc_r2.validation --dataset path/to/arc_training.json --limit 20
```

The validation command writes a JSON report containing the dataset SHA-256,
Python/platform details, commit identifier when supplied by CI, fixed search
limits, per-fold predictions' exact-match status, candidate evaluations, and
runtime.

## Interpretation rules

- Held-out pair accuracy measures within-task consistency, not independent-task
  generalization.
- It is not an official ARC-AGI-2 evaluation score.
- Training-fit accuracy and held-out accuracy must be reported separately.
- Candidate evaluations and wall-clock time must be reported together.
- All variants receive the same declared per-fold limits, but this does not
  guarantee identical compute or wall-clock budgets across search strategies.
- Use the same dataset bytes, task ordering, limits, and seed when comparing
  variants.
- Preserve failed runs and record their causes; do not overwrite results.
- No paid API calls or network access are required by the evaluator itself.

## Current evidence before the new evaluator

On 2026-10-31, GitHub Actions run
[41174383949](https://github.com/ytpokerspace/Arc-2-solver-/actions/runs/41174383949)
passed compilation, 11 unit/integration tests, and the synthetic CLI smoke
test. That smoke test fit 1 of 2 synthetic training examples, evaluated 32
candidates, and completed in approximately 0.0043 seconds.

This is a software smoke test only. No official ARC dataset is currently
present in the repository, and no ARC-AGI-2 baseline score has been measured.

## Next evidence gate

Run the new evaluator against a version-pinned public ARC-AGI-2 training
dataset. First use a small task subset to validate runtime and output format;
then run a larger fixed subset. Keep the dataset hash and task IDs with every
result. Do not use hidden evaluation answers for fitting, tuning, or selection.

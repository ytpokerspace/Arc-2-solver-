# ARC-R2 validation protocol

The validation command performs leave-one-training-pair-out checks. For each
eligible task, it hides one training pair, fits on the remaining pairs, and
scores the hidden pair only after choosing a program.

Run:

```bash
python -m unittest discover -s tests -v
python -m arc_r2.validation --dataset path/to/training --limit 20
```

The dataset can be either a JSON file or a directory of task JSON files.
Directory files are sorted by filename. The report records a SHA-256 digest of
the selected data, fixed solver limits, per-fold exact-match results,
candidate counts, runtime, and the current Git commit when supplied by CI.

Interpretation rules:

- This measures within-task rule consistency, not independent-task
  generalization.
- It is not an official ARC-AGI-2 leaderboard score.
- Training fit and held-out pair accuracy are reported separately.
- Compare runs only when dataset bytes, task ordering, solver limits, and seed
  are held constant.
- A search score is not a calibrated probability of unseen-task success.
- The evaluator uses only the public training split and does not read official
  evaluation answers.

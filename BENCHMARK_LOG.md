# Benchmark log

No ARC-AGI-2 dataset benchmark has been run. Synthetic unit/integration tests are not benchmark scores.

The local comparison harness is available as:

```bash
python -m arc_r2.benchmark --dataset path/to/tasks.json --limit 20
```

It compares identity, Beam-only, MCTS-only, hybrid, hybrid without PRTE, and hybrid without the object/relation scoring signal. Its current metric is fit to the provided training examples, plus runtime and candidate evaluations. This is a search/ablation diagnostic, not held-out accuracy or an official ARC-AGI-2 score.

For every real run, record dataset identity, split, task count, configuration, runtime, exact-match results, candidate counts, and limitations. Do not claim generalization from training-example fit.

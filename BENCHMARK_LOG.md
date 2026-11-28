# Benchmark log

No ARC-AGI-2 dataset benchmark has been run. Synthetic unit/integration tests are not benchmark scores.

The local comparison harness is available as:

```bash
python -m arc_r2.benchmark --dataset path/to/tasks.json --limit 20
```

It compares identity, Beam-only, MCTS-only, hybrid, hybrid without PRTE, and hybrid without the object/relation scoring signal. Its current metric is fit to the provided training examples, plus runtime and candidate evaluations. This is a search/ablation diagnostic, not held-out accuracy or an official ARC-AGI-2 score.

For every real run, record dataset identity, split, task count, configuration, runtime, exact-match results, candidate counts, and limitations. Do not claim generalization from training-example fit.


## Full orchestration comparison

The opt-in multi-family pipeline is available through arc_r2.full_orchestration. It runs Shape Radar, AHPSE, the existing primitive fitter, and the object-level solver under a three-stage HMSE procedure. Candidate families must exactly fit the supplied fit split; conflicting predictions cause abstention.

Run the controlled comparison with:

    python -m arc_r2.full_orchestration_benchmark --dataset path/to/training --limit 20

The benchmark compares full orchestration with the existing primitive fitter on identical leave-one-visible-training-pair-out folds. Persistent memory is disabled. The held-out output is read only by the outer evaluator after prediction. No official hidden evaluation labels are read.

Do not record an accuracy claim until the benchmark report is produced and reviewed. The per-family candidate caps mean the ensemble may consume more total computation than the single-family baseline; report runtime and candidate counts alongside accuracy.

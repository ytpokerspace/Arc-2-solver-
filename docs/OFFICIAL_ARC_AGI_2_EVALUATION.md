# Official ARC-AGI-2 evaluation-input run

The evaluation runner is deliberately separate from the leave-one-training-pair-out benchmark.

## Dataset and run

- Repository: `arcprize/ARC-AGI-2`
- Pinned revision: `f3283f727488ad98fe575ea6a5ac981e4a188e49`
- Split: `data/evaluation` (120 task JSON files at the pinned revision)
- Inputs: every task's `train` examples and every `test[].input`
- Labels: the runner never reads `test[].output`
- Search: deterministic primitive fit, candidate cap 1,500 per task, maximum depth 2
- Memory: disabled; each task is fit independently without persistent solver memory

Run locally or in a checked-out environment:

```bash
python -m arc_r2.evaluation_benchmark \
  --dataset arc-agi-2-data/data/evaluation \
  --candidate-cap 1500 \
  --max-depth 2 \
  --output arc_r2_evaluation_predictions.json \
  --report arc_r2_evaluation_report.json
```

The GitHub Actions workflow exposes this as an opt-in `workflow_dispatch` job named `official-evaluation`. Enable `run_official_evaluation` to run all evaluation tasks. The ordinary PR test workflow does not launch the full evaluation job.

## Interpretation

The report records task/test-input counts, exact-fit coverage on the supplied training pairs, candidate counts, runtime, prediction coverage, and abstentions. `test_accuracy` is always null because expected test outputs are not read or available in this public split.

The predictions JSON is a research artifact mapping task IDs to one predicted grid per test input; an abstention is represented by null. It is not represented as an independently scored result.

## Semantic identity status

The current `semantic_identity.py` matcher associates objects between a reference state and a target grid. The primitive program fitter does not currently invoke it while discovering or applying programs. Therefore this first official evaluation run is explicitly the primitive baseline; it does not claim semantic-identity-enabled ARC performance. A valid semantic ablation requires integrating identity into rule discovery/prediction and rerunning the exact same evaluation inputs and solver limits.

# ARC-2 Solver (ARC-R2)

A research prototype containing a deterministic primitive baseline and separate experimental QBS-guided object/grid search. The primitive benchmark remains a deterministic baseline; the integration branch adds predictive MCTS and leave-one-training-pair-out validation as separately reported experiments.

## Current status

This is a research prototype, not a solved ARC-AGI-2 system. Unit tests validate implementation behavior, not benchmark generalization. Public training results are reported as leave-one-training-pair-out validation, not official hidden-test accuracy. Probability estimates from visible training pairs are descriptive fit statistics, not calibrated probabilities of unseen-task success.

## Adaptive cognitive calibration preflight

The integration branch adds an Adaptive Cognitive Calibration Layer for QBS:

- Ten long-form question prompts across scientific and technical domains live in research/calibration/questions.json.
- The generator receives prompt text only; a separate evaluator must hold the reference answers privately.
- Long-form answer quality is scored on factual correctness (35%), completeness (25%), reasoning (20%), relevance (10%), and consistency (10%).
- The score starts at 150, changes by +5/-5, is bounded to 0-300, and stops after three consecutive correct answers or ten questions.
- Confidence calibration is adjusted separately in five-point steps; it is not treated as IQ or proof of increased intelligence.
- The reference-answer key is intentionally not committed to this public repository.

The API in arc_r2/cognitive_calibration.py provides run_calibrated_session, which runs the preflight before calling the QBS brainstorming callback. A production deployment must keep the answer evaluator and its key isolated from the generator using separate processes or services. The callbacks alone are not a security boundary. The existing offline CLI does not invoke a model or evaluator, so this API must be supplied with those callbacks by the application that hosts QBS.

See research/ADAPTIVE_COGNITIVE_CALIBRATION.md for the protocol and limitations.

## Automatic reasoning-memory capture

The local CLI now wraps each baseline or object-engine solve with the structured reasoning recorder. It writes a persistent `reasoning_memory.json` beside the run log under the selected `--output` directory. Records are keyed by engine, task label, and a fingerprint of the training pairs, so repeating the same task updates its record rather than duplicating it.

Each run records the training fit and explicitly labels its evidence scope. A perfect fit on visible training pairs is marked `tested`, not `validated`. The runner also counts matching validated patterns for reporting, but does not yet inject those patterns into candidate generation or alter predictions. This keeps the solver's behavior unchanged while the memory pipeline is tested.

The memory file is runtime output and should be kept with the experiment artifacts, not committed as source code.

## Run tests

Python 3.10+; standard library only.

```bash
python -m compileall -q arc_r2 tests
python -m unittest discover -s tests -v
```

## Evidence boundaries

- Exact output equality remains the acceptance gate for ARC predictions.
- A held-out training pair must not be visible during candidate generation for its fold.
- Catalog size, executable operator count, candidate count, and solved-task count are distinct metrics.
- No official evaluation answers are read by the public-data workflows.
- Experimental branches with negative held-out results remain research evidence; they are not promoted to the default solver without new validation.

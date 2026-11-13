# Adaptive Cognitive Calibration Layer

## Purpose
Run a bounded, long-form reasoning pre-test before QBS accepts a new user
question. It measures answer quality and confidence calibration; it is not an
IQ test and does not independently increase model intelligence.

## Protocol
1. Select up to ten unseen, long-form questions from the public question bank
   in research/calibration/questions.json.
2. Keep the reference-answer key outside the generator's accessible files and
   credentials. The generator receives question text only.
3. A separate evaluator compares the generated answer with its private
   reference answer. Score factual correctness (35%), completeness (25%),
   reasoning (20%), relevance (10%), and internal consistency (10%).
4. Award +5 for a correct answer and -5 for an incorrect answer, bounded by
   0 and 300. Start at 150.
5. Adjust confidence calibration separately in five-percentage-point steps
   when confidence differs from rubric quality by more than ten points.
6. Stop after three consecutive correct answers or after ten questions.
7. Preserve per-question evidence, rubric scores, confidence, score changes,
   and stop reason. Do not turn a short smoke test into a broad calibration claim.

## Integration contract
Use run_calibrated_session as the QBS entry point. It completes the pre-test
before invoking the brainstorming callback and passes the calibration report
to that callback. The generator callback receives prompt text only. The
evaluator callback owns access to reference answers. A production deployment
must enforce this boundary with separate processes or services and access
controls. Separate callbacks in one Python process are an interface boundary,
not a security boundary. Never claim the test is genuinely blind if the
generator can read the answer key.

## QBS use
The report may inform confidence weights and hypothesis-selection strategy.
It must not silently change the ARC solver's exact-match acceptance gate.
Training-pair fit estimates remain distinct from unseen-task success
probabilities. Calibration changes remain proposals until fresh held-out
evaluation demonstrates improvement.

## Validation limits
Ten questions are an initial smoke test, not enough to establish robust
calibration. Use fresh held-out question sets and report sample size, domain
coverage, rubric distributions, and calibration error before claiming
measured improvement. The answer key is deliberately not committed to this
public repository.

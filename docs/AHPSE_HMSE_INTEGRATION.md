# AHPSE through HMSE — explicit opt-in

## Purpose

`arc_r2.ahpse_hmse` connects the bounded AHPSE proposer to the existing HMSE execution and evidence-recording framework. It does not change the default solver path.

## Usage

```python
from arc_r2.ahpse import TrainingExample
from arc_r2.ahpse_hmse import run_ahpse_with_hmse

examples = (
    TrainingExample.from_values(input_a, output_a),
    TrainingExample.from_values(input_b, output_b),
)
result, execution = run_ahpse_with_hmse(
    examples, max_depth=2, candidate_cap=500
)

if not result.abstained:
    prediction = result.best.apply(test_input)
else:
    prediction = None
```

The test input is applied only after synthesis. Its output is never supplied to the synthesis or HMSE validation functions.

## Evidence semantics

HMSE records the method ID, partition, evaluation count, and AHPSE evidence: generated/checked candidates, constraint checks/rejections, exact-fit count, abstention flag, and reason.

The constraint stage first checks transformed grid dimensions and whether the same cell-wise color map explains every visible training pair. These are conservative prefilters, not general ARC semantic constraints.

The HMSE validator's score of 1.0 means only that at least one candidate exactly fits the supplied training pairs. It is **not** a calibrated probability or a claim of held-out generalization.

## Safety and promotion gates

- The adapter is opt-in; the default primitive-first solver and benchmark are unchanged.
- Only visible training examples may be passed to synthesis.
- Held-out training-pair scoring must happen in an outer evaluator that withholds that pair's output.
- A failed synthesis is an explicit abstention; HMSE does not select it.
- Promote to broader orchestration only after controlled leave-one-training-pair-out and independent-task comparisons against the same baseline and budget.

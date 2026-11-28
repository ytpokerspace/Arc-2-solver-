# AHPSE — Adaptive Hypothesis and Program Synthesis Engine

## First milestone

AHPSE is an opt-in, deterministic, standard-library-only hypothesis proposer for ARC-R2. It is intentionally separate from the default primitive-first benchmark until it earns integration through controlled evaluation.

It currently proposes bounded programs from:

- D4 grid geometry: rotations, horizontal/vertical reflection, transpose.
- Foreground bounding-box crop (background convention: color 0).
- Consistent source-to-target color remapping inferred from aligned cells.
- Short compositions of these operations.

A candidate is accepted only when it exactly reproduces every supplied training pair. If no candidate does, AHPSE abstains. It reports all exact-fit candidates rather than silently treating the first fit as uniquely correct. Before full pixel equality checks, a constraint stage rejects candidates whose transformed dimensions do not match each target or whose single global color map cannot explain every visible training pair. The result records constraint checks and rejections.

## Usage

```python
from arc_r2.ahpse import TrainingExample, synthesize

examples = (
    TrainingExample.from_values(input_a, output_a),
    TrainingExample.from_values(input_b, output_b),
)
result = synthesize(examples, max_depth=2, candidate_cap=500)

if not result.abstained:
    candidate = result.best
    prediction = candidate.apply(test_input)
else:
    prediction = None
```

## Validation protocol

For leave-one-training-pair-out evaluation, call `synthesize` using only the non-held-out training pairs. Apply the selected candidate to the held-out input and compare its prediction to the held-out output in the outer evaluator. Never pass the held-out output into synthesis.

A one-pair exact fit is only a hypothesis, not evidence of generalization. Report task-level and fold-level results separately. Do not use hidden ARC test outputs for fitting, candidate selection, or parameter tuning.

## Current limits

- This is an initial operator family, not complete ARC rule coverage.
- Candidate generation is bounded and deterministic; no learned priors or adaptive search allocation yet.
- Current constraints are deliberately narrow (grid dimensions and globally consistent cell-wise color mapping); they do not model object relations or general ARC semantics.
- The color map is inferred from cell-aligned transformed grids and cannot express object-dependent recoloring.
- Cropping assumes color 0 is background.
- No graph, object identity, constraint solver, or HMSE execution adapter is connected yet.
- The default solver and benchmark behavior remain unchanged.

## Next integration gates

1. Add more typed operators only with unit tests and explicit preconditions.
2. Add an HMSE adapter that stores declarative operation IDs and parameters.
3. Expand constraint-based pruning only when each new constraint has a soundness test and measurable benefit.
4. Compare AHPSE against the primitive baseline on identical leave-one-pair-out folds and candidate/runtime budgets.
5. Promote it into a broader pipeline only if independent-task results justify the complexity.

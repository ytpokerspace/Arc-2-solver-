# Full ARC orchestration

## Purpose

The arc_r2.full_orchestration module is the opt-in multi-family orchestration path. It does not replace or silently alter the primitive-first default solver.

## Execution stages

1. **Perception:** Shape Radar scans each visible training input, its visible training output, and each requested prediction input. Pairwise object/relation deltas are passed into synthesis as structural evidence. No hidden target output is provided to perception.
2. **Candidate synthesis:** AHPSE, the existing primitive fitter, and registered rule families receive only visible fit pairs. The object-level solver is a fallback and runs only when those families produce no exact-fit candidate. Under unanimous consensus, adding another candidate after a candidate already exists cannot improve coverage; it can only preserve agreement or create disagreement. Persistent memory is disabled.
3. **Consensus and abstention:** Every exact-fit candidate is applied to each requested input. A prediction is returned only when all surviving candidates agree. If families disagree, or no valid exact-fit candidate exists, the pipeline abstains.

The HMSE trace records all three stages, selected methods, and evaluation counts. Family-specific evidence records candidate counts and measured synthesis time for AHPSE, the primitive fitter, registered rules, and object search. The benchmark aggregates this profile across folds so expensive families that contribute no candidates are visible.

## Usage

    from arc_r2.ahpse import TrainingExample
    from arc_r2.full_orchestration import run_full_orchestration

    examples = (
        TrainingExample.from_values(input_a, output_a),
        TrainingExample.from_values(input_b, output_b),
    )
    result = run_full_orchestration(
        examples,
        test_inputs=(test_input,),
        max_depth=2,
        candidate_cap=500,
    )
    prediction = result.decision.predictions[0]  # None means abstain

## Controlled comparison

Run leave-one-visible-training-pair-out comparison against the repository's primitive baseline:

    python -m arc_r2.full_orchestration_benchmark \
      --dataset path/to/ARC-AGI-2/data/training \
      --limit 20 \
      --candidate-cap 500 \
      --max-depth 2 \
      --output full_orchestration_report.json

For each fold, both methods receive the same remaining input/output pairs and the held-out input. The held-out output is used only by the outer benchmark scorer after predictions are complete. The orchestrator itself never receives it. Memory is disabled in both engines to prevent cross-fold leakage.

The report separates held-out exact accuracy, abstentions, ambiguity, candidate counts, and runtime. Candidate caps are per family per fold; they are not a claim of equal total computation across all families. The comparison is diagnostic, not an official hidden-test score.

## Guardrails and limitations

- Only public/visible training pairs may be used for fitting or tuning.
- Never pass ARC evaluation outputs into this module or its benchmark.
- Exact training fit is necessary for a candidate to enter consensus, but does not establish generalization.
- Disagreement is handled by abstention, not arbitrary tie-breaking.
- max_depth=0 disables object-level search; object search begins at depth 1 and is skipped when another family has already supplied an exact-fit candidate.
- This is a bounded operator-family ensemble, not complete ARC rule induction.
- The default solver remains unchanged until controlled results justify promotion.

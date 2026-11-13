# ARC-2 Coherent Engine Integration

This branch consolidates the strongest reusable components from the existing research pull requests into one solver path.

## Intended pipeline

1. Parse each training task and preserve the complete set of input/output pairs.
2. Generate candidate programs from the existing grid DSL and the object-centric representation.
3. Use the top ten scored hypotheses as the first QBS seed set.
4. Perform one bounded hybrid-expansion round from those seeds.
5. Continue predictive MCTS over candidate programs with fixed limits and duplicate-state pruning.
6. Accept a program only when it reproduces every supplied training output exactly.
7. Store exact task/program matches in single-stage memory; revalidate a memory hit against the current task before reuse.
8. Use leave-one-training-pair-out validation and pinned public training data for evaluation.

## Integrated components

- Object state, object transforms, scene renderer, and grid adapters from the object-centric research branch.
- Original 200-entry primitive catalog as a catalog, not a claim that all 200 entries have executable implementations.
- Typed primitive registry and QBS search scaffolding.
- Predictive MCTS with top-hypothesis scoring and one bounded hybrid round.
- Lossless grid compression codec for exact state fingerprints.
- Reproducible validation runner and public-data workflow from the validation branch.

## Research components deliberately not enabled by default

HIAN reranking, robust test-time augmentation, Emoji Alchemy, and unified QBS were kept out of the default execution path because their recorded public-data pilots did not demonstrate held-out exact-match gains. Their negative results remain useful evidence; they are not erased.

## Verification rules

- Exact output equality is the acceptance gate.
- Held-out training pairs are never visible to candidate generation for their fold.
- Probability estimates are training-fit summaries, not calibrated unseen-task probabilities.
- Catalog count, executable operator count, candidate count, and solved-task count are reported separately.
- No official evaluation answers are read by this workflow.

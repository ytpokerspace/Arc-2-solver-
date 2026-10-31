# Preliminary failure analysis — baseline research branch

Date: 2026-10-31
Branch: `research/baseline-measurement`

## Evidence status

This is a preliminary source-code and small-sample analysis. It is not a task-by-task failure taxonomy.

### Observed 20-task held-out run

The previously completed deterministic 20-task sample reported:

| Configuration | Exact held-out pairs |
|---|---:|
| Identity | 0 / 72 |
| Beam | 3 / 72 |
| MCTS | 3 / 72 |
| Hybrid | 3 / 72 |
| Hybrid without PRTE | 3 / 72 |
| Hybrid without object signal | 3 / 72 |

Only 1 of 20 tasks reportedly had all held-out pairs correct for the search-based configurations. The hybrid run evaluated 2,289 candidates in approximately 1.26 seconds.

Interpretation: this is weak within-task held-out performance on one deterministic 20-task training sample. It is not official ARC-AGI-2 accuracy, independent-task generalization, or evidence that the variants are statistically equivalent.

### 100-task run status

The pinned 100-task GitHub Actions run is still marked in progress at the time of this note. No completed report artifact is available yet. Do not treat the 20-task figures as the 100-task result.

## Source-inspection findings

These are verified from the current implementation; they are not frequencies measured from the task failures.

### 1. DSL expressivity is a major likely bottleneck

The active operation vocabulary in `arc_r2/transforms.py` contains seven operations:

- horizontal reversal
- vertical reversal
- 180-degree rotation
- main-diagonal reflection
- anti-diagonal reflection
- fixed 0/1 polarity negation
- PRTE polarity reversal

The DSL composes these operations into short sequences. It has no parameterized operations for object translation, object selection, copying, deletion, recoloring arbitrary colors, scaling, line/shape construction, counting-based output construction, or conditional rules.

Therefore, many ARC transformations cannot be represented by the current search space. Search tuning cannot discover a program that the DSL cannot express.

### 2. Search explores a narrow fixed vocabulary

Beam search and MCTS both search sequences from the same seven-operation vocabulary. MCTS changes the exploration strategy, not the representational capacity of the DSL.

The 20-task ablation showed identical held-out pair counts (3/72) for beam, MCTS, hybrid, hybrid without PRTE, and hybrid without object signal. This is a useful diagnostic observation, but the sample is too small to conclude that the methods are generally equivalent.

### 3. Object analysis is currently a ranking signal, not an action language

Object extraction uses four-connected same-color components and infers the background as the most frequent color by default. The relation graph uses bounding-box relations. The solver uses object count and relation similarity as small scoring terms, but the DSL does not currently express object-specific actions.

Potential failure modes include:
- diagonal components split into separate objects;
- the most frequent color is not the true background;
- multi-color objects are fragmented by color;
- object identity and correspondence across input/output are not modeled;
- bbox relations can miss cell-level structure.

### 4. Candidate scoring may rank plausible-looking but wrong programs

The current score is primarily exact training-example fit (0.80), with cell agreement (0.12), object-count agreement (0.05), and relation similarity (0.03) when object signals are enabled.

This heuristic can help rank near misses, but it is not a learned or validated probability of correctness. Partial cell agreement may favor a program that looks close while failing the exact transformation rule.

### 5. The current benchmark cannot yet attribute each miss to a cause

The held-out report records the selected program, exact-match result, training fit, and candidate count per fold. It does not yet label a miss as:
- transformation absent from DSL;
- wrong program ranking;
- search budget exhausted;
- object/background perception error;
- ambiguous or underspecified examples;
- execution/implementation defect.

We need a reviewed failure-labeling pass before reporting category frequencies.

## Current bottleneck hypothesis ranking

These are qualitative priorities based on source inspection, not measured prevalence:

1. **DSL coverage gap — highest priority to investigate.** The operation set is demonstrably narrow.
2. **Object and background representation gap — high priority.** Perception is simplistic and not connected to object-level actions.
3. **Candidate ranking and search policy — medium priority.** Could matter when a valid program is expressible, but current evidence does not isolate it.
4. **IRP invariant/reverse reasoning — promising supporting layer, not yet the first proven fix.** IRP can reduce ambiguity and guide search, but cannot make an absent transformation expressible.

## Recommended next experiment

Do not immediately change MCTS/beam parameters or add the full IRP layer.

First, after the 100-task report completes:

1. Inspect per-fold selected programs and misses.
2. Manually label a stratified sample of failures using the taxonomy above.
3. Check whether each failure's required transformation is expressible by the existing DSL.
4. Measure the proportion of failures that are clearly outside the DSL.
5. Add a small, carefully selected set of missing parameterized primitives only if the audit supports it.
6. Then test IRP as a separate ablation, holding DSL and search budgets constant.

This separates two different questions:
- Can the solver represent the rule?
- Given that it can represent the rule, can its reasoning/search find it?

## Decision

The current evidence does not justify claiming IRP will improve accuracy. It does justify prioritizing a DSL-coverage audit before a major IRP integration. Keep the existing solver as the control and preserve all baseline results.

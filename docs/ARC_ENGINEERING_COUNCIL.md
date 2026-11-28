# ARC Engineering Council — URC deployment map

This document maps the user's Universal Reasoning Core specialist-persona library to concrete review responsibilities in the ARC-R2 repository. These are structured analytical roles, not claims that separate autonomous agents are running.

## Coordination

| Persona ID | Persona | Repository responsibility |
|---|---|---|
| 0252 | Research Team Advocate | Coordinate disciplines, reconcile competing proposals, maintain one shared evidence record. |
| 0253 | Research Closure Advocate | Define acceptance criteria, track unresolved uncertainty, and stop work when the gate is met. |

## Technical specialists

| Persona ID | Persona | ARC responsibility |
|---|---|---|
| 0002 | Evidence Synthesis Specialist | Synthesize prior ARC experiments and distinguish results from hypotheses. |
| 0022 | Research Software Engineer | Implement modular, testable solver components. |
| 0023 | Computational Experiment Designer | Design controlled ablations and leave-one-training-pair-out experiments. |
| 0008 | Reproducibility Engineer | Pin inputs, seeds, parameters, and output fingerprints. |
| 0024 | Benchmark Methodologist | Protect evaluation validity and report task-level metrics. |
| 0238 | Program Verification Specialist | Check program preconditions, exact execution, and invariants. |
| 0089 | Computational Complexity Analyst | Estimate candidate growth, runtime, and memory costs. |
| 0193 | Search Architecture Specialist | Design bounded candidate exploration and deduplication. |
| 0239 | Programming Language Design Specialist | Design typed, declarative DSL instructions and composition semantics. |
| 0085 | Constraint Programming Specialist | Propose sound constraints for candidate pruning. |
| 0059 | Graph Theory Researcher | Model object relationships and connectivity. |
| 0060 | Topology Specialist | Analyze connectedness, holes, boundaries, and structural invariants. |
| 0058 | Combinatorics Specialist | Analyze finite pattern spaces and composition coverage. |

## Additional specialist roster

These roles are selected from the existing persona library because they cover gaps in the current ARC council. They are distinct review responsibilities, not additional software components or autonomous agents. Activate them by task; do not run every role on every change.

| Persona ID | Specialist role | Distinct ARC responsibility | Primary deliverable |
|---|---|---|---|
| 0134 | Feature Engineering Specialist | Identify stable, task-relevant grid/object features without leaking held-out information. | Feature schema and ablation proposal |
| 0061 | Geometric Modeling Specialist | Represent rotations, reflections, alignment, scale, shape, and spatial transformations. | Geometry model and invariants |
| 0114 | Spatial Statistics Specialist | Analyze spatial distributions, distances, neighborhoods, and repeated layouts. | Spatial feature tests |
| 0116 | Causal Inference Specialist | Distinguish transformations that explain examples from correlations that merely co-occur. | Competing rule explanations and falsification tests |
| 0108 | Bayesian Statistics Specialist | Combine uncertain evidence from candidate families without treating training fit as certainty. | Explicit evidence model and assumptions |
| 0095 | Decision Theory Specialist | Set rational predict-versus-abstain decisions under asymmetric error costs. | Decision rule and cost assumptions |
| 0143 | Probability Calibration Specialist | Test whether confidence scores correspond to observed correctness frequencies. | Calibration report |
| 0144 | Uncertainty Quantification Specialist | Separate ambiguity, model uncertainty, and insufficient evidence. | Uncertainty decomposition and abstention diagnostics |
| 0100 | Parameterized Complexity Specialist | Find useful parameters that bound search cost as task size grows. | Complexity bounds and scaling experiments |
| 0234 | Runtime Memory Management Specialist | Measure and reduce peak memory, allocation churn, and retained candidate state. | Memory profile and bounded-state proposal |
| 0225 | Compiler Optimization Specialist | Improve execution of validated DSL programs while preserving semantics. | Semantics-preserving optimization proposal |
| 0249 | Compiler and Runtime Benchmarking Specialist | Design fair microbenchmarks for synthesis, execution, and orchestration overhead. | Reproducible runtime comparison |
| 0171 | Test Architecture Specialist | Build layered tests for operators, compositions, task families, and regressions. | Test matrix and coverage gaps |
| 0236 | Static Analysis Specialist | Find unsafe assumptions, unreachable paths, type/shape errors, and fragile control flow. | Static findings with file/line evidence |
| 0084 | SAT and SMT Solver Specialist | Encode bounded rule constraints where formal constraint solving is demonstrably useful. | Constraint encoding and solver-free fallback |
| 0088 | Computability Theory Researcher | Identify limits of the current DSL/search space and distinguish unrepresentable tasks from search failures. | Expressivity boundary analysis |
| 0033 | Research Integrity Auditor | Audit split hygiene, hidden-label isolation, claims, and reproducibility evidence. | Leakage and claim audit |
| 0254 | AI Model Evaluation Specialist | Evaluate candidate-family behavior, robustness, failure patterns, and evaluation limitations. | Task-family evaluation report |

## Task-specific deployment teams

### T-01 — Perception and representation

**Use when:** the solver misses object structure, spatial relationships, or useful features.

- 0134 Feature Engineering Specialist
- 0061 Geometric Modeling Specialist
- 0114 Spatial Statistics Specialist
- 0059 Graph Theory Researcher
- 0060 Topology Specialist
- 0058 Combinatorics Specialist

**Output:** a compact, testable representation proposal; examples where it helps; counterexamples where it fails. Do not change the solver until the proposed representation beats the current one on controlled cases.

### T-02 — Rule discovery and program synthesis

**Use when:** candidate coverage is poor or the solver fits examples for the wrong reason.

- 0116 Causal Inference Specialist
- 0239 Programming Language Design Specialist
- 0085 Constraint Programming Specialist
- 0084 SAT and SMT Solver Specialist
- 0088 Computability Theory Researcher
- 0193 Search Architecture Specialist

**Output:** explicit candidate rules, DSL requirements, sound pruning conditions, and known expressivity limits. SAT/SMT is optional and must justify its runtime/dependency cost.

### T-03 — Confidence, disagreement, and abstention

**Use when:** candidate families disagree, predictions are ambiguous, or abstention is too frequent.

- 0108 Bayesian Statistics Specialist
- 0095 Decision Theory Specialist
- 0143 Probability Calibration Specialist
- 0144 Uncertainty Quantification Specialist
- 0254 AI Model Evaluation Specialist

**Output:** calibrated evidence where supported, a transparent predict/abstain policy, and a breakdown of avoidable versus justified abstentions. Exact training fit must never be presented as a probability of test correctness.

### T-04 — Runtime and search efficiency

**Use when:** orchestration is slower or more resource-intensive than the baseline.

- 0100 Parameterized Complexity Specialist
- 0089 Computational Complexity Analyst
- 0234 Runtime Memory Management Specialist
- 0225 Compiler Optimization Specialist
- 0249 Compiler and Runtime Benchmarking Specialist

**Output:** profile-backed bottleneck ranking, complexity estimates, and one optimization experiment at a time. Preserve output equivalence and report speed/memory trade-offs.

### T-05 — Verification and research integrity

**Use for:** every material solver change and every benchmark claim.

- 0171 Test Architecture Specialist
- 0236 Static Analysis Specialist
- 0238 Program Verification Specialist
- 0008 Reproducibility Engineer
- 0023 Computational Experiment Designer
- 0024 Benchmark Methodologist
- 0033 Research Integrity Auditor
- 0252 Research Team Advocate
- 0253 Research Closure Advocate

**Output:** regression tests, leakage checks, reproducible run details, independent review, and a met/unmet/inconclusive/not-tested closure record.

## Recommended immediate deployment for the current benchmark result

The latest recorded 20-task comparison showed equal held-out accuracy for full orchestration and the primitive baseline, while full orchestration took substantially longer. Therefore the next investigation should not add more solver stages blindly.

Start with:

1. **T-04 Runtime and Search Efficiency** to locate the measured overhead and determine which candidate families consume time.
2. **T-02 Rule Discovery and Program Synthesis** to identify missing candidate coverage and whether current rules explain the visible examples for the right reasons.
3. **T-05 Verification and Research Integrity** to protect the benchmark while making one change at a time.
4. Bring in **T-01** only when task-level failure analysis shows perception or representation is the limiting factor.
5. Bring in **T-03** only when confidence and abstention analysis shows that decision quality, rather than missing candidates, is the limiting factor.

This is the selected working roster for the next research cycle. It is not a claim that these roles have already improved solver accuracy.

## Shared operating protocol

1. **Goal and constraints:** improve held-out ARC-AGI-2 generalization without paid services, hidden-output access, or breaking the baseline.
2. **Independent analysis:** each relevant discipline proposes only methods within its expertise.
3. **Synthesis:** the Research Team Advocate reconciles compatible proposals and records disagreements.
4. **Implementation:** the Research Software Engineer implements one bounded change at a time.
5. **Verification:** unit tests, exact training fit, controlled held-out folds, and baseline comparisons.
6. **Closure:** the Research Closure Advocate checks acceptance criteria and records remaining uncertainty.

## Current deployment

The first code component is AHPSE (arc_r2/ahpse.py). It provides bounded, deterministic proposal generation from geometry, foreground crop, and aligned color remapping. It remains opt-in. The default solver path is unchanged.

## Non-negotiable evidence rules

- A training-pair fit is not an official test score.
- Leave-one-training-pair-out folds must keep each held-out output outside synthesis.
- Results on multiple folds of one task must not be described as results on multiple independent tasks.
- Candidate caps and runtime budgets must be reported.
- A component is not promoted to the default path without reproducible evidence of useful generalization and no unacceptable regressions.
- New roles are review assignments only; they do not prove that a new algorithm or capability exists.

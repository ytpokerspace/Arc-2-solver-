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

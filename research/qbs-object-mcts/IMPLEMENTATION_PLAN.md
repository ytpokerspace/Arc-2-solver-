# QBS + Object-Centric DSL + Dual MCTS Implementation

## Status
Experimental scaffold on `research/qbs-object-mcts`. The baseline branch and `main` are unchanged.

## Implemented in this scaffold
- Immutable object state with local pixel coordinates, grid origin, and explicit pose fields.
- Pure object movement, rotation, reflection, and integer scale-up operations.
- Deterministic renderer with explicit bounds and collision policies.
- Typed grid-to-grid primitive registry.
- QBS seed-policy and compatibility-gated hybrid data structures.
- Two independent MCTS tracks with family-prioritized action ordering and progressive widening.

## Deliberately not claimed
- The user's exact 200 primitive catalog has not yet been recovered and is not fabricated here.
- The ten QBS ideas have not been generated or filled in by guesswork.
- The current object extractor remains monochrome/4-connected; multi-color object correspondence is not solved by this scaffold.
- The dual-track search is a research scaffold, not yet integrated into the production solver or benchmarked on ARC-AGI-2.

## Required next inputs
1. Load the exact 200 primitive definitions, each with a stable name, family, type signature, parameters, preconditions, cost, and executable implementation.
2. Load the actual ten QBS seed ideas and their family/tag mappings.
3. Define which QBS hybrids are admissible; the implementation only combines pairs with shared compatibility tags.
4. Add unit tests and a deterministic benchmark comparing baseline, seed-only MCTS, hybrid-only MCTS, and combined beam reranking.

## Initial experiment
Use identical training tasks, deterministic seeds, candidate caps, and per-track budgets. Record exact held-out pair accuracy, all-example task accuracy, candidates evaluated, duplicate states, pruning reasons, runtime, and memory. Report synthetic tests separately from ARC results.

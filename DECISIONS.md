# Decisions

- Python standard library only for the first implementation.
- Offline/local-only: no HTTP client, model API, credentials, or dataset downloader.
- Search has explicit depth, candidate, iteration, and task caps.
- Spatial reversal, geometric reflection, and color polarity swaps are separate operations.
- PRTE is a separate heuristic polarity operator, with a switch for ablation.
- Beam Search and MCTS share one evaluator and operation vocabulary; selectable identity, Beam-only, MCTS-only, and hybrid strategies support comparisons.
- Candidate scoring uses exact training-example match first, then cell agreement, object-count agreement, and object-relation agreement as small secondary signals.
- Duplicate transformations that produce identical outputs on all supplied training inputs reuse a cached score while retaining distinct programs. This avoids repeated scoring but does not prune distinct programs that may generalize differently.
- Training examples select programs; held-out examples are prediction targets only.
- Benchmark reporting must distinguish training-example fit from held-out or official evaluation.

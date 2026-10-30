# Limitations

- This is an initial symbolic baseline, not a general ARC solver.
- Object segmentation, object relations, compositional program induction, and learned priors are not implemented.
- The operation vocabulary is small; many ARC tasks require transformations outside it.
- The initial MCTS explores operator sequences, not a rich state/action model.
- Candidate ranking uses exact-example match plus cell agreement; this heuristic requires ablation and calibration.
- No official ARC-AGI-2 dataset has been evaluated in this commit.

- PRTE's first version infers polarity from the most frequent color and swaps it with the most frequent non-background color; this heuristic can be wrong when the dominant color is not the true background.
- Object extraction currently uses four-neighbor connectivity and treats the inferred background as empty; diagonal connectivity and multi-layer object semantics are not modeled.
- Object-count agreement is only a small ranking signal, not a substitute for relational or semantic understanding.

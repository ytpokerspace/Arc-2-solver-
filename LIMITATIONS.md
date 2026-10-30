# Limitations

- This is an initial symbolic baseline, not a general ARC solver.
- Object segmentation, object relations, compositional program induction, and learned priors are not implemented.
- The operation vocabulary is small; many ARC tasks require transformations outside it.
- The initial MCTS explores operator sequences, not a rich state/action model.
- Candidate ranking uses exact-example match plus cell agreement; this heuristic requires ablation and calibration.
- No official ARC-AGI-2 dataset has been evaluated in this commit.

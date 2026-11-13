# ARC-R2 Research Notes and Knowledge Map

**Scope:** ARC-AGI-2 only. This document maps selected research references to the existing ARC-R2 implementation. It does not imply that the full books have been read, that every suggested method is implemented, or that any benchmark score improved.

## Existing implementation inspected

The active public-training benchmark is `arc_r2.primitive_benchmark`. It performs leave-one-training-pair-out evaluation, uses deterministic primitive enumeration, and explicitly reports that MCTS and beam search are not used. The repository also contains separate experimental object-level and search modules. Keep those paths distinct in reports.

Existing relevant modules:
- `arc_r2/primitives.py`: explicit grid transformations and object/perception primitives.
- `arc_r2/objects.py`: four-connected monochrome component extraction.
- `arc_r2/object_ops.py`: object normalization, translation, scaling, rotation, and reflection.
- `arc_r2/relations.py`: pairwise spatial relation signatures and relation graph.
- `arc_r2/object_dsl.py`, `arc_r2/object_solver.py`: parameterized object programs and bounded experimental search.
- `arc_r2/primitive_benchmark.py`: deterministic leave-one-training-pair-out benchmark.
- `BENCHMARK_LOG.md`, `DECISIONS.md`, `DEVELOPMENT_LOG.md`, `LIMITATIONS.md`: project evidence and design notes.

## Source-to-capability map

| Source | Relevant concepts | ARC-R2 mapping | Status |
|---|---|---|---|
| Norvig, *Paradigms of Artificial Intelligence Programming* | Symbolic AI, rule-based programs, search, unification, constraint-based scene labeling | Typed DSL, explicit candidate programs, transformations | Foundational reference; adaptation hypothesis |
| Dechter, *Constraint Processing* | Constraint networks, consistency, look-ahead/look-back, hybrid inference | Exact fit across every training pair; future pruning layer | Foundational reference; no CSP engine added |
| Tsang, *Foundations of Constraint Satisfaction* | CSP formalization, domains, constraints, reduction, propagation | Potential typed object/color/geometry constraints | Foundational reference; proposed extension |
| Soille, *Morphological Image Analysis* | Morphology, regions, boundaries, erosion/dilation | Potential shape and region operators | Research lead; not implemented by this change |
| Bollobás, *Modern Graph Theory* | Connectivity, matching, coloring, graph structure | Existing object relation graph; future graph matching | Existing relation graph is simpler than full graph matching |
| Knuth, *TAOCP Vol. 4 Fascicle 7: Constraint Satisfaction* | Backtracking, CSP, SAT, graph labeling, variable ordering | Bounded enumeration and future candidate pruning | Reference for future search design |
| Joffe & Eliasmith, *Vector Symbolic Algebras for ARC* | VSA representations, object-centric program synthesis | Candidate alternative representation for objects and relations | External research lead; not integrated |

## Engineering interpretation

1. Preserve the primitive-first benchmark as the baseline.
2. Keep object-level search and other experimental engines separate until an apples-to-apples comparison exists.
3. Use exact training-pair consistency as a hard validity gate. Heuristic scores may rank valid candidates but must not disguise a failed example.
4. Explore graph matching only after defining stable object identities and correspondence rules.
5. Add morphology only with unit tests for border handling, connectivity, holes, and color/background assumptions.
6. Treat VSA as an optional representation experiment. Do not add a dependency or replace the current grid representation without measured benefit.
7. Use complexity/description length only as a tie-breaker among candidates that satisfy all training pairs.

## Book and paper references

1. Peter Norvig, *Paradigms of Artificial Intelligence Programming: Case Studies in Common Lisp*, Morgan Kaufmann, 1992. Author resource and contents: https://www.norvig.com/paip.html
2. Rina Dechter, *Constraint Processing*, Morgan Kaufmann, 2003. Author's book page: https://ics.uci.edu/~dechter/books/
3. Edward Tsang, *Foundations of Constraint Satisfaction*, Academic Press, 1993. Author's reference page: https://www.bracil.net/edward/FCS.html
4. Pierre Soille, *Morphological Image Analysis: Principles and Applications*, 2nd ed., Springer, 2004. DOI: https://doi.org/10.1007/978-3-662-05088-0
5. Béla Bollobás, *Modern Graph Theory*, Springer, 1998. DOI: https://doi.org/10.1007/978-1-4612-0619-4
6. Donald E. Knuth, *The Art of Computer Programming, Volume 4, Fascicle 7: Constraint Satisfaction*, Addison-Wesley, 2025. Publisher page: https://www.pearson.com/en-us/subject-catalog/p/the-art-of-computer-programming-volume-4-fascicle-7/P200000012509/9780135328248
7. Isaac Joffe and Chris Eliasmith, “Vector Symbolic Algebras for the Abstraction and Reasoning Corpus,” arXiv:2511.08747 (2025): https://arxiv.org/abs/2511.08747. Associated code (MIT license shown in repository): https://github.com/ijoffe/ARC-VSA-2025

## Access and copyright notes

Publisher and author pages establish bibliographic details and topic relevance; they do not establish that a free full-text copy is available. This repository stores citations and original engineering notes, not copied book chapters. Check each book's institutional access and license before obtaining or redistributing it. The VSA repository is linked as an external implementation, not copied into this project.

## Evidence status

- This catalog is metadata and a research-to-code mapping.
- It does not claim to have imported book text into the solver.
- It does not add a CSP solver, morphology engine, graph matcher, or VSA dependency.
- No ARC dataset benchmark was launched as part of this change.
- No new ARC accuracy or performance result is claimed.

## Next experiments

1. Add property-based or table-driven tests for object transforms and relation invariants.
2. Prototype a small constraint representation for candidate programs; compare candidate counts and runtime against the unchanged primitive baseline.
3. Prototype graph correspondence on synthetic pairs with known object matches.
4. Evaluate morphology operators independently on hand-constructed grids.
5. Evaluate VSA only as a separate experiment with fixed data, seed, and resource limits.
6. Promote a method into the active benchmark only after tests and a reproducible held-out comparison.

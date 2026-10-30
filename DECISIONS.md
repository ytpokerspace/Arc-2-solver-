# Decisions

- Python standard library only for the first implementation.
- Offline/local-only: no HTTP client, model API, credentials, or dataset downloader.
- Search has explicit depth, candidate, iteration, and task caps.
- Spatial reversal, geometric reflection, and color polarity swaps are separate operations.
- Beam Search and MCTS share one evaluator and operation vocabulary.
- Training examples select programs; held-out examples are prediction targets only.

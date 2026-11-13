# Glyphic Path Optimization (GPO)

GPO is an experimental semantic-memory layer for ARC-R2. It represents a reasoning trace as a small sequence of stable glyphs linked to typed events, tags, and preserved source evidence.

Example path:

`🔍 → 🟦 → 🔗 → 📐 → 🔄 → 🎯`

The sequence is a human-readable index, not a lossless replacement for the underlying data.

## Current prototype

Module: `arc_r2/glyphic_path_memory.py`

- Versioned glyph codebook with stable meanings.
- Typed event records and concatenated glyph paths.
- Persistent JSON storage with duplicate-ID protection.
- Tag-based Jaccard retrieval with explicit evidence status filters.
- Lazy hydration of original source evidence, guarded by SHA-256 verification.
- Verified-path hybridization using maximal suffix/prefix overlap.
- Every hybrid result is labelled `hypothesis`; it must be tested independently.

## Evidence states

- `verified`: exact evidence check was completed by the caller and the status is explicitly assigned.
- `hypothesis`: candidate interpretation or hybrid not yet established.
- `rejected`: contradicted candidate retained for audit, excluded from default retrieval.

The memory API does not itself prove an ARC transformation correct. The caller is responsible for assigning `verified` only after an explicit exact check against all relevant training examples.

## Example

```python
from arc_r2.glyphic_path_memory import GlyphicPathMemory, make_event

memory = GlyphicPathMemory("artifacts/glyphic_paths.json")
memory.add_path(
    "task-001-path-01",
    [
        make_event("observe", "detect colored region"),
        make_event("object", "extract connected component"),
        make_event("transform", "horizontal reflection", operation="FLIP_H"),
        make_event("verify", "all training pairs match exactly"),
    ],
    tags=["object", "reflection", "geometry"],
    status="verified",
    source_evidence={"training_pairs": [...]},
)
matches = memory.retrieve(["object", "reflection"], limit=5)
evidence = memory.hydrate("task-001-path-01")
```

## Scope and limitations

This is a first storage/retrieval prototype. It does not yet seed or alter MCTS candidate generation, does not claim that emoji strings compress arbitrary data losslessly, and does not change the default ARC solver. Full evidence remains stored in the memory record; future work can move large evidence blobs to a separate content-addressed store. Retrieval scores are tag-overlap heuristics, not calibrated semantic probabilities.

## Tests

```bash
python -m unittest tests.test_glyphic_path_memory -v
python -m compileall -q arc_r2 tests
python -m unittest discover -s tests -v
```

Measure storage bytes, retrieval latency, reconstruction integrity, and matched-budget held-out ARC performance before considering default integration.

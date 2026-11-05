# ARC-R2 Emoji Representation Experiment

This experiment adds two reversible symbolic views without changing the solver's
canonical integer-grid representation.

## Representations

- **A — Integer baseline:** the existing immutable `Grid` tuple.
- **B — Emoji grid:** one emoji token per cell, with a fixed bijection for ARC
  colors 0–9. The emoji view is for inspection and experiments, not model input.
- **C — Emoji object graph:** connected components represented by an emoji color,
  grid origin, normalized cell coordinates, and spatial-relation metadata.

## Losslessness contract

Every representation must decode to the exact original integer grid. Tests
check all ten ARC colors, shape preservation, unknown tokens, object bounds,
and object-graph reconstruction. A candidate only counts as an exact solution
when the reconstructed integer grid equals the target grid cell by cell.

## Experimental boundary

This commit adds encoding/decoding and tests only. It does **not** replace ARC
search operators, change candidate ranking, or claim a speed/accuracy gain.
Emoji are variable-width Unicode glyphs; they are not a compact storage format.
The integer grid remains canonical so rendering differences cannot affect
solver correctness.

## Next benchmark

Compare the existing integer path against emoji-grid and emoji-object-graph
round trips on identical fixed inputs. Record exact reconstruction, encode/decode
time, and serialized representation size separately. Only after this codec gate
passes should a separate solver-search experiment evaluate whether symbolic
features improve candidate coverage or solve rate.

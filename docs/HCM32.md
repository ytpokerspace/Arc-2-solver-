# HCM-32 v1.0 — Data Model and Prototype

## Purpose

HCM-32 is a separate, persistent knowledge-memory layer for ARC-R2. Its 32 stable primary regions organize concepts; a sparse typed graph links concepts and regions across any family or hierarchy level; declarative DSL procedures preserve reusable methods; evidence records distinguish hypotheses from demonstrated results.

HCM-32 does not change the active solver, HMSE selector, or SingleStageMemory. The latter remains an exact-training-problem cache. HCM-32 is a standalone prototype until its tests and later transfer experiments justify integration.

## 1. Region catalog

There are eight knowledge families, four primary quadrants per family (32 total), and four initial subquadrants per primary quadrant (128 total).

| Primary IDs | Family | Four primary-region labels |
|---|---|---|
| Q01–Q04 | Perception | object identity; shape/geometry; color/attributes; position/orientation |
| Q05–Q08 | Structure | spatial relations; symmetry/patterns; topology/connectivity; constraints/boundaries |
| Q09–Q12 | Combinatorial | pairwise combinations; multi-rule combinations; permutation/sequencing; hybridization/composition |
| Q13–Q16 | Time | operation order; temporal progression; learning history/recency; duration/cycles/repetition |
| Q17–Q20 | Methodology | problem decomposition; hypothesis formation; experimental design; search/decision strategies |
| Q21–Q24 | Technique | DSL primitives; transformation operators; inverse operations; program composition |
| Q25–Q28 | Relationships | similarity/distance; causal/functional relations; generalization/transfer; alternatives/contradictions |
| Q29–Q32 | Verification and memory | evidence/confidence; counterexamples/failures; reusable procedures; recall/refinement |

Every primary quadrant has four generic subquadrants:

- S01: identify
- S02: represent
- S03: evaluate and connect
- S04: retain and retrieve

Example: Q22.S02 is the representation subquadrant for transformation operators. IDs are stable within schema version 1. Deeper hierarchy can be added in a future schema version; v1.0 does not pretend to implement unlimited recursive depth.

A knowledge node can reference multiple regions. It is stored once, not copied into each region.

## 2. Core entities

### Knowledge node

A node represents a concept, object pattern, rule, observation, or learned abstraction.

Required fields: node_id, node_type, name, quadrant_refs, status, version.

quadrant_refs accepts primary or subquadrant IDs. Initial status is hypothesis. A node can become supported or verified only after a passed evidence record is attached; it can become rejected only after failed evidence is attached.

### Connection

A connection endpoint is a typed reference:

```json
{"kind": "node", "id": "KN-REFLECTION"}
```

or

```json
{"kind": "region", "id": "Q22.S02"}
```

Connections can therefore join node-to-node, node-to-region, region-to-node, or region-to-region at any hierarchy level. A connection stores its relationship type, direction, optional strength, strength basis, status, use counters, and evidence references.

Supported relationship types: similar, composable, inverse, dependency, equivalent, contradictory, generalizes, derived, and sequential.

Strength is optional and always separate from correctness. A similarity score does not establish composability. New connections start as hypotheses; evidence is required for promotion.

### Procedure

A procedure stores an ordered list of declarative DSL operations, parameters, input constraints, DSL version, inverse procedure reference, status, evidence references, and reuse outcomes.

Saved procedures contain no Python, shell, or other executable source. A trusted runtime must separately register and validate each DSL operator. Operation order is significant.

An inverse reference can point to the same procedure for self-inverse operations such as reflection. Lossy operations must not claim an inverse unless sufficient side information is preserved.

### Evidence

Evidence records identify the subject, evidence type, result, optional input/expected/actual SHA-256 fingerprints, and JSON-safe details. Results are passed, failed, or inconclusive. Training fit, held-out validation, round-trip checks, transfer tests, and counterexamples should use distinct evidence types.

A passed record is required before a procedure is marked verified or a candidate validated. Failed evidence is preserved and can support rejection/deprecation.

### Candidate combination

A candidate records source procedures, an ordered operation sequence, generation method, status, and evidence references. Candidates start as generated hypotheses. Only a passed evidence record permits validated; a failed record permits rejected.

### Temporal event

Procedure applications append ordered events and increment separate success/failure reuse counters. Problem-time operation order lives in the procedure sequence; learning-time history lives in evidence and event records.

## 3. Persistence and integrity

- JSON, schema name hcm32, version 1.
- Stable IDs and explicit references.
- Atomic file replacement via a temporary sibling file.
- Malformed or incompatible files raise HCM32FormatError; they are not silently reset or overwritten.
- SHA-256 fingerprints are provenance fields, not proof by themselves. The evaluator must compare the actual outputs and record the result.
- No confidence probability is invented. Connection strength must state its basis.

## 4. Minimal usage

```python
from arc_r2.hcm32 import HCM32Memory

memory = HCM32Memory("hcm32-memory.json")

memory.add_node(
    "KN-REFLECTION",
    "transformation_rule",
    "horizontal reflection",
    ["Q22.S02", "Q30.S03"],
)

memory.add_procedure(
    "PR-REFLECT-H",
    "horizontal_reflection",
    [{"operator": "REFLECT", "parameters": {"axis": "left-right"}}],
    inverse_procedure_id="PR-REFLECT-H",
)

memory.add_connection(
    "CN-REFLECT-SYMMETRY",
    {"kind": "node", "id": "KN-REFLECTION"},
    {"kind": "region", "id": "Q06.S01"},
    "composable",
)

memory.save()
```

## 5. Prototype boundaries

Implemented in arc_r2/hcm32.py:

- Fixed 32-region and 128-subregion catalog.
- Typed knowledge nodes and cross-level graph edges.
- Declarative procedure records and inverse references.\n- A deterministic HCM32Runtime with a whitelist of trusted REFLECT, ROTATE, and TRANSPOSE operators.\n- Verified-only procedure execution and exact-fit selection with ambiguity abstention.
- Evidence records and evidence-gated status transitions.
- Combinatorial candidate records.
- Procedure application history and reuse counters.
- JSON persistence, detached snapshots, and fail-closed loading.

Not yet implemented:

- Automatic concept extraction or quadrant assignment.
- Similarity computation or graph traversal ranking.
- Automatic inverse derivation or proof generation.
- Candidate combination search or automatic memory compression.
- Integration with HMSE or the active ARC solver.
- Recursive hierarchy expansion beyond the initial 128 subregions.

Those are later stages. This prototype first establishes stable storage semantics and tests.

## 6. Runtime execution and selection

`HCM32Runtime` executes only explicitly registered trusted operators; persisted procedure records remain declarative and are never evaluated as code. The built-in v1 operators are REFLECT (`left-right` or `top-bottom`), ROTATE (one, two, or three clockwise quarter-turns), and TRANSPOSE. Inputs must be non-empty rectangular grids. Unknown operators, constraints, or malformed parameters fail closed.\n\nSelection examines only procedures with verified status attached to a knowledge node. It returns a procedure only when exactly one candidate reproduces the demonstration output; zero matches or multiple matches return `None`. Application does not itself promote a procedure or record success: the caller must compare against an independently established expected output, then attach evidence and record the outcome.\n\n## 7. Tests

Run:

```bash
python -m unittest discover -s tests -p 'test_hcm32.py' -v
python -m unittest discover -s tests -v
```

Acceptance focuses on catalog counts, cross-level references, duplicate/invalid IDs, persistence, evidence-gated promotion, preservation of failed applications, and corrupt-file rejection. Passing these tests validates the memory data structures, not ARC generalization or solver accuracy.

# ARC-2 Solver (ARC-R2)

A transparent, primitive-first ARC research solver. The active benchmark path uses **explicit elementary transformations and deterministic exhaustive enumeration only**. It does not invoke Monte Carlo Tree Search (MCTS), beam search, neural networks, external APIs, or paid services.

## Current status

This is a research prototype, not a solved ARC-AGI-2 system. A successful unit test means the primitive or harness behaves as specified; it does not mean an ARC task was solved. Public benchmark results are reported as leave-one-training-pair-out cross-validation, not official hidden-test accuracy.

## Run the primitive solver

Python 3.10+; standard library only.

```bash
python -m unittest discover -s tests -v
python -m arc_r2.primitive_benchmark \
  --dataset path/to/ARC-AGI-2/data/training \
  --limit 20 \
  --max-depth 2 \
  --candidate-cap 5000 \
  --output arc_r2_primitive_benchmark.json
```

The harness reads each task's training input/output pairs. It holds one training pair out, fits on the remaining pairs, and checks the held-out pair. It does not parse or use task test outputs.

## Elementary primitive library

All active primitive implementations are in `arc_r2/primitives.py`.

| Family | Operations |
|---|---|
| Grid geometry | rotate 90/180/270, horizontal flip, vertical flip, transpose, anti-diagonal reflection |
| Colors | recolor, swap two colors, invert dominant foreground/background |
| Regions | crop to a color, crop non-background content |
| Size/layout | nearest-neighbor scale, repeat/tile, alternating row/column/checkerboard tile transforms, mask-based pattern stamping, append-prefix repetition |
| Motion | gravity/compact up, down, left, right |
| Completion | fill active rows, fill active columns, fill enclosed holes, mirror-complete horizontally/vertically, fill object bounding boxes, keep/recolor largest object using a learned size/shape-to-color rule |
| Perception | color counts, dominant background, foreground palette, 4-connected and 8-connected object extraction, object size and bounding box |
| Program model | immutable primitive instructions, deterministic program execution, canonical operation sequence |

The vocabulary is parameterized from colors and dimensions visible in the fit inputs. Programs contain zero, one, or two operations. The harness tests candidates deterministically, removes duplicate behaviors, and selects the shortest exact fit. If no candidate reproduces every fit pair exactly, it abstains; it does not return a partial-fit guess.

## Harness and evidence

- `arc_r2/primitive_benchmark.py`: dataset loader and leave-one-training-pair-out benchmark.
- `tests/test_primitives.py`: primitive correctness, object extraction, exact-fit and abstention tests.
- `.github/workflows/tests.yml`: compile and test gate.
- `.github/workflows/benchmark.yml`: pinned public ARC-AGI-2 training benchmark and JSON artifact.

The benchmark reports exact held-out training-pair accuracy, fit coverage, candidate counts, runtime, and per-fold predictions. A task with fewer than two training examples is skipped because it cannot support leave-one-out evaluation.

## Shape Radar (multi-scale perception)

`arc_r2/shape_radar.py` provides a perception-only scan shared by ARC-1 and
ARC-2 task inputs. It reports grid/palette statistics, foreground objects under
4- and 8-connectivity, object cells and bounding boxes, size/aspect ratio,
perimeter, enclosed holes, reflection symmetries, rotation/reflection-invariant
shape signatures, repeated-shape groups, and pairwise bounding-box relations.
The JSON-serializable result deliberately leaves `interpretation` unset:
perception supplies evidence; the hypothesis engine decides what it means.

Example:

```python
from arc_r2.shape_radar import scan_grid

report = scan_grid([[0, 2, 0], [2, 2, 0], [0, 0, 3]])
print(report["objects"])
print(report["repeated_shapes"])
```

Run the focused tests with
`python -m unittest discover -s tests -p 'test_shape_radar.py' -v`.

## Hierarchical Method Synthesis Engine (HMSE)

`arc_r2/hmse.py` is the orchestration layer for reusable, multi-stage
procedures. It is deliberately separate from any one DSL or solver:

- A procedure is an ordered schedule of dependent stages. Each stage declares
  required inputs, an output key, a validator, and a bounded candidate budget.
- Each stage partitions candidate methods into named search families. Candidates
  are evaluated round-robin across partitions so one large family cannot consume
  the whole budget before alternatives are considered.
- The highest-scoring accepted candidate is selected (ties preserve search
  order). A stage can stop early at its configured score threshold.
- Later stages receive earlier outputs. Missing dependencies and unvalidated
  stages fail closed rather than silently continuing.
- Procedures can be composed or have one stage replaced without rewriting the
  rest. `MethodLibrary` saves declarative JSON specifications; executable
  functions are registered separately and are never loaded from saved files.
  Completed runs can be saved with their selected methods, candidate evidence,
  and evaluation counts using `MethodLibrary.save_completed`.
- `register_arc_methods(registry)` plugs the current Shape Radar and primitive
  DSL into the registry as `arc.shape_radar.scan` and
  `arc.primitive.apply`. Additional DSLs can register their own method IDs.
- The procedure exposes an estimated stage count and an upper bound on candidate
  evaluations. The execution trace records the actual selected method, partition,
  candidate evidence, and evaluations.

Minimal example:

```python
from arc_r2.hmse import (
    MethodRegistry, MethodSpec, SearchPartition, Stage, Procedure,
    execute_procedure, register_arc_methods,
)

registry = MethodRegistry()
register_arc_methods(registry)
registry.register_method(
    "count_cells",
    lambda state, params: sum(len(row) for row in state["grid"]),
)
procedure = Procedure("count-grid", (
    Stage("measure", "cell_count", (
        SearchPartition("geometry", (MethodSpec("count_cells"),)),
    ), requires=("grid",)),
), input_keys=("grid",))
result = execute_procedure(procedure, {"grid": [[1, 2], [3, 4]]}, registry)
assert result.state["cell_count"] == 4
```

The HMSE executes schedules formulated by upstream perception/rule-discovery
components. It does not yet automatically invent the complete schedule or infer
new ARC rules; that is the next integration layer. Run its tests with
`python -m unittest discover -s tests -p 'test_hmse.py' -v`.

## Explicit non-goals for this implementation

- No MCTS.
- No beam search in the active primitive benchmark.
- No random exploration or hidden-test fitting.
- No claims of complete ARC primitive coverage.
- No claim that the system has solved ARC-AGI-2.

The existing experimental modules remain in the repository for historical comparison, but the primitive benchmark and its workflow do not import or execute them.

## Shape Radar + HMSE demonstration

A pinned, single-task ARC-AGI-2 demonstration integrates Shape Radar with the
Hierarchical Method Synthesis Engine and compares the result with the existing
primitive-fitting baseline. It uses training-only data and leave-one-training-
pair-out validation.

See [docs/HMSE_ARC_DEMO.md](docs/HMSE_ARC_DEMO.md) for the protocol and run:

```bash
python -m arc_r2.hmse_demo \
  --task-json tests/fixtures/arc_agi2_00d62c1b_train.json \
  --candidate-cap 1500 \
  --max-depth 2 \
  --output arc_hmse_demo.json
```

This is a single-task integration demonstration, not an ARC benchmark score.
The HMSE schedule is authored for the demonstration; automatic schedule
invention and general ARC rule induction are not claimed.


## Second Shape Radar + HMSE demonstration: spatial relocation

A second pinned ARC-AGI-2 training task tests a different rule family:
four isolated color markers are moved into the cells of a central 2x2 anchor
according to their original diagonal quadrants. The fixture contains only the
three training pairs; test inputs and outputs are excluded.

See [docs/HMSE_SPATIAL_DEMO.md](docs/HMSE_SPATIAL_DEMO.md) for the protocol.
Run:

\`\`\`bash
python -m arc_r2.hmse_spatial_demo \\
  --task-json tests/fixtures/arc_agi2_d89b689b_train.json \\
  --candidate-cap 1500 \\
  --max-depth 2 \\
  --output arc_hmse_spatial_demo.json
\`\`\`

The spatial proposer and three-stage schedule are task-specific. This is a
second integration demonstration, not evidence of general ARC accuracy or
automatic rule discovery.


## Shared rule selector: choose from training evidence

The shared selector evaluates both registered rule families (hole-filling and
anchor-quadrant relocation) against every visible training pair. It selects
only a unique exact fit; unsupported or ambiguous cases remain unresolved.

\`\`\`bash
python -m arc_r2.hmse_rule_selector \\
  --task-json tests/fixtures/arc_agi2_00d62c1b_train.json \\
  --task-json tests/fixtures/arc_agi2_d89b689b_train.json \\
  --candidate-cap 1500 \\
  --max-depth 2 \\
  --output arc_hmse_selector_report.json
\`\`\`

See [docs/HMSE_RULE_SELECTOR.md](docs/HMSE_RULE_SELECTOR.md). This tests
selection between two authored rule families, not open-ended rule discovery.

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
| Size/layout | nearest-neighbor scale, repeat/tile, alternating row/column/checkerboard tile transforms |
| Motion | gravity/compact up, down, left, right |
| Completion | fill active rows, fill active columns, mirror-complete horizontally/vertically, fill object bounding boxes |
| Perception | color counts, dominant background, foreground palette, 4-connected and 8-connected object extraction, object size and bounding box |
| Program model | immutable primitive instructions, deterministic program execution, canonical operation sequence |

The vocabulary is parameterized from colors and dimensions visible in the fit inputs. Programs contain zero, one, or two operations. The harness tests candidates deterministically, removes duplicate behaviors, and selects the shortest exact fit. If no candidate reproduces every fit pair exactly, it abstains; it does not return a partial-fit guess.

## Harness and evidence

- `arc_r2/primitive_benchmark.py`: dataset loader and leave-one-training-pair-out benchmark.
- `tests/test_primitives.py`: primitive correctness, object extraction, exact-fit and abstention tests.
- `.github/workflows/tests.yml`: compile and test gate.
- `.github/workflows/benchmark.yml`: pinned public ARC-AGI-2 training benchmark and JSON artifact.

The benchmark reports exact held-out training-pair accuracy, fit coverage, candidate counts, runtime, and per-fold predictions. A task with fewer than two training examples is skipped because it cannot support leave-one-out evaluation.

## Explicit non-goals for this implementation

- No MCTS.
- No beam search in the active primitive benchmark.
- No random exploration or hidden-test fitting.
- No claims of complete ARC primitive coverage.
- No claim that the system has solved ARC-AGI-2.

The existing experimental modules remain in the repository for historical comparison, but the primitive benchmark and its workflow do not import or execute them.

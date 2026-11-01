"""Reliability systems for ARC-R2: isolation, budgets, provenance, verification and learning.

This module is deliberately dependency-light. It never reads hidden test outputs,
executes only the finite typed DSL, and treats confidence as a diagnostic rather
than a correctness claim.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
import math
import re
import time
from typing import Callable, Iterable, Sequence

from .dsl import Program
from .grid import Grid, as_grid, colors, shape
from .objects import GridObject, detect_objects
from .relations import relation
from .transforms import OPERATIONS, apply_operation


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    train: tuple[tuple[Grid, Grid], ...]
    test_inputs: tuple[Grid, ...]
    split: str = "development"

    @classmethod
    def from_mapping(cls, task_id: str, raw: dict, split: str = "development") -> "TaskRecord":
        if split not in {"development", "validation", "official_test"}:
            raise ValueError("unknown split label")
        train_raw = raw.get("train")
        test_raw = raw.get("test")
        if not isinstance(train_raw, list) or not train_raw:
            raise ValueError("task must contain non-empty training pairs")
        if not isinstance(test_raw, list):
            raise ValueError("task must contain a test input list")
        train = tuple((as_grid(pair["input"]), as_grid(pair["output"])) for pair in train_raw)
        # A test target is forbidden at this boundary: do not even retain it.
        if any("output" in item for item in test_raw):
            raise ValueError("test outputs are forbidden; only test inputs may be loaded")
        test_inputs = tuple(as_grid(item["input"]) for item in test_raw)
        return cls(str(task_id), train, test_inputs, split)


class TaskDataManager:
    """Stores task inputs and visible training labels without exposing hidden labels."""
    def __init__(self):
        self._tasks: dict[str, TaskRecord] = {}

    def add(self, task: TaskRecord) -> None:
        if task.task_id in self._tasks:
            raise ValueError(f"duplicate task id: {task.task_id}")
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> TaskRecord:
        return self._tasks[task_id]

    def training_pairs(self, task_id: str) -> tuple[tuple[Grid, Grid], ...]:
        return self.get(task_id).train

    def prediction_inputs(self, task_id: str) -> tuple[Grid, ...]:
        return self.get(task_id).test_inputs

    def summary(self) -> dict:
        return {
            "task_count": len(self._tasks),
            "tasks": [{"task_id": t.task_id, "split": t.split,
                       "training_pairs": len(t.train), "test_inputs": len(t.test_inputs)}
                      for t in sorted(self._tasks.values(), key=lambda x: x.task_id)],
            "hidden_test_outputs_loaded": False,
        }


@dataclass(frozen=True)
class SceneNode:
    index: int
    color: int
    bbox: tuple[int, int, int, int]
    size: int
    height: int
    width: int
    shape_signature: tuple[tuple[int, int], ...]
    symmetric_horizontal: bool
    symmetric_vertical: bool


@dataclass(frozen=True)
class SceneGraph:
    nodes: tuple[SceneNode, ...]
    edges: tuple[tuple[int, int, tuple[str, ...]], ...]


def _normalized_shape(obj: GridObject) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = obj.bbox
    return tuple(sorted((r-r0, c-c0) for r, c in obj.cells))


def build_scene_graph(grid: Grid) -> SceneGraph:
    """Extract object attributes and directed spatial/containment/adjacency edges."""
    objects = detect_objects(grid)
    nodes = []
    for i, obj in enumerate(objects):
        points = _normalized_shape(obj)
        point_set = set(points)
        max_r = max(r for r, _ in points)
        max_c = max(c for _, c in points)
        hflip = tuple(sorted((r, max_c-c) for r, c in points))
        vflip = tuple(sorted((max_r-r, c) for r, c in points))
        nodes.append(SceneNode(i, obj.color, obj.bbox, obj.size, obj.height, obj.width,
                               points, points == hflip, points == vflip))
    edges = []
    for i, a in enumerate(objects):
        for j, b in enumerate(objects):
            if i == j:
                continue
            labels = set(relation(a, b))
            ar0, ac0, ar1, ac1 = a.bbox
            br0, bc0, br1, bc1 = b.bbox
            if br0 <= ar0 and bc0 <= ac0 and ar1 <= br1 and ac1 <= bc1:
                labels.add("A_INSIDE_B")
            if ar0 <= br0 and ac0 <= bc0 and br1 <= ar1 and bc1 <= ac1:
                labels.add("B_INSIDE_A")
            if not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0):
                if any(abs(r1-r2) + abs(c1-c2) == 1
                       for r1, c1 in a.cells for r2, c2 in b.cells):
                    labels.add("ADJACENT")
            if labels:
                edges.append((i, j, tuple(sorted(labels))))
    return SceneGraph(tuple(nodes), tuple(edges))


@dataclass
class SearchBudget:
    max_expansions: int = 2000
    max_executions: int = 5000
    max_program_depth: int = 4
    max_seconds: float = 10.0
    max_hybrids: int = 20
    max_memory_bytes: int = 64 * 1024 * 1024
    started_at: float = field(default_factory=time.monotonic)
    expansions: int = 0
    executions: int = 0
    hybrids: int = 0
    estimated_memory_bytes: int = 0
    peak_estimated_memory_bytes: int = 0
    stopped_reason: str | None = None

    def __post_init__(self):
        if min(self.max_expansions, self.max_executions, self.max_program_depth, self.max_hybrids) < 0:
            raise ValueError("budget limits cannot be negative")
        if self.max_seconds < 0 or self.max_memory_bytes < 0:
            raise ValueError("time and memory limits cannot be negative")

    def consume_expansion(self) -> bool:
        if self.expansions >= self.max_expansions:
            self.stopped_reason = "expansion_budget"
            return False
        if time.monotonic() - self.started_at >= self.max_seconds:
            self.stopped_reason = "time_budget"
            return False
        self.expansions += 1
        return True

    def consume_execution(self, depth: int) -> bool:
        if depth > self.max_program_depth:
            self.stopped_reason = "program_depth"
            return False
        if self.executions >= self.max_executions:
            self.stopped_reason = "execution_budget"
            return False
        if time.monotonic() - self.started_at >= self.max_seconds:
            self.stopped_reason = "time_budget"
            return False
        self.executions += 1
        return True

    def consume_hybrid(self) -> bool:
        if self.hybrids >= self.max_hybrids:
            self.stopped_reason = "hybrid_budget"
            return False
        self.hybrids += 1
        return True

    def consume_memory(self, estimated_bytes: int) -> bool:
        if estimated_bytes < 0:
            raise ValueError("estimated memory cannot be negative")
        if estimated_bytes > self.max_memory_bytes:
            self.stopped_reason = "estimated_memory_budget"
            return False
        self.estimated_memory_bytes = estimated_bytes
        self.peak_estimated_memory_bytes = max(self.peak_estimated_memory_bytes, estimated_bytes)
        return True

    def snapshot(self) -> dict:
        return {"expansions": self.expansions, "executions": self.executions,
                "hybrids": self.hybrids, "elapsed_seconds": round(time.monotonic()-self.started_at, 6),
                "estimated_memory_bytes": self.estimated_memory_bytes,
                "peak_estimated_memory_bytes": self.peak_estimated_memory_bytes,
                "memory_limit_is_estimated_not_rss": True,
                "stopped_reason": self.stopped_reason}


class SafeDSLExecutor:
    """Fail-closed typed DSL boundary with instruction, grid and output limits."""
    def __init__(self, max_depth: int = 8, max_side: int = 30, max_steps: int = 8):
        if min(max_depth, max_side, max_steps) < 0:
            raise ValueError("execution limits must be non-negative")
        self.max_depth, self.max_side, self.max_steps = max_depth, max_side, max_steps

    def execute(self, program: Program, grid: Grid) -> Grid:
        source = as_grid(grid)
        if len(program.instructions) > min(self.max_depth, self.max_steps):
            raise ValueError("program exceeds execution depth limit")
        if max(shape(source)) > self.max_side:
            raise ValueError("input grid exceeds maximum side length")
        result = source
        for instruction in program.instructions:
            operation = instruction.operation
            if operation not in OPERATIONS and not re.fullmatch(r"COLOR\.SWAP\.[0-9]\.[0-9]", operation):
                raise ValueError(f"operation is not in the approved DSL: {operation}")
            result = as_grid(apply_operation(result, operation))
            if max(shape(result)) > self.max_side:
                raise ValueError("operation produced an oversized grid")
        return result


@dataclass
class CandidateRecord:
    candidate_id: str
    program: tuple[str, ...]
    parent_id: str | None = None
    origin: str = "search"
    hybrid_parents: tuple[str, ...] = ()
    qbs_scores: dict[str, float] = field(default_factory=dict)
    exact_training_matches: int = 0
    training_examples: int = 0
    rejection_reason: str | None = None
    execution_count: int = 0
    elapsed_seconds: float = 0.0


class CandidateLedger:
    def __init__(self):
        self.records: dict[str, CandidateRecord] = {}

    @staticmethod
    def fingerprint(program: Program) -> str:
        payload = json.dumps(program.canonical(), separators=(",", ":")).encode()
        return sha256(payload).hexdigest()[:20]

    def add(self, program: Program, **metadata) -> CandidateRecord:
        candidate_id = self.fingerprint(program)
        if candidate_id not in self.records:
            self.records[candidate_id] = CandidateRecord(candidate_id, program.canonical(), **metadata)
        else:
            record = self.records[candidate_id]
            origin = metadata.get("origin")
            if origin and origin not in record.origin.split("+"):
                record.origin = "+".join(sorted(set(record.origin.split("+")) | {origin}))
            record.qbs_scores.update(metadata.get("qbs_scores", {}))
            if metadata.get("parent_id") and record.parent_id is None:
                record.parent_id = metadata["parent_id"]
            if metadata.get("hybrid_parents"):
                record.hybrid_parents = tuple(sorted(set(record.hybrid_parents) | set(metadata["hybrid_parents"])))
        return self.records[candidate_id]

    def reject(self, candidate_id: str, reason: str) -> None:
        if candidate_id not in self.records:
            raise KeyError(candidate_id)
        self.records[candidate_id].rejection_reason = reason

    def export(self) -> list[dict]:
        return [asdict(self.records[key]) for key in sorted(self.records)]


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    exact_matches: int
    total: int
    counterexample_indices: tuple[int, ...]
    errors: tuple[str, ...] = ()


class IndependentVerifier:
    """Replays each operation independently of Program.apply and requires exact grids."""
    def __init__(self, executor: SafeDSLExecutor | None = None):
        self.executor = executor or SafeDSLExecutor()

    def verify(self, program: Program, examples: Sequence[tuple[Grid, Grid]]) -> VerificationResult:
        matches, bad, errors = 0, [], []
        if not examples:
            return VerificationResult(False, 0, 0, (), ("no training examples",))
        for index, (source, expected) in enumerate(examples):
            try:
                actual = self.executor.execute(program, source)
                target = as_grid(expected)
                if actual == target:
                    matches += 1
                else:
                    bad.append(index)
            except Exception as exc:
                bad.append(index)
                errors.append(f"example {index}: {type(exc).__name__}: {exc}")
        return VerificationResult(matches == len(examples), matches, len(examples),
                                  tuple(bad), tuple(errors))


@dataclass(frozen=True)
class CEGISResult:
    solved: bool
    program: Program | None
    iterations: int
    counterexamples: tuple[int, ...]
    rejection_count: int
    stopped_reason: str


def cegis_refine(
    examples: Sequence[tuple[Grid, Grid]],
    candidates: Iterable[Program],
    *,
    budget: SearchBudget | None = None,
    ledger: CandidateLedger | None = None,
    refine: Callable[[Program, tuple[int, ...]], Iterable[Program]] | None = None,
    max_iterations: int = 100,
) -> CEGISResult:
    """Counterexample-guided candidate loop; optional refiner proposes new programs.

    Candidates are never accepted on partial fit. The refiner receives failing
    training-pair indices, not hidden test data.
    """
    budget = budget or SearchBudget()
    ledger = ledger or CandidateLedger()
    verifier = IndependentVerifier(SafeDSLExecutor(max_depth=budget.max_program_depth))
    queue = list(candidates)
    seen: set[tuple[str, ...]] = set()
    last_bad: tuple[int, ...] = ()
    rejected = iterations = 0
    while queue and iterations < max_iterations:
        if not budget.consume_expansion():
            break
        program = queue.pop(0)
        key = program.canonical()
        if key in seen:
            continue
        seen.add(key)
        record = ledger.add(program, origin="cegis")
        started = time.monotonic()
        if not budget.consume_execution(program.complexity):
            ledger.reject(record.candidate_id, budget.stopped_reason or "budget")
            break
        result = verifier.verify(program, examples)
        record.execution_count += len(examples)
        record.elapsed_seconds += time.monotonic() - started
        record.exact_training_matches = result.exact_matches
        record.training_examples = result.total
        iterations += 1
        if result.passed:
            return CEGISResult(True, program, iterations, (), rejected, "verified_exact_fit")
        rejected += 1
        last_bad = result.counterexample_indices
        ledger.reject(record.candidate_id, "counterexample:" + ",".join(map(str, last_bad)))
        if refine is not None:
            for proposal in refine(program, last_bad):
                if proposal.canonical() not in seen and budget.consume_hybrid():
                    child = ledger.add(proposal, parent_id=record.candidate_id, origin="cegis_refinement")
                    queue.append(proposal)
    reason = budget.stopped_reason or ("iteration_limit" if iterations >= max_iterations else "candidate_exhausted")
    return CEGISResult(False, None, iterations, last_bad, rejected, reason)



def compatible_hybrid(left: Program, right: Program, examples: Sequence[tuple[Grid, Grid]], *, max_depth: int = 8) -> tuple[Program | None, dict]:
    """Concatenate two parent programs only when the composition exactly verifies."""
    combined = Program(left.instructions + right.instructions)
    if combined.complexity > max_depth:
        return None, {"compatible": False, "reason": "depth_limit"}
    result = IndependentVerifier(SafeDSLExecutor(max_depth=max_depth)).verify(combined, examples)
    return (combined if result.passed else None), {
        "compatible": result.passed, "exact_matches": result.exact_matches,
        "total": result.total, "counterexample_indices": list(result.counterexample_indices),
        "reason": "exact_training_verification" if result.passed else "training_counterexample",
    }


def disagreement_report(predictions: Sequence[Grid]) -> dict:
    """Quantify candidate disagreement without converting it into correctness."""
    if not predictions:
        return {"uncertain": True, "reason": "no_predictions", "unique_outputs": 0, "agreement": 0.0}
    counts: dict[Grid, int] = {}
    for prediction in predictions:
        grid = as_grid(prediction)
        counts[grid] = counts.get(grid, 0) + 1
    agreement = max(counts.values()) / len(predictions)
    return {"uncertain": len(counts) > 1, "unique_outputs": len(counts),
            "agreement": agreement, "candidate_count": len(predictions),
            "reason": "candidate_disagreement" if len(counts) > 1 else "unanimous_candidates"}


def calibration_metrics(confidence_labels: Sequence[tuple[float, bool]], bins: int = 10) -> dict:
    """Compute Brier score and expected calibration error from held-out outcomes."""
    if bins < 1:
        raise ValueError("bins must be positive")
    if not confidence_labels:
        return {"count": 0, "brier_score": None, "ece": None}
    rows = []
    for confidence, correct in confidence_labels:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0,1]")
        rows.append((confidence, 1.0 if correct else 0.0))
    brier = sum((p-y)**2 for p, y in rows) / len(rows)
    ece = 0.0
    for index in range(bins):
        low, high = index / bins, (index + 1) / bins
        bucket = [row for row in rows if low <= row[0] < high or (index == bins-1 and row[0] == 1)]
        if bucket:
            ece += len(bucket) / len(rows) * abs(sum(p for p, _ in bucket)/len(bucket)
                                                  - sum(y for _, y in bucket)/len(bucket))
    return {"count": len(rows), "brier_score": brier, "ece": ece}


@dataclass
class Abstraction:
    abstraction_id: str
    program: tuple[str, ...]
    validation_tasks: int
    validation_exact: int
    reuse_count: int = 0


class AbstractionLibrary:
    """Stores only reusable routines with explicit held-out validation evidence."""
    def __init__(self, min_validation_tasks: int = 3, min_exact_rate: float = 0.8):
        if min_validation_tasks < 1 or not 0 <= min_exact_rate <= 1:
            raise ValueError("invalid abstraction acceptance thresholds")
        self.min_validation_tasks = min_validation_tasks
        self.min_exact_rate = min_exact_rate
        self.items: dict[str, Abstraction] = {}

    def propose(self, program: Program, validation_outcomes: Sequence[tuple[str, bool]]) -> bool:
        task_ids = [str(task_id) for task_id, _ in validation_outcomes]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("abstraction validation requires unique task IDs")
        if len(task_ids) < self.min_validation_tasks:
            return False
        exact = sum(bool(correct) for _, correct in validation_outcomes)
        if exact / len(task_ids) < self.min_exact_rate:
            return False
        key = CandidateLedger.fingerprint(program)
        self.items.setdefault(key, Abstraction(key, program.canonical(), len(task_ids), exact))
        return True

    def record_reuse(self, abstraction_id: str) -> None:
        self.items[abstraction_id].reuse_count += 1

    def export(self) -> list[dict]:
        return [asdict(self.items[key]) for key in sorted(self.items)]


class JSONLTelemetry:
    """Append-only, deterministic JSONL event writer for benchmark inspection."""
    def __init__(self):
        self.events: list[dict] = []

    def emit(self, event: str, **fields) -> None:
        self.events.append({"event": event, **fields})

    def to_jsonl(self) -> str:
        return "".join(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n"
                       for item in self.events)

    def summary(self) -> dict:
        return {"event_count": len(self.events), "event_types": sorted({e["event"] for e in self.events}),
                "jsonl": self.to_jsonl()}

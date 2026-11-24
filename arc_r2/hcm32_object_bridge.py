"""Experimental adapter for the existing parameterized object DSL and HCM-32.

The object DSL remains the source of truth for instruction semantics. HCM-32
stores only JSON-safe declarative instructions; the trusted adapter registers
the interpreter locally. No persisted code is evaluated.
"""
from __future__ import annotations

import hashlib
import json
from typing import Iterable

from .hcm32 import HCM32Memory, HCM32Runtime
from .grid import as_grid
from .object_dsl import ObjectInstruction, ObjectProgram, apply_object_instruction

REGION_ID = "Q22"  # Technique / transformation operators
_INPUT_CONSTRAINT = "non_empty_rectangular_grid"
_OPERATOR = "OBJECT_DSL_INSTRUCTION"


def _instruction_parameters(instruction: ObjectInstruction) -> dict:
    return {
        "operation": instruction.operation,
        "object_index": instruction.object_index,
        "factor": instruction.factor,
        "turns": instruction.turns,
        "axis": instruction.axis,
        "delta_row": instruction.delta_row,
        "delta_col": instruction.delta_col,
        "top": instruction.top,
        "left": instruction.left,
    }


def _fingerprint(pairs: Iterable[tuple]) -> str:
    payload = [
        [[list(row) for row in source], [list(row) for row in target]]
        for source, target in pairs
    ]
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class HCM32ObjectDSLBridge:
    """Persist, reload, and retrieve exact-fit ObjectProgram procedures."""

    def __init__(self, path):
        self.memory = HCM32Memory(path)
        self.runtime = HCM32Runtime(self.memory)
        self.runtime.register_operator(_OPERATOR, self._apply_object_instruction)

    @staticmethod
    def _apply_object_instruction(grid, parameters):
        instruction = ObjectInstruction(**dict(parameters))
        return apply_object_instruction(as_grid(grid), instruction)

    @staticmethod
    def encode(program: ObjectProgram) -> list[dict]:
        if not isinstance(program, ObjectProgram) or not program.instructions:
            raise ValueError("a non-empty ObjectProgram is required")
        return [
            {"operator": _OPERATOR, "parameters": _instruction_parameters(item)}
            for item in program.instructions
        ]

    def store_program(self, program: ObjectProgram, examples) -> str | None:
        pairs = tuple((as_grid(source), as_grid(target)) for source, target in examples)
        if not pairs or not all(program.apply(source) == target for source, target in pairs):
            return None
        operations = self.encode(program)
        digest = hashlib.sha256(
            json.dumps(operations, separators=(",", ":"), sort_keys=True).encode("utf-8")
        ).hexdigest()[:20]
        procedure_id = "PR-OBJ-" + digest
        node_id = "KN-OBJ-" + digest
        if procedure_id in self.memory.data["procedures"]:
            return None
        fingerprint = _fingerprint(pairs)
        self.memory.add_procedure(
            procedure_id, str(program), operations,
            input_constraints=[_INPUT_CONSTRAINT],
        )
        self.memory.add_evidence(
            "EVO-" + digest, "procedure", procedure_id,
            "exact_fit_on_visible_training_pairs", "passed",
            input_fingerprint=fingerprint,
            expected_fingerprint=fingerprint,
            actual_fingerprint=fingerprint,
            details={"training_pair_count": len(pairs), "scope": "fit pairs only"},
        )
        self.memory.set_procedure_status(procedure_id, "verified")
        self.memory.add_node(
            node_id, "object_transformation", str(program), [REGION_ID],
            description="Existing object DSL program stored declaratively in HCM-32.",
            procedure_refs=[procedure_id],
        )
        self.memory.add_evidence(
            "EVON-" + digest, "node", node_id,
            "linked_verified_object_dsl_procedure", "passed",
            input_fingerprint=fingerprint,
            expected_fingerprint=fingerprint,
            actual_fingerprint=fingerprint,
            details={"procedure_id": procedure_id},
        )
        self.memory.set_node_status(node_id, "supported")
        self.memory.save()
        return procedure_id

    def find_program(self, examples) -> tuple[str, ObjectProgram] | None:
        pairs = tuple((as_grid(source), as_grid(target)) for source, target in examples)
        if not pairs:
            return None
        matches = []
        for node in self.memory.nodes_in_region(REGION_ID):
            for procedure in self.memory.procedures_for_node(node["node_id"]):
                if procedure.get("status") != "verified":
                    continue
                try:
                    exact = all(
                        self.runtime.apply_procedure(procedure["procedure_id"], source) == target
                        for source, target in pairs
                    )
                except (KeyError, TypeError, ValueError, IndexError):
                    exact = False
                if not exact:
                    continue
                instructions = []
                try:
                    for operation in procedure["operations"]:
                        if operation.get("operator") != _OPERATOR:
                            raise ValueError("not an object DSL procedure")
                        instructions.append(ObjectInstruction(**operation.get("parameters", {})))
                    if instructions:
                        matches.append((
                            procedure["procedure_id"], ObjectProgram(tuple(instructions))
                        ))
                except (TypeError, ValueError):
                    continue
        return matches[0] if len(matches) == 1 else None

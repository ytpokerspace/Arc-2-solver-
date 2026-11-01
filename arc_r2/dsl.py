"""Small typed DSL for deterministic grid programs."""
from __future__ import annotations
from dataclasses import dataclass
import re
from .grid import Grid
from .transforms import apply_operation, inverse_of, OPERATIONS

@dataclass(frozen=True)
class Instruction:
    operation: str

@dataclass(frozen=True)
class Program:
    instructions: tuple[Instruction, ...] = ()

    @property
    def complexity(self) -> int:
        return len(self.instructions)

    def apply(self, grid: Grid) -> Grid:
        result = grid
        for instruction in self.instructions:
            result = apply_operation(result, instruction.operation)
        return result

    def then(self, operation: str) -> "Program":
        return Program(self.instructions + (Instruction(operation),))

    def canonical(self) -> tuple[str, ...]:
        return tuple(i.operation for i in self.instructions)

    def __str__(self) -> str:
        if not self.instructions:
            return "IDENTITY"
        return " -> ".join(i.operation for i in self.instructions)

def parse_program(source: str) -> Program:
    """Parse operation chains such as 'REVERSE.HORIZONTAL -> POLARITY.NEGATE'."""
    text = source.strip()
    if not text or text.upper() == "IDENTITY":
        return Program()
    parts = [p.strip().upper() for p in text.split("->")]
    allowed = set(OPERATIONS)
    instructions = []
    for part in parts:
        if part.startswith("INVERSE."):
            base = part[len("INVERSE."):]
            part = "INVERSE." + inverse_of(base)
        if part not in allowed and not part.startswith("INVERSE.") and not re.fullmatch(r"COLOR\.SWAP\.[0-9]\.[0-9]", part):
            raise ValueError(f"unsupported DSL operation: {part}")
        instructions.append(Instruction(part))
    return Program(tuple(instructions))

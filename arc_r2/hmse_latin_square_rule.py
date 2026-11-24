"""Conservative Latin-square completion rule family for ARC grids."""
from __future__ import annotations

from dataclasses import dataclass

from .primitives import as_grid


@dataclass(frozen=True)
class LatinSquareCompletionProgram:
    """Complete a square grid as a Latin square over a fixed symbol alphabet.

    Existing nonzero cells are immutable. The program accepts only a unique
    completion and stops after a bounded number of search nodes.
    """

    symbols: tuple[int, ...]
    max_search_nodes: int = 10000

    def apply(self, value):
        grid = as_grid(value)
        size = len(grid)
        if size == 0 or any(len(row) != size for row in grid):
            raise ValueError("expected a nonempty square grid")
        if len(self.symbols) != size or len(set(self.symbols)) != size:
            raise ValueError("symbol alphabet must contain exactly one symbol per row")

        allowed = set(self.symbols)
        rows = [set() for _ in range(size)]
        columns = [set() for _ in range(size)]
        for r in range(size):
            for c in range(size):
                symbol = grid[r][c]
                if symbol == 0:
                    continue
                if symbol not in allowed:
                    raise ValueError("grid contains a symbol outside the learned alphabet")
                if symbol in rows[r] or symbol in columns[c]:
                    raise ValueError("duplicate symbol violates Latin-square constraints")
                rows[r].add(symbol)
                columns[c].add(symbol)

        working = [list(row) for row in grid]
        solutions = []
        visited = 0
        exhausted = False

        def search():
            nonlocal visited, exhausted
            if len(solutions) > 1 or exhausted:
                return
            visited += 1
            if visited > self.max_search_nodes:
                exhausted = True
                return

            best_cell = None
            best_options = None
            for r in range(size):
                for c in range(size):
                    if working[r][c] != 0:
                        continue
                    options = allowed - rows[r] - columns[c]
                    if not options:
                        return
                    if best_options is None or len(options) < len(best_options):
                        best_cell = (r, c)
                        best_options = options
                        if len(options) == 1:
                            break
                if best_options is not None and len(best_options) == 1:
                    break

            if best_cell is None:
                solutions.append(tuple(tuple(row) for row in working))
                return

            r, c = best_cell
            for symbol in sorted(best_options):
                working[r][c] = symbol
                rows[r].add(symbol)
                columns[c].add(symbol)
                search()
                rows[r].remove(symbol)
                columns[c].remove(symbol)
                working[r][c] = 0
                if len(solutions) > 1 or exhausted:
                    return

        search()
        if exhausted:
            raise ValueError("Latin-square search exceeded its node budget")
        if len(solutions) != 1:
            raise ValueError("grid has no unique Latin-square completion")
        return solutions[0]


def propose_latin_square_completion(pairs):
    """Propose only when one bounded Latin-square program fits every pair."""
    if not pairs:
        return None, "no-training-pairs"

    dimensions = set()
    symbols = set()
    for source, target in pairs:
        source = as_grid(source)
        target = as_grid(target)
        if not source or any(len(row) != len(source) for row in source):
            return None, "input-not-square"
        if len(target) != len(source) or any(len(row) != len(source) for row in target):
            return None, "output-dimensions-changed"
        dimensions.add(len(source))
        symbols.update(cell for row in source for cell in row if cell != 0)
        for r, row in enumerate(source):
            for c, cell in enumerate(row):
                if cell != 0 and target[r][c] != cell:
                    return None, "existing-cell-changed"

    if len(dimensions) != 1:
        return None, "training-grid-sizes-differ"
    size = next(iter(dimensions))
    alphabet = tuple(sorted(symbols))
    if len(alphabet) != size:
        return None, "symbol-count-does-not-match-grid-size"

    program = LatinSquareCompletionProgram(alphabet)
    try:
        if not all(program.apply(source) == as_grid(target) for source, target in pairs):
            return None, "proposal-failed-exact-fit"
    except (ValueError, TypeError, IndexError):
        return None, "proposal-failed-or-ambiguous"
    return program, "exact-fit"

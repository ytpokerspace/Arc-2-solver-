import unittest
from types import SimpleNamespace
from unittest.mock import patch

from arc_r2.dsl import Program, parse_program
from arc_r2.grid import as_grid
from arc_r2.object_dsl import ObjectProgram
from arc_r2.transforms import apply_operation
from arc_r2.robust_solver import (
    _OPERATION_CACHE,
    _pack,
    _unpack,
    batch_apply_operation,
    batch_apply_programs,
    exact_fit_candidates,
    solve_two_attempts,
    OBJECT_SEARCH_CANDIDATE_CAP,
    OBJECT_SEARCH_PARAMETER_CAP,
    OBJECT_SEARCH_MAX_DEPTH,
)


class RobustSolverTests(unittest.TestCase):
    def test_vectorized_geometry_matches_scalar_reference(self):
        grids = (
            as_grid([[1, 2, 3], [4, 5, 6]]),
            as_grid([[0, 1], [2, 3], [4, 5]]),
        )
        tensor, shapes = _pack(grids)
        for operation in (
            "REVERSE.HORIZONTAL",
            "REVERSE.VERTICAL",
            "ROTATE.180",
            "REFLECT.MAIN_DIAGONAL",
            "REFLECT.ANTI_DIAGONAL",
        ):
            actual, actual_shapes = batch_apply_operation(
                tensor, shapes, operation, use_cache=False
            )
            decoded = _unpack(actual, actual_shapes)
            expected = tuple(apply_operation(grid, operation) for grid in grids)
            self.assertEqual(decoded, expected, operation)

    def test_batch_program_execution_matches_scalar_program(self):
        source = as_grid([[1, 0, 2], [3, 4, 5]])
        programs = (
            Program(),
            Program().then("REVERSE.HORIZONTAL"),
            Program().then("REFLECT.MAIN_DIAGONAL").then("REVERSE.VERTICAL"),
            Program().then("COLOR.SWAP.1.4"),
        )
        actual = batch_apply_programs(source, programs)
        expected = tuple(program.apply(source) if program.canonical() != ("COLOR.SWAP.1.4",)
                         else ((4, 0, 2), (3, 1, 5)) for program in programs)
        self.assertEqual(actual, expected)
        parsed_swap = parse_program("COLOR.SWAP.1.4")
        self.assertEqual(parsed_swap.apply(source), actual[-1])

    def test_exact_filter_rejects_contradictory_training_pairs(self):
        source = as_grid([[1, 0], [0, 0]])
        left = as_grid([[0, 1], [0, 0]])
        right = as_grid([[1, 0], [0, 0]])
        self.assertEqual(exact_fit_candidates(((source, left), (source, right))), ())

    def test_exact_filter_keeps_only_programs_matching_all_pairs(self):
        source_a = as_grid([[1, 0, 0], [0, 2, 0]])
        target_a = as_grid([[0, 0, 1], [0, 2, 0]])
        source_b = as_grid([[0, 3, 0], [4, 0, 0]])
        target_b = as_grid([[0, 3, 0], [0, 0, 4]])
        survivors = exact_fit_candidates(((source_a, target_a), (source_b, target_b)))
        self.assertTrue(survivors)
        for candidate in survivors:
            self.assertEqual(candidate.outputs, (target_a, target_b))
            self.assertEqual(candidate.complexity, len(candidate.program))

    def test_hash_cache_reuses_operation_results(self):
        _OPERATION_CACHE.clear()
        grid = as_grid([[1, 2, 3], [4, 5, 6]])
        tensor, shapes = _pack((grid,))
        first, first_shapes = batch_apply_operation(tensor, shapes, "ROTATE.180")
        cache_size = len(_OPERATION_CACHE)
        second, second_shapes = batch_apply_operation(tensor, shapes, "ROTATE.180")
        self.assertGreater(cache_size, 0)
        self.assertEqual(_unpack(first, first_shapes), _unpack(second, second_shapes))

    def test_tta_returns_only_exact_fit_votes_and_at_most_two_attempts(self):
        source = as_grid([[1, 0, 0], [0, 2, 0]])
        target = as_grid([[0, 0, 1], [0, 2, 0]])
        test_input = as_grid([[0, 3, 0], [4, 0, 0]])
        report = solve_two_attempts(((source, target),), test_input, max_depth=1)
        self.assertEqual(report["views"], 24)
        self.assertGreater(report["views_with_exact_fit"], 0)
        self.assertGreater(report["object_candidates_evaluated"], 0)
        self.assertLessEqual(len(report["attempts"]), 2)
        for attempt in report["attempts"]:
            self.assertTrue(attempt["exact_training_fit"])
            self.assertGreater(attempt["votes"], 0)
            self.assertTrue(attempt["program_family"])

    def test_object_search_runs_when_grid_dsl_has_no_exact_fit(self):
        source = as_grid([[1, 0], [0, 0]])
        # Force the independent grid candidate source to return no survivors.
        object_result = SimpleNamespace(
            program=ObjectProgram(),
            candidates_evaluated=1,
            exact_examples=1,
        )
        with patch("arc_r2.robust_solver.exact_fit_candidates", return_value=()), patch("arc_r2.robust_solver.ObjectHybridSolver") as object_solver:
            object_solver.return_value.solve.return_value = object_result
            report = solve_two_attempts(((source, source),), source, max_depth=1)

        self.assertEqual(object_solver.return_value.solve.call_count, report["views"])
        object_solver.assert_called_with(
            max_depth=OBJECT_SEARCH_MAX_DEPTH,
            candidate_cap=OBJECT_SEARCH_CANDIDATE_CAP,
            parameter_cap=OBJECT_SEARCH_PARAMETER_CAP,
        )
        self.assertGreater(OBJECT_SEARCH_CANDIDATE_CAP, OBJECT_SEARCH_PARAMETER_CAP + 1)
        self.assertGreater(report["object_candidates_evaluated"], 0)
        self.assertEqual(report["views_with_exact_fit"], report["views"])
        self.assertTrue(report["attempts"])
        self.assertEqual(tuple(report["attempts"][0]["program_family"]), ("object:identity",))

    def test_second_attempt_must_have_different_family(self):
        source = as_grid([[1, 0], [0, 0]])
        target = as_grid([[0, 1], [0, 0]])
        report = solve_two_attempts(((source, target),), source, max_depth=1)
        attempts = report["attempts"]
        if len(attempts) == 2:
            self.assertNotEqual(attempts[0]["program_family"], attempts[1]["program_family"])


if __name__ == "__main__":
    unittest.main()

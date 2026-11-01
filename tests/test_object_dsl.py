import unittest

from arc_r2.grid import as_grid
from arc_r2.object_dsl import (
    ObjectInstruction, ObjectProgram, apply_object_instruction,
    generate_object_instructions, object_descriptors,
)
from arc_r2.object_solver import ObjectExample, ObjectHybridSolver


class ParameterizedObjectDslTests(unittest.TestCase):
    def setUp(self):
        self.grid = as_grid([
            [0, 0, 0, 0, 0],
            [0, 3, 3, 0, 0],
            [0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ])

    def test_descriptors_include_size_and_shape(self):
        descriptors = object_descriptors(self.grid)
        self.assertEqual(len(descriptors), 1)
        self.assertEqual(descriptors[0]["size"], (3, 2, 2))
        self.assertEqual(descriptors[0]["shape"], ((0, 0), (0, 1), (1, 0)))

    def test_parameterized_move_and_program(self):
        instruction = ObjectInstruction("MOVE", 0, delta_row=1, delta_col=1)
        expected = as_grid([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        self.assertEqual(apply_object_instruction(self.grid, instruction), expected)
        self.assertEqual(ObjectProgram((instruction,)).apply(self.grid), expected)

    def test_rotation_and_scale(self):
        rotated = ObjectProgram((ObjectInstruction("ROTATE90", 0, turns=1),)).apply(self.grid)
        self.assertEqual(rotated[1][1], 3)
        scaled = ObjectProgram((ObjectInstruction("SCALE", 0, factor=2),)).apply(self.grid)
        self.assertEqual(sum(cell == 3 for row in scaled for cell in row), 12)

    def test_bounds_and_collisions_fail_closed(self):
        with self.assertRaises(ValueError):
            apply_object_instruction(self.grid, ObjectInstruction("MOVE", 0, delta_row=-2))
        with self.assertRaises(ValueError):
            apply_object_instruction(self.grid, ObjectInstruction("MOVE", 0, delta_col=3))
        with self.assertRaises(ValueError):
            apply_object_instruction(self.grid, ObjectInstruction("SCALE", 0, factor=3))

    def test_generation_is_deterministic_and_capped(self):
        first = generate_object_instructions((self.grid,), parameter_cap=17)
        second = generate_object_instructions((self.grid,), parameter_cap=17)
        self.assertEqual(first, second)
        self.assertLessEqual(len(first), 17)

    def test_solver_learns_translation_from_training_pair(self):
        target = as_grid([
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 3, 3, 0],
            [0, 0, 3, 0, 0],
            [0, 0, 0, 0, 0],
        ])
        result = ObjectHybridSolver(
            beam_width=30, max_depth=1, candidate_cap=500, parameter_cap=500
        ).solve((ObjectExample(self.grid, target),))
        self.assertEqual(result.exact_examples, 1)
        self.assertEqual(result.program.apply(self.grid), target)
        self.assertLessEqual(result.candidates_evaluated, 500)


if __name__ == "__main__":
    unittest.main()

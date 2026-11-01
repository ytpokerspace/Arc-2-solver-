import unittest

from arc_r2.grid import as_grid
from arc_r2.object_dsl import ObjectInstruction, ObjectProgram
from arc_r2.object_solver import ObjectExample, _score


class ObjectSolverScoringTests(unittest.TestCase):
    def test_program_that_fails_on_one_training_example_is_invalid(self):
        object_grid = as_grid([[0, 3, 0], [0, 0, 0]])
        empty_grid = as_grid([[0, 0], [0, 0]])
        examples = (
            ObjectExample(object_grid, object_grid),
            ObjectExample(empty_grid, empty_grid),
        )
        program = ObjectProgram((
            ObjectInstruction("MOVE", 0, delta_row=0, delta_col=1),
        ))
        score, exact = _score(program, examples)
        self.assertEqual(score, -1.0)
        self.assertEqual(exact, 0)

    def test_wrong_sized_prediction_gets_no_cell_credit(self):
        source = as_grid([[0, 0], [0, 0]])
        target = as_grid([[0]])
        score, exact = _score(ObjectProgram(), (ObjectExample(source, target),))
        self.assertEqual(score, 0.0)
        self.assertEqual(exact, 0)


if __name__ == "__main__":
    unittest.main()

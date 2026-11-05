import tempfile
import unittest
from pathlib import Path

from arc_r2.predictive_mcts import search
from arc_r2.primitives import Primitive, Program, as_grid, flip_horizontal, recolor, generate_primitives


class PredictivePairMCTSTests(unittest.TestCase):
    def test_exact_pair_match_stops_search_and_is_saved(self):
        source = as_grid([[0, 1, 0], [2, 0, 0]])
        target = flip_horizontal(source)
        with tempfile.TemporaryDirectory() as directory:
            memory = Path(directory) / "rules.json"
            first = search(((source, target),), iterations=100,
                           branch_width=10, max_depth=3, memory_path=memory)
            self.assertTrue(first.exact_fit)
            self.assertEqual(first.solved_pairs, 1)
            self.assertEqual(len(first.program.operations), 1)
            self.assertTrue(first.first_hypotheses)
            self.assertIn("p_est=", first.first_hypotheses[0])
            self.assertIn("train_exact_rate=", first.first_hypotheses[0])
            self.assertEqual(first.best_probability_estimate, 2 / 3)
            self.assertEqual(first.best_feasibility, 1.0)
            self.assertEqual(first.best_training_pair_success_rate, 1.0)
            self.assertTrue(first.memory_saved)

            second = search(((source, target),), iterations=100,
                            branch_width=10, max_depth=3, memory_path=memory)
            self.assertTrue(second.exact_fit)
            self.assertTrue(second.memory_hit)
            self.assertEqual(second.iterations, 0)
            self.assertEqual(second.nodes_created, 0)

    def test_hybrid_composes_two_steps_to_reach_target(self):
        source = as_grid([[0, 1, 0], [2, 0, 0]])
        target = flip_horizontal(recolor(source, 1, 3))
        result = search(((source, target),), iterations=100,
                        branch_width=10, max_depth=3)
        self.assertTrue(result.exact_fit)
        self.assertEqual(len(result.program.operations), 2)
        self.assertEqual(result.program.apply(source), target)
        self.assertTrue(result.hybrid_candidates)
        self.assertGreater(result.hybrid_seeds, 0)
        self.assertGreater(result.hybrid_nodes_created, 0)

    def test_every_training_pair_must_match(self):
        source_a = as_grid([[0, 1], [0, 0]])
        target_a = flip_horizontal(source_a)
        source_b = as_grid([[0, 2], [0, 0]])
        target_b = flip_horizontal(source_b)
        result = search(((source_a, target_a), (source_b, target_b)),
                        iterations=100, branch_width=10, max_depth=3)
        self.assertTrue(result.exact_fit)
        self.assertEqual(result.solved_pairs, 2)
        self.assertEqual(result.program.apply(source_a), target_a)
        self.assertEqual(result.program.apply(source_b), target_b)

    def test_does_not_return_a_partial_fit_as_a_solution(self):
        source = as_grid([[0, 1]])
        result = search(((source, as_grid([[1, 0]])),
                         (source, as_grid([[0, 1]]))),
                        iterations=50, branch_width=10, max_depth=2)
        self.assertFalse(result.exact_fit)
        self.assertIsNone(result.program)

    def test_object_dsl_operations_share_the_predictive_program_path(self):
        source = as_grid([[0, 0, 0, 0], [0, 2, 2, 0], [0, 0, 0, 0]])
        target = as_grid([[0, 0, 0, 0], [0, 0, 2, 2], [0, 0, 0, 0]])
        operation = Primitive("OBJECT_MOVE", (0, 1, 0))
        self.assertEqual(Program((operation,)).apply(source), target)
        candidates = generate_primitives((source,), (target,))
        self.assertIn(operation, candidates)

    def test_only_top_ten_hypotheses_seed_the_hybrid_round(self):
        source = as_grid([[0, 1, 0], [2, 0, 0]])
        target = flip_horizontal(recolor(source, 1, 3))
        result = search(((source, target),), iterations=100,
                        branch_width=15, max_depth=3, hypothesis_count=10)
        self.assertLessEqual(result.hybrid_seeds, 10)
        self.assertGreater(result.hypotheses_scored, 1)


if __name__ == "__main__":
    unittest.main()

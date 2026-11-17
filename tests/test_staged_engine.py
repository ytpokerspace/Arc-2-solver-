import unittest
from unittest.mock import patch

from arc_r2.grid import as_grid
from arc_r2.dsl import Program as DSLProgram
from arc_r2.search import SearchLimits
from arc_r2.staged_engine import solve_staged


class StagedEngineTests(unittest.TestCase):
    def test_stage_a_wins_and_test_input_is_only_predicted_after_fit(self):
        source = as_grid([[0, 1], [0, 0]])
        target = as_grid([[1, 0], [0, 0]])
        result = solve_staged(
            [(source, target)], [source],
            total_candidate_budget=5000, stage_a_cap=5000,
        )
        self.assertEqual(result.stage, "A")
        self.assertEqual(result.status, "solved_on_training")
        self.assertEqual(result.predictions[0], target)
        self.assertEqual([s.stage for s in result.stages], ["A"])

    def test_stage_b_runs_only_after_stage_a_fails(self):
        source = as_grid([[0, 1], [0, 0]])
        target = as_grid([[1, 0], [0, 0]])
        # Force A to exhaust a one-candidate budget before it can reach the
        # horizontal reversal primitive; B then gets the remaining budget.
        result = solve_staged(
            [(source, target)], [source],
            total_candidate_budget=100, stage_a_cap=1,
            limits=SearchLimits(beam_width=20, max_depth=1, candidate_cap=99,
                                mcts_iterations=50, seed=0, include_prte=False),
        )
        self.assertEqual(result.stages[0].stage, "A")
        self.assertEqual(result.stages[0].status, "no_exact_fit")
        self.assertEqual(result.stages[1].stage, "B")
        self.assertEqual(result.stage, "B")
        self.assertEqual(result.predictions[0], target)

    def test_shared_budget_never_exceeded(self):
        source = as_grid([[0, 1], [0, 0]])
        target = as_grid([[1, 0], [0, 0]])
        result = solve_staged(
            [(source, target)], [],
            total_candidate_budget=1, stage_a_cap=1,
            limits=SearchLimits(max_depth=4, candidate_cap=1),
        )
        self.assertLessEqual(result.candidates_evaluated, 1)
        self.assertEqual(result.stages[1].status, "skipped_no_budget")


    def test_stage_c_runs_only_after_stage_b_fails(self):
        source = as_grid([[0, 1], [0, 0]])
        target = as_grid([[1, 0], [0, 0]])

        def fake_beam(score, limits, stats, state_key=None):
            stats.evaluated = 1
            return []

        def fake_mcts(score, limits, stats, state_key=None):
            stats.evaluated = 1
            program = DSLProgram().then("REVERSE.HORIZONTAL")
            return [(score(program), program)]

        with patch("arc_r2.staged_engine.beam_search", side_effect=fake_beam) as beam:
            with patch("arc_r2.staged_engine.mcts_search", side_effect=fake_mcts) as mcts:
                result = solve_staged(
                    [(source, target)], [source],
                    total_candidate_budget=3, stage_a_cap=1,
                    limits=SearchLimits(beam_width=2, max_depth=1, candidate_cap=3,
                                        mcts_iterations=2, seed=0),
                )

        self.assertEqual(result.status, "solved_on_training")
        self.assertEqual(result.stage, "C")
        self.assertEqual([stage.stage for stage in result.stages], ["A", "B", "C"])
        self.assertEqual(result.predictions[0], target)
        self.assertEqual(result.candidates_evaluated, 3)
        beam.assert_called_once()
        mcts.assert_called_once()



if __name__ == "__main__":
    unittest.main()

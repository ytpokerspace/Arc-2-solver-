import unittest

from arc_r2.dsl import Program
from arc_r2.hian import analyze_hypotheses, evaluate_hypothesis
from arc_r2.grid import as_grid


class HIANTests(unittest.TestCase):
    def test_exact_evidence_supports_hypothesis(self):
        source = as_grid([[1, 0], [0, 0]])
        expected = as_grid([[0, 1], [0, 0]])
        hypothesis, evidence = evaluate_hypothesis(
            Program().then("REVERSE.HORIZONTAL"),
            [("train-0", source, expected)],
        )
        self.assertEqual(hypothesis.exact_matches, 1)
        self.assertEqual(hypothesis.supporting_sources, ("train-0",))
        self.assertEqual(hypothesis.contradicting_sources, ())
        self.assertTrue(evidence[0].exact_match)

    def test_contradictory_examples_are_explicit(self):
        source_a = as_grid([[1, 0]])
        expected_a = as_grid([[1, 0]])
        source_b = as_grid([[1, 0]])
        expected_b = as_grid([[0, 1]])
        hypothesis, _ = evaluate_hypothesis(
            Program(),
            [
                ("train-a", source_a, expected_a),
                ("train-b", source_b, expected_b),
            ],
        )
        self.assertEqual(hypothesis.supporting_sources, ("train-a",))
        self.assertEqual(hypothesis.contradicting_sources, ("train-b",))
        self.assertEqual(hypothesis.exact_matches, 1)

    def test_hierarchy_groups_candidates_and_selects_exact_fit(self):
        source = as_grid([[1, 0]])
        expected = as_grid([[0, 1]])
        report = analyze_hypotheses(
            [Program(), Program().then("REVERSE.HORIZONTAL")],
            [("train-0", source, expected)],
        )
        self.assertEqual(report["candidate_count"], 2)
        self.assertEqual(
            report["selected_hypothesis"]["program"],
            ["REVERSE.HORIZONTAL"],
        )
        self.assertIn("IDENTITY", report["hierarchy"])
        self.assertIn("REVERSE.HORIZONTAL", report["hierarchy"])

    def test_empty_candidates_and_invalid_top_k_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one candidate"):
            analyze_hypotheses([], [])
        with self.assertRaisesRegex(ValueError, "top_k"):
            analyze_hypotheses([Program()], [], top_k=0)


if __name__ == "__main__":
    unittest.main()

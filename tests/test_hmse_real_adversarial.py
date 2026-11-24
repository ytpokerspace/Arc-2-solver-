"""Real, unmocked adversarial tests for HMSE rule selection.

These tests use the registered candidate implementations and the actual
selector. They deliberately separate fitting visible examples from predicting
a held-out output. They are controls, not ARC benchmark accuracy claims.
"""
import unittest

from arc_r2.hmse_rule_selector import _select_rule
from arc_r2.shape_radar import scan_grid


def select_from_visible_pairs(pairs):
    perception = tuple(scan_grid(source, connectivity=4) for source, _ in pairs)
    return _select_rule(
        {"training_pairs": tuple(pairs), "perception": perception},
        {},
    )


class HmseRealAdversarialTests(unittest.TestCase):
    def setUp(self):
        # Two examples support left-right reflection. A third example uses
        # top-bottom reflection. The third output is never shown during fit.
        self.left_right_a = (
            ((1, 0, 0), (0, 2, 0), (0, 0, 0)),
            ((0, 0, 1), (0, 2, 0), (0, 0, 0)),
        )
        self.left_right_b = (
            ((0, 3, 0), (0, 0, 0), (4, 0, 0)),
            ((0, 3, 0), (0, 0, 0), (0, 0, 4)),
        )
        self.top_bottom_c = (
            ((5, 0, 0), (0, 0, 6), (0, 0, 0)),
            ((0, 0, 0), (0, 0, 6), (5, 0, 0)),
        )

    def test_two_visible_examples_can_overfit_a_held_out_counterexample(self):
        examples = (self.left_right_a, self.left_right_b, self.top_bottom_c)
        selected = 0
        abstained = 0
        exact = 0
        outcomes = []

        for held_index, held_pair in enumerate(examples):
            visible = tuple(pair for i, pair in enumerate(examples) if i != held_index)
            result = select_from_visible_pairs(visible)
            if result.selected_program is None:
                abstained += 1
                prediction = None
            else:
                selected += 1
                prediction = result.selected_program.apply(held_pair[0])
            matched = prediction == held_pair[1]
            exact += int(matched)
            outcomes.append((result.status, result.selected_rule, matched))

        # Only the fold that hides the contradictory example selects
        # left-right reflection; that prediction is confidently wrong.
        self.assertEqual(selected, 1)
        self.assertEqual(abstained, 2)
        self.assertEqual(exact, 0)
        self.assertEqual(
            outcomes,
            [
                ("no-rule-family-exact-fit", None, False),
                ("no-rule-family-exact-fit", None, False),
                ("selected-unique-exact-fit", "reflection", False),
            ],
        )

    def test_symmetric_evidence_does_not_choose_a_reflection_axis(self):
        symmetric = (
            ((1, 0, 1), (0, 2, 0), (1, 0, 1)),
            ((1, 0, 1), (0, 2, 0), (1, 0, 1)),
        )
        result = select_from_visible_pairs((symmetric,))
        reflection = next(
            candidate for candidate in result.candidates
            if candidate.rule_id == "reflection"
        )
        self.assertFalse(reflection.accepted)
        self.assertEqual(reflection.reason, "reflection-axis-ambiguous")
        self.assertIsNone(result.selected_rule)
        self.assertIsNone(result.selected_program)


if __name__ == "__main__":
    unittest.main()

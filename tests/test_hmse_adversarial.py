"""Adversarial checks for conservative HMSE rule selection.

These tests target ambiguity and unsupported evidence, not benchmark accuracy.
"""
import unittest
from unittest.mock import patch

from arc_r2.hmse_rule_selector import (
    RuleCandidate,
    _select_rule,
)


class HmseAdversarialSelectionTests(unittest.TestCase):
    def setUp(self):
        self.state = {
            "training_pairs": (((1, 0), (1, 1)),),
            "perception": (),
        }

    def candidates(self, accepted_ids=()):
        ids = (
            "hole-fill",
            "anchor-quadrant-relocation",
            "reflection",
            "latin-square-completion",
        )
        return {
            rule_id: RuleCandidate(
                rule_id, object() if rule_id in accepted_ids else None,
                rule_id in accepted_ids,
                "exact-fit" if rule_id in accepted_ids else "rejected",
                1,
            )
            for rule_id in ids
        }

    def select_with(self, accepted_ids):
        candidates = self.candidates(accepted_ids)
        with (
            patch("arc_r2.hmse_rule_selector._hole_fill_candidate",
                  return_value=candidates["hole-fill"]),
            patch("arc_r2.hmse_rule_selector._spatial_candidate",
                  return_value=candidates["anchor-quadrant-relocation"]),
            patch("arc_r2.hmse_rule_selector._reflection_candidate",
                  return_value=candidates["reflection"]),
            patch("arc_r2.hmse_rule_selector._latin_square_candidate",
                  return_value=candidates["latin-square-completion"]),
        ):
            return _select_rule(self.state, {})

    def test_two_plausible_exact_families_force_abstention(self):
        result = self.select_with({"hole-fill", "reflection"})
        self.assertEqual(result.status, "ambiguous-multiple-exact-fits")
        self.assertIsNone(result.selected_rule)
        self.assertIsNone(result.selected_program)

    def test_all_families_rejected_force_abstention(self):
        result = self.select_with(set())
        self.assertEqual(result.status, "no-rule-family-exact-fit")
        self.assertIsNone(result.selected_rule)
        self.assertIsNone(result.selected_program)

    def test_one_exact_family_is_selected(self):
        result = self.select_with({"latin-square-completion"})
        self.assertEqual(result.status, "selected-unique-exact-fit")
        self.assertEqual(result.selected_rule, "latin-square-completion")
        self.assertIsNotNone(result.selected_program)

    def test_conflicting_training_examples_cannot_be_hidden_by_one_fit(self):
        from arc_r2.hmse_latin_square_rule import propose_latin_square_completion

        pairs = (
            ([[0, 4, 2, 3], [4, 1, 0, 2], [0, 3, 4, 0], [3, 0, 1, 4]],
             [[1, 4, 2, 3], [4, 1, 3, 2], [2, 3, 4, 1], [3, 2, 1, 4]]),
            ([[0, 4, 2, 3], [4, 1, 0, 2], [0, 3, 4, 0], [3, 0, 1, 4]],
             [[1, 4, 2, 3], [4, 1, 3, 2], [2, 3, 4, 1], [3, 2, 1, 3]]),
        )
        program, reason = propose_latin_square_completion(pairs)
        self.assertIsNone(program)
        self.assertEqual(reason, "existing-cell-changed")


if __name__ == "__main__":
    unittest.main()

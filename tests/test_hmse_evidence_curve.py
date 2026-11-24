"""Evidence-count experiment using a pinned ARC-AGI-2 training task.

For each evidence level, the selector sees only a prefix of training pairs.
Remaining inputs are predicted afterward; their outputs are consulted only by
the outer test assertions. This is a small task-level control, not a benchmark.
"""
import json
import unittest
from pathlib import Path

from arc_r2.hmse_rule_selector import _select_rule
from arc_r2.shape_radar import scan_grid


def select_from_visible_pairs(visible_pairs):
    perception = tuple(
        scan_grid(source, connectivity=4)
        for source, _ in visible_pairs
    )
    return _select_rule(
        {
            "training_pairs": tuple(visible_pairs),
            "perception": perception,
        },
        {},
    )


class HmseEvidenceCurveTests(unittest.TestCase):
    def setUp(self):
        fixture = (
            Path(__file__).parent
            / "fixtures"
            / "arc_agi2_4cd1b7b2_train.json"
        )
        self.task = json.loads(fixture.read_text(encoding="utf-8"))
        self.pairs = tuple(
            (pair["input"], pair["output"])
            for pair in self.task["train"]
        )

    def test_prefix_evidence_curve_on_pinned_latin_square_task(self):
        # At each level, the selector receives only the visible prefix.
        # Held-out outputs are deliberately not included in selector state.
        expected = {
            1: {"held_out": 2, "selected": 2, "exact": 2},
            2: {"held_out": 1, "selected": 1, "exact": 1},
        }
        observed = {}

        for evidence_count in (1, 2):
            visible = self.pairs[:evidence_count]
            held_out = self.pairs[evidence_count:]
            result = select_from_visible_pairs(visible)

            self.assertEqual(result.status, "selected-unique-exact-fit")
            self.assertEqual(result.selected_rule, "latin-square-completion")
            self.assertEqual(len(held_out), expected[evidence_count]["held_out"])

            predictions = [
                result.selected_program.apply(held_input)
                for held_input, _held_output in held_out
            ]
            exact = sum(
                prediction == held_output
                for prediction, (_held_input, held_output)
                in zip(predictions, held_out)
            )
            observed[evidence_count] = {
                "held_out": len(held_out),
                "selected": len(predictions),
                "exact": exact,
            }

        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()

    def test_extra_consistent_example_does_not_reveal_hidden_exception(self):
        # Synthetic adversarial control: A and B support left-right reflection;
        # C is a hidden top-bottom exception. This is not an ARC task score.
        examples = (
            (
                ((1, 0, 0), (0, 2, 0), (0, 0, 0)),
                ((0, 0, 1), (0, 2, 0), (0, 0, 0)),
            ),
            (
                ((0, 3, 0), (0, 0, 0), (4, 0, 0)),
                ((0, 3, 0), (0, 0, 0), (0, 0, 4)),
            ),
            (
                ((5, 0, 0), (0, 0, 6), (0, 0, 0)),
                ((0, 0, 0), (0, 0, 6), (5, 0, 0)),
            ),
        )
        expected = {
            1: {"held_out": 2, "selected": 2, "exact": 1},
            2: {"held_out": 1, "selected": 1, "exact": 0},
        }
        observed = {}

        for evidence_count in (1, 2):
            visible = examples[:evidence_count]
            held_out = examples[evidence_count:]
            result = select_from_visible_pairs(visible)

            self.assertEqual(result.status, "selected-unique-exact-fit")
            self.assertEqual(result.selected_rule, "reflection")
            predictions = [
                result.selected_program.apply(held_input)
                for held_input, _held_output in held_out
            ]
            exact = sum(
                prediction == held_output
                for prediction, (_held_input, held_output)
                in zip(predictions, held_out)
            )
            observed[evidence_count] = {
                "held_out": len(held_out),
                "selected": len(predictions),
                "exact": exact,
            }

        self.assertEqual(observed, expected)

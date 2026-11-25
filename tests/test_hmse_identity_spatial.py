"""Tests for reversible identity-aware removal and HMSE spatial execution."""
import json
import unittest
from pathlib import Path

from arc_r2.object_identity_dsl import IdentityInstruction, IdentityObjectProgram
from arc_r2.hmse_identity_spatial import run_comparison


class IdentityRemovalTests(unittest.TestCase):
    def setUp(self):
        self.grid = (
            (0, 0, 5, 0, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 0, 8, 8, 0, 0),
            (0, 0, 8, 8, 0, 0),
            (0, 0, 0, 0, 0, 0),
            (0, 2, 0, 0, 0, 0),
        )

    def test_remove_preserves_identity_and_inverse_restores_exact_grid(self):
        initial = IdentityObjectProgram.initialize(self.grid)
        anchor_id = next(
            object_id for object_id, obj in initial.objects if obj.size == 4
        )
        program = IdentityObjectProgram((IdentityInstruction(anchor_id, "REMOVE"),))
        removed = program.apply_state(initial)
        self.assertNotIn(anchor_id, removed.active_object_ids())
        self.assertIn(anchor_id, removed.object_ids())
        self.assertEqual(len(removed.archived_objects), 1)
        restored = program.inverse().apply_state(removed)
        self.assertEqual(restored.to_grid(), self.grid)
        self.assertEqual(restored.object_ids(), initial.object_ids())

    def test_remove_then_move_markers_and_inverse_restores_grid(self):
        initial = IdentityObjectProgram.initialize(self.grid)
        anchor_id, anchor = next(
            (object_id, obj) for object_id, obj in initial.objects if obj.size == 4
        )
        marker_id, marker = next(
            (object_id, obj) for object_id, obj in initial.objects if obj.size == 1
        )
        row, col = next(iter(marker.cells))
        top, left = anchor.bbox[:2]
        program = IdentityObjectProgram((
            IdentityInstruction(anchor_id, "REMOVE"),
            IdentityInstruction(marker_id, "MOVE",
                                delta_row=top-row, delta_col=left-col),
        ))
        transformed = program.apply_state(initial)
        self.assertEqual(len(transformed.archived_objects), 1)
        self.assertEqual(program.inverse().apply_state(transformed).to_grid(), self.grid)

    def test_three_way_arc_training_fold_comparison(self):
        fixture = Path(__file__).parent / "fixtures" / "arc_agi2_d89b689b_train.json"
        task = json.loads(fixture.read_text(encoding="utf-8"))
        report = run_comparison(task)
        self.assertEqual(report["protocol"], "leave-one-training-pair-out")
        self.assertEqual(report["training_pairs"], 3)
        self.assertEqual(report["results"]["hmse"]["held_out_exact"], 3)
        self.assertEqual(report["results"]["hybrid"]["held_out_exact"], 3)
        for fold in report["fold_results"]:
            self.assertTrue(all(
                item["inverse_restores_input"]
                for item in fold["hybrid_visible_pair_evidence"]
            ))


if __name__ == "__main__":
    unittest.main()

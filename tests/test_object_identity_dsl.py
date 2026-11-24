"""Tests for stable object identity across composed transformations."""
import unittest

from arc_r2.object_identity_dsl import (
    IdentityInstruction,
    IdentityObjectProgram,
)


class IdentityObjectDSLTests(unittest.TestCase):
    def setUp(self):
        self.source = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 0, 2, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    def test_identity_survives_spatial_reordering_and_inverse(self):
        state = IdentityObjectProgram.initialize(self.source)
        self.assertEqual(state.object_ids(), ("obj-0000", "obj-0001"))

        program = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=5),
            IdentityInstruction("obj-0000", "ROTATE90", turns=1),
        ))
        transformed_state = program.apply_state(state)
        self.assertEqual(transformed_state.object_ids(), state.object_ids())

        # The objects have exchanged left-to-right order, but IDs did not change.
        positions = {
            object_id: obj.bbox[1] for object_id, obj in transformed_state.objects
        }
        self.assertGreater(positions["obj-0000"], positions["obj-0001"])

        restored = program.inverse().apply_state(transformed_state)
        self.assertEqual(restored.to_grid(), self.source)
        self.assertEqual(restored.object_ids(), state.object_ids())

    def test_identity_mode_matches_existing_single_object_move(self):
        source = ((0, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 0))
        program = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=1),
        ))
        self.assertEqual(
            program.apply(source),
            ((0, 0, 0, 0), (0, 0, 1, 0), (0, 0, 0, 0)),
        )

    def test_ambiguous_or_missing_identity_fails_closed(self):
        state = IdentityObjectProgram.initialize(((0, 0, 0), (0, 1, 0), (0, 0, 0)))
        program = IdentityObjectProgram((
            IdentityInstruction("obj-does-not-exist", "MOVE", delta_col=1),
        ))
        with self.assertRaisesRegex(ValueError, "unknown object identity"):
            program.apply_state(state)

    def test_collisions_and_grid_bounds_fail_closed(self):
        state = IdentityObjectProgram.initialize(self.source)
        collision = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=2),
        ))
        with self.assertRaisesRegex(ValueError, "collide"):
            collision.apply_state(state)

        outside = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=-5),
        ))
        with self.assertRaisesRegex(ValueError, "leave the grid"):
            outside.apply_state(state)

    def test_raw_grid_inverse_is_not_silently_treated_as_identity_state(self):
        state = IdentityObjectProgram.initialize(self.source)
        program = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=5),
        ))
        transformed = program.apply_state(state)
        with self.assertRaisesRegex(TypeError, "IdentityState"):
            program.inverse().apply_state(transformed.to_grid())

    def test_noninvertible_operations_are_rejected_by_inverse(self):
        for operation in (
            IdentityInstruction("obj-0000", "SCALE", factor=2),
            IdentityInstruction("obj-0000", "PLACE", top=1, left=1),
        ):
            with self.subTest(operation=operation.operation):
                with self.assertRaises(ValueError):
                    operation.inverse()


if __name__ == "__main__":
    unittest.main()

"""Focused tests for identity-preserving recolor and program composition."""
import unittest

from arc_r2.object_identity_dsl import (
    IdentityInstruction,
    IdentityObjectProgram,
    IdentityState,
)
from arc_r2.semantic_identity import match_identities


class IdentityCompositionTests(unittest.TestCase):
    def setUp(self):
        self.reference_grid = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )

    def test_identity_survives_move_rotation_reflection_and_recolor_inverse(self):
        state = IdentityObjectProgram.initialize(self.reference_grid)
        object_id = state.object_ids()[0]
        program = IdentityObjectProgram((
            IdentityInstruction(object_id, "MOVE", delta_col=4),
            IdentityInstruction(object_id, "ROTATE90", turns=1),
            IdentityInstruction(object_id, "REFLECT", axis="horizontal"),
            IdentityInstruction(
                object_id, "RECOLOR", color=4, previous_color=1,
            ),
        ))

        transformed = program.apply_state(state)
        self.assertEqual(transformed.object_ids(), state.object_ids())
        transformed_object = dict(transformed.objects)[object_id]
        self.assertEqual(transformed_object.color, 4)

        restored = program.inverse().apply_state(transformed)
        self.assertEqual(restored.to_grid(), self.reference_grid)
        self.assertEqual(restored.object_ids(), state.object_ids())

    def test_recolor_fails_closed_when_expected_previous_color_is_wrong(self):
        state = IdentityObjectProgram.initialize(self.reference_grid)
        object_id = state.object_ids()[0]
        program = IdentityObjectProgram((
            IdentityInstruction(
                object_id, "RECOLOR", color=4, previous_color=3,
            ),
        ))
        with self.assertRaisesRegex(ValueError, "RECOLOR expected color"):
            program.apply_state(state)

    def test_two_independent_programs_compose_in_order(self):
        state = IdentityObjectProgram.initialize(self.reference_grid)
        object_id = state.object_ids()[0]
        geometry_program = IdentityObjectProgram((
            IdentityInstruction(object_id, "MOVE", delta_col=1),
        ))
        appearance_program = IdentityObjectProgram((
            IdentityInstruction(object_id, "ROTATE90", turns=1),
            IdentityInstruction(
                object_id, "RECOLOR", color=4, previous_color=1,
            ),
        ))

        composed = geometry_program.compose(appearance_program)
        chained = geometry_program.then(
            IdentityInstruction(object_id, "ROTATE90", turns=1)
        ).then(
            IdentityInstruction(
                object_id, "RECOLOR", color=4, previous_color=1,
            )
        )
        self.assertEqual(composed.canonical(), chained.canonical())
        self.assertEqual(composed.complexity, 3)
        self.assertEqual(
            composed.inverse().apply_state(composed.apply_state(state)).to_grid(),
            self.reference_grid,
        )

    def test_composed_program_transfers_after_object_order_changes(self):
        reference = IdentityObjectProgram.initialize(self.reference_grid)
        reference_ids = reference.object_ids()
        self.assertEqual(reference_ids, ("obj-0000", "obj-0001"))

        # Same two objects, but their scan order is reversed and both are relocated.
        heldout_grid = (
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )
        # Place the reference L-shape at the bottom-right of the held-out grid.
        heldout_rows = [list(row) for row in heldout_grid]
        heldout_rows[16][10] = 1
        heldout_rows[16][11] = 1
        heldout_rows[17][10] = 1
        heldout_grid = tuple(tuple(row) for row in heldout_rows)

        heldout = IdentityObjectProgram.initialize(heldout_grid)
        match = match_identities(reference, heldout_grid)
        self.assertEqual(len(match.matches), 2)
        target_objects = [obj for _, obj in heldout.objects]
        matched_objects = {
            item.object_id: target_objects[item.target_index]
            for item in match.matches
        }
        aligned = IdentityState(
            heldout.height,
            heldout.width,
            heldout.background,
            tuple((object_id, matched_objects[object_id]) for object_id in reference_ids),
        )

        first = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "MOVE", delta_col=1),
        ))
        second = IdentityObjectProgram((
            IdentityInstruction("obj-0000", "ROTATE90", turns=1),
            IdentityInstruction(
                "obj-0000", "RECOLOR", color=4, previous_color=1,
            ),
        ))
        transferred = first.compose(second).apply_state(aligned)
        self.assertEqual(transferred.object_ids(), reference_ids)
        self.assertEqual(dict(transferred.objects)["obj-0000"].color, 4)


if __name__ == "__main__":
    unittest.main()

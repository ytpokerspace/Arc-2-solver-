"""Tests for conservative semantic identity matching."""
import unittest
from arc_r2.object_identity_dsl import IdentityObjectProgram
from arc_r2.semantic_identity import match_identities

class SemanticIdentityTests(unittest.TestCase):
    def test_matches_distinct_objects_after_spatial_reordering(self):
        reference_grid = (
            (0,0,0,0,0,0,0,0,0),
            (0,1,0,2,2,0,0,0,0),
            (0,0,0,2,0,0,0,0,0),
            (0,0,0,0,0,0,0,0,0),
        )
        target_grid = (
            (0,0,0,0,0,0,0,0,0),
            (0,0,0,0,0,0,0,1,0),
            (0,0,0,0,0,0,0,0,0),
            (0,2,2,0,0,0,0,0,0),
            (0,2,0,0,0,0,0,0,0),
        )
        ref = IdentityObjectProgram.initialize(reference_grid)
        result = match_identities(ref, target_grid)
        by_id = {m.object_id: m.target_index for m in result.matches}
        self.assertEqual(set(by_id), {"obj-0000", "obj-0001"})
        self.assertEqual(by_id["obj-0000"], 1)

    def test_identical_candidates_are_reported_ambiguous(self):
        grid = (
            (0,0,0,0,0,0,0),
            (0,1,0,0,0,0,0),
            (0,0,0,0,0,0,0),
            (0,0,0,0,1,0,0),
            (0,0,0,0,0,0,0),
        )
        result = match_identities(IdentityObjectProgram.initialize(grid), grid)
        self.assertEqual(result.matches, ())
        self.assertEqual(set(result.ambiguous_reference), {"obj-0000", "obj-0001"})

    def test_no_candidate_abstains(self):
        ref = IdentityObjectProgram.initialize(((0,0,0),(0,1,0),(0,0,0)))
        result = match_identities(ref, ((0,0,0),(0,2,2),(0,0,0)))
        self.assertEqual(result.matches, ())
        self.assertEqual(result.unmatched_reference, ("obj-0000",))

    def test_invalid_threshold_is_rejected(self):
        ref = IdentityObjectProgram.initialize(((0,0,0),(0,1,0),(0,0,0)))
        with self.assertRaises(ValueError):
            match_identities(ref, ((0,0,0),(0,1,0),(0,0,0)), threshold=1.1)

if __name__ == "__main__":
    unittest.main()

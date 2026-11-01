import unittest

from arc_r2.objects import GridObject
from arc_r2.object_ops import (
    move_object,
    normalize_object,
    object_size,
    place_object,
    reflect_object,
    rotate_object90,
    scale_object,
    shape_signature,
)


def make_object(cells, color=3):
    points = frozenset(cells)
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return GridObject(color, points, (min(rows), min(cols), max(rows), max(cols)))


class ObjectTransformTests(unittest.TestCase):
    def setUp(self):
        self.obj = make_object({(4, 7), (4, 8), (5, 7)})

    def test_normalize_and_shape_are_translation_invariant(self):
        shifted = move_object(self.obj, 10, -3)
        self.assertEqual(shape_signature(self.obj), ((0, 0), (0, 1), (1, 0)))
        self.assertEqual(shape_signature(self.obj), shape_signature(shifted))
        self.assertEqual(normalize_object(shifted).bbox, (0, 0, 1, 1))

    def test_size_descriptor(self):
        self.assertEqual(object_size(self.obj), (3, 2, 2))

    def test_move_and_absolute_placement(self):
        moved = move_object(self.obj, -2, 5)
        self.assertEqual(moved.bbox, (2, 12, 3, 13))
        placed = place_object(self.obj, 1, 2)
        self.assertEqual(placed.bbox, (1, 2, 2, 3))
        self.assertEqual(placed.color, self.obj.color)

    def test_scale_preserves_pattern_and_multiplies_area(self):
        scaled = scale_object(self.obj, 2)
        self.assertEqual(scaled.size, self.obj.size * 4)
        self.assertEqual(scaled.bbox, (0, 0, 3, 3))
        self.assertEqual(scaled.color, self.obj.color)

    def test_rotate_clockwise_and_inverse(self):
        once = rotate_object90(self.obj)
        self.assertEqual(once.bbox, (0, 0, 1, 1))
        self.assertEqual(rotate_object90(once, 3), normalize_object(self.obj))
        self.assertEqual(rotate_object90(self.obj, 4), normalize_object(self.obj))

    def test_reflections(self):
        self.assertEqual(
            shape_signature(reflect_object(self.obj, "horizontal")),
            ((0, 0), (0, 1), (1, 1)),
        )
        self.assertEqual(
            shape_signature(reflect_object(self.obj, "vertical")),
            ((0, 0), (1, 0), (1, 1)),
        )
        with self.assertRaises(ValueError):
            reflect_object(self.obj, "diagonal")

    def test_invalid_scale_rejected(self):
        for factor in (0, -1, True, 1.5):
            with self.subTest(factor=factor), self.assertRaises(ValueError):
                scale_object(self.obj, factor)


if __name__ == "__main__":
    unittest.main()

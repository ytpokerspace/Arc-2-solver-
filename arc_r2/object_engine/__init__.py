"""Object-centric ARC representation and transformations."""
from .model import ObjectState, Scene, extract_scene
from .operators import move, rotate90, flip_horizontal, flip_vertical, scale_up
from .renderer import render_scene
from .adapters import (
    transform_object, move_object, rotate_object, flip_object_horizontal,
    flip_object_vertical, scale_object_up, object_exists,
)

__all__ = [
    "ObjectState", "Scene", "extract_scene", "move", "rotate90",
    "flip_horizontal", "flip_vertical", "scale_up", "render_scene",
]

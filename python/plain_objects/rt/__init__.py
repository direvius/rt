from enum import Enum
from .vector import Vector3  # noqa
from .scene import Rgb, Camera, SceneBuilder  # noqa
from .optics import Material  # noqa


class Materials(Enum):
    METAL = Material(0.1, 0.9)

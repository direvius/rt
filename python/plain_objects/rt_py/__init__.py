from enum import Enum
from .vector import Vector3  # noqa
from .scene import Rgb, Camera, SceneBuilder  # noqa
from .optics import Material  # noqa


class Materials(Enum):
    METAL = Material(0.1, attenuation=Vector3(0.8, 0.8, 0.3))
    PLASTIC = Material(0.8, attenuation=Vector3(0.6, 0.4, 0.5))

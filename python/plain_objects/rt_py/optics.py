from typing import Protocol
from attrs import frozen
from .vector import Vector3
from random import random


@frozen
class Ray:
    """
    A class used to represent a ray with an origin and direction.

    Attributes:
    origin (Vector3): The origin point of the ray
    direction (Vector3): The direction vector of the ray (should be normalized)
    """

    origin: Vector3
    direction: Vector3

    def __attrs_post_init__(self):
        """
        Ensures the direction is normalized after initialization.
        """
        object.__setattr__(self, "direction", self.direction.normalize())

    def point_at(self, p: float) -> Vector3:
        """
        Returns the point at parameter p along the ray.

        Args:
        p (float): The parameter to compute the point at

        Returns:
        Vector3: The point at parameter 'p' along the ray
        """
        return self.origin + p * self.direction


@frozen
class CollideResult:
    r: Ray
    attenuation: Vector3


class Collider(Protocol):
    def collide(self, point: Vector3, normal: Vector3, r: Ray) -> CollideResult:
        ...


@frozen
class Material:
    """
    A class used to represent a material with properties such as fuzz and attenuation coefficients.

    Attributes:
    fuzz (float): The roughness of material
    attenuation (Vector3): The attenuation of each color component in vector form

    If you want to extend the functionality of this class, consider watching this videos:

    https://www.youtube.com/watch?v=HPNW0we-ft0
    https://www.youtube.com/watch?v=oa3Yo7Ro02A
    https://www.youtube.com/watch?v=R9iZzaXUaK4
    """

    fuzz: float
    attenuation: Vector3

    def collide(self, point: Vector3, normal: Vector3, r: Ray) -> CollideResult:
        collided = Ray(
            origin=point,
            direction=r.direction.reflect_direction(normal) + Vector3.random_in_unit_sphere() * self.fuzz)
        return CollideResult(collided, self.attenuation)

    @staticmethod
    def random():
        return Material(random(), Vector3(random(), random(), random()))

from attrs import frozen
from .math import Vector3
from random import choices


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
    color: Vector3

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
class Material:
    """
    A class used to represent a material with properties such as diffuse, reflection, and refraction coefficients.

    Attributes:
    diffuse (float): The diffuse coefficient of the material
    reflection (float): The reflection coefficient of the material
    refraction (float): The refraction coefficient of the material
    """

    diffuse: float
    reflection: float
    refraction: float

    @staticmethod
    def reflect_direction(v: Vector3, n: Vector3) -> Vector3:
        return v - 2 * n * v.dot(n)

    @staticmethod
    def refract_direction(v: Vector3, n: Vector3) -> Vector3:
        # TODO: rewrite
        return v + 2 * n * v.dot(n) * 0.7

    @staticmethod
    def diffuse_direction(v: Vector3, n: Vector3) -> Vector3:
        return Material.reflect_direction(v, n) + Vector3.random_in_unit_sphere()

    def collide(self, point: Vector3, normal: Vector3, r: Ray):
        weights = [
            self.diffuse,
            self.reflection,
            self.refraction,
        ]
        strategies = [
            Material.diffuse_direction,
            Material.reflect_direction,
            Material.refract_direction,
        ]

        # reflection
        collided = Ray(
            origin=point,
            direction=choices(strategies, weights)[0](r.direction, normal),
            color=r.color / 2
        )
        return collided

from attrs import frozen
from random import choices
from .vector import Vector3
from . import vector


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
        object.__setattr__(self, "direction", vector.normalize(self.direction))

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
    """

    diffuse: float
    reflection: float

    @staticmethod
    def reflect_direction(v: Vector3, n: Vector3) -> Vector3:
        return v - 2 * n * v.dot(n)

    @staticmethod
    def diffuse_direction(v: Vector3, n: Vector3) -> Vector3:
        return Material.reflect_direction(v, n) + vector.random_in_unit_sphere()

    def collide(self, point: Vector3, normal: Vector3, r: Ray) -> Ray:
        weights = [
            self.diffuse,
            self.reflection,
        ]
        strategies = [
            Material.diffuse_direction,
            Material.reflect_direction,
        ]

        # reflection
        collided = Ray(origin=point, direction=choices(strategies, weights)[0](r.direction, normal), color=r.color / 2)
        return collided

# https://www.youtube.com/watch?v=HPNW0we-ft0
# https://www.youtube.com/watch?v=oa3Yo7Ro02A
# https://www.youtube.com/watch?v=R9iZzaXUaK4

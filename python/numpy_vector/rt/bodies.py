from typing import Protocol
from attrs import frozen
from math import sqrt
from .vector import Vector3
from . import vector
from .optics import Material, Ray


@frozen
class HitResult:
    """
    A class used to represent the result of a ray hitting an object.

    Attributes:
    p (Vector3): The point where the ray hit the object
    n (Vector3): The normal vector at the point of hit
    t (float): The parameter at which the ray hit the object
    material (Material): The material of the hit object
    """

    p: Vector3
    n: Vector3
    t: float
    material: Material

    def collide(self, r: Ray) -> Ray:
        """
        Returns a new ray representing a reflection after collision.

        Args:
        r (Ray): The incoming ray that collided

        Returns:
        Ray: The reflected ray after collision
        """
        return self.material.collide(self.p, self.n, r)


class Hitable(Protocol):
    """
    An interface that represents something a ray can hit.
    """

    def hit(self, r: Ray) -> HitResult | None:
        """
        Checks if a ray hits this object and returns the hit result.

        Args:
        r (Ray): The ray to check

        Returns:
        HitResult | None: The hit result if the ray hits, None otherwise
        """
        ...


@frozen
class Sphere:
    """
    A class used to represent a sphere that can be hit by a ray.

    Attributes:
    center (Vector3): The center of the sphere
    radius (float): The radius of the sphere
    material (Material): The material of the sphere
    """

    center: Vector3
    radius: float
    material: Material

    def hit(self, r: Ray) -> HitResult | None:
        """
        Calculates the hit result if a ray hits the sphere.

        Args:
        r (Ray): The ray to check

        Returns:
        HitResult | None: The hit result if the ray hits, None otherwise
        """
        oc = r.origin - self.center
        a = r.direction.dot(r.direction)
        b = oc.dot(r.direction) * 2.0
        c = oc.dot(oc) - self.radius**2
        d = b * b - a * c * 4.0
        if d >= 0.0:
            t = (-b - sqrt(d)) / (a * 2.0)
            if t > 0.001:
                return HitResult(
                    p=r.point_at(t),
                    n=vector.normalize(r.point_at(t) - self.center),
                    t=t,
                    material=self.material,
                )
            else:
                return None
        else:
            return None

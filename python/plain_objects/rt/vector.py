from __future__ import annotations
from random import random
from typing import Iterator
from attrs import frozen
import math


@frozen
class Vector3:
    """
    A class used to represent a three-dimensional vector.

    Attributes:
    x (float): x-coordinate of the vector
    y (float): y-coordinate of the vector
    z (float): z-coordinate of the vector
    """

    x: float
    y: float
    z: float

    def __add__(self, other):
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __abs__(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __mul__(self, other):
        other = float(other)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other: float):
        other = float(other)
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __iter__(self) -> Iterator[float]:
        return iter((self.x, self.y, self.z))

    def normalize(self) -> Vector3:
        return self / abs(self)

    def dot(self, other: Vector3) -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    @staticmethod
    def random_in_unit_sphere() -> Vector3:
        while True:
            v = 2 * Vector3(random(), random(), random()) - Vector3(1, 1, 1)
            if abs(v) < 1:
                return v

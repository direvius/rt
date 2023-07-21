from __future__ import annotations
from typing import Iterator, Self
import png  # type: ignore
import functools as ft
import math
from attrs import frozen, field
from typing import Protocol


@frozen
class Vector3:
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


@frozen
class Rgb:
    r: int
    g: int
    b: int

    def __iter__(self) -> Iterator[int]:
        return iter((self.r, self.g, self.b))

    @staticmethod
    def from_vector3(v: Vector3) -> Rgb:
        v = v * 255
        return Rgb(int(v.x), int(v.y), int(v.z))


@frozen
class Material:
    diffuse: float
    reflection: float
    refraction: float


METAL = Material(0.1, 0.9, 0)


@frozen
class Ray:
    origin: Vector3
    direction: Vector3

    def __attrs_post_init__(self):
        object.__setattr__(self, "direction", self.direction.normalize())

    def point_at(self, p: float) -> Vector3:
        return self.origin + p * self.direction


@frozen
class HitResult:
    p: Vector3
    n: Vector3
    t: float
    material: Material

    @staticmethod
    def reflect_direction(v: Vector3, n: Vector3) -> Vector3:
        return v - 2 * n * v.dot(n)

    def collide(self, r: Ray) -> Ray:
        return Ray(origin=self.p, direction=self.reflect_direction(r.direction, self.n))


class Hitable(Protocol):
    def hit(self, r: Ray) -> HitResult | None:
        ...


@frozen
class Camera:
    geometry: Hitable
    width: int = 400
    height: int = 200
    ex: Vector3 = Vector3(1e-2, 0, 0)
    ey: Vector3 = Vector3(0, 1e-2, 0)
    origin: Vector3 = Vector3(0, 0, 0)
    center: Vector3 = Vector3(0, 0, -2)

    @property
    def upper_left(self) -> Vector3:
        return self.center - self.ex * (self.width / 2) + self.ey * (self.height / 2)

    def get_ray(self, i: int, j: int) -> Ray:
        return Ray(self.origin, self.upper_left + self.ex * i - self.ey * j)

    def color(self, r: Ray) -> Vector3:
        if hr := self.geometry.hit(r):
            return self.color(hr.collide(r)) / 1.2
        else:
            return self.env_color(r)

    def render(self) -> list[list[tuple[int, int, int]]]:
        img = [[(0, 0, 0) for _ in range(self.width)] for _ in range(self.height)]
        for iy in range(self.height):
            for ix in range(self.width):
                r = self.get_ray(ix, iy)
                img[iy][ix] = tuple(Rgb.from_vector3(self.color(r)))
        return img

    @staticmethod
    def env_color(r: Ray) -> Vector3:
        t = (r.direction.y + 1) * 0.5
        return (1 - t) * Vector3(1, 1, 1) + t * Vector3(0.5, 0.7, 1)

    def write_png(self):
        with open("test.png", "wb") as f:
            w = png.Writer(self.width, self.height, greyscale=False)  # type: ignore
            w.write(f, [ft.reduce(lambda a, b: a + b, row) for row in self.render()])  # type: ignore


@frozen
class Sphere:
    center: Vector3
    radius: float
    material: Material

    def hit(self, r: Ray) -> HitResult | None:
        oc = r.origin - self.center
        a = r.direction.dot(r.direction)
        b = oc.dot(r.direction) * 2.0
        c = oc.dot(oc) - self.radius**2
        d = b * b - a * c * 4.0
        if d >= 0.0:
            t = (-b - math.sqrt(d)) / (a * 2.0)
            if t > 0.001:
                return HitResult(
                    p=r.point_at(t),
                    n=(r.point_at(t) - self.center).normalize(),
                    t=t,
                    material=self.material,
                )
            else:
                return None
        else:
            return None


@frozen
class Scene:
    geometry: list[Hitable]

    def hit(self, r: Ray) -> HitResult | None:
        hits = [hr for g in self.geometry if (hr := g.hit(r))]
        if len(hits) > 0:
            return min(hits, key=lambda hr: hr.t)
        else:
            return None


@frozen
class SceneBuilder:
    geometry: list[Hitable] = field(factory=list)

    def add_geometry(self, g: Hitable) -> Self:
        self.geometry.append(g)
        return self

    def create(self) -> Scene:
        return Scene(self.geometry)


scene = (
    SceneBuilder()
    .add_geometry(Sphere(Vector3(0, 0, -4), 1, METAL))
    .add_geometry(Sphere(Vector3(0, -5, -4), 4, METAL))
    .create()
)
camera = Camera(scene)
camera.write_png()

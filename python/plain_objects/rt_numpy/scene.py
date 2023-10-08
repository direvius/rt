from __future__ import annotations
import functools as ft
import png  # type: ignore
from attrs import frozen, field
from typing import Iterator
from random import random
from .vector import Vector3
from .optics import Material, Ray, Collider
from .bodies import Hitable, HitResult, Sphere
import numpy as np


@frozen
class Rgb:
    """
    A class used to represent an RGB color.

    Attributes:
    r (int): Red component of the color
    g (int): Green component of the color
    b (int): Blue component of the color
    """

    r: int
    g: int
    b: int

    def __iter__(self) -> Iterator[int]:
        """
        Allows iteration over the color components.

        Returns:
        Iterator[int]: An iterator over the r, g, b components
        """
        return iter((self.r, self.g, self.b))

    @staticmethod
    def from_vector3(v: Vector3, gamma_correction: bool = False) -> Rgb:
        """
        Converts a Vector3 to an RGB color.

        Args:
        v (Vector3): The vector to convert

        Returns:
        Rgb: The RGB color corresponding to 'v'
        """
        if gamma_correction:
            return Rgb(*(np.sqrt(v) * 255).astype(np.int64))
        else:
            return Rgb(*(v * 255).astype(np.int64))


@frozen
class Camera:
    """
    A class used to represent a camera, with methods to generate rays and render the scene.

    Attributes:
    geometry (Hitable): The geometry to render
    width (int): The width of the image to render (in pixels)
    height (int): The height of the image to render (in pixels)
    ex (Vector3): The x-axis of the image plane (in world coordinates)
    ey (Vector3): The y-axis of the image plane (in world coordinates)
    origin (Vector3): The origin of the camera (in world coordinates)
    center (Vector3): The center of the image plane (in world coordinates)
    """

    geometry: Hitable
    width: int = 400
    height: int = 200
    ex: Vector3 = np.array([1e-2, 0, 0])
    ey: Vector3 = np.array([0, 1e-2, 0])
    origin: Vector3 = np.zeros(3)
    center: Vector3 = np.array([0, 0, -1])
    jitter_passes: int = 64

    @property
    def upper_left(self) -> Vector3:
        """
        Returns the position of the upper left corner of the image plane.

        Returns:
        Vector3: The position of the upper left corner of the image plane
        """
        return self.center - self.ex * (self.width / 2) + self.ey * (self.height / 2)

    def get_ray(self, i: int, j: int) -> Ray:
        """
        Generates a ray for pixel (i, j) on the image plane.

        Args:
        i (int): The x-coordinate of the pixel
        j (int): The y-coordinate of the pixel

        Returns:
        Ray: The ray corresponding to pixel (i, j)
        """
        return Ray(
            self.origin,
            self.upper_left + self.ex * (i + random() - 0.5) - self.ey * (j + random() - 0.5),
        )

    def trace(self, r: Ray, depth_limit: int = 32) -> Vector3:
        """
        Calculates the color of a ray by tracing it through the scene.

        Args:
        r (Ray): The ray to trace
        depth_limit (int): Maximum count of body interactions

        Returns:
        Vector3: The color of the ray after tracing
        """
        if hr := self.geometry.hit(r):
            if depth_limit > 0:
                collide_result = hr.collide(r)
                return self.trace(collide_result.r, depth_limit-1) * collide_result.attenuation
            else:
                return np.zeros(3)
        return self.env_color(r)

    def render(self) -> list[list[tuple[int, int, int]]]:
        """
        Renders the scene into an image.

        Returns:
        list[list[tuple[int, int, int]]]: The rendered image
        """
        img = [[(0, 0, 0) for _ in range(self.width)] for _ in range(self.height)]
        for iy in range(self.height):
            for ix in range(self.width):
                color = np.zeros(3)
                for _ in range(self.jitter_passes):
                    r = self.get_ray(ix, iy)
                    color += self.trace(r)
                img[iy][ix] = tuple(Rgb.from_vector3(color / self.jitter_passes, gamma_correction=False))  # type: ignore
        return img  # type: ignore

    @staticmethod
    def env_color(r: Ray) -> Vector3:
        """
        Calculates the environmental color for a ray.

        Args:
        r (Ray): The ray to calculate the color for

        Returns:
        Vector3: The environmental color of the ray
        """
        t = (r.direction[1] + 1) * 0.5
        env_color = (1 - t) * np.ones(3) + t * np.array([0.5, 0.7, 1])
        return env_color

    def write_png(self):
        """
        Writes the rendered image to a PNG file.
        """
        with open("test.png", "wb") as f:
            w = png.Writer(self.width, self.height, greyscale=False)  # type: ignore
            w.write(f, [ft.reduce(lambda a, b: a + b, row) for row in self.render()])  # type: ignore


@frozen
class Scene:
    """
    A class used to represent a collection of Hitable objects.

    Attributes:
    geometry (list[Hitable]): The list of hitable objects in the scene
    """

    geometry: list[Hitable]

    def hit(self, r: Ray) -> HitResult | None:
        """
        Calculates the first hit result for a ray in the scene.

        Args:
        r (Ray): The ray to check

        Returns:
        HitResult | None: The first hit result if the ray hits any object, None otherwise
        """
        hits = [hr for g in self.geometry if (hr := g.hit(r))]
        if len(hits) > 0:
            return min(hits, key=lambda hr: hr.t)
        else:
            return None


@frozen
class SceneBuilder:
    """
    A builder class for the Scene class.

    Attributes:
    geometry (list[Hitable]): The list of hitable objects to add to the scene
    """

    geometry: list[Hitable] = field(factory=list)

    def add_geometry(self, g: Hitable) -> SceneBuilder:
        """
        Adds a Hitable object to the scene.

        Args:
        g (Hitable): The hitable object to add

        Returns:
        SceneBuilder: This builder instance (for method chaining)
        """
        self.geometry.append(g)
        return self

    def add_sphere(self, cx: float, cy: float, cz: float, r: float, material: Collider | None = None) -> SceneBuilder:
        """
        Adds a Sphere object to the scene.

        Args:
        cx, cy, cz (float): coordinates of the center
        r (float): radius
        material (Material): sphere material

        Returns:
        SceneBuilder: This builder instance (for method chaining)
        """
        if material is None:
            material = Material.random()
        self.add_geometry(Sphere(np.array([cx, cy, cz]), r, material))
        return self

    def create(self) -> Scene:
        """
        Creates a Scene object from the current state of the builder.

        Returns:
        Scene: The created Scene object
        """
        return Scene(self.geometry)

    def camera(self, passes: int = 64) -> Camera:
        return Camera(self.create(), jitter_passes=64)

from __future__ import annotations
from typing import Iterator, Self
import png  # type: ignore
import functools as ft
import math
from attrs import frozen, field
from typing import Protocol


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
        """
        Overloads the '+' operator for vector addition.

        Args:
        other (Vector3): The other vector to be added

        Returns:
        Vector3: A new vector which is the sum of this vector and 'other'
        """
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other):
        """
        Overloads the '-' operator for vector subtraction.

        Args:
        other (Vector3): The other vector to be subtracted

        Returns:
        Vector3: A new vector which is the subtraction of this vector by 'other'
        """
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __abs__(self):
        """
        Overloads the 'abs' operator to get the magnitude of the vector.

        Returns:
        float: The magnitude of the vector
        """
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def __mul__(self, other):
        """
        Overloads the '*' operator for multiplication with a scalar.

        Args:
        other (float): The scalar to multiply with

        Returns:
        Vector3: A new vector which is the product of this vector and 'other'
        """
        other = float(other)
        return Vector3(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        """
        Overloads the '*' operator for multiplication with a scalar (reversed).

        Args:
        other (float): The scalar to multiply with

        Returns:
        Vector3: A new vector which is the product of this vector and 'other'
        """
        return self * other

    def __truediv__(self, other: float):
        """
        Overloads the '/' operator for division by a scalar.

        Args:
        other (float): The scalar to divide by

        Returns:
        Vector3: A new vector which is the division of this vector by 'other'
        """
        other = float(other)
        return Vector3(self.x / other, self.y / other, self.z / other)

    def __iter__(self) -> Iterator[float]:
        """
        Allows iteration over the vector components.

        Returns:
        Iterator[float]: An iterator over the x, y, z components
        """
        return iter((self.x, self.y, self.z))

    def normalize(self) -> Vector3:
        """
        Normalizes the vector (makes it have a magnitude of 1).

        Returns:
        Vector3: A normalized version of the vector
        """
        return self / abs(self)

    def dot(self, other: Vector3) -> float:
        """
        Computes the dot product with another vector.

        Args:
        other (Vector3): The other vector to compute the dot product with

        Returns:
        float: The dot product of this vector and 'other'
        """
        return self.x * other.x + self.y * other.y + self.z * other.z


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
    def from_vector3(v: Vector3) -> Rgb:
        """
        Converts a Vector3 to an RGB color.

        Args:
        v (Vector3): The vector to convert

        Returns:
        Rgb: The RGB color corresponding to 'v'
        """
        v = v * 255
        return Rgb(int(v.x), int(v.y), int(v.z))


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


METAL = Material(0.1, 0.9, 0)


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

    @staticmethod
    def reflect_direction(v: Vector3, n: Vector3) -> Vector3:
        """
        Calculates the reflection direction of a vector against a normal.

        Args:
        v (Vector3): The vector to reflect
        n (Vector3): The normal vector to reflect against

        Returns:
        Vector3: The reflection of 'v' against 'n'
        """
        return v - 2 * n * v.dot(n)

    def collide(self, r: Ray) -> Ray:
        """
        Returns a new ray representing a reflection after collision.

        Args:
        r (Ray): The incoming ray that collided

        Returns:
        Ray: The reflected ray after collision
        """
        return Ray(origin=self.p, direction=self.reflect_direction(r.direction, self.n))


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
    ex: Vector3 = Vector3(1e-2, 0, 0)
    ey: Vector3 = Vector3(0, 1e-2, 0)
    origin: Vector3 = Vector3(0, 0, 0)
    center: Vector3 = Vector3(0, 0, -2)

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
        return Ray(self.origin, self.upper_left + self.ex * i - self.ey * j)

    def color(self, r: Ray) -> Vector3:
        """
        Calculates the color of a ray by tracing it through the scene.

        Args:
        r (Ray): The ray to trace

        Returns:
        Vector3: The color of the ray after tracing
        """
        if hr := self.geometry.hit(r):
            return self.color(hr.collide(r)) / 1.2
        else:
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
                r = self.get_ray(ix, iy)
                img[iy][ix] = tuple(Rgb.from_vector3(self.color(r)))
        return img

    @staticmethod
    def env_color(r: Ray) -> Vector3:
        """
        Calculates the environmental color for a ray.

        Args:
        r (Ray): The ray to calculate the color for

        Returns:
        Vector3: The environmental color of the ray
        """
        t = (r.direction.y + 1) * 0.5
        return (1 - t) * Vector3(1, 1, 1) + t * Vector3(0.5, 0.7, 1)

    def write_png(self):
        """
        Writes the rendered image to a PNG file.
        """
        with open("test.png", "wb") as f:
            w = png.Writer(self.width, self.height, greyscale=False)  # type: ignore
            w.write(f, [ft.reduce(lambda a, b: a + b, row) for row in self.render()])  # type: ignore


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

    def add_geometry(self, g: Hitable) -> Self:
        """
        Adds a Hitable object to the scene.

        Args:
        g (Hitable): The hitable object to add

        Returns:
        SceneBuilder: This builder instance (for method chaining)
        """
        self.geometry.append(g)
        return self

    def create(self) -> Scene:
        """
        Creates a Scene object from the current state of the builder.

        Returns:
        Scene: The created Scene object
        """
        return Scene(self.geometry)


scene = (
    SceneBuilder()
    .add_geometry(Sphere(Vector3(0, 0, -4), 1, METAL))
    .add_geometry(Sphere(Vector3(0, -5, -4), 4, METAL))
    .create()
)
camera = Camera(scene)
camera.write_png()

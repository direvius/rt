from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore
from typing import Annotated, Literal
from attrs import frozen, field
from loguru import logger


Vector3 = Annotated[npt.NDArray[np.float64], Literal[3]]


@frozen
class Rays:
    origins: npt.NDArray
    directions: npt.NDArray


@frozen
class Spheres:
    centers: npt.NDArray
    radia: npt.NDArray[np.float64]
    fuzzes: npt.NDArray[np.float64]
    colors: npt.NDArray


@frozen
class Camera:
    scene: Spheres
    jitter_passes: int = 64
    width: int = 400
    height: int = 200

    def get_rays(self) -> Rays:
        upper_left = (
            np.array([0, 0, -1]) -
            np.array([1e-2, 0, 0]) * (self.width / 2) +
            np.array([0, 1e-2, 0]) * (self.height / 2)
        )

        # grid with jitter
        directions = (
            np.mgrid[0:self.width, 0:-self.height:-1].reshape(2, -1) +
            np.random.random((2, self.width*self.height))
        ) * 1e-2

        # third dimension and transition
        directions = np.r_[directions, np.zeros(self.width*self.height)[np.newaxis]].T + upper_left

        # normalization
        directions /= np.sqrt(np.einsum("ij, ij -> i", directions, directions))[:, np.newaxis]

        return Rays(np.zeros((self.width * self.height, 3)), directions)

    def write_png(self):
        logger.info("pass #1")
        rays = self.get_rays()
        env = hit(rays, self.scene)
        for i in range(self.jitter_passes - 1):
            logger.info("pass #{}", i+2)
            rays = self.get_rays()
            env += hit(rays, self.scene)
        env = env / self.jitter_passes
        plt.figure()
        plt.imshow(np.swapaxes(env.reshape(self.width, self.height, 3), 0, 1))
        plt.savefig("test.png")


@frozen
class SceneBuilder:

    centers: list[Vector3] = field(factory=list)
    radia: list[float] = field(factory=list)
    fuzzes: list[float] = field(factory=list)
    colors: list[Vector3] = field(factory=list)

    def add_sphere(self, cx: float, cy: float, cz: float, r: float) -> SceneBuilder:
        self.centers.append(np.array([cx, cy, cz]))
        self.radia.append(r)
        self.fuzzes.append(np.random.random())
        self.colors.append(np.random.random(3))
        return self

    def create(self) -> Spheres:
        return Spheres(
            np.array(self.centers),
            np.array(self.radia),
            np.array(self.fuzzes),
            np.array(self.colors)
        )

    def camera(self, width=400, height=200, jitter_passes=64) -> Camera:
        return Camera(self.create(), jitter_passes=jitter_passes, width=width, height=height)


def normalize(vector_batch: npt.NDArray) -> npt.NDArray:
    return vector_batch / np.sqrt(np.einsum("ij,ij->i", vector_batch, vector_batch))[:, np.newaxis]


def random_in_unit_sphere(length: int) -> npt.NDArray:
    dim = (length, 1)
    lengths = (np.random.random(size=dim) ** (1/3))
    u = np.random.random(size=dim)
    v = np.random.random(size=dim)
    ex = np.hstack([np.ones(dim), np.zeros(dim), np.zeros(dim)])
    ey = np.hstack([np.zeros(dim), np.ones(dim), np.zeros(dim)])
    ez = np.hstack([np.zeros(dim), np.ones(dim), np.zeros(dim)])
    theta = 2 * np.pi * u
    phi = np.arccos(2 * v - 1)
    sin_phi = np.sin(phi)
    sin_theta = np.sin(theta)
    cos_phi = np.cos(phi)
    cos_theta = np.cos(theta)
    return (
        ex * sin_phi * cos_theta +
        ey * sin_phi * sin_theta +
        ez * cos_phi
    ) * lengths


def env_colors(rays: Rays) -> npt.NDArray:
    t = (rays.directions[:, 1] + 1) * 0.5
    env_colors = (1 - t)[:, np.newaxis] * np.ones([len(rays.directions), 3]) + t[:, np.newaxis] * np.array([0.5, 0.7, 1])
    return env_colors


def hit(rays: Rays, spheres: Spheres, max_depth: int = 32) -> npt.NDArray:
    if max_depth <= 0:
        return np.zeros(rays.directions.shape)
    origins = rays.origins
    dim = len(origins)
    acc = np.full(dim, np.inf)
    indices = np.zeros(dim, dtype=int)
    normals = np.zeros((dim, 3))
    hitpoints = np.zeros((dim, 3))
    env = env_colors(rays)
    for i, (center, radius) in enumerate(zip(spheres.centers, spheres.radia)):
        oc = origins - center
        b = np.einsum("ij,ij->i", oc, rays.directions) * 2.0
        c = np.einsum("ij,ij->i", oc, oc) - radius**2
        d = b * b - c * 4.0
        hits = d > 0.0
        t = np.full(dim, np.inf)
        t[hits] = (-b[hits] - np.sqrt(d[hits])) / 2
        t[t < 1e-2] = np.inf
        should_update = t < acc
        acc[should_update] = t[should_update]
        indices[should_update] = i + 1
        hitpoints[should_update] = origins[should_update] + rays.directions[should_update] * t[should_update, np.newaxis]
        normals[should_update] = normalize(hitpoints[should_update] - center)
    updated = indices != 0
    if any(updated):
        env[updated] = hit(
            Rays(
                origins=hitpoints[updated],
                directions=fuzzy_reflect(
                    rays.directions[updated],
                    normals[updated],
                    spheres.fuzzes[indices[updated] - 1, np.newaxis])
            ),
            spheres=spheres,
            max_depth=max_depth - 1
        ) * spheres.colors[indices[updated] - 1]
    return env


def reflect(directions: npt.NDArray, normals: npt.NDArray) -> npt.NDArray:
    reflect_direction = (
        directions - 2 * normals *
        np.einsum("ij,ij->i", directions, normals)[:, np.newaxis]
    )
    return normalize(reflect_direction)


def fuzzy_reflect(directions: npt.NDArray, normals: npt.NDArray, fuzz: float | npt.NDArray = 0.7) -> npt.NDArray:
    reflect_direction = (
        directions -
        2 * normals *
        np.einsum("ij,ij->i", directions, normals)[:, np.newaxis] +
        random_in_unit_sphere(directions.shape[0]) * fuzz
    )
    return normalize(reflect_direction)

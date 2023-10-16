from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore
import time
from typing import Annotated, Literal
from attrs import frozen, field
from loguru import logger

from numba import njit  # type: ignore


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
        start_time = time.perf_counter()
        rays = self.get_rays()
        env = hit(rays.origins, rays.directions, self.scene.centers, self.scene.radia, self.scene.fuzzes, self.scene.colors)
        end_time = time.perf_counter()
        logger.info("first pass (with JIT) took {:3f} seconds", end_time - start_time)
        for i in range(self.jitter_passes - 1):
            logger.info("pass #{}", i+2)
            rays = self.get_rays()
            env += hit(rays.origins, rays.directions, self.scene.centers, self.scene.radia, self.scene.fuzzes, self.scene.colors)
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

    def camera(self, width=400, height=200) -> Camera:
        return Camera(self.create(), jitter_passes=64, width=width, height=height)


@njit
def normalize(vector_batch: npt.NDArray) -> npt.NDArray:
    return vector_batch / np.sqrt(dot_batch(vector_batch, vector_batch))[:, np.newaxis]


@njit
def random_in_unit_sphere(length: int) -> npt.NDArray:
    dim = (length, 1)
    lengths = (np.random.random(size=dim) ** (1/3))
    u = np.random.random(size=dim)
    v = np.random.random(size=dim)
    ex = np.hstack((np.ones(dim), np.zeros(dim), np.zeros(dim)))
    ey = np.hstack((np.zeros(dim), np.ones(dim), np.zeros(dim)))
    ez = np.hstack((np.zeros(dim), np.ones(dim), np.zeros(dim)))
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


@njit
def env_colors(directions) -> npt.NDArray:
    t = (directions[:, 1] + 1) * 0.5
    env_colors = (1 - t)[:, np.newaxis] * np.ones((len(directions), 3)) + t[:, np.newaxis] * np.array([0.5, 0.7, 1])
    return env_colors


@njit
def hit(origins, directions, centers, radia, fuzzes, colors, max_depth: int = 32) -> npt.NDArray:
    if max_depth <= 0:
        return np.zeros(directions.shape)
    directions = directions.copy()  # because it is more efficient to dot contigous arrays later
    dim = len(origins)
    acc = np.full(dim, np.inf)
    indices = np.zeros(dim, dtype=np.int64)
    normals = np.zeros((dim, 3))
    hitpoints = np.zeros((dim, 3))
    env = env_colors(directions)
    for i, (center, radius) in enumerate(zip(centers, radia)):
        oc = origins - center
        b = dot_batch(oc, directions) * 2.0
        c = dot_batch(oc, oc) - radius**2
        d = b * b - c * 4.0
        hits = d > 0.0
        t = np.full(dim, np.inf)
        t[hits] = (-b[hits] - np.sqrt(d[hits])) / 2
        t[t < 1e-2] = np.inf
        should_update = t < acc
        acc[should_update] = t[should_update]
        indices[should_update] = i + 1
        hitpoints[should_update] = origins[should_update] + directions[should_update] * t[should_update, np.newaxis]
        normals[should_update] = normalize(hitpoints[should_update] - center)
    updated = indices != 0
    if np.any(updated):
        env[updated] = hit(
            origins=hitpoints[updated],
            directions=fuzzy_reflect(
                directions[updated],
                normals[updated],
                fuzzes[indices[updated] - 1, np.newaxis]),
            centers=centers,
            radia=radia,
            fuzzes=fuzzes,
            colors=colors,
            max_depth=max_depth - 1
        ) * colors[indices[updated] - 1]
    return env


@njit
def dot_batch(a: npt.NDArray, b: npt.NDArray) -> npt.NDArray:
    dotted = np.zeros(len(a))
    for i in range(len(a)):
        dotted[i] = np.dot(a[i], b[i])
    return dotted


@njit
def reflect(directions: npt.NDArray, normals: npt.NDArray) -> npt.NDArray:
    reflect_direction = (
        directions - 2 * normals *
        dot_batch(directions, normals)[:, np.newaxis]
    )
    return normalize(reflect_direction)


@njit
def fuzzy_reflect(directions: npt.NDArray, normals: npt.NDArray, fuzz: float | npt.NDArray = 0.7) -> npt.NDArray:
    reflect_direction = (
        directions -
        2 * normals *
        dot_batch(directions, normals)[:, np.newaxis] +
        random_in_unit_sphere(directions.shape[0]) * fuzz
    )
    return normalize(reflect_direction)

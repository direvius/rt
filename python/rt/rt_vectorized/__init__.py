from __future__ import annotations
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt  # type: ignore
from typing import Annotated, Literal
from attrs import frozen, field
from loguru import logger

width = 400
height = 200

Vector3 = Annotated[npt.NDArray[np.float64], Literal[3]]


@frozen
class Rays:
    origins: npt.NDArray
    directions: npt.NDArray


@frozen
class Spheres:
    centers: npt.NDArray
    radia: npt.NDArray[np.float64]


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

        return Rays(np.zeros((width * height, 3)), directions)

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
        plt.imshow(np.swapaxes(env.reshape(width, height, 3), 0, 1))
        plt.savefig("test.png")


@frozen
class SceneBuilder:

    centers: list[Vector3] = field(factory=list)
    radia: list[np.float64] = field(factory=list)

    def add_sphere(self, cx: float, cy: float, cz: float, r: float) -> SceneBuilder:
        self.centers.append(np.array([cx, cy, cz]))
        self.radia.append(np.float64(r))
        return self

    def create(self) -> Spheres:
        return Spheres(np.array(self.centers), np.array(self.radia))

    def camera(self) -> Camera:
        return Camera(self.create(), jitter_passes=64)


def env_colors(rays: Rays) -> npt.NDArray:
    t = (rays.directions[:, 1] + 1) * 0.5
    env_colors = (1 - t)[:, np.newaxis] * np.ones([len(rays.directions), 3]) + t[:, np.newaxis] * np.array([0.5, 0.7, 1])
    return env_colors


def hit(rays: Rays, spheres: Spheres) -> npt.NDArray:
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
        t[hits] = -b[hits] - np.sqrt(d[hits]) / 2
        t[t < 0.001] = np.inf
        should_update = t < acc
        acc[should_update] = t[should_update]
        indices[should_update] = i + 1
        hitpoints[should_update] = origins[should_update] + rays.directions[should_update] * t[should_update, np.newaxis]
        normals[should_update] = hitpoints[should_update] - center
        normals[should_update] /= np.einsum("ij,ij->i", normals[should_update], normals[should_update])[:, np.newaxis]
    updated = indices != 0
    updated_len = sum(updated)
    reflected_directions = (
        rays.directions[updated] -
        2 * normals[updated] *
        np.einsum(
            "ij,ij->i",
            rays.directions[updated],
            normals[updated]
        )[:, np.newaxis]
    )
    if updated_len > 0:
        env[updated] = hit(
            Rays(
                origins=hitpoints[updated],
                directions=reflected_directions
            ),
            spheres=spheres
        ) / 1.5
    return env

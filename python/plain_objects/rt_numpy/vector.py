from typing import Annotated, Literal
import numpy.typing as npt
import numpy as np

Vector3 = Annotated[npt.NDArray[np.float64], Literal[3]]


def normalize(v: Vector3) -> Vector3:
    return v / np.linalg.norm(v)


def random_in_unit_sphere() -> Vector3:
    while True:
        v = 2 * np.random.rand(3) - np.ones(3)
        if np.linalg.norm(v) < 1:
            return v

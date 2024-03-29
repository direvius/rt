import pytest
import numpy as np
from . import random_in_unit_sphere, reflect, normalize


def test_normalize():
    vectors = np.array(
        [
            [1, 0, 0],
            [2, 0, 0],
            [1, 1, 0],
            [25, 25, 0],
        ],
        dtype=np.float64
    )

    expected_vectors = normalize(np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [0.7071067812, 0.7071067812, 0],
            [0.7071067812, 0.7071067812, 0],
        ],
        dtype=np.float64
    ))
    np.testing.assert_allclose(normalize(vectors), expected_vectors)


@pytest.mark.parametrize(
    "direction,normal,expected,msg",
    [
        ([1, 0, 0], [-1, 0, 0], [-1, 0, 0], "horizontal reflection"),
        ([2, 0, 0], [-1, 0, 0], [-1, 0, 0], "horizontal reflection and normalization"),
        ([1, 1, 0], [-1, 0, 0], [-1, 1, 0], "horizontal reflection with vertical component"),
        ([1, 0, 0], [-1, -1, 0], [0, -1, 0], "right to down reflection"),
        ([1, 0, 0], [1, 0, 0], [-1, 0, 0], "horizontal reflection (back side)"),
    ]
)
def test_reflection(direction, normal, expected, msg):
    directions = np.array([direction], dtype=np.float64)
    normals = normalize(np.array([normal], dtype=np.float64))
    expected_directions = normalize(np.array([expected], dtype=np.float64))
    np.testing.assert_allclose(
        reflect(directions=directions, normals=normals),
        expected_directions,
        atol=1e-10,
        err_msg=msg
    )


def test_random_in_unit_sphere():
    random = random_in_unit_sphere(1000)

    assert random.shape == (1000, 3)

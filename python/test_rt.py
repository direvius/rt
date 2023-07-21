import pytest
from rt import Vector3


class TestVector:
    def test_add(self):
        a = Vector3(0, 1, 2)
        b = Vector3(2, 3, 4)
        result = a + b
        expected = Vector3(2, 4, 6)
        assert result == expected

    def test_sub(self):
        a = Vector3(0.1, 1, 2)
        b = Vector3(2, 3, 4)
        result = a - b
        expected = Vector3(-1.9, -2, -2)
        assert result == expected

    def test_abs(self):
        a = Vector3(2, -3, 4)
        result = abs(a)
        expected = 5.3851648071
        assert result == pytest.approx(expected)

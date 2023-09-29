from __future__ import annotations
from rt_py import SceneBuilder, Camera, Material


def main():
    scene = (
        SceneBuilder()
        .add_sphere(0, 0, -1.5, 0.5, Material.random())  # central
        .add_sphere(1, 0, -1.5, 0.3, Material.random())  # right
        .add_sphere(0.3, 0, -1, 0.1, Material.random())
        .add_sphere(0.8, -0.3, -1.5, 0.1, Material.random())
        .add_sphere(-1, 0, -1.5, 0.3, Material.random())  # left
        .add_sphere(-2, 0, -1.5, 0.1, Material.random())
        .add_sphere(0, -100.5, -1.5, 100, Material.random())  # The Earth
        .create()
    )
    camera = Camera(scene, jitter_passes=64)
    camera.write_png()


if __name__ == "__main__":
    main()

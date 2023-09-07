from __future__ import annotations
from rt import SceneBuilder, Camera, Materials


def main():
    scene = (
        SceneBuilder()
        .add_sphere(0, 0, -1, 0.5, Materials.METAL.value)
        .add_sphere(0, -100.5, -1, 100, Materials.METAL.value)
        .create()
    )
    camera = Camera(scene)
    camera.write_png()


if __name__ == "__main__":
    main()

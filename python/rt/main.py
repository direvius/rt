from __future__ import annotations
import click
from rt_numpy import SceneBuilder as NumpySceneBuilder
from rt_py import SceneBuilder as PlainSceneBuilder
from rt_vectorized import SceneBuilder as VectorSceneBuilder
import time
from loguru import logger


ALGORITHMS = {
    "numpy": NumpySceneBuilder,
    "plain": PlainSceneBuilder,
    "vector": VectorSceneBuilder,
}


@click.command()
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice((*ALGORITHMS,)),
    help="Raytracing implementation",
    default="plain",
)
def main(algorithm: str):
    logger.info("Using {} implementation. Creating scene...", algorithm)
    camera = (
        ALGORITHMS[algorithm]()
        .add_sphere(0, 0, -1.5, 0.5)  # central
        .add_sphere(1, 0, -1.5, 0.3)  # right
        .add_sphere(0.3, 0, -1, 0.1)
        .add_sphere(0.8, -0.3, -1.5, 0.1)
        .add_sphere(-1, 0, -1.5, 0.3)  # left
        .add_sphere(-2, 0, -1.5, 0.1)
        .add_sphere(0, -100.5, -1.5, 100)  # The Earth
        .camera()
    )
    logger.info("Tracing...")
    start_time = time.perf_counter()
    camera.write_png()
    end_time = time.perf_counter()
    logger.info("Tracing finished. It took {:.3f} seconds", end_time - start_time)


if __name__ == "__main__":
    main()

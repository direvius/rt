from __future__ import annotations
import click
from rt_numpy import SceneBuilder as NumpySceneBuilder
from rt_py import SceneBuilder as PlainSceneBuilder
import logging
import time


logger = logging.getLogger(__name__)


ALGORITHMS = {
    "numpy": NumpySceneBuilder,
    "plain": PlainSceneBuilder,
}


@click.command()
@click.option(
    "--algorithm",
    "-a",
    type=click.Choice(["numpy", "plain"]),
    help="Raytracing implementation",
    default="plain",
)
def main(algorithm: str):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger.info(f"Using {algorithm} implementation. Creating scene...")
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
    logger.info(f"Tracing finished. It took {end_time - start_time:.3f} seconds")


if __name__ == "__main__":
    main()

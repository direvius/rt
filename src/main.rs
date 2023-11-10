use crate::geometry::Material;
use crate::geometry::{Geometry, Sphere};
use crate::render::Camera;
use cgmath::Vector3;
use image::{buffer::ConvertBuffer, RgbImage};

mod geometry;
mod render;

fn main() {
    let mut g: Geometry = Default::default();
    let c: Camera = Default::default();

    g
        .add_sphere(0.0, 0.0, -1.5, 0.5)
        .add_sphere(1.0, 0.0, -1.5, 0.3)
        .add_sphere(0.3, 0.0, -1.0, 0.1)
        .add_sphere(0.8, -0.3, -1.5, 0.1)
        .add_sphere(-1.0, 0.0, -1.5, 0.3)
        .add_sphere(-2.0, 0.0, -1.5, 0.1)
        .add_sphere(0.0, -100.5, -1.5, 100.0);

    let ib: RgbImage = c.render(&g).convert();

    ib.save("test.png").unwrap();
}

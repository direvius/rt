use crate::geometry::{Geometry, Sphere};
use crate::render::Camera;
use cgmath::Vector3;
use image::{buffer::ConvertBuffer, RgbImage};

mod geometry;
mod render;

fn main() {
    let mut g: Geometry = Default::default();
    let c: Camera = Default::default();

    g.bodies.push(Box::new(Sphere {
        center: Vector3::new(0.0f32, 0.0, -1.0),
        radius: 0.5,
    }));

    g.bodies.push(Box::new(Sphere {
        center: Vector3::new(0.0f32, -100.5, -1.0),
        radius: 100.0f32,
    }));

    let ib: RgbImage = c.render(&g).convert();

    ib.save("test.png").unwrap();
}

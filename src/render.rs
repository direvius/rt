use crate::geometry::Scalar;
use crate::geometry::{Geometry, Hit, HitResult, Ray};
use cgmath::{InnerSpace, Vector3};
use image::{ImageBuffer, Rgb};
use rand::random;
use std::iter::repeat_with;

pub struct AASettings {
    passes: usize,
}

pub struct Camera {
    width: u32,
    height: u32,
    origin: Vector3<Scalar>,
    lower_left: Vector3<Scalar>,
    ux: Vector3<Scalar>,
    uy: Vector3<Scalar>,
    aa_settings: AASettings,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            width: 400,
            height: 200,
            lower_left: Vector3::new(-2.0, -1.0, -1.0),
            origin: Vector3::new(0.0, 0.0, 0.0),
            ux: Vector3::new(4.0, 0.0, 0.0),
            uy: Vector3::new(0.0, 2.0, 0.0),
            aa_settings: AASettings { passes: 64 },
        }
    }
}

impl Camera {
    pub fn get_ray(&self, i: u32, j: u32) -> Ray {
        let (u, v) = (
            i as Scalar / self.width as Scalar,
            1.0 - j as Scalar / self.height as Scalar,
        );
        let x_jitter = random::<Scalar>() / self.width as Scalar;
        let y_jitter = random::<Scalar>() / self.height as Scalar;
        Ray {
            origin: self.origin,
            direction: self.lower_left + self.ux * (u + x_jitter) + self.uy * (v + y_jitter),
        }
    }
    fn env_color(r: Ray) -> Vector3<Scalar> {
        let t = (r.direction.normalize().y + 1.0) * 0.5;
        (1.0 - t) * Vector3::new(1.0, 1.0, 1.0) + t * Vector3::new(0.5, 0.7, 1.0)
    }
    fn random_in_unit_sphere() -> Vector3<Scalar> {
        loop {
            let v = 2.0f32
                * Vector3::new(random::<Scalar>(), random::<Scalar>(), random::<Scalar>())
                - Vector3::new(1.0, 1.0, 1.0);
            if v.magnitude2() < 1.0 {
                return v;
            }
        }
    }
    fn reflect(v: Vector3<Scalar>, n: Vector3<Scalar>) -> Vector3<Scalar> {
        v - 2.0f32 * n * v.dot(n)
    }
    fn color_normal(r: Ray, g: &Geometry) -> Vector3<Scalar> {
        if let Some(HitResult { p: _, n, t: _ }) = g.hit(r) {
            (n + Vector3::new(1.0f32, 1.0, 1.0)) / 2.0
        } else {
            Self::env_color(r)
        }
    }
    fn color_diffuse(r: Ray, g: &Geometry) -> Vector3<Scalar> {
        if let Some(HitResult { p, n, t: _ }) = g.hit(r) {
            let target = p + n + Self::random_in_unit_sphere();
            Self::color_diffuse(
                Ray {
                    origin: p,
                    direction: target - p,
                },
                g,
            ) / 2.0
        } else {
            Self::env_color(r)
        }
    }
    fn color_metal(r: Ray, g: &Geometry) -> Vector3<Scalar> {
        if let Some(HitResult { p, n, t: _ }) = g.hit(r) {
            let reflected =
                Self::reflect(r.direction.normalize(), n) + Self::random_in_unit_sphere() * 0.1;
            Self::color_metal(
                Ray {
                    origin: p,
                    direction: reflected,
                },
                g,
            ) / 1.5
        } else {
            Self::env_color(r)
        }
    }
    pub fn render(&self, g: &Geometry) -> ImageBuffer<Rgb<Scalar>, Vec<Scalar>> {
        let mut imgbuf = ImageBuffer::new(self.width, self.height);

        for (i, j, p) in imgbuf.enumerate_pixels_mut() {
            let rgbv3 = repeat_with(|| self.get_ray(i, j))
                .take(self.aa_settings.passes)
                .map(|r| Self::color_metal(r, g))
                .fold(Vector3::new(0.0f32, 0.0, 0.0), |acc, p| acc + p);
            *p = Rgb((rgbv3 / self.aa_settings.passes as f32).into());
        }
        imgbuf
    }
}

use cgmath::{InnerSpace, Vector3};

pub type Scalar = f32;

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vector3<Scalar>,
    pub direction: Vector3<Scalar>,
}

impl Ray {
    pub fn point_at(&self, p: Scalar) -> Vector3<Scalar> {
        self.direction * p + self.origin
    }
}

pub struct Material {
    attenuation: Scalar,
    refraction: Scalar,
    fuzz: Scalar,
}

pub struct Geometry {
    pub bodies: Vec<Box<dyn Hit>>,
}

impl Default for Geometry {
    fn default() -> Self {
        Self { bodies: Vec::new() }
    }
}

impl Hit for Geometry {
    fn hit(&self, r: Ray) -> Option<HitResult> {
        self.bodies
            .iter()
            .fold(None, |accum, item| -> Option<HitResult> {
                match item.hit(r) {
                    Some(b) => match accum {
                        None => Some(b),
                        Some(a) => {
                            if a.t < b.t {
                                Some(a)
                            } else {
                                Some(b)
                            }
                        }
                    },
                    None => accum,
                }
            })
    }
}

pub struct HitResult {
    pub p: Vector3<Scalar>,
    pub t: Scalar,
    pub n: Vector3<Scalar>,
}

pub trait Hit {
    fn hit(&self, r: Ray) -> Option<HitResult>;
}

pub trait Scatter {
    fn scatter(&self, r: Ray) -> Vec<Ray>;
}

pub struct Sphere {
    pub center: Vector3<Scalar>,
    pub radius: Scalar,
    pub material: Material,
}

impl Scatter for Sphere {
    fn scatter(&self, r: Ray) -> Vec<Ray> {
        todo!()
    }
}

impl Hit for Sphere {
    fn hit(&self, r: Ray) -> Option<HitResult> {
        let oc = r.origin - self.center;
        let a = r.direction.dot(r.direction);
        let b = oc.dot(r.direction) * 2.0;
        let c = oc.dot(oc) - self.radius * self.radius;
        let d = b * b - a * c * 4.0;
        if d >= 0.0 {
            let t = (-b - d.sqrt()) / (a * 2.0);
            if t > 0.001 {
                Some(HitResult {
                    p: r.point_at(t),
                    n: (r.point_at(t) - self.center).normalize(),
                    t,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

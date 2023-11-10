use cgmath::{InnerSpace, Vector3};
use rand::Rng;

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

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub attenuation: Vector3<Scalar>,
    pub fuzz: Scalar,
}

pub struct Geometry {
    pub bodies: Vec<Box<dyn Hit>>,
}

impl Geometry {
    pub fn add_sphere(&mut self, x: Scalar, y: Scalar, z: Scalar, r: Scalar) -> &mut Geometry {
        let mut rng = rand::thread_rng();
        self.bodies.push(Box::new(Sphere {
            center: Vector3::new(x, y, z),
            radius: r,
            material: Material {
                attenuation: Vector3::new(rng.gen(), rng.gen(), rng.gen()),
                fuzz: rng.gen(),
            },
        }));
        return self;
    }
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
    pub m: Material,
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
                    m: self.material,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}

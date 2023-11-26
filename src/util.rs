use std::cell::{RefCell, RefMut};

use nalgebra::SMatrix;
use rand::{rngs::SmallRng, Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use super::model::network::{ActivityVector, WeightMatrix};

pub struct Random {
    rng: RefCell<SmallRng>,
    activity_noise: Normal<f32>,
    weight_noise: Normal<f32>,
}

impl Random {
    pub fn new(activity_noise: f32, weight_noise: f32, seed: Option<u64>) -> Random {
        let rng = if let Some(seed) = seed {
            SmallRng::seed_from_u64(seed)
        } else {
            SmallRng::from_entropy()
        };

        Random {
            rng: RefCell::new(rng),
            activity_noise: Normal::new(0.0, activity_noise).unwrap(),
            weight_noise: Normal::new(0.0, weight_noise).unwrap(),
        }
    }

    pub fn rng(&self) -> RefMut<impl Rng> {
        self.rng.borrow_mut()
    }

    fn noisify<const N: usize, const M: usize>(
        &self,
        dist: impl Distribution<f32>,
        matrix: &SMatrix<f32, { N }, { M }>,
    ) -> SMatrix<f32, { N }, { M }> {
        let mut rng = self.rng.borrow_mut();
        matrix.map(|x| (x + dist.sample(&mut *rng)).clamp(0.0, 1.0))
    }

    pub fn noisify_weights<const N: usize, const M: usize>(
        &self,
        weights: &WeightMatrix<{ N }, { M }>,
    ) -> WeightMatrix<{ N }, { M }> {
        self.noisify(&self.weight_noise, weights)
    }

    pub fn noisify_activity<const N: usize>(
        &self,
        activity: &ActivityVector<{ N }>,
    ) -> ActivityVector<{ N }> {
        self.noisify(&self.activity_noise, activity)
    }

    pub fn noisy_sigmoid<const N: usize>(
        &self,
        activity: &ActivityVector<{ N }>,
        slope: f32,
        bias: f32,
    ) -> ActivityVector<{ N }> {
        self.noisify(
            &self.activity_noise,
            &activity.map(|x| 1.0 / (1.0 + (-(x * slope - bias)).exp())),
        )
    }
}

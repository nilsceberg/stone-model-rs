use nalgebra::{SMatrix, SVector};
use rand_distr::{Distribution, Normal};

pub struct Noise {
    activity_noise: Normal<f32>,
    weight_noise: Normal<f32>,
}

impl Noise {
    pub fn new(activity_noise: f32, weight_noise: f32) -> Noise {
        Noise {
            activity_noise: Normal::new(0.0, activity_noise).unwrap(),
            weight_noise: Normal::new(0.0, weight_noise).unwrap(),
        }
    }

    fn noisify<const N: usize, const M: usize>(
        dist: impl Distribution<f32>,
        matrix: &SMatrix<f32, { N }, { M }>,
    ) -> SMatrix<f32, { N }, { M }> {
        matrix.map(|x| (x + dist.sample(&mut rand::thread_rng())).clamp(0.0, 1.0))
    }

    pub fn noisify_weights<const N: usize, const M: usize>(
        &self,
        weights: &WeightMatrix<{ N }, { M }>,
    ) -> WeightMatrix<{ N }, { M }> {
        Self::noisify(&self.weight_noise, weights)
    }

    pub fn noisify_activity<const N: usize>(
        &self,
        activity: &ActivityVector<{ N }>,
    ) -> ActivityVector<{ N }> {
        Self::noisify(&self.activity_noise, activity)
    }

    pub fn noisy_sigmoid<const N: usize>(
        &self,
        activity: &ActivityVector<{ N }>,
        slope: f32,
        bias: f32,
    ) -> ActivityVector<{ N }> {
        Self::noisify(&self.activity_noise, &activity.map(|x| 1.0 / (1.0 + (-(x * slope - bias)).exp())))
    }
}

pub type ActivityVector<const N: usize> = SVector<f32, N>;
pub type WeightMatrix<const N: usize, const M: usize> = SMatrix<f32, N, M>;

pub trait Weights<const TO: usize, const FROM: usize> {
    /// Weights may dynamically change depending on input.
    fn update(&mut self, input: &ActivityVector<FROM>) -> &WeightMatrix<TO, FROM>;
}

pub trait Layer<const N: usize> {
    fn update(&mut self, input: ActivityVector<N>, noise: &Noise) -> ActivityVector<N>;
}

#[derive(Clone)]
pub struct StaticWeights<const TO: usize, const FROM: usize>(pub &'static WeightMatrix<TO, FROM>);

impl<const TO: usize, const FROM: usize> StaticWeights<TO, FROM> {
    pub fn noisy(noise: &Noise, weights: &WeightMatrix<TO, FROM>) -> StaticWeights<TO, FROM> {
        let weights = Box::new(noise.noisify_weights(weights));
        StaticWeights(Box::leak(weights))
    }

    pub fn matrix(&self) -> &'static WeightMatrix<TO, FROM> {
        self.0
    }
}

impl<const TO: usize, const FROM: usize> Weights<TO, FROM> for StaticWeights<TO, FROM> {
    fn update(&mut self, _input: &ActivityVector<FROM>) -> &'static WeightMatrix<TO, FROM> {
        self.matrix()
    }
}

impl<const TO: usize, const FROM: usize> From<&'static WeightMatrix<TO, FROM>>
    for StaticWeights<TO, FROM>
{
    fn from(value: &'static WeightMatrix<TO, FROM>) -> Self {
        StaticWeights(value)
    }
}

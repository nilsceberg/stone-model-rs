use nalgebra::{SMatrix, SVector};

use crate::util::Random;

pub type ActivityVector<const N: usize> = SVector<f32, N>;
pub type WeightMatrix<const N: usize, const M: usize> = SMatrix<f32, N, M>;

pub trait Weights<const TO: usize, const FROM: usize> {
    /// Weights may dynamically change depending on input.
    fn update(&mut self, input: &ActivityVector<FROM>) -> &WeightMatrix<TO, FROM>;
}

pub trait Layer<const N: usize> {
    fn update(&mut self, input: ActivityVector<N>, noise: &Random) -> ActivityVector<N>;
}

#[derive(Clone)]
pub struct StaticWeights<const TO: usize, const FROM: usize>(pub &'static WeightMatrix<TO, FROM>);

impl<const TO: usize, const FROM: usize> StaticWeights<TO, FROM> {
    pub fn noisy(random: &Random, weights: &WeightMatrix<TO, FROM>) -> StaticWeights<TO, FROM> {
        let weights = Box::new(random.noisify_weights(weights));
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

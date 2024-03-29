use super::{constants::N_CPU4, Config, CX};
use nalgebra::SVector;

/// Reference implementation of memory acculumating in the CPU4 cells as in the Stone et al. (2017) paper.
pub mod reference {
    use nalgebra::SVector;

    use crate::{
        model::{
            constants::{self, CPU4_MEM_FADE, N_CPU4},
            network::{ActivityVector, Layer},
            Config, CX,
        },
        util::Random,
    };

    use super::MemoryRecorder;

    pub struct AbstractCpu4 {
        memory: ActivityVector<N_CPU4>,
    }

    impl AbstractCpu4 {
        pub fn new() -> AbstractCpu4 {
            AbstractCpu4 {
                memory: ActivityVector::repeat(0.5),
            }
        }
    }

    impl Layer<N_CPU4> for AbstractCpu4 {
        fn update(
            &mut self,
            input: ActivityVector<N_CPU4>,
            random: &Random,
        ) -> ActivityVector<N_CPU4> {
            let mem_update = input.map(|x| x.clamp(0.0, 1.0) - CPU4_MEM_FADE);
            self.memory =
                (&self.memory + mem_update * constants::CPU4_MEM_GAIN).map(|x| x.clamp(0.0, 1.0));

            random.noisy_sigmoid(
                &self.memory,
                constants::CPU4_SLOPE_TUNED,
                constants::CPU4_BIAS_TUNED,
            )
        }
    }

    pub struct AbstractMemoryRecorder;
    impl<C: Config<Cpu4Layer = AbstractCpu4>> MemoryRecorder<C> for AbstractMemoryRecorder {
        fn record(cx: &CX<C>) -> SVector<f32, N_CPU4> {
            cx.cpu4_layer.memory
        }
    }
}

pub mod weights {
    use std::fmt::Debug;

    use nalgebra::SVector;

    use crate::{
        model::{
            constants::{self, N_CPU4},
            network::{ActivityVector, Layer, WeightMatrix, Weights},
            Config, CX,
        },
        util::Random,
    };

    use super::MemoryRecorder;

    pub trait Dynamics: Clone {
        fn dwdt(&self, w: f32, r: f32) -> f32;
    }

    #[derive(Clone)]
    pub struct AffineDynamics {
        pub beta: f32,
    }
    impl Dynamics for AffineDynamics {
        fn dwdt(&self, _w: f32, r: f32) -> f32 {
            const H: f32 = 0.0025;
            let k: f32 = 0.125 + self.beta;
            H * (-k + r)
        }
    }

    #[derive(Clone)]
    pub struct LogisticDynamics {
        pub h: f32,
    }
    impl Dynamics for LogisticDynamics {
        fn dwdt(&self, w: f32, r: f32) -> f32 {
            self.h * r * w * (1.0 - w)
        }
    }

    pub struct DynamicWeights<D: Dynamics, const TO: usize, const FROM: usize> {
        dynamics: D,
        connectivity: &'static WeightMatrix<TO, FROM>,
        weights: WeightMatrix<TO, FROM>,
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> DynamicWeights<D, TO, FROM> {
        pub fn new(
            dynamics: &D,
            connectivity: &'static WeightMatrix<TO, FROM>,
            initial: WeightMatrix<TO, FROM>,
        ) -> Self {
            Self {
                dynamics: dynamics.clone(),
                connectivity,
                weights: initial.component_mul(connectivity),
            }
        }
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> Weights<TO, FROM>
        for DynamicWeights<D, TO, FROM>
    {
        fn update(&mut self, input: &ActivityVector<FROM>) -> &WeightMatrix<TO, FROM> {
            // Each row in the weight matrix represents one synapse per input rate,
            // so each row gets element-wise multiplied with the connectivity and the current weights.
            //let signal = self.connectivity * WeightMatrix::from_diagonal(input);
            //self.weights += self
            //    .weights
            //    .zip_map(&signal, |w, r| self.dynamics.dwdt(w, r))
            //    .component_mul(self.connectivity);

            for j in 0..FROM {
                for i in 0..TO {
                    let index = j * TO + i;
                    self.weights[index] = (self.weights[index]
                        + self.connectivity[index]
                            * self.dynamics.dwdt(self.weights[index], input[j]))
                    .clamp(0.0, 1.0);
                }
            }

            &self.weights
        }

        fn matrix(&self) -> &WeightMatrix<TO, FROM> {
            &self.weights
        }
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> Debug for DynamicWeights<D, TO, FROM> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(format_args!("{:?}\n", self.weights))
            //debug_struct("DynamicWeights").field("dynamics", &self.dynamics).field("connectivity", &self.connectivity).field("weights", &self.weights).finish()
        }
    }

    pub struct StatelessCpu4 {
        beta: f32,
    }

    impl StatelessCpu4 {
        pub fn new(beta: f32) -> Self {
            Self { beta }
        }
    }

    impl Layer<N_CPU4> for StatelessCpu4 {
        fn update(
            &mut self,
            input: ActivityVector<N_CPU4>,
            random: &Random,
        ) -> ActivityVector<N_CPU4> {
            random
                .noisy_sigmoid(
                    &input,
                    constants::CPU4_SLOPE_TUNED,
                    constants::CPU4_BIAS_TUNED,
                )
                .map(|x| (x + self.beta).clamp(0.0, 1.0))
        }
    }

    pub struct PontineWeightMemoryRecorder;
    impl<C: Config> MemoryRecorder<C> for PontineWeightMemoryRecorder {
        fn record(cx: &CX<C>) -> SVector<f32, N_CPU4> {
            cx.w_cpu4_pontine.matrix().diagonal()
        }
    }
}

pub trait MemoryRecorder<C: Config> {
    fn record(cx: &CX<C>) -> SVector<f32, N_CPU4>;
}

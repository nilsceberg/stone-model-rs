/// Reference implementation of memory acculumating in the CPU4 cells as in the Stone et al. (2017) paper.
pub mod reference {
    use crate::{
        model::{
            constants::{self, CPU4_MEM_FADE, N_CPU4},
            network::{ActivityVector, Layer},
        },
        util::Random,
    };

    pub struct AbstractCpu4 {
        memory: ActivityVector<{ N_CPU4 }>,
    }

    impl AbstractCpu4 {
        pub fn new() -> AbstractCpu4 {
            AbstractCpu4 {
                memory: ActivityVector::repeat(0.5),
            }
        }
    }

    impl Layer<{ N_CPU4 }> for AbstractCpu4 {
        fn update(
            &mut self,
            input: ActivityVector<{ N_CPU4 }>,
            random: &Random,
        ) -> ActivityVector<{ N_CPU4 }> {
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
}

pub mod weights {
    use std::fmt::Debug;

    use crate::{
        model::{
            constants::{self, N_CPU4},
            network::{ActivityVector, Layer, WeightMatrix, Weights},
        },
        util::Random,
    };

    pub trait Dynamics: Clone {
        fn dwdt(&self, w: f32, r: f32) -> f32;
    }

    #[derive(Clone)]
    pub struct LinearDynamics {
        pub beta: f32,
    }
    impl Dynamics for LinearDynamics {
        fn dwdt(&self, _w: f32, r: f32) -> f32 {
            const H: f32 = 0.0025;
            let k: f32 = 0.125 + self.beta;
            H * (-k + r)
        }
    }

    //#[derive(Clone)]
    //pub struct LogisticDynamics {
    //    pub : f32,
    //}
    //impl Dynamics for LinearDynamics {
    //    fn dwdt(&self, _w: f32, r: f32) -> f32 {
    //        const H: f32 = 0.0025;
    //        let k: f32 = 0.125 + self.beta;
    //        H * (-k + r)
    //    }
    //}

    pub struct DynamicWeights<D: Dynamics, const TO: usize, const FROM: usize> {
        dynamics: D,
        connectivity: &'static WeightMatrix<{ TO }, { FROM }>,
        weights: WeightMatrix<{ TO }, { FROM }>,
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> DynamicWeights<D, { TO }, { FROM }> {
        pub fn new(
            dynamics: &D,
            connectivity: &'static WeightMatrix<{ TO }, { FROM }>,
            initial: WeightMatrix<{ TO }, { FROM }>,
        ) -> Self {
            Self {
                dynamics: dynamics.clone(),
                connectivity,
                weights: initial.component_mul(connectivity),
            }
        }
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> Weights<{ TO }, { FROM }>
        for DynamicWeights<D, { TO }, { FROM }>
    {
        fn update(&mut self, input: &ActivityVector<{ FROM }>) -> &WeightMatrix<{ TO }, { FROM }> {
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
                    self.weights[index] += self.connectivity[index]
                        * self.dynamics.dwdt(self.weights[index], input[j]);
                }
            }

            &self.weights
        }
    }

    impl<D: Dynamics, const TO: usize, const FROM: usize> Debug
        for DynamicWeights<D, { TO }, { FROM }>
    {
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

    impl Layer<{ N_CPU4 }> for StatelessCpu4 {
        fn update(
            &mut self,
            input: ActivityVector<{ N_CPU4 }>,
            random: &Random,
        ) -> ActivityVector<{ N_CPU4 }> {
            // Like in the thesis:
            random
                .noisy_sigmoid(
                    &input,
                    constants::CPU4_SLOPE_TUNED,
                    constants::CPU4_BIAS_TUNED,
                )
                .map(|x| (x + self.beta).clamp(0.0, 1.0))

            // TODO: Does this also work?
            // let mut input = input * (1.0 - self.beta);
            // input.add_scalar_mut(self.beta);
            // random.noisy_sigmoid(
            //     &input,
            //     constants::CPU4_SLOPE_TUNED,
            //     constants::CPU4_BIAS_TUNED,
            // )
        }
    }
}

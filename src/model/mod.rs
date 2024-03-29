//! Ported rate-based CX with pontine cells and swappable CPU4 layer and downstream weights, from Stone et al. (2017).

pub mod connectomics;
pub mod constants;
pub mod memory;
pub mod network;

use nalgebra::{matrix, SVector, Vector2};
use ndarray::{prelude::*, Axis};

use crate::{
    movement::PhysicalState,
    util::{self, Random},
};
use constants::{N_CL1, N_CPU1A, N_CPU1B, N_CPU4, N_PONTINE, N_TB1, N_TL2, N_TN1, N_TN2};
use network::*;

use self::{constants::N_AMP, memory::MemoryRecorder};

pub trait Config: Sized {
    type Cpu4Layer: Layer<N_CPU4>;
    type Cpu4AmpWeights: Weights<N_AMP, N_CPU4>;
    type Cpu4PontineWeights: Weights<N_PONTINE, N_CPU4>;
    type AmpLayer: Layer<N_AMP>;
    type MemoryRecorder: MemoryRecorder<Self>;
}

pub struct CX<'a, C: Config> {
    pub w_cl1_tb1: StaticWeights<N_TB1, N_CL1>,
    pub w_tb1_tb1: StaticWeights<N_TB1, N_TB1>,

    pub w_tb1_cpu1a: StaticWeights<N_CPU1A, N_TB1>,
    pub w_tb1_cpu1b: StaticWeights<N_CPU1B, N_TB1>,
    pub w_tb1_cpu4: StaticWeights<N_CPU4, N_TB1>,

    pub w_tn1_cpu4: StaticWeights<N_CPU4, N_TN1>,
    pub w_tn2_cpu4: StaticWeights<N_CPU4, N_TN2>,

    pub w_cpu4_pontine: C::Cpu4PontineWeights,
    pub w_cpu4_amp: C::Cpu4AmpWeights,

    pub w_pontine_amp: StaticWeights<N_AMP, N_PONTINE>,

    pub w_amp_cpu1a: StaticWeights<N_CPU1A, N_AMP>,
    pub w_amp_cpu1b: StaticWeights<N_CPU1B, N_AMP>,

    pub w_cpu1a_motor: StaticWeights<2, N_CPU1A>,
    pub w_cpu1b_motor: StaticWeights<2, N_CPU1B>,

    pub tb1: ActivityVector<N_TB1>,
    pub cpu4_layer: C::Cpu4Layer,
    pub amp_layer: C::AmpLayer,

    turn_sharpness: f32,

    tn_prefs: f32,
    tl2_prefs: SVector<f32, N_TL2>,
    random: &'a Random,
}

impl<'a, C: Config> CX<'a, C> {
    pub fn new(
        random: &'a Random,
        turn_sharpness: f32,
        cpu4: C::Cpu4Layer,
        amp: C::AmpLayer,
        w_cpu4_amp: C::Cpu4AmpWeights,
        w_cpu4_pontine: C::Cpu4PontineWeights,
    ) -> Self {
        CX {
            w_cl1_tb1: StaticWeights::noisy(random, &connectomics::W_CL1_TB1),

            w_tb1_tb1: StaticWeights::noisy(random, &connectomics::generate_tb_tb_weights()),
            w_tb1_cpu1a: StaticWeights::noisy(random, &connectomics::W_TB1_CPU1A),
            w_tb1_cpu1b: StaticWeights::noisy(random, &connectomics::W_TB1_CPU1B),
            w_tb1_cpu4: StaticWeights::noisy(random, &connectomics::W_TB1_CPU4),

            w_tn1_cpu4: StaticWeights::noisy(random, &connectomics::W_TN1_CPU4),
            w_tn2_cpu4: StaticWeights::noisy(random, &connectomics::W_TN2_CPU4),

            w_cpu4_amp,
            w_cpu4_pontine,

            w_pontine_amp: StaticWeights::noisy(random, &connectomics::W_PONTINE_AMP),

            w_amp_cpu1a: StaticWeights::noisy(random, &connectomics::W_AMP_CPU1A),
            w_amp_cpu1b: StaticWeights::noisy(random, &connectomics::W_AMP_CPU1B),

            w_cpu1a_motor: StaticWeights::noisy(random, &connectomics::W_CPU1A_MOTOR),
            w_cpu1b_motor: StaticWeights::noisy(random, &connectomics::W_CPU1B_MOTOR),

            tb1: ActivityVector::zeros(),
            cpu4_layer: cpu4,
            amp_layer: amp,

            turn_sharpness,

            tn_prefs: std::f32::consts::PI / 4.0,
            tl2_prefs: Self::generate_tl2_prefs(),
            random,
        }
    }

    pub fn update(&mut self, physical_state: &PhysicalState) -> f32 {
        // Sensory inputs: heading
        let tl2 = self.tl2_output(physical_state.heading);
        let cl1 = self.cl1_output(&tl2);

        // Sensory inputs: optical flow / speed
        let flow = self.get_flow(physical_state);
        let tn1 = self.tn1_output(&flow);
        let tn2 = self.tn2_output(&flow);

        // Compass ring attractor
        self.tb1 = self.tb1_output(&cl1);

        // Allocentric re-projection
        let cpu4 = self.cpu4_update(&tn1, &tn2);

        // Steering system
        let pontine = self.pontine_output(&cpu4);
        let amp = self.amp_output(&cpu4, &pontine);
        let cpu1a = self.cpu1a_output(&amp);
        let cpu1b = self.cpu1b_output(&amp);

        self.turn_sharpness * self.motor_output(&cpu1a, &cpu1b)
    }

    fn generate_tl2_prefs() -> SVector<f32, N_TL2> {
        let tl2_prefs = Array::linspace(0.0, 2.0 * std::f32::consts::PI, constants::N_TB1 + 1);
        let tl2_prefs = ndarray::concatenate(
            Axis(0),
            &[tl2_prefs.slice(s![..-1]), tl2_prefs.slice(s![..-1])],
        )
        .unwrap();
        SVector::<f32, N_TL2>::from_vec(tl2_prefs.into_raw_vec())
    }

    fn get_flow(&self, PhysicalState { heading, velocity }: &PhysicalState) -> Vector2<f32> {
        // TODO: figure out what this is supposed to be (currently opposite Stone?)
        let right = heading - self.tn_prefs;
        let left = heading + self.tn_prefs;
        let sensitivity = matrix![
            right.sin(), right.cos();
            left.sin(), left.cos();
        ];
        sensitivity * velocity
    }

    fn tl2_output(&self, heading: f32) -> ActivityVector<N_TL2> {
        let input = self.tl2_prefs.map(|pref| (heading - pref).cos());
        self.random.noisy_sigmoid(
            &input,
            constants::TL2_SLOPE_TUNED,
            constants::TL2_BIAS_TUNED,
        )
    }

    fn cl1_output(&self, tl2: &ActivityVector<N_TL2>) -> ActivityVector<N_CL1> {
        let input = -tl2;
        self.random.noisy_sigmoid(
            &input,
            constants::CL1_SLOPE_TUNED,
            constants::CL1_BIAS_TUNED,
        )
    }

    fn tb1_output(&self, cl1: &ActivityVector<N_CL1>) -> ActivityVector<N_TB1> {
        let prop_cl1 = 0.667f32;
        let prop_tb1 = 1.0 - prop_cl1;

        let input = prop_cl1 * self.w_cl1_tb1.matrix() * cl1
            - prop_tb1 * self.w_tb1_tb1.matrix() * self.tb1;

        self.random.noisy_sigmoid(
            &input,
            constants::TB1_SLOPE_TUNED,
            constants::TB1_BIAS_TUNED,
        )
    }

    fn tn1_output(&self, flow: &Vector2<f32>) -> ActivityVector<N_TN2> {
        self.random
            .noisify_activity(&(flow.map(|x| (1.0 - x) / 2.0)))
    }

    fn tn2_output(&self, flow: &Vector2<f32>) -> ActivityVector<N_TN2> {
        self.random.noisify_activity(flow)
    }

    fn cpu4_update(
        &mut self,
        _tn1: &ActivityVector<N_TN1>,
        tn2: &ActivityVector<N_TN2>,
    ) -> ActivityVector<N_CPU4> {
        let input = self.w_tn2_cpu4.matrix() * tn2 - self.w_tb1_cpu4.matrix() * self.tb1;

        self.cpu4_layer.update(input, &self.random)
    }

    fn pontine_output(&mut self, cpu4: &ActivityVector<N_CPU4>) -> ActivityVector<N_PONTINE> {
        let input = self.w_cpu4_pontine.update(&cpu4) * cpu4;

        // The activation function has been changed from a sigmoid
        // that is approximately linear in [0, 1] to a rectified linear curve
        // to work with the amplification layer, which requires the means of the inputs to cancel out.
        // To be fair to the original model, it gets this update as well.

        //self.random.noisy_linear(
        util::activation::linear(
            &input,
            constants::PONTINE_SLOPE_TUNED_LINEAR,
            constants::PONTINE_BIAS_TUNED_LINEAR,
        )
    }

    fn amp_output(
        &mut self,
        cpu4: &ActivityVector<N_CPU4>,
        pontine: &ActivityVector<N_PONTINE>,
    ) -> ActivityVector<N_AMP> {
        let input = 0.5 * self.w_cpu4_amp.update(&cpu4) * cpu4
            - 0.5 * self.w_pontine_amp.matrix() * pontine;

        self.amp_layer.update(input, &self.random)
    }

    fn cpu1a_output(&mut self, amp: &ActivityVector<N_AMP>) -> ActivityVector<N_CPU1A> {
        let input = self.w_amp_cpu1a.update(&amp) * amp - self.w_tb1_cpu1a.matrix() * self.tb1;

        self.random.noisy_sigmoid(
            &input,
            constants::CPU1_SLOPE_TUNED,
            constants::CPU1_BIAS_TUNED,
        )
    }

    fn cpu1b_output(&mut self, amp: &ActivityVector<N_AMP>) -> ActivityVector<N_CPU1B> {
        let input = self.w_amp_cpu1b.update(&amp) * amp - self.w_tb1_cpu1b.matrix() * self.tb1;

        self.random.noisy_sigmoid(
            &input,
            constants::CPU1_SLOPE_TUNED,
            constants::CPU1_BIAS_TUNED,
        )
    }

    fn motor_output(
        &self,
        cpu1a: &ActivityVector<N_CPU1A>,
        cpu1b: &ActivityVector<N_CPU1B>,
    ) -> f32 {
        let motor = self.w_cpu1a_motor.matrix() * cpu1a + self.w_cpu1b_motor.matrix() * cpu1b;
        motor[0] - motor[1]
    }
}

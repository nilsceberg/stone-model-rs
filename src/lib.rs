use serde::Serialize;
use std::marker::PhantomData;

use model::{
    connectomics::{W_CPU4_AMP, W_CPU4_PONTINE},
    constants::{AMP_BIAS_TUNED, AMP_SLOPE_TUNED, N_AMP, N_CPU4, N_PONTINE},
    memory::{
        self,
        reference::AbstractMemoryRecorder,
        weights::{Dynamics, LinearDynamics, LogisticDynamics, PontineWeightMemoryRecorder},
        MemoryRecorder,
    },
    network::{PassthroughLayer, SigmoidLayer, StaticWeights, WeightMatrix},
    Config, CX,
};
use movement::{PhysicalState, DEFAULT_DRAG};
use nalgebra::SVector;
use util::Random;

pub mod model;
pub mod movement;
pub mod stats;
pub mod util;

pub const COMMON_SEED: Option<u64> = Some(64172527321326);

pub struct ReferenceConfig;
impl model::Config for ReferenceConfig {
    type Cpu4Layer = memory::reference::AbstractCpu4;
    type AmpLayer = PassthroughLayer;
    type Cpu4AmpWeights = StaticWeights<N_AMP, N_CPU4>;
    type Cpu4PontineWeights = StaticWeights<N_PONTINE, N_CPU4>;
    type MemoryRecorder = AbstractMemoryRecorder;
}

pub fn create_reference_cx<'a>(random: &'a Random) -> CX<'a, ReferenceConfig> {
    CX::new(
        random,
        0.25,
        memory::reference::AbstractCpu4::new(),
        PassthroughLayer,
        StaticWeights::noisy(random, &W_CPU4_AMP),
        StaticWeights::noisy(random, &W_CPU4_PONTINE),
    )
}

pub struct WeightConfig<D: Dynamics>(PhantomData<D>);
impl<D: Dynamics> model::Config for WeightConfig<D> {
    type Cpu4Layer = memory::weights::StatelessCpu4;
    type AmpLayer = PassthroughLayer;
    type Cpu4AmpWeights = memory::weights::DynamicWeights<D, N_AMP, N_CPU4>;
    type Cpu4PontineWeights = memory::weights::DynamicWeights<D, N_PONTINE, N_CPU4>;
    type MemoryRecorder = PontineWeightMemoryRecorder;
}

pub struct WeightAmpConfig<D: Dynamics>(PhantomData<D>);
impl<D: Dynamics> model::Config for WeightAmpConfig<D> {
    type Cpu4Layer = memory::weights::StatelessCpu4;
    type AmpLayer = SigmoidLayer;
    type Cpu4AmpWeights = memory::weights::DynamicWeights<D, N_AMP, N_CPU4>;
    type Cpu4PontineWeights = memory::weights::DynamicWeights<D, N_PONTINE, N_CPU4>;
    type MemoryRecorder = PontineWeightMemoryRecorder;
}

pub fn create_weight_cx<'a, D: Dynamics>(
    random: &'a Random,
    dynamics: &D,
    beta: f32,
    initial_weight: f32,
    turn_sharpness: f32,
) -> CX<'a, WeightConfig<D>> {
    CX::new(
        random,
        turn_sharpness,
        memory::weights::StatelessCpu4::new(beta),
        PassthroughLayer,
        memory::weights::DynamicWeights::new(
            dynamics,
            &W_CPU4_AMP,
            WeightMatrix::repeat(initial_weight),
        ),
        memory::weights::DynamicWeights::new(
            dynamics,
            &W_CPU4_PONTINE,
            WeightMatrix::repeat(initial_weight),
        ),
    )
}

pub fn create_weight_linear_cx<'a>(
    random: &'a Random,
    beta: f32,
) -> CX<'a, WeightConfig<LinearDynamics>> {
    let dynamics = LinearDynamics { beta };
    let initial_weight = 0.5;
    CX::new(
        random,
        0.25,
        memory::weights::StatelessCpu4::new(beta),
        PassthroughLayer,
        memory::weights::DynamicWeights::new(
            &dynamics,
            &W_CPU4_AMP,
            WeightMatrix::repeat(initial_weight),
        ),
        memory::weights::DynamicWeights::new(
            &dynamics,
            &W_CPU4_PONTINE,
            WeightMatrix::repeat(initial_weight),
        ),
    )
}

pub fn create_weight_logistic_cx<'a>(
    random: &'a Random,
    h: f32,
    w0: f32,
    beta: f32,
) -> CX<'a, WeightConfig<LogisticDynamics>> {
    let dynamics = LogisticDynamics { h };
    CX::new(
        random,
        0.25,
        memory::weights::StatelessCpu4::new(beta),
        PassthroughLayer,
        memory::weights::DynamicWeights::new(&dynamics, &W_CPU4_AMP, WeightMatrix::repeat(w0)),
        memory::weights::DynamicWeights::new(&dynamics, &W_CPU4_PONTINE, WeightMatrix::repeat(w0)),
    )
}

pub fn create_weight_logistic_amp_cx<'a>(
    random: &'a Random,
    h: f32,
    w0: f32,
    beta: f32,
) -> CX<'a, WeightAmpConfig<LogisticDynamics>> {
    let dynamics = LogisticDynamics { h };
    CX::new(
        random,
        -0.25,
        memory::weights::StatelessCpu4::new(beta),
        SigmoidLayer {
            slope: AMP_SLOPE_TUNED,
            bias: AMP_BIAS_TUNED,
        },
        memory::weights::DynamicWeights::new(&dynamics, &W_CPU4_AMP, WeightMatrix::repeat(w0)),
        memory::weights::DynamicWeights::new(&dynamics, &W_CPU4_PONTINE, WeightMatrix::repeat(w0)),
    )
}

#[derive(Clone, Serialize)]
pub struct Setup {
    pub inbound_steps: usize,
    pub outbound_steps: usize,
    pub vary_speed: bool,
    pub acceleration_out: f32,
    pub acceleration_in: f32,
    pub record_memory: bool,
}

impl Setup {
    pub fn generate_outbound(&self, random: &Random) -> Vec<PhysicalState> {
        // Generate an outbound path
        let rng = &mut *random.rng();
        movement::generate_outbound(
            rng,
            self.outbound_steps,
            self.acceleration_out,
            self.vary_speed,
        )
    }
}

#[derive(Serialize)]
pub struct FlightData {
    pub setup: Setup,
    pub physical_states: Vec<PhysicalState>,
    pub memory_record: Option<Vec<SVector<f32, N_CPU4>>>,
}

impl FlightData {
    pub fn print(&self) {
        println!("{}", serde_json::to_string(&self).unwrap());
    }
}

pub fn run_homing_trial<C: Config>(
    setup: &Setup,
    cx: &mut CX<C>,
    outbound: Vec<PhysicalState>,
) -> FlightData {
    let mut physical_states = outbound;
    let mut memory_record: Option<Vec<SVector<f32, N_CPU4>>> = if setup.record_memory {
        Some(Vec::with_capacity(
            setup.outbound_steps + setup.inbound_steps,
        ))
    } else {
        None
    };

    // Simulate the agent flying along the outbound path
    for state in
        &physical_states[..physical_states.len()-1 /* leave last state for start of homing */]
    {
        cx.update(&state);

        if let Some(ref mut memory_record) = memory_record {
            memory_record.push(C::MemoryRecorder::record(&cx));
        }
    }

    // Let the agent home, using the model's motor output to steer
    physical_states.reserve(setup.inbound_steps);
    for _ in 0..setup.inbound_steps {
        let physical_state = physical_states.last().unwrap();
        let motor = cx.update(physical_state);
        physical_states.push(physical_state.next(motor, setup.acceleration_in, DEFAULT_DRAG));

        if let Some(ref mut memory_record) = memory_record {
            memory_record.push(C::MemoryRecorder::record(&cx));
        }
    }

    FlightData {
        setup: setup.clone(),
        physical_states,
        memory_record,
    }
}

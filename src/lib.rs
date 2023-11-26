use model::{
    connectomics::{W_CPU4_CPU1A, W_CPU4_CPU1B, W_CPU4_PONTINE},
    constants::{N_CPU1A, N_CPU1B, N_CPU4, N_PONTINE},
    network::StaticWeights,
    AbstractCpu4, Config, CX,
};
use movement::{PhysicalState, DEFAULT_DRAG};
use util::Random;

pub mod model;
pub mod movement;
pub mod util;

pub struct ReferenceConfig;
impl model::Config for ReferenceConfig {
    type Cpu4Layer = AbstractCpu4;
    type Cpu4Cpu1aWeights = StaticWeights<{ N_CPU1A }, { N_CPU4 }>;
    type Cpu4Cpu1bWeights = StaticWeights<{ N_CPU1B }, { N_CPU4 }>;
    type Cpu4PontineWeights = StaticWeights<{ N_PONTINE }, { N_CPU4 }>;
}

pub fn create_reference_cx<'a>(random: &'a Random) -> CX<'a, ReferenceConfig> {
    CX::new(
        random,
        1.0,
        AbstractCpu4::new(),
        StaticWeights::noisy(random, &W_CPU4_CPU1A),
        StaticWeights::noisy(random, &W_CPU4_CPU1B),
        StaticWeights::noisy(random, &W_CPU4_PONTINE),
    )
}

pub struct Setup {
    pub inbound_steps: usize,
    pub outbound_steps: usize,
    pub vary_speed: bool,
    pub acceleration_out: f32,
    pub acceleration_in: f32,
}

pub struct Result {
    pub physical_states: Vec<PhysicalState>,
}

pub fn run_homing_trial(cx: &mut CX<impl Config>, random: &Random, setup: &Setup) -> Result {
    // Generate an outbound path
    let mut physical_states = {
        let rng = &mut *random.rng();
        movement::generate_outbound(rng, setup.outbound_steps, setup.acceleration_out, setup.vary_speed)
    };

    // Simulate the agent flying along it
    for state in &physical_states[..physical_states.len()-1 /* leave last state for start of homing */] {
        cx.update(&state);
    }

    // Let the agent home, using the model's motor output to steer
    physical_states.reserve(setup.inbound_steps);
    for _ in 0..setup.inbound_steps {
        let physical_state = physical_states.last().unwrap();
        let motor = cx.update(physical_state);
        physical_states.push(physical_state.next(motor, setup.acceleration_in, DEFAULT_DRAG));
    }

    Result { physical_states }
}
